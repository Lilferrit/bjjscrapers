#!/usr/bin/env python3
"""
Smoothcomp Match Scraper

Produces one Parquet row per completed match for BJJ events on smoothcomp.com
within a given year range.

Data comes from the server-rendered matchlist pages:
    /en/event/{id}/schedule/matchlist?page=N

Category headers encode belt, weight class, age division, and gi/no-gi.
Each match yields: winner/loser name, ID, team; score; result method; duration.

Note: Weigh-in (measured) weight is NOT available without authentication.
      weight_class from the category header IS included.

Usage:
    smoothcomp-bjj --min-year 2023
    smoothcomp-bjj --min-year 2022 --max-year 2023 --output matches_2022_2023
    smoothcomp-bjj --min-year 2025 --max-id 21600 --max-probe 10 --output test_run
"""

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import platformdirs
import polars as pl
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

SMOOTHCOMP_BASE = "https://smoothcomp.com"
CACHE_PATH = Path(platformdirs.user_cache_dir("bjjnerdshit")) / "smoothcomp_cache.json"

NOGI_KEYWORDS = {"no-gi", "no gi", "nogi", "no kimono", "submission only"}

# Lowercase English belt names as they appear on Smoothcomp
BELT_NAMES = frozenset(
    {
        "white",
        "grey",
        "gray",  # gray = US spelling of grey
        "yellow-grey",
        "yellow-gray",
        "yellow",
        "orange",
        "green-orange",
        "green-gray",
        "green",
        "blue",
        "purple",
        "brown",
        "black",
    }
)

BELT_TRANSLATIONS = {
    "white": "White",
    "grey": "Grey",
    "gray": "Grey",  # US spelling
    "yellow-grey": "Yellow-Grey",
    "yellow-gray": "Yellow-Grey",
    "yellow": "Yellow",
    "orange": "Orange",
    "green-orange": "Green-Orange",
    "green-gray": "Green-Grey",
    "green": "Green",
    "blue": "Blue",
    "purple": "Purple",
    "brown": "Brown",
    "black": "Black",
}

GENDER_MALE_WORDS = frozenset({"boys", "men", "male", "masculino", "masc", "man"})
GENDER_FEMALE_WORDS = frozenset(
    {"girls", "women", "female", "feminino", "fem", "woman"}
)

# Canonical result method names
_METHOD_MAP = {
    "points": "points",
    "advantage": "advantages",
    "advantages": "advantages",
    "submission": "submission",
    "sub": "submission",
    "dq": "disqualification",
    "disqualification": "disqualification",
    "referee": "decision",
    "decision": "decision",
    "walkover": "walkover",
    "w/o": "walkover",
    "forfeit": "forfeit",
    "no contest": "no_contest",
    "draw": "draw",
}

_SCHEMA = {
    "data_source": pl.Utf8,  # always "smoothcomp"
    "event_id": pl.Utf8,
    "event_name": pl.Utf8,
    "event_year": pl.Int32,
    "event_date": pl.Utf8,  # date string as shown on event page
    "gi": pl.Boolean,
    "gender": pl.Utf8,
    "age_division": pl.Utf8,
    "belt": pl.Utf8,
    "weight_class": pl.Utf8,
    "category_raw": pl.Utf8,  # full unparsed header for debugging
    "result_method": pl.Utf8,  # "points", "submission", "disqualification", …
    "score_winner": pl.Int32,  # first competitor's score (assumed winner)
    "score_loser": pl.Int32,
    "match_duration": pl.Utf8,  # "MM:SS"
    "winner_id": pl.Utf8,  # smoothcomp profile ID
    "winner_name": pl.Utf8,
    "winner_team": pl.Utf8,
    "loser_id": pl.Utf8,
    "loser_name": pl.Utf8,
    "loser_team": pl.Utf8,
    "bracket_id": pl.Utf8,
    "is_absolute": pl.Boolean,  # True when weight_class is absent (open-class bracket)
    "winner_inferred_weight": pl.Utf8,  # nullable — only set for is_absolute rows
    "loser_inferred_weight": pl.Utf8,  # nullable
    "winner_inferred_belt": pl.Utf8,  # nullable — only set for is_absolute rows
    "loser_inferred_belt": pl.Utf8,  # nullable
}

logger = logging.getLogger(__name__)


def _setup_logging(log_path: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8")],
    )


# ---------------------------------------------------------------------------
# HTTP layer  (mirrors scrape_absolute.py RateLimitedSession)
# ---------------------------------------------------------------------------


class RateLimitedSession:
    """requests.Session with per-request delay and exponential-backoff retry."""

    def __init__(self, delay: float = 1.0, retries: int = 3):
        self._session = requests.Session()
        self._session.headers["User-Agent"] = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
        self.delay = delay
        self.retries = retries

    def get(self, url: str) -> Optional[requests.Response]:
        time.sleep(self.delay)
        for attempt in range(1, self.retries + 1):
            try:
                resp = self._session.get(url, timeout=30)
                if resp.status_code == 200:
                    return resp
                if resp.status_code == 404:
                    return None
                if resp.status_code in (503, 504):
                    logger.warning("server overloaded (%s): %s", resp.status_code, url)
                    return None
                if resp.status_code >= 500:
                    wait = 30 * attempt
                    logger.warning(
                        "HTTP %s, retry in %ss: %s", resp.status_code, wait, url
                    )
                    time.sleep(wait)
            except (requests.ConnectionError, requests.Timeout) as exc:
                wait = 30 * attempt
                logger.warning(
                    "connection error (%s), retry in %ss: %s", exc, wait, url
                )
                time.sleep(wait)
        return None


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"events": {}}


def save_cache(cache: dict) -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "w") as f:
            json.dump(cache, f, indent=2)
    except OSError as exc:
        logger.warning("could not save cache: %s", exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_year(text: str) -> Optional[int]:
    m = re.search(r"\b(20\d{2})\b", text)
    return int(m.group(1)) if m else None


def _is_nogi(text: str) -> bool:
    return any(kw in text.lower() for kw in NOGI_KEYWORDS)


def _normalize_belt(raw: str) -> str:
    return BELT_TRANSLATIONS.get(raw.lower().strip(), raw.strip())


def _normalize_result_method(text: str) -> str:
    """Map 'Won by points', 'Won by Submission', etc. to canonical form."""
    t = re.sub(r"^won\s+by\s+", "", text.lower().strip()).rstrip(".")
    for key, val in _METHOD_MAP.items():
        if key in t:
            return val
    return t


def _parse_category_header(text: str) -> Optional[dict]:
    """
    Parse a Smoothcomp category header such as:
        'Boys Gi / White / PEE WEE 1 (4 & 5 yrs) / -21 kg (Saturday)'
    Returns dict with gi, gender, belt, age_division, weight_class, category_raw.
    Returns None if it doesn't look like a valid category header (no belt found).
    """
    parts = [p.strip() for p in text.split("/")]
    if len(parts) < 3:
        return None

    gi = not _is_nogi(parts[0])

    first_words = set(parts[0].lower().split())
    if first_words & GENDER_MALE_WORDS:
        gender = "Male"
    elif first_words & GENDER_FEMALE_WORDS:
        gender = "Female"
    else:
        gender = ""

    belt = ""
    weight_class = ""
    age_division = ""
    belt_idx = -1
    weight_idx = -1

    for i, part in enumerate(parts):
        p_low = part.lower().strip()
        if p_low in BELT_NAMES:
            belt = _normalize_belt(part)
            belt_idx = i
        elif re.search(r"\d+(?:\.\d+)?\s*(?:kg|lbs)|(?:above|below)\s+\d+", p_low):
            # Strip trailing parenthetical (day-of-week, label, etc.)
            weight_class = re.sub(r"\s*\([^)]*\)\s*$", "", part).strip()
            weight_idx = i

    if not belt:
        return None

    # Age division = the part that isn't index 0, belt, or weight
    for i, part in enumerate(parts):
        if i in (0, belt_idx, weight_idx):
            continue
        candidate = re.sub(r"\s*\([^)]*\)\s*$", "", part).strip()
        if candidate:
            age_division = candidate
            break

    return {
        "gi": gi,
        "gender": gender,
        "belt": belt,
        "age_division": age_division,
        "weight_class": weight_class,
        "category_raw": text,
    }


# ---------------------------------------------------------------------------
# Matchlist page parser
# ---------------------------------------------------------------------------

_PROFILE_ID_RE = re.compile(r"/profile/(\d+)")
_BRACKET_RE = re.compile(r"/bracket/(\d+)")
_SCORE_RE = re.compile(r"(\d+)\s*-\s*(\d+)")
_DURATION_RE = re.compile(r"\b(\d{1,2}:\d{2})\b")
_METHOD_RE = re.compile(r"won\s+by\s+\w+(?:\s+\w+)?", re.IGNORECASE)
_ABSOLUTE_RE = re.compile(
    r"\b(?:absolute|absoluto|open\s+class|open\s+weight)\b", re.IGNORECASE
)


def _participant_name(span) -> str:
    """Direct text node(s) of a span.participant, excluding child span text."""
    return "".join(t for t in span.children if isinstance(t, str)).strip()


def _parse_matchlist_page(html: str, event_info: dict) -> list[dict]:
    """
    Parse one matchlist page into match row dicts.

    Smoothcomp matchlist HTML structure (server-side rendered):
      div.matches-list
        div.category-row   ← category metadata for the next match
        div.match-row       ← one completed match
          div.number        ← "winner_score-loser_score"
          span.participant  ← first competitor (winner — has span.text-success)
            span.club       ← team name
            span.text-success ← "Won by X - MM:SS"
          span.participant  ← second competitor (loser)
            span.club
          div.collapse
            a.profile       ← links for profile IDs (in participant order)
            a[href*=bracket] ← bracket link
    """
    soup = BeautifulSoup(html, "lxml")

    matches_list = soup.find("div", class_="matches-list")
    if not matches_list:
        logger.debug(
            "event %s: no matches-list div on page", event_info.get("event_id")
        )
        return []

    rows: list[dict] = []
    current_category: dict = {}

    for child in matches_list.children:
        if not hasattr(child, "get") or child.name != "div":
            continue
        classes: list[str] = child.get("class", [])

        # ---- Category header ----
        if "category-row" in classes:
            raw = re.sub(r"\s+", " ", child.get_text(" ", strip=True)).strip()
            cat = _parse_category_header(raw)
            if cat:
                current_category = cat
                logger.debug(
                    "event %s: category %s", event_info.get("event_id"), raw[:80]
                )
            else:
                logger.debug(
                    "event %s: unparsed category %r",
                    event_info.get("event_id"),
                    raw[:80],
                )
            continue

        # ---- Match row ----
        if "match-row" not in classes:
            continue

        # Score
        num_div = child.find("div", class_="number")
        score_str = num_div.get_text(strip=True) if num_div else ""

        # Competitors
        participants = child.find_all("span", class_="participant")
        if len(participants) < 2:
            logger.warning(
                "event %s: match-row with %d participant(s)",
                event_info.get("event_id"),
                len(participants),
            )
            continue

        # Winner = participant with span.text-success; loser = the other
        ts_spans = [p for p in participants if p.find("span", class_="text-success")]
        if ts_spans:
            winner_span = ts_spans[0]
            loser_span = next(p for p in participants if p is not winner_span)
        else:
            # Both unfinished or bye — treat first as winner
            winner_span, loser_span = participants[0], participants[1]

        winner_name = _participant_name(winner_span)
        loser_name = _participant_name(loser_span)
        winner_club = winner_span.find("span", class_="club")
        loser_club = loser_span.find("span", class_="club")
        winner_team = winner_club.get_text(strip=True) if winner_club else ""
        loser_team = loser_club.get_text(strip=True) if loser_club else ""

        # Result method + duration
        result_method = ""
        match_duration = ""
        ts = winner_span.find("span", class_="text-success")
        if ts:
            ts_text = re.sub(r"\s+", " ", ts.get_text(strip=True))
            mm = _METHOD_RE.search(ts_text)
            if mm:
                result_method = _normalize_result_method(mm.group(0))
            dm = _DURATION_RE.search(ts_text)
            if dm:
                match_duration = dm.group(1)

        # Score → assign winner/loser based on which participant came first
        score_winner: Optional[int] = None
        score_loser: Optional[int] = None
        sm = _SCORE_RE.search(score_str)
        if sm:
            a_score, b_score = int(sm.group(1)), int(sm.group(2))
            if participants[0] is winner_span:
                score_winner, score_loser = a_score, b_score
            else:
                score_winner, score_loser = b_score, a_score

        # Profile IDs and bracket from collapsed section (same participant order)
        winner_id = ""
        loser_id = ""
        bracket_id = ""
        collapse_div = child.find("div", class_="collapse")
        if collapse_div:
            profile_links = collapse_div.find_all("a", class_="profile")
            # Profile links follow the original participant order, not winner-first
            ordered = [None, None]
            for i, pl in enumerate(profile_links[:2]):
                href = pl.get("href", "")
                m = _PROFILE_ID_RE.search(href)
                if m:
                    ordered[i] = m.group(1)

            if participants[0] is winner_span:
                winner_id, loser_id = ordered[0] or "", ordered[1] or ""
            else:
                winner_id, loser_id = ordered[1] or "", ordered[0] or ""

            bl = collapse_div.find("a", href=lambda h: h and "bracket" in h)
            if bl:
                bm = _BRACKET_RE.search(bl.get("href", ""))
                if bm:
                    bracket_id = bm.group(1)

        if not winner_name or not loser_name:
            logger.warning(
                "event %s: empty name — winner=%r loser=%r",
                event_info.get("event_id"),
                winner_name,
                loser_name,
            )
            continue

        rows.append(
            {
                "data_source": "smoothcomp",
                "event_id": event_info["event_id"],
                "event_name": event_info["event_name"],
                "event_year": event_info.get("event_year"),
                "event_date": event_info.get("event_date", ""),
                "gi": current_category.get("gi"),
                "gender": current_category.get("gender", ""),
                "age_division": current_category.get("age_division", ""),
                "belt": current_category.get("belt", ""),
                "weight_class": current_category.get("weight_class", ""),
                "category_raw": current_category.get("category_raw", ""),
                "result_method": result_method,
                "score_winner": score_winner,
                "score_loser": score_loser,
                "match_duration": match_duration,
                "winner_id": winner_id,
                "winner_name": winner_name,
                "winner_team": winner_team,
                "loser_id": loser_id,
                "loser_name": loser_name,
                "loser_team": loser_team,
                "bracket_id": bracket_id,
            }
        )

    return rows


def _count_pages(html: str) -> int:
    """Find the highest page number in pagination links."""
    soup = BeautifulSoup(html, "lxml")
    max_page = 1
    for link in soup.find_all("a", href=re.compile(r"[?&]page=(\d+)")):
        m = re.search(r"[?&]page=(\d+)", link["href"])
        if m:
            n = int(m.group(1))
            if n > max_page:
                max_page = n
    return max_page


# ---------------------------------------------------------------------------
# Event info and match fetching
# ---------------------------------------------------------------------------


def _fetch_event_info(session: RateLimitedSession, event_id: int) -> Optional[dict]:
    """
    Fetch /en/event/{id} and return metadata dict.
    Returns None if the event doesn't exist (404).
    """
    url = f"{SMOOTHCOMP_BASE}/en/event/{event_id}"
    resp = session.get(url)
    if resp is None:
        return None

    soup = BeautifulSoup(resp.text, "lxml")
    name = ""

    # og:title is most reliable; strip "- Smoothcomp" suffix
    og = soup.find("meta", {"property": "og:title"})
    if og and og.get("content"):
        name = re.sub(r"\s*[-|]\s*[Ss]moothcomp.*$", "", og["content"]).strip()

    if not name:
        title_el = soup.find("title")
        if title_el:
            name = re.sub(
                r"\s*[-|]\s*[Ss]moothcomp.*$", "", title_el.get_text(strip=True)
            ).strip()

    if not name:
        h1 = soup.find("h1")
        if h1:
            name = h1.get_text(strip=True)

    if not name:
        name = f"event_{event_id}"
        logger.debug("could not extract name for event %d", event_id)

    year = _extract_year(name)
    if not year:
        year = _extract_year(soup.get_text())

    date_str = ""
    # Normalize whitespace before searching for date patterns
    page_text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
    for pat in (
        r"\d{1,2}\s+\w{3,9}\s*[-–]\s*\d{1,2}\s+\w{3,9}\s+\d{4}",
        r"\d{1,2}\s+\w{3,9}\s+\d{4}",
        r"\d{1,2}\s+\w{3}\s*[-–]\s*\d{1,2}\s+\w{3}",
    ):
        dm = re.search(pat, page_text)
        if dm:
            date_str = re.sub(r"\s+", " ", dm.group(0)).strip()
            break

    return {
        "event_id": str(event_id),
        "event_name": name,
        "event_year": year,
        "event_date": date_str,
    }


def _fetch_event_matches(
    session: RateLimitedSession,
    event_info: dict,
) -> list[dict]:
    """Paginate through all matchlist pages for one event."""
    event_id = event_info["event_id"]
    base_url = f"{SMOOTHCOMP_BASE}/en/event/{event_id}/schedule/matchlist"

    resp = session.get(f"{base_url}?page=1")
    if resp is None:
        logger.warning("event %s: matchlist page 1 unavailable", event_id)
        return []

    total_pages = _count_pages(resp.text)
    logger.info(
        "event %s (%s): %d page(s)", event_id, event_info["event_name"], total_pages
    )

    rows = _parse_matchlist_page(resp.text, event_info)

    for page in tqdm(
        range(2, total_pages + 1),
        desc=f"  {event_info['event_name'][:35]}",
        leave=False,
        disable=(total_pages <= 1),
        file=sys.stderr,
    ):
        resp = session.get(f"{base_url}?page={page}")
        if resp is None:
            logger.warning("event %s: page %d unavailable", event_id, page)
            continue
        rows.extend(_parse_matchlist_page(resp.text, event_info))

    logger.info("event %s: %d match rows collected", event_id, len(rows))
    return rows


# ---------------------------------------------------------------------------
# Event discovery
# ---------------------------------------------------------------------------


def _discover_events(
    session: RateLimitedSession,
    min_year: int,
    max_year: int,
    max_id: int,
    max_probe: int,
    sport_filter: str,
    cache: dict,
    max_events: Optional[int] = None,
) -> list[dict]:
    """
    Probe event IDs from max_id downward to find events in [min_year, max_year].

    Stops after max_probe consecutive IDs with year < min_year - 1 (well before
    the target range), to avoid scanning the entire ID history.
    Stops early once max_events matching events are found.
    All probed IDs are cached in cache['events'] to speed up subsequent runs.
    """
    events_cache = cache.setdefault("events", {})
    found: list[dict] = []
    consecutive_old = 0

    pbar = tqdm(desc="Discovering events", unit="id", file=sys.stderr)

    for event_id in range(max_id, 0, -1):
        pbar.update(1)
        key = str(event_id)

        if key in events_cache:
            info = events_cache[key]
        else:
            info = _fetch_event_info(session, event_id)
            events_cache[key] = info
            if len(events_cache) % 50 == 0:
                save_cache(cache)

        if info is None:
            # 404 — event doesn't exist, doesn't affect age counter
            continue

        year = info.get("event_year")
        name = info.get("event_name", "")

        if year is None:
            logger.debug(
                "event %d: no year found in %r — skipping", event_id, name[:60]
            )
            continue

        # Count how far back we've gone; stop if too old
        if year < min_year - 1:
            consecutive_old += 1
            if consecutive_old >= max_probe:
                logger.info(
                    "Stopping discovery: %d consecutive events before %d",
                    max_probe,
                    min_year - 1,
                )
                break
            continue
        else:
            consecutive_old = 0

        # Apply sport keyword filter on event name
        if sport_filter and sport_filter.lower() not in name.lower():
            continue

        if min_year <= year <= max_year:
            found.append(info)
            pbar.set_postfix(found=len(found))
            logger.info("event %d: %s (%d) — adding", event_id, name, year)
            if max_events is not None and len(found) >= max_events:
                logger.info(
                    "Stopping discovery: reached --max-events limit (%d)", max_events
                )
                break

    pbar.close()
    save_cache(cache)
    tqdm.write(
        f"Discovery complete: {len(found)} event(s) in [{min_year}–{max_year}]",
        file=sys.stderr,
    )
    return found


# ---------------------------------------------------------------------------
# DataFrame builder
# ---------------------------------------------------------------------------


def build_dataframe(rows: list[dict]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=_SCHEMA)
    for row in rows:
        for col in _SCHEMA:
            row.setdefault(col, None)
    return pl.from_dicts(rows, schema=_SCHEMA)


def _infer_absolute_fields(df: pl.DataFrame) -> pl.DataFrame:
    """
    Populate is_absolute and per-competitor inferred weight/belt columns.

    is_absolute uses keyword detection on category_raw ("absolute", "open class",
    etc.) rather than relying on empty weight_class, which can be empty for
    unusual weight formats that the parser misses.
    Inferred weight/belt come from the same competitor's non-absolute bracket
    appearance at the same event — pure post-processing, no extra HTTP requests.
    Inference columns are null for non-absolute rows.
    """
    df = df.with_columns(
        pl.col("category_raw")
        .str.contains("(?i)" + _ABSOLUTE_RE.pattern, literal=False)
        .alias("is_absolute")
    )

    has_weight = df.filter(~pl.col("is_absolute"))
    weight_map = (
        pl.concat(
            [
                has_weight.select(
                    ["event_id", "winner_id", "weight_class", "belt"]
                ).rename({"winner_id": "competitor_id"}),
                has_weight.select(
                    ["event_id", "loser_id", "weight_class", "belt"]
                ).rename({"loser_id": "competitor_id"}),
            ]
        )
        .filter(pl.col("competitor_id").is_not_null() & (pl.col("competitor_id") != ""))
        .unique(subset=["event_id", "competitor_id"], keep="first")
    )

    # Drop the all-null placeholders so joins can add real values
    df = df.drop(
        [
            "winner_inferred_weight",
            "loser_inferred_weight",
            "winner_inferred_belt",
            "loser_inferred_belt",
        ]
    )

    df = df.join(
        weight_map.rename(
            {
                "competitor_id": "winner_id",
                "weight_class": "winner_inferred_weight",
                "belt": "winner_inferred_belt",
            }
        ),
        on=["event_id", "winner_id"],
        how="left",
    )
    df = df.join(
        weight_map.rename(
            {
                "competitor_id": "loser_id",
                "weight_class": "loser_inferred_weight",
                "belt": "loser_inferred_belt",
            }
        ),
        on=["event_id", "loser_id"],
        how="left",
    )

    # Null out inferred columns for non-absolute rows
    df = df.with_columns(
        [
            pl.when(pl.col("is_absolute"))
            .then(pl.col("winner_inferred_weight"))
            .otherwise(None)
            .alias("winner_inferred_weight"),
            pl.when(pl.col("is_absolute"))
            .then(pl.col("loser_inferred_weight"))
            .otherwise(None)
            .alias("loser_inferred_weight"),
            pl.when(pl.col("is_absolute"))
            .then(pl.col("winner_inferred_belt"))
            .otherwise(None)
            .alias("winner_inferred_belt"),
            pl.when(pl.col("is_absolute"))
            .then(pl.col("loser_inferred_belt"))
            .otherwise(None)
            .alias("loser_inferred_belt"),
        ]
    )

    return df.select(list(_SCHEMA.keys()))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scrape Smoothcomp match results → Parquet.\n\n"
            "Probes event IDs starting from --max-id (default 25000) downward,\n"
            "collecting all matches from events in [--min-year, --max-year].\n"
            f"Discovered event IDs are cached to {CACHE_PATH}.\n\n"
            "First run is slow (up to --max-id HTTP requests); subsequent runs\n"
            "reuse the cache and only fetch matchlist pages for new events.\n\n"
            "Weigh-in (measured) weight is not available without authentication."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--min-year",
        type=int,
        required=True,
        help="Earliest event year to include.",
    )
    parser.add_argument(
        "--max-year",
        type=int,
        default=datetime.now().year,
        help="Latest event year to include (default: current year).",
    )
    parser.add_argument(
        "--output",
        default="smoothcomp_results",
        help=(
            "Output file root (default: smoothcomp_results). "
            "Writes <root>.parquet and <root>.log."
        ),
    )
    parser.add_argument(
        "--slow",
        action="store_true",
        help="5 s delay between requests (default: 1 s).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="HTTP retry attempts (default: 3).",
    )
    parser.add_argument(
        "--max-id",
        type=int,
        default=25000,
        help=(
            "Starting event ID for discovery — probe downward from here "
            "(default: 25000). Lower this to narrow the search range."
        ),
    )
    parser.add_argument(
        "--max-probe",
        type=int,
        default=500,
        help=(
            "Stop after this many consecutive events with year < min-year − 1 "
            "(default: 500). Lower values speed up search at the cost of possibly "
            "missing sporadic late-submitted events."
        ),
    )
    parser.add_argument(
        "--sport-filter",
        default="bjj",
        help=(
            "Case-insensitive keyword to match in event names (default: 'bjj'). "
            "Pass an empty string to include all sports."
        ),
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Cap the number of events scraped (default: no limit). Useful for test runs.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore cached event discovery; re-probe all IDs from scratch.",
    )
    args = parser.parse_args()

    out_base = Path(args.output).with_suffix("")
    out_parquet = out_base.with_suffix(".parquet")
    out_log = out_base.with_suffix(".log")

    _setup_logging(out_log)
    logger.info(
        "Starting: min_year=%d max_year=%d sport_filter=%r max_id=%d max_probe=%d",
        args.min_year,
        args.max_year,
        args.sport_filter,
        args.max_id,
        args.max_probe,
    )
    tqdm.write(f"Logging to {out_log}", file=sys.stderr)

    delay = 5.0 if args.slow else 1.0
    session = RateLimitedSession(delay=delay, retries=args.retries)
    cache = {} if args.no_cache else load_cache()

    # Phase 1: discover events
    tqdm.write("Phase 1: discovering events…", file=sys.stderr)
    events = _discover_events(
        session,
        min_year=args.min_year,
        max_year=args.max_year,
        max_id=args.max_id,
        max_probe=args.max_probe,
        sport_filter=args.sport_filter,
        cache=cache,
        max_events=args.max_events,
    )

    if not events:
        tqdm.write(
            "No events found. Try --no-cache, a higher --max-id, "
            "a wider year range, or a different --sport-filter.",
            file=sys.stderr,
        )
        logger.warning("No events found.")
        sys.exit(1)

    tqdm.write(
        f"Found {len(events)} event(s). Phase 2: scraping matches…",
        file=sys.stderr,
    )

    # Phase 2: scrape matches for each event
    all_rows: list[dict] = []
    for event_info in tqdm(events, desc="Events", file=sys.stderr):
        all_rows.extend(_fetch_event_matches(session, event_info))

    logger.info("Total rows: %d", len(all_rows))
    if not all_rows:
        tqdm.write("No match data collected.", file=sys.stderr)
        logger.warning("No match rows.")
        sys.exit(1)

    df = build_dataframe(all_rows)
    df = _infer_absolute_fields(df)
    df.write_parquet(out_parquet, compression="zstd")

    msg = f"Written → {out_parquet}  ({len(df):,} rows × {df.shape[1]} columns)"
    logger.info(msg)
    tqdm.write(msg, file=sys.stderr)

    by_year = df.group_by("event_year").len().sort("event_year")
    for row in by_year.iter_rows(named=True):
        line = f"  {row['event_year']}: {row['len']:,} rows"
        logger.info(line)
        tqdm.write(line, file=sys.stderr)


if __name__ == "__main__":
    main()
