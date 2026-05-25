#!/usr/bin/env python3
"""
Smoothcomp Match Scraper

Produces one Parquet row per completed match for BJJ events on smoothcomp.com
within a given year range.

Event discovery uses the paginated past-events listing:
    /en/events/past?sport=bjj&page=N  (40 events/page, sorted newest-first)

Match data comes from the server-rendered matchlist pages:
    /en/event/{id}/schedule/matchlist?page=N

Category headers encode belt, weight class, age division, and gi/no-gi.
Each match yields: winner/loser name, ID, team; score; result method; duration.

Note: Weigh-in (measured) weight is NOT available without authentication.
      weight_class from the category header IS included.

Usage:
    smoothcomp-bjj --min-year 2023
    smoothcomp-bjj --min-year 2022 --max-year 2023 --output matches_2022_2023
    smoothcomp-bjj --min-year 2025 --max-events 10 --output test_run
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
# Listing page parser (event discovery)
# ---------------------------------------------------------------------------

def _format_listing_date(startdate: str, enddate: str) -> str:
    try:
        start = datetime.strptime(startdate, "%Y-%m-%d")
        end = datetime.strptime(enddate, "%Y-%m-%d") if enddate else start
        if start == end:
            return start.strftime("%-d %b %Y")
        if start.month == end.month and start.year == end.year:
            return f"{start.day}-{end.strftime('%-d %b %Y')}"
        return f"{start.strftime('%-d %b')} - {end.strftime('%-d %b %Y')}"
    except ValueError:
        return startdate


def _parse_listing_page(html: str) -> list[dict]:
    """Extract event_info dicts from the var events array embedded in a listing page."""
    marker = "var events = ["
    idx = html.find(marker)
    if idx < 0:
        return []
    start = idx + len(marker) - 1  # points at '['
    depth = 0
    in_str = False
    esc = False
    end = -1
    for j in range(start, len(html)):
        c = html[j]
        if esc:
            esc = False
            continue
        if c == "\\" and in_str:
            esc = True
            continue
        if c == '"' and not esc:
            in_str = not in_str
            continue
        if not in_str:
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    end = j
                    break
    if end < 0:
        return []
    try:
        items = json.loads(html[start : end + 1])
    except json.JSONDecodeError:
        return []
    result = []
    for item in items:
        event_id = item.get("id")
        if not event_id:
            continue
        startdate = item.get("startdate", "")
        year = int(startdate[:4]) if len(startdate) >= 4 else None
        result.append(
            {
                "event_id": str(event_id),
                "event_name": item.get("title", f"event_{event_id}"),
                "event_year": year,
                "event_date": _format_listing_date(
                    startdate, item.get("enddate", "")
                ),
            }
        )
    return result


# ---------------------------------------------------------------------------
# Match fetching
# ---------------------------------------------------------------------------


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
    max_probe: int,
    sport_filter: str,
    cache: dict,
    max_events: Optional[int] = None,
) -> list[dict]:
    """
    Page through /en/events/past to find events in [min_year, max_year].

    Events are returned newest-first by the listing, so once max_probe
    consecutive events older than min_year-1 are seen, discovery stops.
    Stops early once max_events matching events are found.
    All discovered event metadata is cached in cache['events'].
    """
    events_cache = cache.setdefault("events", {})
    found: list[dict] = []
    consecutive_old = 0
    page = 1

    pbar = tqdm(desc="Discovering events", unit="event", file=sys.stderr)

    while True:
        params = f"?sport={sport_filter.lower()}&page={page}" if sport_filter else f"?page={page}"
        resp = session.get(f"{SMOOTHCOMP_BASE}/en/events/past{params}")
        if resp is None:
            break

        page_events = _parse_listing_page(resp.text)
        if not page_events:
            break

        for info in page_events:
            pbar.update(1)
            key = info["event_id"]

            if key not in events_cache:
                events_cache[key] = info
                if len(events_cache) % 50 == 0:
                    save_cache(cache)

            year = info.get("event_year")
            name = info.get("event_name", "")

            if year is None:
                logger.debug("event %s: no year found — skipping", key)
                continue

            if year < min_year - 1:
                consecutive_old += 1
                if consecutive_old >= max_probe:
                    logger.info(
                        "Stopping discovery: %d consecutive events before %d",
                        max_probe,
                        min_year - 1,
                    )
                    pbar.close()
                    save_cache(cache)
                    tqdm.write(
                        f"Discovery complete: {len(found)} event(s) in [{min_year}–{max_year}]",
                        file=sys.stderr,
                    )
                    return found
                continue
            else:
                consecutive_old = 0

            if min_year <= year <= max_year:
                found.append(info)
                pbar.set_postfix(found=len(found))
                logger.info("event %s: %s (%d) — adding", key, name, year)
                if max_events is not None and len(found) >= max_events:
                    logger.info(
                        "Stopping discovery: reached --max-events limit (%d)", max_events
                    )
                    pbar.close()
                    save_cache(cache)
                    tqdm.write(
                        f"Discovery complete: {len(found)} event(s) in [{min_year}–{max_year}]",
                        file=sys.stderr,
                    )
                    return found

        page += 1

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
            "Pages through smoothcomp.com/en/events/past to find events in\n"
            "[--min-year, --max-year], then scrapes match data for each.\n"
            f"Discovered event metadata is cached to {CACHE_PATH}.\n\n"
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
        "--delay",
        type=float,
        default=10.0,
        help="Seconds between HTTP requests (default: 10). Values below 10 trigger a warning.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="HTTP retry attempts (default: 3).",
    )
    parser.add_argument(
        "--max-probe",
        type=int,
        default=80,
        help=(
            "Stop after this many consecutive events older than min-year−1 "
            "(default: 80). Events are date-sorted on the listing page, so "
            "this threshold is crossed quickly once past the target range."
        ),
    )
    parser.add_argument(
        "--sport-filter",
        default="bjj",
        help=(
            "Sport slug passed as ?sport= to the past-events listing "
            "(default: 'bjj'). Pass an empty string to include all sports."
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
        help="Ignore cached event metadata; re-fetch all listing pages.",
    )
    args = parser.parse_args()

    out_base = Path(args.output).with_suffix("")
    out_parquet = out_base.with_suffix(".parquet")
    out_log = out_base.with_suffix(".log")

    _setup_logging(out_log)
    logger.info(
        "Starting: min_year=%d max_year=%d sport_filter=%r max_probe=%d delay=%.1fs",
        args.min_year,
        args.max_year,
        args.sport_filter,
        args.max_probe,
        args.delay,
    )
    tqdm.write(f"Logging to {out_log}", file=sys.stderr)

    if args.delay < 10.0:
        msg = (
            f"--delay {args.delay}s is below the recommended 10 s minimum — are you"
            " fr just going to disrespect their robots.txt like that? 😤😤😤"
        )
        logger.warning(msg)
        tqdm.write(f"WARNING: {msg}", file=sys.stderr)

    session = RateLimitedSession(delay=args.delay, retries=args.retries)
    cache = {} if args.no_cache else load_cache()

    # Phase 1: discover events
    tqdm.write("Phase 1: discovering events…", file=sys.stderr)
    events = _discover_events(
        session,
        min_year=args.min_year,
        max_year=args.max_year,
        max_probe=args.max_probe,
        sport_filter=args.sport_filter,
        cache=cache,
        max_events=args.max_events,
    )

    if not events:
        tqdm.write(
            "No events found. Try --no-cache, a wider year range, or a different --sport-filter.",
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
