#!/usr/bin/env python3
"""
IBJJF Absolute Bracket Scraper

Produces one Parquet row per match for the open-class (absolute) bracket.

Two data sources:
  - ibjjfdb.com   Historical placement data (Gold/Silver/Bronze) scraped from
                  ibjjf.com/events/results. Yields inferred rows for the final
                  and two semis (the matches we can reconstruct with certainty).
                  is_inferred=True.
  - bjjcompsystem.com  Live bracket data for currently-active tournaments.
                  Yields full per-match rows. is_inferred=False.

The year filters work for both sources. For historical years you will get
inferred matches; for currently live tournaments you get the full bracket.

Usage:
    python -m bjjnerdshit.scrape_absolute --min-year 2022 --gi gi --gender both
    python -m bjjnerdshit.scrape_absolute --min-year 2024 --max-year 2024 \\
        --gi both --gender male --output matches
"""

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import platformdirs
import polars as pl
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BJJCOMP_BASE = "https://www.bjjcompsystem.com"
IBJJF_RESULTS_URL = "https://ibjjf.com/events/results"
IBJJFDB_BASE = "https://www.ibjjfdb.com"
CACHE_PATH = Path(platformdirs.user_cache_dir("bjjnerdshit")) / "ibjjf_cache.json"

NOGI_KEYWORDS = {"no-gi", "no gi", "nogi", "sem kimono", "submission only"}

# Weight labels that identify the absolute bracket on bjjcompsystem
OPEN_CLASS_LABELS = {
    "open class",
    "absoluto",
    "open class leve",
    "open class pesado",
    "open class light",
    "open class heavy",
}

# Belt name normalization (Portuguese → English title case; English uppercase → title case)
BELT_TRANSLATIONS = {
    # Portuguese
    "branca": "White",
    "cinza": "Grey",
    "amarela cinza": "Yellow-Grey",
    "amarela": "Yellow",
    "laranja": "Orange",
    "verde laranja": "Green-Orange",
    "verde": "Green",
    "azul": "Blue",
    "roxa": "Purple",
    "marrom": "Brown",
    "preta": "Black",
    # English (bjjcompsystem returns uppercase)
    "white": "White",
    "grey": "Grey",
    "yellow-grey": "Yellow-Grey",
    "yellow": "Yellow",
    "orange": "Orange",
    "green-orange": "Green-Orange",
    "green": "Green",
    "blue": "Blue",
    "purple": "Purple",
    "brown": "Brown",
    "black": "Black",
}

# Output column schema — shared by both sources
_SCHEMA = {
    "data_source": pl.Utf8,  # "bjjcompsystem" | "ibjjfdb"
    "is_inferred": pl.Boolean,  # True for ibjjfdb placement-derived rows
    "tournament_id": pl.Utf8,  # bjjcompsystem int or ibjjfdb int as string
    "tournament_name": pl.Utf8,
    "tournament_championship": pl.Utf8,  # e.g. "World Jiu-Jitsu IBJJF Championship"
    "tournament_year": pl.Int32,
    "gi": pl.Boolean,
    "gender": pl.Utf8,
    "age_division": pl.Utf8,
    "belt": pl.Utf8,
    "weight_class": pl.Utf8,
    "category_id": pl.Utf8,
    "match_number": pl.Int32,
    "fight_number": pl.Int32,
    "is_final": pl.Boolean,
    "match_datetime": pl.Utf8,  # ISO string; cast to Datetime at write time
    "match_location": pl.Utf8,
    "winner_id": pl.Utf8,
    "winner_name": pl.Utf8,
    "winner_team": pl.Utf8,
    "winner_seed": pl.Int32,
    "winner_weight": pl.Utf8,  # individual weight class (best-effort)
    "winner_medal": pl.Utf8,
    "winner_note": pl.Utf8,
    "loser_id": pl.Utf8,
    "loser_name": pl.Utf8,
    "loser_team": pl.Utf8,
    "loser_seed": pl.Int32,
    "loser_weight": pl.Utf8,
    "loser_medal": pl.Utf8,
    "loser_note": pl.Utf8,
    "is_bye": pl.Boolean,
    "is_unfinished": pl.Boolean,
}

logger = logging.getLogger(__name__)


def _setup_logging(log_path: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8")],
    )


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------


class RateLimitedSession:
    """requests.Session with per-request delay and exponential-backoff retry.

    Pattern mirrors rate_limit_get() from weisserw/ibjjf-elo.
    """

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
    return {"tournaments": {}}


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


def _is_nogi(name: str) -> bool:
    return any(kw in name.lower() for kw in NOGI_KEYWORDS)


def _extract_year(text: str) -> Optional[int]:
    m = re.search(r"\b(20\d{2})\b", text)
    return int(m.group(1)) if m else None


def _normalize_belt(raw: str) -> str:
    return BELT_TRANSLATIONS.get(raw.lower().strip(), raw)


def _parse_category_heading(heading: str) -> Optional[dict]:
    """
    Parse an ibjjfdb category heading like 'Adult / Male / Black / Open Class'.
    Returns dict with age_division, gender, belt, weight_class or None.
    """
    parts = [p.strip() for p in heading.split("/")]
    if len(parts) < 4:
        return None
    return {
        "age_division": parts[0],
        "gender": parts[1],
        "belt": _normalize_belt(parts[2]),
        "weight_class": parts[3],
    }


def _parse_bjjcomp_timestamp(text: str, year: int) -> Optional[datetime]:
    """Parse bjjcompsystem timestamp like 'Sat 05/16 at 05:02 PM'."""
    text = text.strip()
    # Remove day-of-week prefix
    text = re.sub(r"^\w{2,4}\s+", "", text)
    m = re.match(r"(\d{1,2})/(\d{1,2})\s+at\s+(\d{1,2}):(\d{2})\s*(AM|PM)", text, re.I)
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        hour, minute = int(m.group(3)), int(m.group(4))
        ampm = m.group(5).upper()
        if ampm == "PM" and hour != 12:
            hour += 12
        elif ampm == "AM" and hour == 12:
            hour = 0
        try:
            return datetime(year, month, day, hour, minute)
        except ValueError:
            pass
    # ISO fallback
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    return None


# ---------------------------------------------------------------------------
# ibjjfdb.com — historical placement data
# ---------------------------------------------------------------------------


def discover_ibjjfdb_tournaments(
    session: RateLimitedSession,
    min_year: int,
    max_year: int,
) -> list[dict]:
    """
    Scrape ibjjf.com/events/results to get a list of tournaments with their
    ibjjfdb.com IDs, filtered to [min_year, max_year].
    """
    logger.info("Fetching ibjjf.com/events/results")
    resp = session.get(IBJJF_RESULTS_URL)
    if not resp:
        logger.warning("Failed to fetch ibjjf.com/events/results")
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    results: list[dict] = []

    for section in soup.find_all("section", class_="event"):
        # Championship name: first significant text in the section
        champ_name = ""
        for el in section.descendants:
            if isinstance(el, str):
                t = el.strip()
                if t and len(t) > 5 and not re.match(r"^\d{4}$", t):
                    champ_name = t
                    break
        if not champ_name:
            champ_name = section.get_text(strip=True)[:50]

        is_gi = not _is_nogi(champ_name)

        # Year links within the section
        for year_div in section.find_all("div", class_="year"):
            link = year_div.find("a", href=lambda h: h and "ibjjfdb.com" in h)
            if not link:
                continue
            year_text = year_div.get_text(strip=True)
            year = _extract_year(year_text)
            if not year or not (min_year <= year <= max_year):
                continue

            m = re.search(r"/ChampionshipResults/(\d+)/", link["href"])
            if not m:
                continue

            results.append(
                {
                    "ibjjfdb_id": m.group(1),
                    "championship": champ_name,
                    "year": year,
                    "gi": is_gi,
                    "name": f"{champ_name} {year}",
                    "url": link["href"],
                }
            )

    logger.info(
        "Found %d ibjjfdb tournaments in [%d, %d]", len(results), min_year, max_year
    )
    return results


def _infer_matches_from_placements(
    placements: list[dict],
    category_meta: dict,
) -> list[dict]:
    """
    Given a sorted list of placement dicts, infer match rows.

    Standard IBJJF elimination: two Bronze medalists = the semifinal losers.
    We can reconstruct:
      match 1 (semi): Gold beat Bronze-A
      match 2 (semi): Silver beat Bronze-B
      match 3 (final): Gold beat Silver

    If there are no bronze medalists (2-person bracket) we only generate the final.
    If > 4 athletes in semifinal pool, we only generate the final + semis from the
    top 4 (positions 1, 2, 3, 3).
    """
    by_pos: dict[str, list[dict]] = {}
    for p in placements:
        by_pos.setdefault(p["position"], []).append(p)

    gold = by_pos.get("1", [])
    silver = by_pos.get("2", [])
    bronze = by_pos.get("3", [])

    if not gold or not silver:
        return []

    g = gold[0]
    s = silver[0]
    rows: list[dict] = []

    def _row(
        winner: dict, loser: dict, match_num: int, winner_medal: str, loser_medal: str
    ) -> dict:
        return {
            **category_meta,
            "match_number": match_num,
            "fight_number": None,
            "is_final": match_num == 3,
            "match_datetime": None,
            "match_location": None,
            "winner_id": None,
            "winner_name": winner["name"],
            "winner_team": winner["team"],
            "winner_seed": None,
            "winner_weight": None,
            "winner_medal": winner_medal,
            "winner_note": None,
            "loser_id": None,
            "loser_name": loser["name"],
            "loser_team": loser["team"],
            "loser_seed": None,
            "loser_weight": None,
            "loser_medal": loser_medal,
            "loser_note": None,
            "is_bye": False,
            "is_unfinished": False,
        }

    if len(bronze) >= 2:
        rows.append(_row(g, bronze[0], match_num=1, winner_medal="1", loser_medal="3"))
        rows.append(_row(s, bronze[1], match_num=2, winner_medal="2", loser_medal="3"))
    elif len(bronze) == 1:
        rows.append(_row(g, bronze[0], match_num=1, winner_medal="1", loser_medal="3"))

    rows.append(_row(g, s, match_num=3, winner_medal="1", loser_medal="2"))
    return rows


def scrape_ibjjfdb(
    session: RateLimitedSession,
    tournaments: list[dict],
    gender_filter: str,
    gi_filter: str,
) -> list[dict]:
    """
    Fetch ibjjfdb.com results for each tournament and return inferred match rows
    for the absolute bracket.
    """
    gender_accept = {"male", "female"} if gender_filter == "both" else {gender_filter}

    filtered = [
        t
        for t in tournaments
        if not (gi_filter == "gi" and not t["gi"])
        and not (gi_filter == "nogi" and t["gi"])
    ]

    all_rows: list[dict] = []

    for t in tqdm(filtered, desc="ibjjfdb", unit="tournament", file=sys.stderr):
        logger.info("[ibjjfdb %s] %s", t["ibjjfdb_id"], t["name"])
        resp = session.get(t["url"])
        if not resp:
            logger.warning("Failed to fetch %s", t["url"])
            continue

        soup = BeautifulSoup(resp.text, "lxml")

        for h4 in soup.find_all("h4"):
            heading = h4.get_text(strip=True)
            if "open class" not in heading.lower() and "absolut" not in heading.lower():
                continue

            cat = _parse_category_heading(heading)
            if not cat:
                continue

            if cat["gender"].lower() not in gender_accept:
                continue

            # Find placement list (next sibling div.list)
            list_div = h4.find_next_sibling("div", class_="list")
            if not list_div:
                continue

            placements: list[dict] = []
            for item in list_div.find_all("div", class_="athlete-item"):
                pos_el = item.find("div", class_="position-athlete")
                name_el = item.find("div", class_="name")
                if not pos_el or not name_el:
                    continue
                p_el = name_el.find("p")
                if not p_el:
                    continue
                span = p_el.find("span")
                team = span.get_text(strip=True) if span else ""
                name = p_el.get_text(strip=True)
                if team:
                    name = name[: name.rfind(team)].strip() if team in name else name
                placements.append(
                    {
                        "position": pos_el.get_text(strip=True),
                        "name": name.strip(),
                        "team": team.strip(),
                    }
                )

            if not placements:
                continue

            category_meta = {
                "data_source": "ibjjfdb",
                "is_inferred": True,
                "tournament_id": t["ibjjfdb_id"],
                "tournament_name": t["name"],
                "tournament_championship": t["championship"],
                "tournament_year": t["year"],
                "gi": t["gi"],
                "gender": cat["gender"],
                "age_division": cat["age_division"],
                "belt": cat["belt"],
                "weight_class": cat["weight_class"],
                "category_id": f"ibjjfdb-{t['ibjjfdb_id']}-{heading[:30]}",
            }

            inferred = _infer_matches_from_placements(placements, category_meta)
            logger.info(
                "  %s: %d athletes → %d inferred match(es)",
                heading,
                len(placements),
                len(inferred),
            )
            all_rows.extend(inferred)

    return all_rows


# ---------------------------------------------------------------------------
# bjjcompsystem.com — live tournament match data
# ---------------------------------------------------------------------------


def _probe_bjjcomp_tournament(session: RateLimitedSession, tid: int) -> Optional[dict]:
    """Fetch /tournaments/{id}/categories and extract name, year, gi."""
    url = f"{BJJCOMP_BASE}/tournaments/{tid}/categories"
    resp = session.get(url)
    if resp is None:
        return None

    soup = BeautifulSoup(resp.text, "lxml")

    # Tournament name: check navbar dropdown option that matches current page
    # Also check page title (usually just "IBJJF")
    # Best bet: look for notice banners that often contain event name
    # As a fallback, we mark name=None; the dropdown text is more reliable
    name_el = soup.find("h1", class_=re.compile(r"tournament", re.I)) or soup.find("h1")
    title_el = soup.find("title")
    raw_name = (
        name_el.get_text(separator=" ", strip=True)
        if name_el
        else (title_el.get_text(strip=True) if title_el else "")
    )
    # Strip generic suffix
    raw_name = re.sub(
        r"\s*[|\-–]\s*(BJJ|Comp|IBJJF).*$", "", raw_name, flags=re.I
    ).strip()

    # If name is just "IBJJF" or empty, the page doesn't expose the name
    if raw_name.lower() in ("ibjjf", "bjjcompsystem", ""):
        raw_name = None

    year = _extract_year(raw_name) if raw_name else None
    return {
        "id": tid,
        "name": raw_name,
        "year": year,
        "gi": not _is_nogi(raw_name) if raw_name else True,
        "source": "probe",
    }


def discover_bjjcomp_tournaments(
    session: RateLimitedSession,
    min_year: int,
    max_year: int,
    cache: dict,
    max_probe: int = 200,
) -> list[dict]:
    """
    Discover currently active bjjcompsystem.com tournaments (those with categories).

    Strategy:
      1. Seed from the /tournaments dropdown (recent active events with names).
      2. Probe IDs downward from the highest known ID; stop once we've seen
         15 consecutive tournaments whose year is below min_year.
    """
    tournaments: dict[str, dict] = cache.get("tournaments", {})

    # Step 1: dropdown
    logger.info("Fetching bjjcompsystem.com tournament dropdown")
    resp = session.get(f"{BJJCOMP_BASE}/tournaments")
    if resp:
        soup = BeautifulSoup(resp.text, "lxml")
        select = soup.find("select", id="tournament_id")
        if select:
            for opt in select.find_all("option"):
                val = opt.get("value", "").strip()
                if not val:
                    continue
                name = opt.get_text(strip=True)
                year = _extract_year(name)
                key = str(val)
                if key not in tournaments:
                    tournaments[key] = {
                        "id": int(val),
                        "name": name,
                        "year": year,
                        "gi": not _is_nogi(name),
                        "source": "dropdown",
                    }
                    logger.info("  dropdown [%s] %s (%s)", val, name, year)

    # Step 2: probe downward
    known_ids = sorted(int(k) for k in tournaments if tournaments[k].get("name"))
    start_id = (max(known_ids) if known_ids else 3200) - 1
    consecutive_below = 0
    logger.info("Probing IDs downward from %d (max %d)", start_id, max_probe)

    probe_range = range(start_id, max(start_id - max_probe - 1, 0), -1)
    with tqdm(
        total=max_probe, desc="Probing bjjcomp IDs", unit="id", file=sys.stderr
    ) as pbar:
        for tid in probe_range:
            key = str(tid)
            if key in tournaments:
                yr = tournaments[key].get("year")
                if yr is not None and yr < min_year:
                    consecutive_below += 1
                    if consecutive_below >= 15:
                        logger.info("Reached min_year boundary, stopping probe.")
                        break
                else:
                    consecutive_below = 0
                pbar.update(1)
                continue

            meta = _probe_bjjcomp_tournament(session, tid)
            pbar.update(1)
            if meta:
                tournaments[key] = meta
                if meta["name"]:
                    logger.info("  probe [%d] %s (%s)", tid, meta["name"], meta["year"])
                yr = meta.get("year")
                if yr is not None and yr < min_year:
                    consecutive_below += 1
                    if consecutive_below >= 15:
                        break
                else:
                    consecutive_below = 0
            else:
                tournaments[key] = {
                    "id": tid,
                    "name": None,
                    "year": None,
                    "gi": None,
                    "source": "probe",
                }

    save_cache({**cache, "tournaments": tournaments})

    return [
        t
        for t in tournaments.values()
        if t.get("name") and t.get("year") and min_year <= t["year"] <= max_year
    ]


def _parse_bjjcomp_categories(html: str, gender_label: str) -> list[dict]:
    """Parse the categories listing page and return open-class category dicts."""
    soup = BeautifulSoup(html, "lxml")
    results: list[dict] = []

    for card in soup.find_all("li", class_="categories-grid__category"):
        link_el = card.find("a")
        if not link_el or not link_el.get("href"):
            continue

        age_el = card.find("div", class_="category-card__age-division")
        belt_el = card.find("span", class_="category-card__belt-label")
        weight_el = card.find("span", class_="category-card__weight-label")

        age = age_el.get_text(strip=True) if age_el else ""
        belt = _normalize_belt(belt_el.get_text(strip=True)) if belt_el else ""
        weight = weight_el.get_text(strip=True) if weight_el else ""

        if weight.lower() not in OPEN_CLASS_LABELS:
            continue

        m = re.search(r"/categories/(\d+)", link_el["href"])
        if not m:
            continue

        results.append(
            {
                "category_id": m.group(1),
                "href": link_el["href"],
                "age_division": age,
                "belt": belt,
                "weight": weight,
                "gender": gender_label,
            }
        )

    return results


def _parse_bjjcomp_competitor(comp_el) -> dict:
    """Extract competitor fields from a match-card__competitor div."""
    if comp_el is None:
        return {}

    comp_id_raw = comp_el.get("id", "")
    comp_id = comp_id_raw.split("-")[-1] if comp_id_raw else None

    # Loser class is on the inner span.match-card__competitor-description
    desc_el = comp_el.find("span", class_="match-card__competitor-description")
    is_loser = desc_el is not None and "match-competitor--loser" in (
        desc_el.get("class") or []
    )
    is_bye = bool(comp_el.find("div", class_="match-card__bye"))

    seed_el = comp_el.find("span", class_="match-card__competitor-n")
    name_el = comp_el.find("div", class_="match-card__competitor-name")
    team_el = comp_el.find("div", class_="match-card__club-name")
    dq_el = comp_el.find("i", class_="match-card__disqualification")

    seed_text = seed_el.get_text(strip=True) if seed_el else ""
    seed = int(seed_text) if seed_text.isdigit() else None

    return {
        "id": comp_id,
        "seed": seed,
        "is_loser": is_loser,
        "is_bye": is_bye,
        "name": name_el.get_text(strip=True) if name_el else None,
        "team": team_el.get_text(strip=True) if team_el else None,
        "note": dq_el.get("title", "").strip() if dq_el else "",
    }


def _parse_bjjcomp_medals(soup) -> dict[str, str]:
    """Return {name: place_string} from the podium section."""
    medals: dict[str, str] = {}
    for step in soup.find_all("div", class_="podium__step"):
        name_el = step.find("div", class_="podium__competitor-name")
        place_el = step.find("span", class_="podium__place")
        if name_el and place_el:
            medals[name_el.get_text(strip=True)] = place_el.get_text(strip=True)
    return medals


def build_bjjcomp_weight_map(
    session: RateLimitedSession,
    tournament_id: int,
    gender_suffixes: list[str],
) -> dict[str, str]:
    """
    Build competitor_id → weight_class by scanning non-absolute category brackets.
    Best-effort; competitors who only enter the absolute won't appear.
    """
    weight_map: dict[str, str] = {}

    for suffix in gender_suffixes:
        url = f"{BJJCOMP_BASE}/tournaments/{tournament_id}/categories{suffix}"
        resp = session.get(url)
        if not resp:
            continue

        soup = BeautifulSoup(resp.text, "lxml")
        non_abs_cats = []
        for card in soup.find_all("li", class_="categories-grid__category"):
            link_el = card.find("a")
            weight_el = card.find("span", class_="category-card__weight-label")
            if not link_el or not weight_el:
                continue
            weight = weight_el.get_text(strip=True)
            if weight.lower() in OPEN_CLASS_LABELS:
                continue
            m = re.search(r"/categories/(\d+)", link_el.get("href", ""))
            if m:
                non_abs_cats.append((m.group(1), weight))

        for cat_id, weight in tqdm(
            non_abs_cats, desc="weight map", unit="cat", leave=False, file=sys.stderr
        ):
            cat_url = f"{BJJCOMP_BASE}/tournaments/{tournament_id}/categories/{cat_id}"
            cat_resp = session.get(cat_url)
            if not cat_resp:
                continue
            cat_soup = BeautifulSoup(cat_resp.text, "lxml")
            for comp in cat_soup.find_all("div", class_="match-card__competitor"):
                cid_raw = comp.get("id", "")
                cid = cid_raw.split("-")[-1] if cid_raw else None
                if cid and cid not in weight_map:
                    weight_map[cid] = weight

    return weight_map


def parse_bjjcomp_matches(
    html: str,
    category_meta: dict,
    weight_map: dict[str, str],
) -> list[dict]:
    """Parse all matches from a bjjcompsystem bracket page."""
    soup = BeautifulSoup(html, "lxml")
    medals = _parse_bjjcomp_medals(soup)
    rows: list[dict] = []

    last_dt: Optional[datetime] = None
    last_dt_count = 0
    year = category_meta["tournament_year"]

    for match_div in soup.find_all("div", class_="tournament-category__match"):
        # Match number: from div.tournament-category__match-card.match-N
        match_card_outer = match_div.find(
            "div", class_="tournament-category__match-card"
        )
        match_num: Optional[int] = None
        if match_card_outer:
            for cls in match_card_outer.get("class", []):
                mm = re.match(r"^match-(\d+)$", cls)
                if mm:
                    match_num = int(mm.group(1))
                    break

        # Final label
        final_el = match_div.find("span", class_="tournament-category__final-label")
        is_final = final_el is not None

        header = match_div.find("div", class_="bracket-match-header")

        # where div contains BOTH fight span and location text
        where_el = (
            header.find("div", class_="bracket-match-header__where") if header else None
        )
        when_el = (
            header.find("div", class_="bracket-match-header__when") if header else None
        )

        # Extract fight number and location separately from the where div
        fight_num: Optional[int] = None
        location: Optional[str] = None
        if where_el:
            fight_span = where_el.find("span", class_="bracket-match-header__fight")
            if fight_span:
                fm = re.search(r"(?:FIGHT|LUTA)\s+(\d+)", fight_span.get_text(), re.I)
                if fm:
                    fight_num = int(fm.group(1))
                fight_text = fight_span.get_text(strip=True)
                rest = where_el.get_text(strip=True)
                location = rest[len(fight_text) :].lstrip(":").strip() or None
            else:
                location = where_el.get_text(strip=True) or None

        # Timestamp
        match_dt: Optional[datetime] = None
        if when_el:
            match_dt = _parse_bjjcomp_timestamp(when_el.get_text(strip=True), year)

        # Deduplication (ibjjf-elo pattern)
        if match_dt is None:
            if last_dt is not None:
                last_dt_count += 1
                match_dt = last_dt + timedelta(seconds=last_dt_count)
        elif match_dt == last_dt:
            last_dt_count += 1
            match_dt = last_dt + timedelta(seconds=last_dt_count)
        else:
            last_dt = match_dt
            last_dt_count = 0

        # Competitors: inside the inner div.match-card
        match_card = match_div.find("div", class_="match-card")
        if not match_card:
            continue

        comp_els = match_card.find_all("div", class_="match-card__competitor")
        if len(comp_els) < 2:
            continue

        red = _parse_bjjcomp_competitor(comp_els[0])
        blue = _parse_bjjcomp_competitor(comp_els[1])
        if not red or not blue:
            continue

        red_wins = not red["is_loser"] and not red["is_bye"]
        blue_wins = not blue["is_loser"] and not blue["is_bye"]

        is_unfinished = (
            not red["is_bye"] and not blue["is_bye"] and red_wins == blue_wins
        )

        is_bye_match = red["is_bye"] or blue["is_bye"]
        default_gold = ""
        if is_bye_match and red_wins:
            default_gold = "DEFAULT_GOLD"

        if red_wins and not blue_wins:
            winner, loser = red, blue
        elif blue_wins and not red_wins:
            winner, loser = blue, red
        else:
            winner, loser = red, blue

        if is_unfinished:
            winner["note"] = winner["note"] or "WINNER_NOT_RECORDED"
            loser["note"] = loser["note"] or "WINNER_NOT_RECORDED"
        if default_gold:
            winner["note"] = default_gold

        wname = winner.get("name") or ""
        lname = loser.get("name") or ""

        rows.append(
            {
                "data_source": "bjjcompsystem",
                "is_inferred": False,
                "tournament_id": str(category_meta["tournament_id"]),
                "tournament_name": category_meta["tournament_name"],
                "tournament_championship": None,
                "tournament_year": category_meta["tournament_year"],
                "gi": category_meta["gi"],
                "gender": category_meta["gender"],
                "age_division": category_meta["age_division"],
                "belt": category_meta["belt"],
                "weight_class": category_meta["weight"],
                "category_id": str(category_meta["category_id"]),
                "match_number": match_num,
                "fight_number": fight_num,
                "is_final": is_final,
                "match_datetime": match_dt.isoformat() if match_dt else None,
                "match_location": location,
                "winner_id": winner.get("id"),
                "winner_name": wname or None,
                "winner_team": winner.get("team"),
                "winner_seed": winner.get("seed"),
                "winner_weight": weight_map.get(winner.get("id") or ""),
                "winner_medal": medals.get(wname),
                "winner_note": winner.get("note") or None,
                "loser_id": loser.get("id"),
                "loser_name": lname or None,
                "loser_team": loser.get("team"),
                "loser_seed": loser.get("seed"),
                "loser_weight": weight_map.get(loser.get("id") or ""),
                "loser_medal": medals.get(lname),
                "loser_note": loser.get("note") or None,
                "is_bye": is_bye_match,
                "is_unfinished": is_unfinished,
            }
        )

    return rows


def scrape_bjjcompsystem(
    session: RateLimitedSession,
    tournaments: list[dict],
    gender_pairs: list[tuple[str, str]],
    gi_filter: str,
    build_weight_map: bool = False,
) -> list[dict]:
    """Scrape full match data for each bjjcompsystem tournament."""
    filtered = [
        t
        for t in tournaments
        if not (gi_filter == "gi" and not t["gi"])
        and not (gi_filter == "nogi" and t["gi"])
    ]

    all_rows: list[dict] = []

    for t in tqdm(filtered, desc="bjjcomp", unit="tournament", file=sys.stderr):
        tid = t["id"]
        gi_label = "gi" if t["gi"] else "nogi"
        logger.info("[bjjcomp %d] %s (%d, %s)", tid, t["name"], t["year"], gi_label)

        weight_map: dict[str, str] = {}
        if build_weight_map:
            logger.info("  Building weight map for tournament %d", tid)
            weight_map = build_bjjcomp_weight_map(
                session, tid, [suffix for _, suffix in gender_pairs]
            )
            logger.info("  Weight map: %d competitors", len(weight_map))

        for gender_label, gender_suffix in gender_pairs:
            cat_url = f"{BJJCOMP_BASE}/tournaments/{tid}/categories{gender_suffix}"
            resp = session.get(cat_url)
            if not resp:
                logger.warning(
                    "  [%s] could not fetch categories for tournament %d",
                    gender_label,
                    tid,
                )
                continue

            abs_cats = _parse_bjjcomp_categories(resp.text, gender_label)
            if not abs_cats:
                logger.info("  [%s] no absolute categories", gender_label)
                continue

            logger.info("  [%s] %d absolute category/-ies", gender_label, len(abs_cats))

            for cat in tqdm(
                abs_cats,
                desc=f"{t['name'][:30]} {gender_label}",
                unit="cat",
                leave=False,
                file=sys.stderr,
            ):
                bracket_url = f"{BJJCOMP_BASE}{cat['href']}"
                label = f"{cat['age_division']} {cat['belt']} {cat['weight']}"
                logger.info("    %s (%s) ...", label, gender_label)

                bracket_resp = session.get(bracket_url)
                if not bracket_resp:
                    logger.warning("    failed to fetch bracket: %s", bracket_url)
                    continue

                category_meta = {
                    "tournament_id": tid,
                    "tournament_name": t["name"],
                    "tournament_year": t["year"],
                    "gi": t["gi"],
                    "gender": gender_label,
                    "age_division": cat["age_division"],
                    "belt": cat["belt"],
                    "weight": cat["weight"],
                    "category_id": cat["category_id"],
                }

                matches = parse_bjjcomp_matches(
                    bracket_resp.text, category_meta, weight_map
                )
                logger.info("    → %d match(es)", len(matches))
                all_rows.extend(matches)

    return all_rows


# ---------------------------------------------------------------------------
# DataFrame builder
# ---------------------------------------------------------------------------


def build_dataframe(rows: list[dict]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=_SCHEMA)

    # Ensure all keys exist in every row
    for row in rows:
        for col in _SCHEMA:
            row.setdefault(col, None)

    df = pl.from_dicts(rows, schema=_SCHEMA)
    df = df.with_columns(
        pl.col("match_datetime")
        .str.to_datetime(format="%Y-%m-%dT%H:%M:%S", strict=False)
        .alias("match_datetime")
    )
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scrape IBJJF absolute bracket results → Parquet.\n\n"
            "Historical data comes from ibjjfdb.com (inferred semis+final from placements).\n"
            "Live/current tournament data comes from bjjcompsystem.com (full match results)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--min-year",
        type=int,
        required=True,
        help="Earliest tournament year to include.",
    )
    parser.add_argument(
        "--max-year",
        type=int,
        default=datetime.now().year,
        help="Latest tournament year (default: current year).",
    )
    parser.add_argument(
        "--gi",
        choices=["gi", "nogi", "both"],
        default="both",
        help="Gi / no-gi filter (default: both).",
    )
    parser.add_argument(
        "--gender",
        choices=["male", "female", "both"],
        default="both",
        help="Gender filter (default: both).",
    )
    parser.add_argument(
        "--output",
        default="results",
        help="Output file root (default: results). Writes <root>.parquet and <root>.log.",
    )
    parser.add_argument(
        "--slow", action="store_true", help="5 s delay between requests."
    )
    parser.add_argument(
        "--retries", type=int, default=3, help="HTTP retry attempts (default: 3)."
    )
    parser.add_argument(
        "--max-probe",
        type=int,
        default=200,
        help="Max new bjjcompsystem IDs to probe (default: 200). "
        "Results cached in ~/.bjjnerdshit_cache.json.",
    )
    parser.add_argument(
        "--build-weight-map",
        action="store_true",
        help="Fetch all weight-class brackets to resolve each competitor's "
        "individual weight class. Significantly slower.",
    )
    parser.add_argument(
        "--no-ibjjfdb",
        action="store_true",
        help="Skip ibjjfdb.com historical data; only scrape bjjcompsystem.com.",
    )
    parser.add_argument(
        "--no-bjjcomp",
        action="store_true",
        help="Skip bjjcompsystem.com; only use ibjjfdb.com historical data.",
    )
    args = parser.parse_args()

    out_base = Path(args.output).with_suffix("")
    out_parquet = out_base.with_suffix(".parquet")
    out_log = out_base.with_suffix(".log")

    _setup_logging(out_log)
    logger.info(
        "Starting scrape: min_year=%d max_year=%d gi=%s gender=%s",
        args.min_year,
        args.max_year,
        args.gi,
        args.gender,
    )
    tqdm.write(f"Logging to {out_log}", file=sys.stderr)

    delay = 5.0 if args.slow else 1.0
    session = RateLimitedSession(delay=delay, retries=args.retries)

    gender_pairs: list[tuple[str, str]] = []
    if args.gender in ("male", "both"):
        gender_pairs.append(("Male", ""))
    if args.gender in ("female", "both"):
        gender_pairs.append(("Female", "?gender_id=2"))

    all_rows: list[dict] = []

    # --- ibjjfdb historical source ---
    if not args.no_ibjjfdb:
        logger.info("=== ibjjfdb.com (historical placements) ===")
        ibjjfdb_tournaments = discover_ibjjfdb_tournaments(
            session, args.min_year, args.max_year
        )
        ibjjfdb_rows = scrape_ibjjfdb(
            session, ibjjfdb_tournaments, args.gender, args.gi
        )
        logger.info("ibjjfdb: %d inferred match row(s)", len(ibjjfdb_rows))
        all_rows.extend(ibjjfdb_rows)

    # --- bjjcompsystem live source ---
    if not args.no_bjjcomp:
        logger.info("=== bjjcompsystem.com (live bracket data) ===")
        cache = load_cache()
        bjjcomp_tournaments = discover_bjjcomp_tournaments(
            session, args.min_year, args.max_year, cache, args.max_probe
        )
        if args.build_weight_map:
            logger.info(
                "--build-weight-map active: each tournament will fetch all weight-class brackets"
            )
        bjjcomp_rows = scrape_bjjcompsystem(
            session,
            bjjcomp_tournaments,
            gender_pairs,
            args.gi,
            build_weight_map=args.build_weight_map,
        )
        logger.info("bjjcompsystem: %d match row(s)", len(bjjcomp_rows))
        all_rows.extend(bjjcomp_rows)

    logger.info("Total rows: %d", len(all_rows))
    if not all_rows:
        tqdm.write("No data found. Check year range and filters.", file=sys.stderr)
        logger.warning("No data found. Check year range and filters.")
        sys.exit(1)

    df = build_dataframe(all_rows)
    df.write_parquet(out_parquet, compression="zstd")
    msg = f"Written → {out_parquet}  ({len(df):,} rows × {df.shape[1]} columns)"
    logger.info(msg)
    tqdm.write(msg, file=sys.stderr)

    counts = df.group_by("data_source", "is_inferred").len().sort("data_source")
    for row in counts.iter_rows(named=True):
        line = f"  {row['data_source']} (inferred={row['is_inferred']}): {row['len']:,} rows"
        logger.info(line)
        tqdm.write(line, file=sys.stderr)


if __name__ == "__main__":
    main()
