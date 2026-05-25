# bjjscrapers

Tools for scraping BJJ competition match data into analysis-ready Parquet files.

## Install

```bash
uv sync
uv pip install -e .
```

## Scrapers

### `ibjjf-absolute` — IBJJF Absolute Bracket Results

**This scraper kinda sucks and doesn't really retrieve any useful info, I would advise against using it 🙃**

Scrapes open-class (absolute) bracket results from IBJJF events.

Two data sources:

- **ibjjfdb.com** — historical placement data going back years; yields inferred semifinal and final rows (gold/silver/bronze positions are known, so the last three matches can be reconstructed)
- **bjjcompsystem.com** — live bracket data for active/recent tournaments; yields full per-match rows with scores, duration, and result method

```bash
scrape-ibjjf-absolute --min-year 2022
scrape-ibjjf-absolute --min-year 2020 --max-year 2023 --gi gi --gender male --output ibjjf_male_gi
```

#### ibjjf-absolute options

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--min-year` | required | Earliest tournament year |
| `--max-year` | current year | Latest tournament year |
| `--gi` | `both` | `gi`, `nogi`, or `both` |
| `--gender` | `both` | `male`, `female`, or `both` |
| `--output` | `results` | Output file root (writes `.parquet` + `.log`) |
| `--build-weight-map` | off | Fetch all weight-class brackets to populate `winner_weight` / `loser_weight`; much slower |
| `--no-ibjjfdb` | off | Skip historical data; only scrape bjjcompsystem.com |
| `--no-bjjcomp` | off | Skip bjjcompsystem.com; only use historical data |
| `--max-probe` | 200 | Max new bjjcompsystem IDs to probe (cached after first run) |
| `--slow` | off | 5 s delay between requests |
| `--retries` | 3 | HTTP retry attempts |

#### Output columns

`data_source`, `is_inferred`, `tournament_id`, `tournament_name`, `tournament_championship`, `tournament_year`, `gi`, `gender`, `age_division`, `belt`, `weight_class`, `category_id`, `match_number`, `fight_number`, `is_final`, `match_datetime`, `match_location`, `winner_id`, `winner_name`, `winner_team`, `winner_seed`, `winner_weight`, `winner_medal`, `winner_note`, `loser_id`, `loser_name`, `loser_team`, `loser_seed`, `loser_weight`, `loser_medal`, `loser_note`, `is_bye`, `is_unfinished`

---

### `smoothcomp-bjj` — Smoothcomp All-Match Results

Scrapes every match (all weight classes and divisions) from BJJ events on Smoothcomp. Discovers events by probing numeric event IDs, then paginates through match lists. Discovered IDs are cached at `~/.smoothcomp_cache.json` so subsequent runs are fast.

```bash
scrape-smoothcomp --min-year 2024
scrape-smoothcomp --min-year 2023 --max-year 2024 --output sc_2023_2024
scrape-smoothcomp --min-year 2025 --max-id 21600 --max-probe 10 --output test_run
```

#### smoothcomp-bjj options

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--min-year` | required | Earliest event year |
| `--max-year` | current year | Latest event year |
| `--output` | `smoothcomp_results` | Output file root (writes `.parquet` + `.log`) |
| `--max-id` | 25000 | Starting event ID; probes downward from here |
| `--max-probe` | 500 | Stop after N consecutive out-of-range/non-BJJ events |
| `--sport-filter` | `bjj` | Keyword matched against event name; `""` to include all sports |
| `--max-events` | no limit | Cap number of events scraped; useful for test runs |
| `--no-cache` | off | Ignore cached event discovery; re-probe from scratch |
| `--slow` | off | 5 s delay between requests |
| `--retries` | 3 | HTTP retry attempts |

#### Output columns

`data_source`, `event_id`, `event_name`, `event_year`, `event_date`, `gi`, `gender`, `age_division`, `belt`, `weight_class`, `category_raw`, `result_method`, `score_winner`, `score_loser`, `match_duration`, `winner_id`, `winner_name`, `winner_team`, `loser_id`, `loser_name`, `loser_team`, `bracket_id`, `is_absolute`, `winner_inferred_weight`, `loser_inferred_weight`, `winner_inferred_belt`, `loser_inferred_belt`

`is_absolute` is `True` for open-class/absolute brackets. The four `inferred_*` columns are populated for absolute rows by looking up each competitor's appearance in a same-event weight-class bracket; `null` if the competitor only entered the absolute or if no profile ID was found.

**Note:** Weigh-in (measured) weight is not available without authentication. `weight_class` reflects the division label from the category header.

---

## Loading results

```python
import polars as pl

df = pl.read_parquet("smoothcomp_results.parquet")

# Absolute division rows with inferred weights
abs_df = df.filter(pl.col("is_absolute"))

# Black belt adults, gi
black_adult = df.filter(
    (pl.col("belt") == "Black") &
    (pl.col("age_division").str.contains("(?i)adult")) &
    pl.col("gi")
)
```

## Caches

Both scrapers write a local JSON cache to avoid re-fetching event metadata:

| Scraper | Cache path |
| ------- | ---------- |
| `ibjjf-absolute` | `~/.bjjnerdshit_cache.json` |
| `smoothcomp-bjj` | macOS: `~/Library/Caches/bjjnerdshit/smoothcomp_cache.json` · Linux: `~/.cache/bjjnerdshit/smoothcomp_cache.json` |

Pass `--no-cache` to force a full re-probe.
