"""Data fetching from AMFI, mfapi.in, and yfinance."""

import logging
import re
import time
from datetime import datetime, timedelta
from typing import Callable, Optional

import pandas as pd
import requests
import yfinance as yf

import io

from config import (
    AMFI_NAV_URL,
    BENCHMARKS,
    EQUITY_CATEGORY_KEYWORDS,
    MFAPI_BASE_URL,
    MFAPI_BATCH_SIZE,
    MFAPI_DELAY_SECONDS,
    NSE_ARCHIVE_URL,
    NSE_DIRECT_INDICES,
)
from db import (
    bulk_insert_benchmark,
    bulk_insert_nav,
    get_connection,
    get_last_benchmark_date,
    get_last_nav_date,
    log_update_finish,
    log_update_start,
    upsert_fund,
)

logger = logging.getLogger(__name__)


def parse_amfi_nav_file(text: str) -> list[dict]:
    """Parse the AMFI NAVAll.txt file to extract scheme list with categories.

    Returns list of dicts with keys:
        scheme_code, scheme_name, scheme_category, scheme_type,
        isin_growth, isin_div, nav, date
    """
    schemes = []
    current_category = None
    current_type = None

    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        # Category header lines look like:
        # "Open Ended Schemes(Equity Scheme - Large Cap Fund)"
        # "Open Ended Schemes ( Equity Scheme - Large Cap Fund )"
        header_match = re.match(
            r"^(Open Ended Schemes|Close Ended Schemes|Interval Fund Schemes)"
            r"\s*\(\s*(.+?)\s*\)\s*$",
            line, re.IGNORECASE
        )
        if header_match:
            current_type = header_match.group(1).strip()
            current_category = header_match.group(2).strip()
            continue

        # Skip the column header line
        if line.startswith("Scheme Code"):
            continue

        # Data lines: code;isin_div_payout/isin_growth;isin_div_reinvest;name;nav;date
        parts = line.split(";")
        if len(parts) >= 4:
            try:
                scheme_code = int(parts[0].strip())
            except ValueError:
                continue

            isin_field = parts[1].strip() if len(parts) > 1 else ""
            isin_div = parts[2].strip() if len(parts) > 2 else ""
            scheme_name = parts[3].strip() if len(parts) > 3 else ""
            nav_str = parts[4].strip() if len(parts) > 4 else ""
            date_str = parts[5].strip() if len(parts) > 5 else ""

            try:
                nav = float(nav_str) if nav_str and nav_str != "N.A." else None
            except ValueError:
                nav = None

            schemes.append({
                "scheme_code": scheme_code,
                "scheme_name": scheme_name,
                "scheme_category": current_category,
                "scheme_type": current_type,
                "isin_growth": isin_field,
                "isin_div": isin_div,
                "nav": nav,
                "date": date_str,
            })

    return schemes


def fetch_scheme_list() -> list[dict]:
    """Fetch and parse the complete scheme list from AMFI."""
    logger.info("Fetching scheme list from AMFI...")
    resp = requests.get(AMFI_NAV_URL, timeout=60)
    resp.raise_for_status()
    return parse_amfi_nav_file(resp.text)


_DIRECT_GROWTH_EXCLUDE = [
    "idcw", "dividend", "income distribution", "bonus", "reinvest",
    "quarterly", "payout", "regular", " - institutional", "institutional plan",
    "eco plan", "standard plan", "wealth plan", "plan b", "plan c",
    "annual ", "monthly ", "series i", "series ii",
]


def is_direct_growth_plan(scheme_name: str) -> bool:
    """Return True only for Direct Growth plans.

    Keeps schemes whose name contains 'direct' and none of the IDCW /
    dividend / regular / bonus / payout exclusion keywords.
    """
    n = scheme_name.lower()
    if "direct" not in n:
        return False
    return not any(kw in n for kw in _DIRECT_GROWTH_EXCLUDE)


def filter_equity_schemes(schemes: list[dict]) -> list[dict]:
    """Filter schemes to equity Direct Growth plans only."""
    result = []
    for s in schemes:
        cat = s.get("scheme_category") or ""
        if not any(kw.lower() in cat.lower() for kw in EQUITY_CATEGORY_KEYWORDS):
            continue
        if not is_direct_growth_plan(s["scheme_name"]):
            continue
        result.append(s)
    return result


def save_scheme_list_to_db(schemes: list[dict]):
    """Save scheme metadata to the database."""
    with get_connection() as conn:
        for s in schemes:
            upsert_fund(
                conn,
                scheme_code=s["scheme_code"],
                scheme_name=s["scheme_name"],
                scheme_category=s.get("scheme_category"),
                scheme_type=s.get("scheme_type"),
                fund_house=_extract_fund_house(s["scheme_name"]),
                isin_growth=s.get("isin_growth"),
                isin_div=s.get("isin_div"),
            )
    logger.info(f"Saved {len(schemes)} schemes to database.")


def _extract_fund_house(scheme_name: str) -> str:
    """Extract fund house name from scheme name.

    e.g. "HDFC Top 100 Fund - Direct Plan - Growth" -> "HDFC"
    This is approximate; AMFI doesn't provide fund_house directly in NAVAll.txt.
    """
    # Common patterns: first word(s) before a known suffix
    # We'll grab everything before " Fund", " Scheme", the first " - "
    parts = re.split(r"\s+(?:Fund|Scheme|ETF|FOF|FoF)\b", scheme_name, maxsplit=1)
    if parts:
        house = parts[0].strip()
        # Take the first few words as the AMC name
        words = house.split()
        # Most AMC names are 1-5 words
        if len(words) > 5:
            return " ".join(words[:4])
        return house
    return scheme_name.split(" - ")[0].strip()


def fetch_scheme_nav_history(scheme_code: int, max_retries: int = 3) -> Optional[dict]:
    """Fetch full NAV history for a single scheme from mfapi.in.

    Returns dict with 'meta' and 'data' keys, or None on failure.
    Retries up to max_retries times with exponential backoff on 5xx/timeout errors.
    """
    url = f"{MFAPI_BASE_URL}/{scheme_code}"
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 404:
                return None
            if resp.status_code in (500, 502, 503, 504):
                raise requests.HTTPError(f"{resp.status_code} error", response=resp)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") == "FAIL":
                return None
            return data
        except (requests.RequestException, ValueError) as e:
            if attempt < max_retries:
                wait = 2 ** (attempt + 1)  # 2, 4, 8 seconds
                logger.debug(f"Retry {attempt + 1}/{max_retries} for scheme {scheme_code} after {wait}s: {e}")
                time.sleep(wait)
            else:
                logger.warning(f"Failed to fetch scheme {scheme_code} after {max_retries + 1} attempts: {e}")
                return None


def _parse_mfapi_date(date_str: str) -> Optional[str]:
    """Convert mfapi date format (dd-mm-yyyy) to ISO format (yyyy-mm-dd)."""
    try:
        dt = datetime.strptime(date_str, "%d-%m-%Y")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return None


def download_nav_for_schemes(
    scheme_codes: list[int],
    progress_callback: Optional[Callable] = None,
    stop_flag: Optional[Callable] = None,
) -> int:
    """Download historical NAV data for a list of schemes.

    Args:
        scheme_codes: List of scheme codes to fetch.
        progress_callback: Called with (current_index, total, scheme_code, status_msg).
        stop_flag: Callable that returns True if the operation should stop.

    Returns:
        Number of schemes successfully updated.
    """
    total = len(scheme_codes)
    updated = 0

    with get_connection() as conn:
        log_id = log_update_start(conn, "nav_download")

        for i, code in enumerate(scheme_codes):
            if stop_flag and stop_flag():
                log_update_finish(conn, log_id, updated, "stopped")
                break

            if progress_callback:
                progress_callback(i, total, code, f"Fetching {code}...")

            data = fetch_scheme_nav_history(code)
            if data and data.get("data"):
                meta = data.get("meta", {})
                fund_house = meta.get("fund_house")
                if fund_house:
                    upsert_fund(
                        conn, code, meta.get("scheme_name", ""),
                        scheme_category=meta.get("scheme_category"),
                        scheme_type=meta.get("scheme_type"),
                        fund_house=fund_house,
                    )

                last_date = get_last_nav_date(conn, code)
                records = []
                for entry in data["data"]:
                    iso_date = _parse_mfapi_date(entry["date"])
                    if iso_date is None:
                        continue
                    if last_date and iso_date <= last_date:
                        continue
                    try:
                        nav_val = float(entry["nav"])
                    except (ValueError, TypeError):
                        continue
                    records.append((code, iso_date, nav_val))

                if records:
                    bulk_insert_nav(conn, records)
                    updated += 1

                if (i + 1) % MFAPI_BATCH_SIZE == 0:
                    conn.commit()

            time.sleep(MFAPI_DELAY_SECONDS)

        log_update_finish(conn, log_id, updated, "completed")

    if progress_callback:
        progress_callback(total, total, 0, "Done!")

    return updated


def fetch_benchmark_data(
    benchmarks: Optional[dict] = None,
    start_date: str = "2003-01-01",
    progress_callback: Optional[Callable] = None,
):
    """Fetch benchmark index data from yfinance.

    Args:
        benchmarks: Dict of {display_name: yfinance_ticker}. Defaults to config.
        start_date: Earliest date to fetch. Will be overridden by last stored date.
        progress_callback: Called with (index, total, name, status).
    """
    if benchmarks is None:
        benchmarks = BENCHMARKS

    total = len(benchmarks)

    with get_connection() as conn:
        for i, (name, ticker) in enumerate(benchmarks.items()):
            if progress_callback:
                progress_callback(i, total, name, f"Fetching {name}...")

            last_date = get_last_benchmark_date(conn, name)
            fetch_start = start_date
            if last_date:
                # Fetch from day after last stored date
                last_dt = datetime.strptime(last_date, "%Y-%m-%d")
                fetch_start = (last_dt + timedelta(days=1)).strftime("%Y-%m-%d")

            try:
                df = yf.download(
                    ticker, start=fetch_start,
                    end=datetime.now().strftime("%Y-%m-%d"),
                    progress=False, auto_adjust=True,
                )
                if df.empty:
                    logger.info(f"No new data for {name}")
                    continue

                # Newer yfinance returns MultiIndex columns for single-ticker downloads
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)

                records = []
                for idx, row in df.iterrows():
                    date_str = idx.strftime("%Y-%m-%d")
                    close_val = row["Close"]
                    if pd.isna(close_val):
                        continue
                    records.append((name, date_str, float(close_val)))

                if records:
                    bulk_insert_benchmark(conn, records)
                    logger.info(f"Stored {len(records)} records for {name}")

            except Exception as e:
                logger.warning(f"Failed to fetch {name} ({ticker}): {e}")

    if progress_callback:
        progress_callback(total, total, "", "Done!")


def fetch_nse_index_data(
    indices: Optional[dict] = None,
    start_date: str = "2013-01-01",
    progress_callback: Optional[Callable] = None,
) -> dict[str, int]:
    """Fetch historical index data from NSE daily bulk archive files.

    NSE publishes a CSV of all index closes for every trading day at:
      https://archives.nseindia.com/content/indices/ind_close_all_DDMMYYYY.csv

    Archives go back to January 2013.  Each file is ~16 KB and contains every
    NSE index, so we download once per day and extract all requested indices.

    Args:
        indices: {display_name: csv_index_name}.  Defaults to NSE_DIRECT_INDICES.
        start_date: Earliest date to fetch (YYYY-MM-DD).
        progress_callback: Called with (done, total, date_str, status_msg).

    Returns:
        {display_name: records_inserted} counts.
    """
    if indices is None:
        indices = NSE_DIRECT_INDICES

    # Build reverse lookup: lowercase csv name → display name
    name_map = {v.lower(): k for k, v in indices.items()}

    # Find the latest date already stored for each index
    with get_connection() as conn:
        last_dates: dict[str, Optional[str]] = {}
        for display_name in indices:
            last_dates[display_name] = get_last_benchmark_date(conn, display_name)

    # Determine the start date per index (day after last stored)
    global_start = datetime.strptime(start_date, "%Y-%m-%d")
    per_index_start: dict[str, datetime] = {}
    for display_name in indices:
        last = last_dates[display_name]
        if last:
            per_index_start[display_name] = (
                datetime.strptime(last, "%Y-%m-%d") + timedelta(days=1)
            )
        else:
            per_index_start[display_name] = global_start

    # Overall date range needed
    earliest_needed = min(per_index_start.values())
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    if earliest_needed > today:
        if progress_callback:
            progress_callback(1, 1, "", "All indices up to date.")
        return {k: 0 for k in indices}

    # Generate all candidate weekdays in the range
    dates: list[datetime] = []
    d = earliest_needed
    while d <= today:
        if d.weekday() < 5:   # Mon–Fri only
            dates.append(d)
        d += timedelta(days=1)

    total = len(dates)
    counts: dict[str, int] = {k: 0 for k in indices}

    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": "https://www.nseindia.com/",
    })

    import pandas as pd

    with get_connection() as conn:
        for i, dt in enumerate(dates):
            date_str = dt.strftime("%d%m%Y")   # DDMMYYYY for URL
            iso_date = dt.strftime("%Y-%m-%d")

            if progress_callback and i % 20 == 0:
                progress_callback(i, total, iso_date,
                                  f"Fetching NSE archive {iso_date} ({i}/{total})")

            url = NSE_ARCHIVE_URL.format(date=date_str)
            try:
                r = session.get(url, timeout=15)
                if r.status_code != 200:
                    continue   # holiday / weekend — file doesn't exist

                df = pd.read_csv(io.StringIO(r.text))
                # Normalise column name for close price
                close_col = next(
                    (c for c in df.columns if "clos" in c.lower()), None
                )
                if close_col is None:
                    continue

                # Match rows to requested indices (case-insensitive)
                df["_key"] = df["Index Name"].str.lower().str.strip()
                matched = df[df["_key"].isin(name_map)]

                records = []
                for _, row in matched.iterrows():
                    display_name = name_map[row["_key"]]
                    # Skip if this date is before the per-index start
                    if dt < per_index_start[display_name]:
                        continue
                    try:
                        close_val = float(row[close_col])
                    except (ValueError, TypeError):
                        continue
                    if close_val <= 0:
                        continue
                    records.append((display_name, iso_date, close_val))
                    counts[display_name] += 1

                if records:
                    bulk_insert_benchmark(conn, records)

            except Exception as e:
                logger.debug(f"NSE archive {date_str}: {e}")

            time.sleep(0.05)   # ~50 ms — polite but fast

        conn.commit()

    if progress_callback:
        progress_callback(total, total, "", "Done!")

    return counts


def get_scheme_codes_for_equity() -> list[int]:
    """Get scheme codes for all Direct Growth equity funds in the database."""
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT scheme_code, scheme_name FROM funds
               WHERE scheme_category LIKE '%Equity%'
                  OR scheme_category LIKE '%ELSS%'
               ORDER BY scheme_code"""
        ).fetchall()
    return [r[0] for r in rows if is_direct_growth_plan(r[1])]


def refresh_scheme_list():
    """Fetch latest scheme list from AMFI and update the database.

    Returns (total_schemes, equity_count).
    """
    all_schemes = fetch_scheme_list()
    equity_schemes = filter_equity_schemes(all_schemes)
    save_scheme_list_to_db(equity_schemes)
    return len(all_schemes), len(equity_schemes)
