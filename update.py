#!/usr/bin/env python3
"""
Daily updater for MF Analytics.

Run standalone:  python update.py
Schedule via cron:  0 22 * * 1-5  cd /path/to/mf_analytics && python update.py

Or schedule within Python using the --schedule flag:
    python update.py --schedule    # runs update at 22:00 IST every weekday
"""

import argparse
import logging
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db import init_db
from fetcher import (
    download_nav_for_schemes,
    fetch_benchmark_data,
    get_scheme_codes_for_equity,
    refresh_scheme_list,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "update.log")),
    ],
)
logger = logging.getLogger(__name__)


def run_daily_update():
    """Execute a full daily update cycle."""
    logger.info("=" * 60)
    logger.info("Starting daily update")
    logger.info("=" * 60)

    init_db()

    # 1. Refresh scheme list
    logger.info("Step 1: Refreshing scheme list from AMFI...")
    try:
        total, equity = refresh_scheme_list()
        logger.info(f"Scheme list: {total} total, {equity} equity schemes")
    except Exception as e:
        logger.error(f"Failed to refresh scheme list: {e}")
        return False

    # 2. Download/update NAV data
    logger.info("Step 2: Updating NAV data...")
    codes = get_scheme_codes_for_equity()
    logger.info(f"Updating {len(codes)} equity schemes")

    def progress(current, total, code, msg):
        if current % 100 == 0 or current == total:
            logger.info(f"  [{current}/{total}] {msg}")

    try:
        updated = download_nav_for_schemes(codes, progress_callback=progress)
        logger.info(f"NAV update complete: {updated} schemes updated")
    except Exception as e:
        logger.error(f"NAV update failed: {e}")

    # 3. Update benchmark data
    logger.info("Step 3: Updating benchmark data...")
    try:
        fetch_benchmark_data()
        logger.info("Benchmark update complete")
    except Exception as e:
        logger.error(f"Benchmark update failed: {e}")

    logger.info("Daily update finished")
    logger.info("=" * 60)
    return True


def run_scheduled(run_time: str = "22:00"):
    """Run the updater on a schedule using the schedule library."""
    import schedule

    logger.info(f"Scheduling daily update at {run_time}")
    schedule.every().monday.at(run_time).do(run_daily_update)
    schedule.every().tuesday.at(run_time).do(run_daily_update)
    schedule.every().wednesday.at(run_time).do(run_daily_update)
    schedule.every().thursday.at(run_time).do(run_daily_update)
    schedule.every().friday.at(run_time).do(run_daily_update)

    logger.info("Scheduler started. Waiting for next run...")
    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MF Analytics Daily Updater")
    parser.add_argument("--schedule", action="store_true",
                       help="Run on a recurring schedule instead of once")
    parser.add_argument("--time", default="22:00",
                       help="Time to run daily update (HH:MM, default 22:00)")
    args = parser.parse_args()

    if args.schedule:
        run_scheduled(args.time)
    else:
        success = run_daily_update()
        sys.exit(0 if success else 1)
