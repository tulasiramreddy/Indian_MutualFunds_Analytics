#!/usr/bin/env python3
"""
One-time migration: copy all data from mf_data.db (SQLite) to PostgreSQL.

Usage:
    export DATABASE_URL="postgresql://user:password@host:5432/dbname"
    python migrate_sqlite_to_pg.py

Or pass the SQLite path explicitly:
    python migrate_sqlite_to_pg.py --sqlite mf_data.db
"""

import argparse
import os
import sqlite3
import sys

import psycopg2
import psycopg2.extras

BATCH_SIZE = 5000


def get_pg_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        sys.exit("ERROR: Set DATABASE_URL env var before running this script.")
    return url


def migrate(sqlite_path: str):
    print(f"Source : {sqlite_path}")
    print(f"Target : {os.environ.get('DATABASE_URL', '(from DATABASE_URL)')}\n")

    src = sqlite3.connect(sqlite_path)
    src.row_factory = sqlite3.Row
    dst = psycopg2.connect(get_pg_url())
    dst.autocommit = False

    try:
        _init_schema(dst)
        _migrate_table(src, dst, "funds",
                       "INSERT INTO funds VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                       7)
        _migrate_table(src, dst, "nav_history",
                       "INSERT INTO nav_history VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
                       3)
        _migrate_table(src, dst, "benchmark_data",
                       "INSERT INTO benchmark_data VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
                       3)
        _migrate_table(src, dst, "fund_tags",
                       "INSERT INTO fund_tags VALUES (%s,%s) ON CONFLICT DO NOTHING",
                       2,
                       skip_if_missing=True)
        dst.commit()
        print("\nMigration complete.")
    except Exception as e:
        dst.rollback()
        print(f"\nERROR: {e}")
        raise
    finally:
        src.close()
        dst.close()


def _init_schema(dst):
    from db import SCHEMA_SQL, _PGConn
    wrapped = _PGConn(dst)
    wrapped.executescript(SCHEMA_SQL)
    # fund_tags table (from tagger.py)
    wrapped.executescript("""
        CREATE TABLE IF NOT EXISTS fund_tags (
            scheme_code INTEGER NOT NULL,
            tag         TEXT    NOT NULL,
            PRIMARY KEY (scheme_code, tag),
            FOREIGN KEY (scheme_code) REFERENCES funds(scheme_code)
        )
    """)
    dst.commit()
    print("Schema initialised.")


def _migrate_table(src, dst, table: str, insert_sql: str, ncols: int,
                   skip_if_missing: bool = False):
    cur_src = src.cursor()
    try:
        cur_src.execute(f"SELECT COUNT(*) FROM {table}")
    except sqlite3.OperationalError:
        if skip_if_missing:
            print(f"  {table}: not found in source, skipping.")
            return
        raise

    total = cur_src.fetchone()[0]
    print(f"  {table}: {total:,} rows", end="", flush=True)

    cur_src.execute(f"SELECT * FROM {table}")
    cur_dst = dst.cursor()
    done = 0

    while True:
        batch = cur_src.fetchmany(BATCH_SIZE)
        if not batch:
            break
        psycopg2.extras.execute_batch(cur_dst, insert_sql,
                                      [tuple(row) for row in batch],
                                      page_size=BATCH_SIZE)
        done += len(batch)
        print(f"\r  {table}: {done:,}/{total:,}", end="", flush=True)

    dst.commit()
    print(f"\r  {table}: {done:,} rows migrated. ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate SQLite → PostgreSQL")
    parser.add_argument("--sqlite", default="mf_data.db",
                        help="Path to mf_data.db (default: ./mf_data.db)")
    args = parser.parse_args()

    if not os.path.exists(args.sqlite):
        sys.exit(f"ERROR: SQLite file not found: {args.sqlite}")

    migrate(args.sqlite)
