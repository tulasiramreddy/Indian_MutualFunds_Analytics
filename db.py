"""Database management for MF Analytics — PostgreSQL backend."""

import os
import re
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

import pandas as pd
import psycopg2


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS funds (
    scheme_code INTEGER PRIMARY KEY,
    scheme_name TEXT NOT NULL,
    scheme_category TEXT,
    scheme_type TEXT,
    fund_house TEXT,
    isin_growth TEXT,
    isin_div TEXT
);

CREATE TABLE IF NOT EXISTS nav_history (
    scheme_code INTEGER NOT NULL,
    date TEXT NOT NULL,
    nav REAL NOT NULL,
    PRIMARY KEY (scheme_code, date),
    FOREIGN KEY (scheme_code) REFERENCES funds(scheme_code)
);

CREATE TABLE IF NOT EXISTS benchmark_data (
    index_name TEXT NOT NULL,
    date TEXT NOT NULL,
    close_price REAL NOT NULL,
    PRIMARY KEY (index_name, date)
);

CREATE TABLE IF NOT EXISTS update_log (
    id SERIAL PRIMARY KEY,
    update_type TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    schemes_updated INTEGER DEFAULT 0,
    status TEXT DEFAULT 'running'
);

CREATE INDEX IF NOT EXISTS idx_nav_date ON nav_history(date);
CREATE INDEX IF NOT EXISTS idx_nav_scheme ON nav_history(scheme_code);
CREATE INDEX IF NOT EXISTS idx_funds_category ON funds(scheme_category);
CREATE INDEX IF NOT EXISTS idx_benchmark_date ON benchmark_data(date);
"""


def _get_database_url() -> str:
    """Resolve DATABASE_URL from Streamlit secrets or environment variable."""
    try:
        import streamlit as st
        return st.secrets["DATABASE_URL"]
    except Exception:
        pass
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL not set. Add it to .streamlit/secrets.toml or set the DATABASE_URL env var."
        )
    return url


class _PGConn:
    """Thin wrapper around a psycopg2 connection that provides a sqlite3-compatible API.

    Handles:
      - ? → %s placeholder conversion
      - INSERT OR IGNORE → INSERT ... ON CONFLICT DO NOTHING
      - executescript (not native to psycopg2)
    """

    _INSERT_OR_IGNORE_RE = re.compile(r"INSERT\s+OR\s+IGNORE\s+INTO", re.IGNORECASE)

    def __init__(self, conn):
        self._conn = conn

    def _adapt(self, sql: str) -> str:
        sql = sql.replace("?", "%s")
        if self._INSERT_OR_IGNORE_RE.search(sql):
            sql = self._INSERT_OR_IGNORE_RE.sub("INSERT INTO", sql)
            sql = sql.rstrip().rstrip(";") + " ON CONFLICT DO NOTHING"
        return sql

    def execute(self, sql: str, params=None):
        cur = self._conn.cursor()
        cur.execute(self._adapt(sql), params)
        return cur

    def executemany(self, sql: str, params_list):
        cur = self._conn.cursor()
        cur.executemany(self._adapt(sql), params_list)
        return cur

    def executescript(self, sql: str):
        cur = self._conn.cursor()
        for stmt in sql.split(";"):
            stmt = stmt.strip()
            if stmt:
                cur.execute(stmt)

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def close(self):
        self._conn.close()


@contextmanager
def get_connection(_ignored=None):
    """Context manager yielding a database connection.

    The optional positional argument is accepted but ignored (legacy sqlite compat).
    """
    conn = psycopg2.connect(_get_database_url())
    wrapped = _PGConn(conn)
    try:
        yield wrapped
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Initialize the database schema."""
    with get_connection() as conn:
        conn.executescript(SCHEMA_SQL)


def upsert_fund(conn, scheme_code: int, scheme_name: str,
                scheme_category: str = None, scheme_type: str = None,
                fund_house: str = None, isin_growth: str = None,
                isin_div: str = None):
    """Insert or update a fund record."""
    conn.execute("""
        INSERT INTO funds (scheme_code, scheme_name, scheme_category,
                          scheme_type, fund_house, isin_growth, isin_div)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (scheme_code) DO UPDATE SET
            scheme_name=EXCLUDED.scheme_name,
            scheme_category=COALESCE(EXCLUDED.scheme_category, funds.scheme_category),
            scheme_type=COALESCE(EXCLUDED.scheme_type, funds.scheme_type),
            fund_house=COALESCE(EXCLUDED.fund_house, funds.fund_house),
            isin_growth=COALESCE(EXCLUDED.isin_growth, funds.isin_growth),
            isin_div=COALESCE(EXCLUDED.isin_div, funds.isin_div)
    """, (scheme_code, scheme_name, scheme_category, scheme_type,
          fund_house, isin_growth, isin_div))


def bulk_insert_nav(conn, records: list[tuple]):
    """Insert NAV records, ignoring duplicates.

    records: list of (scheme_code, date_str, nav)
    """
    conn.executemany("""
        INSERT INTO nav_history (scheme_code, date, nav)
        VALUES (%s, %s, %s)
        ON CONFLICT DO NOTHING
    """, records)


def bulk_insert_benchmark(conn, records: list[tuple]):
    """Insert benchmark records, ignoring duplicates.

    records: list of (index_name, date_str, close_price)
    """
    conn.executemany("""
        INSERT INTO benchmark_data (index_name, date, close_price)
        VALUES (%s, %s, %s)
        ON CONFLICT DO NOTHING
    """, records)


def get_last_nav_date(conn, scheme_code: int) -> Optional[str]:
    """Get the most recent NAV date for a scheme."""
    row = conn.execute(
        "SELECT MAX(date) FROM nav_history WHERE scheme_code = %s",
        (scheme_code,)
    ).fetchone()
    return row[0] if row and row[0] else None


def get_last_benchmark_date(conn, index_name: str) -> Optional[str]:
    """Get the most recent date for a benchmark index."""
    row = conn.execute(
        "SELECT MAX(date) FROM benchmark_data WHERE index_name = %s",
        (index_name,)
    ).fetchone()
    return row[0] if row and row[0] else None


def get_fund_nav(scheme_code: int) -> pd.DataFrame:
    """Get NAV history for a fund as a DataFrame."""
    with get_connection() as conn:
        df = pd.read_sql_query(
            "SELECT date, nav FROM nav_history WHERE scheme_code = %s ORDER BY date",
            conn._conn, params=(scheme_code,), parse_dates=["date"]
        )
    return df


def get_benchmark_data(index_name: str) -> pd.DataFrame:
    """Get benchmark price history as a DataFrame."""
    with get_connection() as conn:
        df = pd.read_sql_query(
            "SELECT date, close_price FROM benchmark_data WHERE index_name = %s ORDER BY date",
            conn._conn, params=(index_name,), parse_dates=["date"]
        )
    return df


def get_all_funds() -> pd.DataFrame:
    """Get all funds metadata."""
    with get_connection() as conn:
        df = pd.read_sql_query("SELECT * FROM funds ORDER BY scheme_name", conn._conn)
    return df


def get_equity_funds() -> pd.DataFrame:
    """Get equity fund metadata."""
    with get_connection() as conn:
        df = pd.read_sql_query(
            """SELECT f.*,
                      (SELECT COUNT(*) FROM nav_history n WHERE n.scheme_code = f.scheme_code) as nav_count,
                      (SELECT MIN(date) FROM nav_history n WHERE n.scheme_code = f.scheme_code) as first_date,
                      (SELECT MAX(date) FROM nav_history n WHERE n.scheme_code = f.scheme_code) as last_date
               FROM funds f
               WHERE f.scheme_category LIKE '%%Equity%%' OR f.scheme_category LIKE '%%ELSS%%'
               ORDER BY f.scheme_name""",
            conn._conn
        )
    return df


def get_funds_with_nav() -> pd.DataFrame:
    """Get funds that have NAV data loaded."""
    with get_connection() as conn:
        df = pd.read_sql_query(
            """SELECT f.scheme_code, f.scheme_name, f.scheme_category, f.scheme_type,
                      f.fund_house, f.isin_growth, f.isin_div,
                      COUNT(n.date) as nav_count,
                      MIN(n.date) as first_date,
                      MAX(n.date) as last_date
               FROM funds f
               INNER JOIN nav_history n ON f.scheme_code = n.scheme_code
               GROUP BY f.scheme_code, f.scheme_name, f.scheme_category, f.scheme_type,
                        f.fund_house, f.isin_growth, f.isin_div
               HAVING COUNT(n.date) > 0
               ORDER BY f.scheme_name""",
            conn._conn
        )
    return df


def get_categories() -> list[str]:
    """Get distinct fund categories."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT scheme_category FROM funds WHERE scheme_category IS NOT NULL ORDER BY scheme_category"
        ).fetchall()
    return [r[0] for r in rows]


def get_fund_houses() -> list[str]:
    """Get distinct fund houses."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT fund_house FROM funds WHERE fund_house IS NOT NULL ORDER BY fund_house"
        ).fetchall()
    return [r[0] for r in rows]


def get_db_stats() -> dict:
    """Get database statistics."""
    with get_connection() as conn:
        stats = {}
        stats["total_funds"] = conn.execute("SELECT COUNT(*) FROM funds").fetchone()[0]
        stats["funds_with_nav"] = conn.execute(
            "SELECT COUNT(DISTINCT scheme_code) FROM nav_history"
        ).fetchone()[0]
        stats["total_nav_records"] = conn.execute("SELECT COUNT(*) FROM nav_history").fetchone()[0]
        stats["benchmark_records"] = conn.execute("SELECT COUNT(*) FROM benchmark_data").fetchone()[0]
        stats["benchmarks_loaded"] = [
            r[0] for r in conn.execute(
                "SELECT DISTINCT index_name FROM benchmark_data"
            ).fetchall()
        ]

        last_update = conn.execute(
            "SELECT finished_at, status FROM update_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if last_update:
            stats["last_update"] = last_update[0]
            stats["last_update_status"] = last_update[1]
        else:
            stats["last_update"] = None
            stats["last_update_status"] = None

    return stats


def log_update_start(conn, update_type: str) -> int:
    """Log the start of an update operation. Returns the log ID."""
    cursor = conn.execute(
        "INSERT INTO update_log (update_type, started_at, status) VALUES (%s, %s, 'running') RETURNING id",
        (update_type, datetime.now().isoformat())
    )
    conn.commit()
    return cursor.fetchone()[0]


def log_update_finish(conn, log_id: int, schemes_updated: int, status: str = "completed"):
    """Log the completion of an update operation."""
    conn.execute(
        "UPDATE update_log SET finished_at=%s, schemes_updated=%s, status=%s WHERE id=%s",
        (datetime.now().isoformat(), schemes_updated, status, log_id)
    )
    conn.commit()
