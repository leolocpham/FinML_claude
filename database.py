"""
database.py — FinMine Analytics Engine
Permanent local SQLite database with WAL mode, normalized schema,
and read/write helpers used by every other module.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

DB_PATH = Path("finmine.db")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONNECTION MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=30, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")   # 64 MB page cache
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS companies (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker        TEXT    NOT NULL UNIQUE,
    exchange      TEXT,
    name          TEXT,
    sector        TEXT,
    industry      TEXT,
    updated_at    TEXT
);

CREATE TABLE IF NOT EXISTS historical_prices (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker     TEXT    NOT NULL,
    date       TEXT    NOT NULL,
    open       REAL,
    high       REAL,
    low        REAL,
    close      REAL,
    volume     REAL,
    dividends  REAL,
    splits     REAL,
    UNIQUE(ticker, date)
);
CREATE INDEX IF NOT EXISTS idx_prices_ticker_date ON historical_prices(ticker, date);

CREATE TABLE IF NOT EXISTS financial_fundamentals (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker         TEXT    NOT NULL,
    period_type    TEXT    NOT NULL,   -- 'annual' | 'quarterly'
    period_end     TEXT    NOT NULL,   -- YYYY-MM-DD
    statement_type TEXT    NOT NULL,   -- 'income' | 'balance' | 'cashflow'
    data_json      TEXT    NOT NULL,   -- full line-item dict
    updated_at     TEXT,
    UNIQUE(ticker, period_type, period_end, statement_type)
);
CREATE INDEX IF NOT EXISTS idx_fund_ticker ON financial_fundamentals(ticker);

CREATE TABLE IF NOT EXISTS ml_recommendations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT    NOT NULL UNIQUE,
    score           REAL,
    label           TEXT,
    features_json   TEXT,
    compiled_at     TEXT
);

CREATE TABLE IF NOT EXISTS ai_summaries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker      TEXT    NOT NULL UNIQUE,
    markdown    TEXT    NOT NULL,
    model       TEXT,
    created_at  TEXT
);

CREATE TABLE IF NOT EXISTS mining_checkpoint (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker      TEXT    NOT NULL UNIQUE,
    status      TEXT    NOT NULL,   -- 'pending' | 'prices_done' | 'fundamentals_done' | 'done' | 'error'
    error_msg   TEXT,
    updated_at  TEXT
);
"""

def init_db() -> None:
    with get_conn() as conn:
        for stmt in _DDL.strip().split(";"):
            s = stmt.strip()
            if s:
                conn.execute(s)
    log.info("Database initialised at %s", DB_PATH.resolve())


# ─────────────────────────────────────────────────────────────────────────────
# COMPANIES
# ─────────────────────────────────────────────────────────────────────────────

def upsert_company(ticker: str, exchange: str = "", name: str = "",
                   sector: str = "", industry: str = "") -> None:
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO companies (ticker, exchange, name, sector, industry, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                exchange   = excluded.exchange,
                name       = excluded.name,
                sector     = excluded.sector,
                industry   = excluded.industry,
                updated_at = excluded.updated_at
        """, (ticker, exchange, name, sector, industry, _now()))


def get_all_companies() -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql("SELECT * FROM companies ORDER BY ticker", conn)


def company_exists(ticker: str) -> bool:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM companies WHERE ticker=?", (ticker,)
        ).fetchone()
        return row is not None


# ─────────────────────────────────────────────────────────────────────────────
# HISTORICAL PRICES
# ─────────────────────────────────────────────────────────────────────────────

def upsert_prices(ticker: str, df: pd.DataFrame) -> int:
    """
    df must have columns: date (str YYYY-MM-DD), open, high, low, close,
    volume, dividends (optional), splits (optional).
    Returns number of rows written.
    """
    rows = []
    for _, r in df.iterrows():
        rows.append((
            ticker,
            str(r["date"])[:10],
            _f(r, "open"), _f(r, "high"), _f(r, "low"),
            _f(r, "close"), _f(r, "volume"),
            _f(r, "dividends"), _f(r, "splits"),
        ))
    with get_conn() as conn:
        conn.executemany("""
            INSERT INTO historical_prices
                (ticker, date, open, high, low, close, volume, dividends, splits)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, date) DO UPDATE SET
                open=excluded.open, high=excluded.high, low=excluded.low,
                close=excluded.close, volume=excluded.volume,
                dividends=excluded.dividends, splits=excluded.splits
        """, rows)
    return len(rows)


def get_prices(ticker: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    sql = "SELECT * FROM historical_prices WHERE ticker=?"
    params: list[Any] = [ticker]
    if start:
        sql += " AND date >= ?"
        params.append(start)
    if end:
        sql += " AND date <= ?"
        params.append(end)
    sql += " ORDER BY date"
    with get_conn() as conn:
        df = pd.read_sql(sql, conn, params=params)
    return df


def prices_last_date(ticker: str) -> str | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT MAX(date) as d FROM historical_prices WHERE ticker=?", (ticker,)
        ).fetchone()
        return row["d"] if row and row["d"] else None


def prices_stale(ticker: str, max_age_hours: int = 24) -> bool:
    last = prices_last_date(ticker)
    if not last:
        return True
    cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
    return datetime.fromisoformat(last) < cutoff


# ─────────────────────────────────────────────────────────────────────────────
# FINANCIAL FUNDAMENTALS
# ─────────────────────────────────────────────────────────────────────────────

def upsert_fundamental(ticker: str, period_type: str, period_end: str,
                       statement_type: str, data: dict) -> None:
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO financial_fundamentals
                (ticker, period_type, period_end, statement_type, data_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, period_type, period_end, statement_type) DO UPDATE SET
                data_json  = excluded.data_json,
                updated_at = excluded.updated_at
        """, (ticker, period_type, period_end, statement_type,
              json.dumps(data), _now()))


def get_fundamentals(ticker: str, period_type: str = "annual",
                     statement_type: str | None = None) -> pd.DataFrame:
    sql = ("SELECT period_end, statement_type, data_json FROM financial_fundamentals "
           "WHERE ticker=? AND period_type=?")
    params: list[Any] = [ticker, period_type]
    if statement_type:
        sql += " AND statement_type=?"
        params.append(statement_type)
    sql += " ORDER BY period_end"
    with get_conn() as conn:
        df = pd.read_sql(sql, conn, params=params)
    if df.empty:
        return df
    df["data"] = df["data_json"].apply(json.loads)
    return df


def fundamentals_stale(ticker: str, max_age_days: int = 90) -> bool:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT MAX(updated_at) as d FROM financial_fundamentals WHERE ticker=?",
            (ticker,)
        ).fetchone()
    if not row or not row["d"]:
        return True
    cutoff = datetime.utcnow() - timedelta(days=max_age_days)
    return datetime.fromisoformat(row["d"][:19]) < cutoff


# ─────────────────────────────────────────────────────────────────────────────
# ML RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────

def upsert_ml_recommendation(ticker: str, score: float,
                              label: str, features: dict) -> None:
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO ml_recommendations
                (ticker, score, label, features_json, compiled_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                score         = excluded.score,
                label         = excluded.label,
                features_json = excluded.features_json,
                compiled_at   = excluded.compiled_at
        """, (ticker, score, label, json.dumps(features), _now()))


def get_ml_recommendations(min_score: float = 0.0) -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql(
            "SELECT ticker, score, label, compiled_at "
            "FROM ml_recommendations WHERE score >= ? ORDER BY score DESC",
            conn, params=[min_score]
        )
    return df


def get_ml_recommendation(ticker: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM ml_recommendations WHERE ticker=?", (ticker,)
        ).fetchone()
    if not row:
        return None
    d = dict(row)
    d["features"] = json.loads(d.get("features_json") or "{}")
    return d


# ─────────────────────────────────────────────────────────────────────────────
# AI SUMMARIES
# ─────────────────────────────────────────────────────────────────────────────

def upsert_ai_summary(ticker: str, markdown: str, model: str = "") -> None:
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO ai_summaries (ticker, markdown, model, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                markdown   = excluded.markdown,
                model      = excluded.model,
                created_at = excluded.created_at
        """, (ticker, markdown, model, _now()))


def get_ai_summary(ticker: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM ai_summaries WHERE ticker=?", (ticker,)
        ).fetchone()
    return dict(row) if row else None


# ─────────────────────────────────────────────────────────────────────────────
# MINING CHECKPOINT
# ─────────────────────────────────────────────────────────────────────────────

def checkpoint_set(ticker: str, status: str, error_msg: str = "") -> None:
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO mining_checkpoint (ticker, status, error_msg, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                status     = excluded.status,
                error_msg  = excluded.error_msg,
                updated_at = excluded.updated_at
        """, (ticker, status, error_msg, _now()))


def checkpoint_get(ticker: str) -> str | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT status FROM mining_checkpoint WHERE ticker=?", (ticker,)
        ).fetchone()
    return row["status"] if row else None


def checkpoints_bulk(tickers: list[str]) -> dict[str, str]:
    if not tickers:
        return {}
    placeholders = ",".join("?" * len(tickers))
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT ticker, status FROM mining_checkpoint WHERE ticker IN ({placeholders})",
            tickers,
        ).fetchall()
    return {r["ticker"]: r["status"] for r in rows}


def get_pending_tickers(all_tickers: list[str]) -> list[str]:
    done_statuses = {"done", "prices_done", "fundamentals_done"}
    existing = checkpoints_bulk(all_tickers)
    return [t for t in all_tickers if existing.get(t) not in done_statuses]


# ─────────────────────────────────────────────────────────────────────────────
# STATS / HOUSEKEEPING
# ─────────────────────────────────────────────────────────────────────────────

def db_stats() -> dict:
    with get_conn() as conn:
        stats = {}
        for tbl in ("companies", "historical_prices", "financial_fundamentals",
                    "ml_recommendations", "ai_summaries", "mining_checkpoint"):
            row = conn.execute(f"SELECT COUNT(*) as n FROM {tbl}").fetchone()
            stats[tbl] = row["n"]
        size_bytes = DB_PATH.stat().st_size if DB_PATH.exists() else 0
        stats["db_size_mb"] = round(size_bytes / 1_048_576, 2)
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")

def _f(row: Any, col: str) -> float | None:
    v = row.get(col) if hasattr(row, "get") else getattr(row, col, None)
    if v is None:
        return None
    try:
        f = float(v)
        return None if (f != f) else f   # NaN guard
    except (TypeError, ValueError):
        return None
