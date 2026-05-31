"""
scraper.py — FinMine Analytics Engine
Data mining engine: fetches ticker universe, downloads 10-year OHLCV
and financial statements from yfinance / SEC EDGAR with rate-limiting,
exponential backoff, and checkpoint-based resume logic.
"""

import io
import logging
import time
from datetime import date, datetime, timedelta
from typing import Callable

import pandas as pd
import requests
import yfinance as yf

import database as db

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

YEARS_HISTORY = 10
PRICE_START   = str((date.today() - timedelta(days=365 * YEARS_HISTORY + 30)))
PRICE_END     = str(date.today())

BATCH_SIZE    = 25       # tickers per yfinance batch download
RATE_DELAY    = 0.4      # base sleep between individual ticker fundamentals
MAX_RETRIES   = 5

_USER_AGENT   = "FinMineBot/1.0 leolocpham@gmail.com"

# ─────────────────────────────────────────────────────────────────────────────
# TICKER UNIVERSE
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_wikipedia_sp500() -> list[dict]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; FinMineBot/1.0)"}
    html = requests.get(url, headers=headers, timeout=15).text
    tbl  = pd.read_html(io.StringIO(html), header=0)[0]
    tbl.columns = [str(c).strip() for c in tbl.columns]
    sym_col = next(c for c in tbl.columns if "symbol"   in c.lower())
    sec_col = next((c for c in tbl.columns if "sector"   in c.lower()), None)
    nam_col = next((c for c in tbl.columns if "security" in c.lower()
                    or "name" in c.lower()), None)
    tbl[sym_col] = tbl[sym_col].str.replace(".", "-", regex=False)
    out = []
    for _, r in tbl.iterrows():
        out.append({
            "ticker":   str(r[sym_col]).strip(),
            "name":     str(r[nam_col]).strip() if nam_col else "",
            "sector":   str(r[sec_col]).strip() if sec_col else "",
            "exchange": "NYSE/NASDAQ",
        })
    return out


def _fetch_nasdaq_screener() -> list[dict]:
    """
    Pull all US equities from NASDAQ's public screener API.
    Falls back gracefully if unavailable.
    """
    url = ("https://api.nasdaq.com/api/screener/stocks"
           "?tableonly=true&limit=10000&exchange=nasdaq|nyse|amex")
    headers = {"User-Agent": _USER_AGENT, "Accept": "application/json"}
    try:
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        rows = r.json()["data"]["table"]["rows"]
        out = []
        for row in rows:
            sym = str(row.get("symbol", "")).strip()
            if sym and sym.isalpha():   # skip warrants / rights (contain ^/+)
                out.append({
                    "ticker":   sym,
                    "name":     row.get("name", ""),
                    "sector":   row.get("sector", ""),
                    "exchange": row.get("exchange", ""),
                    "industry": row.get("industry", ""),
                })
        return out
    except Exception as exc:
        log.warning("NASDAQ screener unavailable (%s); falling back to S&P 500 only", exc)
        return []


def get_ticker_universe(use_sp500_only: bool = False) -> list[dict]:
    """
    Returns list of {ticker, name, sector, exchange, industry} dicts.
    By default pulls the full NASDAQ screener (~7-9 k tickers).
    set use_sp500_only=True for faster testing / smaller runs.
    """
    if use_sp500_only:
        return _fetch_wikipedia_sp500()

    tickers = _fetch_nasdaq_screener()
    if not tickers:
        tickers = _fetch_wikipedia_sp500()

    # De-duplicate preserving first occurrence
    seen: set[str] = set()
    result = []
    for t in tickers:
        if t["ticker"] not in seen:
            seen.add(t["ticker"])
            result.append(t)
    log.info("Universe: %d tickers loaded", len(result))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# RETRY / RATE-LIMIT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _retry(fn: Callable, retries: int = MAX_RETRIES, base_delay: float = 1.0):
    """Exponential backoff wrapper; returns None on terminal failure."""
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:
            if attempt == retries - 1:
                log.debug("All %d retries exhausted: %s", retries, exc)
                return None
            wait = base_delay * (2 ** attempt)
            log.debug("Retry %d/%d in %.1fs — %s", attempt + 1, retries, wait, exc)
            time.sleep(wait)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# PRICE DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────

def download_prices_batch(tickers: list[str],
                          progress_cb: Callable[[str], None] | None = None) -> dict[str, int]:
    """
    Batch-downloads OHLCV for a list of tickers using yfinance.
    Returns {ticker: rows_written}.
    """
    results: dict[str, int] = {}

    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i : i + BATCH_SIZE]
        log.info("Price batch %d-%d / %d", i + 1, i + len(batch), len(tickers))

        def _dl():
            return yf.download(
                batch,
                start=PRICE_START, end=PRICE_END,
                interval="1d",
                auto_adjust=True,
                actions=True,
                threads=True,
                progress=False,
                group_by="ticker",
            )

        raw = _retry(_dl)
        if raw is None or raw.empty:
            for t in batch:
                db.checkpoint_set(t, "error", "price download returned empty")
            continue

        for ticker in batch:
            try:
                df_t = _extract_ticker_df(raw, ticker)
                if df_t is None or df_t.empty:
                    db.checkpoint_set(ticker, "error", "no price rows")
                    results[ticker] = 0
                    continue

                df_t = df_t.reset_index()
                df_t.columns = [str(c).lower().replace(" ", "_") for c in df_t.columns]
                if "date" not in df_t.columns and "datetime" in df_t.columns:
                    df_t.rename(columns={"datetime": "date"}, inplace=True)
                df_t["date"] = df_t["date"].astype(str).str[:10]

                # Ensure dividend/splits columns exist
                for col in ("dividends", "stock_splits"):
                    if col not in df_t.columns:
                        df_t[col] = 0.0
                df_t.rename(columns={"stock_splits": "splits"}, inplace=True)

                n = db.upsert_prices(ticker, df_t)
                results[ticker] = n
                db.checkpoint_set(ticker, "prices_done")
                if progress_cb:
                    progress_cb(ticker)

            except Exception as exc:
                log.warning("Price store failed for %s: %s", ticker, exc)
                db.checkpoint_set(ticker, "error", str(exc))
                results[ticker] = 0

        time.sleep(RATE_DELAY)

    return results


def _extract_ticker_df(raw: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    if raw is None or raw.empty:
        return None
    try:
        cols = raw.columns
        if isinstance(cols, pd.MultiIndex):
            lvl0 = cols.get_level_values(0).unique().tolist()
            lvl1 = cols.get_level_values(1).unique().tolist()
            if ticker in lvl0:
                df = raw[ticker].copy()
            elif ticker in lvl1:
                df = raw.xs(ticker, axis=1, level=1).copy()
            else:
                return None
        else:
            df = raw.copy()
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
        needed = {"open", "high", "low", "close", "volume"}
        if not needed.issubset(set(df.columns)):
            return None
        return df.dropna(subset=["close"])
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# FUNDAMENTALS DOWNLOAD (yfinance)
# ─────────────────────────────────────────────────────────────────────────────

_STMT_MAP = {
    "income":   ["income_stmt", "quarterly_income_stmt"],
    "balance":  ["balance_sheet", "quarterly_balance_sheet"],
    "cashflow": ["cashflow", "quarterly_cashflow"],
}

def _yf_stmt_to_records(df: pd.DataFrame) -> list[tuple[str, dict]]:
    """
    Converts a yfinance statement DataFrame (rows=line items, cols=period dates)
    into a list of (period_end_str, {metric: value}) tuples.
    """
    if df is None or df.empty:
        return []
    records = []
    for col in df.columns:
        try:
            period_end = str(col)[:10]
        except Exception:
            continue
        data = {}
        for idx in df.index:
            val = df.loc[idx, col]
            if pd.notna(val):
                try:
                    data[str(idx)] = float(val)
                except (TypeError, ValueError):
                    data[str(idx)] = str(val)
        if data:
            records.append((period_end, data))
    return records


def download_fundamentals_single(ticker: str) -> bool:
    """
    Downloads annual + quarterly financial statements for one ticker.
    Returns True on success.
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        db.upsert_company(
            ticker,
            exchange=info.get("exchange", ""),
            name=info.get("longName", info.get("shortName", "")),
            sector=info.get("sector", ""),
            industry=info.get("industry", ""),
        )

        stmt_mapping = [
            ("annual",    "income",   "income_stmt"),
            ("annual",    "balance",  "balance_sheet"),
            ("annual",    "cashflow", "cashflow"),
            ("quarterly", "income",   "quarterly_income_stmt"),
            ("quarterly", "balance",  "quarterly_balance_sheet"),
            ("quarterly", "cashflow", "quarterly_cashflow"),
        ]

        for period_type, stmt_type, attr in stmt_mapping:
            df_stmt = getattr(t, attr, None)
            if df_stmt is None:
                continue
            if callable(df_stmt):
                df_stmt = df_stmt
            for period_end, data in _yf_stmt_to_records(df_stmt):
                db.upsert_fundamental(ticker, period_type, period_end, stmt_type, data)

        db.checkpoint_set(ticker, "done")
        return True

    except Exception as exc:
        log.warning("Fundamentals failed for %s: %s", ticker, exc)
        db.checkpoint_set(ticker, "error", str(exc))
        return False


def download_fundamentals_batch(
    tickers: list[str],
    progress_cb: Callable[[str, int, int], None] | None = None,
) -> tuple[int, int]:
    """
    Downloads fundamentals for each ticker with rate-limiting.
    Returns (success_count, fail_count).
    """
    ok = fail = 0
    for i, ticker in enumerate(tickers):
        success = _retry(lambda t=ticker: download_fundamentals_single(t), retries=3)
        if success:
            ok += 1
        else:
            fail += 1
            db.checkpoint_set(ticker, "error", "fundamentals exhausted retries")
        if progress_cb:
            progress_cb(ticker, i + 1, len(tickers))
        time.sleep(RATE_DELAY)
    return ok, fail


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATED FULL MINE RUN
# ─────────────────────────────────────────────────────────────────────────────

def run_full_mine(
    use_sp500_only: bool = False,
    price_progress_cb: Callable | None = None,
    fund_progress_cb:  Callable | None = None,
    log_cb:            Callable[[str], None] | None = None,
) -> dict:
    """
    Full end-to-end pipeline:
      1. Load universe
      2. Seed checkpoint table with pending tickers
      3. Download prices for those still needing them
      4. Download fundamentals for those still needing them
    Returns summary stats dict.
    """
    def _log(msg: str):
        log.info(msg)
        if log_cb:
            log_cb(msg)

    db.init_db()

    _log("Loading ticker universe...")
    universe = get_ticker_universe(use_sp500_only=use_sp500_only)
    all_tickers = [u["ticker"] for u in universe]

    # Seed company table and checkpoints
    existing_cp = db.checkpoints_bulk(all_tickers)
    new_tickers  = [t for t in all_tickers if t not in existing_cp]
    _log(f"Universe: {len(all_tickers)} tickers | New: {len(new_tickers)}")

    for u in universe:
        db.upsert_company(
            u["ticker"], u.get("exchange", ""),
            u.get("name", ""), u.get("sector", ""), u.get("industry", ""),
        )
    for t in new_tickers:
        db.checkpoint_set(t, "pending")

    # Phase 1: prices
    need_prices = [
        t for t in all_tickers
        if existing_cp.get(t, "pending") in ("pending", "error", "")
        or existing_cp.get(t) is None
    ]
    _log(f"Phase 1 — downloading prices for {len(need_prices)} tickers")
    price_results = download_prices_batch(need_prices, progress_cb=price_progress_cb)
    price_ok   = sum(1 for v in price_results.values() if v > 0)
    price_fail = len(price_results) - price_ok

    # Phase 2: fundamentals
    cp_after_prices = db.checkpoints_bulk(all_tickers)
    need_funds = [
        t for t in all_tickers
        if cp_after_prices.get(t) in ("prices_done", "pending")
    ]
    _log(f"Phase 2 — downloading fundamentals for {len(need_funds)} tickers")
    fund_ok, fund_fail = download_fundamentals_batch(need_funds, progress_cb=fund_progress_cb)

    stats = {
        "universe_size":  len(all_tickers),
        "prices_ok":      price_ok,
        "prices_fail":    price_fail,
        "funds_ok":       fund_ok,
        "funds_fail":     fund_fail,
    }
    _log(f"Mining complete: {stats}")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-TICKER REFRESH (used by UI on-demand)
# ─────────────────────────────────────────────────────────────────────────────

def refresh_ticker(ticker: str, force: bool = False) -> dict:
    """
    Fetch fresh data for one ticker if stale. Returns status dict.
    """
    db.init_db()
    status = {"ticker": ticker, "prices": "skipped", "fundamentals": "skipped"}

    if force or db.prices_stale(ticker, max_age_hours=24):
        raw = _retry(lambda: yf.Ticker(ticker).history(
            start=PRICE_START, end=PRICE_END, interval="1d",
            auto_adjust=True, actions=True,
        ))
        if raw is not None and not raw.empty:
            raw = raw.reset_index()
            raw.columns = [str(c).lower().replace(" ", "_") for c in raw.columns]
            if "date" not in raw.columns and "datetime" in raw.columns:
                raw.rename(columns={"datetime": "date"}, inplace=True)
            raw["date"] = raw["date"].astype(str).str[:10]
            for col in ("dividends", "stock_splits"):
                if col not in raw.columns:
                    raw[col] = 0.0
            raw.rename(columns={"stock_splits": "splits"}, inplace=True)
            n = db.upsert_prices(ticker, raw)
            status["prices"] = f"{n} rows"
        else:
            status["prices"] = "error"

    if force or db.fundamentals_stale(ticker, max_age_days=90):
        ok = _retry(lambda: download_fundamentals_single(ticker))
        status["fundamentals"] = "ok" if ok else "error"

    return status
