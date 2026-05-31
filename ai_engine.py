"""
ai_engine.py — FinMine Analytics Engine
Claude API integration: assembles 10-year financial data payload,
calls Claude API, and caches the response permanently in ai_summaries.
"""

import json
import logging
from datetime import datetime

import anthropic
import pandas as pd

import database as db

log = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-6"

_SYSTEM_PROMPT = (
    "You are a Senior Institutional Value Investor specializing in corporate forensic "
    "accounting and equity research. Analyze the following 10-year financial data packet "
    "for {ticker}. Provide an objective, data-backed analytical breakdown organized "
    "strictly into these sections:\n\n"
    "1. **Financial Strength Assessment:** Evaluate balance sheet health, solvency risks, "
    "debt servicing capacity, and cash runway stability.\n\n"
    "2. **Operational Efficiency & Moat:** Analyze profit margins, asset efficiency, and "
    "whether the historical trajectory points to a strengthening or decaying competitive advantage.\n\n"
    "3. **Long-Term Investment Outlook:** Synthesize the value metrics (P/E, P/B, FCF Yield) "
    "against growth histories. Conclude with a definitive long-term qualitative posture: "
    "Is the stock structurally positioned as a Long-Term Buy, Hold, or Sell candidate "
    "based strictly on underlying value investing principles? Do not provide short-term "
    "momentum trading advice."
)


# ─────────────────────────────────────────────────────────────────────────────
# DATA PAYLOAD ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────

def _series_from_fundamentals(ann_df: pd.DataFrame, stmt_type: str,
                               key: str) -> dict[str, float]:
    sub = ann_df[ann_df["statement_type"] == stmt_type].copy()
    result = {}
    for _, row in sub.iterrows():
        data = row["data"] if isinstance(row["data"], dict) else json.loads(row.get("data_json", "{}"))
        for k, v in data.items():
            if k.lower().replace(" ", "").replace("_", "") == key.lower().replace(" ", "").replace("_", ""):
                try:
                    result[row["period_end"]] = float(v)
                except (TypeError, ValueError):
                    pass
                break
    return dict(sorted(result.items()))


def _fmt_billions(val: float | None) -> str:
    if val is None:
        return "N/A"
    if abs(val) >= 1e12:
        return f"${val/1e12:.2f}T"
    if abs(val) >= 1e9:
        return f"${val/1e9:.2f}B"
    if abs(val) >= 1e6:
        return f"${val/1e6:.2f}M"
    return f"${val:.0f}"


def _fmt_pct(val: float | None) -> str:
    return f"{val*100:.1f}%" if val is not None else "N/A"


def assemble_payload(ticker: str) -> str:
    """
    Builds a clean text financial summary for the Claude prompt.
    """
    company_df = db.get_all_companies()
    company_row = company_df[company_df["ticker"] == ticker]
    company_name = company_row["name"].values[0] if not company_row.empty else ticker
    sector       = company_row["sector"].values[0] if not company_row.empty else "Unknown"

    ann_df = db.get_fundamentals(ticker, period_type="annual")
    if ann_df.empty:
        return f"No fundamental data available in the local database for {ticker}."

    ann_df["data"] = ann_df["data_json"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )

    def series(stmt: str, key: str) -> dict[str, float]:
        return _series_from_fundamentals(ann_df, stmt, key)

    revenue       = series("income", "TotalRevenue")
    net_income    = series("income", "NetIncome")
    op_income     = series("income", "OperatingIncome")
    total_debt    = series("balance", "TotalDebt")
    total_cash    = series("balance", "CashAndCashEquivalents")
    equity        = series("balance", "StockholdersEquity")
    fcf           = series("cashflow", "FreeCashFlow")
    capex         = series("cashflow", "CapitalExpenditure")

    # Price data for valuation context
    prices_df = db.get_prices(ticker)
    latest_price: float | None = None
    if not prices_df.empty:
        latest_price = float(prices_df.sort_values("date")["close"].iloc[-1])

    # Build tables
    def _table(title: str, d: dict[str, float], fmt_fn=_fmt_billions) -> str:
        if not d:
            return f"\n{title}: No data\n"
        lines = [f"\n{title}:"]
        for yr, val in list(d.items())[-10:]:
            lines.append(f"  {yr[:7]:>7}  {fmt_fn(val)}")
        return "\n".join(lines)

    # Operating margin calculation
    op_margin_lines = ["\nOperating Margin:"]
    for yr in sorted(set(op_income.keys()) & set(revenue.keys()))[-10:]:
        r = revenue.get(yr)
        o = op_income.get(yr)
        if r and r != 0:
            op_margin_lines.append(f"  {yr[:7]:>7}  {_fmt_pct(o/r)}")

    # D/E ratio
    de_lines = ["\nDebt-to-Equity:"]
    for yr in sorted(set(total_debt.keys()) & set(equity.keys()))[-10:]:
        d_val = total_debt.get(yr, 0)
        e_val = equity.get(yr)
        if e_val and e_val != 0:
            de_lines.append(f"  {yr[:7]:>7}  {d_val/e_val:.2f}x")

    # ML recommendation if available
    ml_rec = db.get_ml_recommendation(ticker)
    ml_section = ""
    if ml_rec:
        ml_section = (
            f"\n--- ML ENGINE OUTPUT ---\n"
            f"Confidence Score: {ml_rec['score']:.3f}\n"
            f"Label: {ml_rec['label']}\n"
        )

    payload = (
        f"=== FINMINE 10-YEAR FINANCIAL ANALYSIS PACKET ===\n"
        f"Ticker: {ticker}\n"
        f"Company: {company_name}\n"
        f"Sector: {sector}\n"
        f"Latest Price: {_fmt_billions(latest_price) if latest_price else 'N/A'}\n"
        f"Report Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"{ml_section}"
        f"\n--- INCOME STATEMENT (Annual) ---"
        f"{_table('Revenue', revenue)}"
        f"{_table('Net Income', net_income)}"
        f"{_table('Operating Income', op_income)}"
        f"\n{''.join(op_margin_lines)}"
        f"\n--- BALANCE SHEET (Annual) ---"
        f"{_table('Total Debt', total_debt)}"
        f"{_table('Cash & Equivalents', total_cash)}"
        f"{_table('Stockholders Equity', equity)}"
        f"\n{''.join(de_lines)}"
        f"\n--- CASH FLOW (Annual) ---"
        f"{_table('Free Cash Flow', fcf)}"
        f"{_table('Capital Expenditures', capex)}"
        f"\n\n[End of data packet for {ticker}]"
    )
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# CLAUDE API CALL
# ─────────────────────────────────────────────────────────────────────────────

def generate_ai_summary(ticker: str, api_key: str, force_refresh: bool = False) -> str:
    """
    Returns the AI fundamental summary for ticker.
    Checks cache first; calls Claude API only when needed.
    Persists response to ai_summaries table.
    """
    if not force_refresh:
        cached = db.get_ai_summary(ticker)
        if cached:
            return cached["markdown"]

    payload  = assemble_payload(ticker)
    system   = _SYSTEM_PROMPT.format(ticker=ticker)

    client = anthropic.Anthropic(api_key=api_key)

    try:
        message = client.messages.create(
            model=_MODEL,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": payload}],
        )
        response_text = message.content[0].text
    except anthropic.APIError as exc:
        log.error("Anthropic API error for %s: %s", ticker, exc)
        raise

    db.upsert_ai_summary(ticker, response_text, model=_MODEL)
    return response_text
