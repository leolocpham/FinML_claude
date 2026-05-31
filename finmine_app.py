"""
finmine_app.py — FinMine Analytics Engine
Main Streamlit UI.

Run:  streamlit run finmine_app.py
"""

import json
import logging
import threading
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
import streamlit as st

import database as db
import scraper
import ml_engine
import ai_engine

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(name)s — %(message)s")
log = logging.getLogger(__name__)

db.init_db()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FinMine Analytics Engine",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  [data-testid="stMetricValue"]   { font-size: 1.25rem; }
  .stTabs [data-baseweb="tab"]    { font-size: 0.95rem; }
  code { font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — API KEY + DB STATS
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=56)
    st.title("FinMine Analytics")
    st.caption("Production-grade equity research engine")
    st.divider()

    api_key_input = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Required only for the AI Deep-Dive feature.",
    )

    st.divider()
    st.subheader("Database Status")
    stats = db.db_stats()
    c1, c2 = st.columns(2)
    c1.metric("Companies",    f"{stats['companies']:,}")
    c2.metric("Price Rows",   f"{stats['historical_prices']:,}")
    c1.metric("Fundamentals", f"{stats['financial_fundamentals']:,}")
    c2.metric("ML Recs",      f"{stats['ml_recommendations']:,}")
    c1.metric("AI Summaries", f"{stats['ai_summaries']:,}")
    c2.metric("DB Size",      f"{stats['db_size_mb']} MB")

    st.divider()
    st.caption("All data is stored locally in `finmine.db`. "
               "No data leaves your machine except for AI summary calls.")


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab_mine, tab_stock, tab_ml, tab_help = st.tabs([
    "⛏️ Data Mine",
    "📊 Stock Analysis",
    "🤖 ML Rankings",
    "❓ Help",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA MINE
# ═════════════════════════════════════════════════════════════════════════════

with tab_mine:
    st.header("⛏️ Data Mining Engine")
    st.caption(
        "Download 10 years of OHLCV + financial statements for the full US equity universe "
        "or just the S&P 500. Mining resumes from checkpoint if interrupted."
    )

    col_a, col_b = st.columns([1, 2])
    with col_a:
        universe_mode = st.radio(
            "Universe",
            ["S&P 500 only (~500 tickers, ~10 min)",
             "Full US equity universe (~7,000 tickers, several hours)"],
            index=0,
        )
        run_ml_after = st.checkbox("Run ML pipeline automatically after mine", value=True)
        force_refresh_mine = st.checkbox("Force re-download even if data exists", value=False)

    with col_b:
        st.info(
            "**Checkpoint system:** If mining is interrupted, click 'Start Mine' again — "
            "it will skip tickers already marked `done` and resume where it left off. "
            "No duplicate API calls are made."
        )

    start_mine_btn = st.button("Start Data Mine", type="primary", key="start_mine")

    if start_mine_btn:
        use_sp500 = "S&P 500" in universe_mode

        log_container   = st.empty()
        prog_price      = st.progress(0, text="Prices: waiting...")
        prog_fund       = st.progress(0, text="Fundamentals: waiting...")
        status_box      = st.empty()
        log_lines: list[str] = []

        def _log_cb(msg: str):
            log_lines.append(f"[{datetime.utcnow().strftime('%H:%M:%S')}] {msg}")
            log_container.code("\n".join(log_lines[-20:]))

        price_total   = {"n": 1}
        price_done    = {"n": 0}
        fund_total    = {"n": 1}
        fund_done     = {"n": 0}

        def _price_cb(ticker: str):
            price_done["n"] += 1
            pct = min(price_done["n"] / max(price_total["n"], 1), 1.0)
            prog_price.progress(pct, text=f"Prices: {price_done['n']}/{price_total['n']} — {ticker}")

        def _fund_cb(ticker: str, done: int, total: int):
            fund_total["n"] = total
            fund_done["n"]  = done
            pct = min(done / max(total, 1), 1.0)
            prog_fund.progress(pct, text=f"Fundamentals: {done}/{total} — {ticker}")

        with st.spinner("Mining in progress..."):
            try:
                # Pre-flight: get universe size to set progress denominators
                universe = scraper.get_ticker_universe(use_sp500_only=use_sp500)
                price_total["n"] = len(universe)

                if force_refresh_mine:
                    for u in universe:
                        db.checkpoint_set(u["ticker"], "pending")

                mine_stats = scraper.run_full_mine(
                    use_sp500_only=use_sp500,
                    price_progress_cb=_price_cb,
                    fund_progress_cb=_fund_cb,
                    log_cb=_log_cb,
                )
                status_box.success(
                    f"Mine complete — {mine_stats['universe_size']} tickers | "
                    f"Prices OK: {mine_stats['prices_ok']} | "
                    f"Fundamentals OK: {mine_stats['funds_ok']}"
                )
            except Exception as exc:
                status_box.error(f"Mine failed: {exc}")
                st.stop()

        if run_ml_after:
            ml_log_lines: list[str] = []
            ml_log_box    = st.empty()
            ml_prog       = st.progress(0, text="ML: building features...")
            feat_done     = {"n": 0}
            feat_total    = {"n": 1}

            def _feat_cb(ticker: str, done: int, total: int):
                feat_total["n"] = total
                feat_done["n"]  = done
                pct = min(done / max(total, 1), 1.0)
                ml_prog.progress(pct, text=f"ML features: {done}/{total} — {ticker}")

            def _ml_log(msg: str):
                ml_log_lines.append(f"[{datetime.utcnow().strftime('%H:%M:%S')}] {msg}")
                ml_log_box.code("\n".join(ml_log_lines[-10:]))

            with st.spinner("Running ML pipeline..."):
                ml_result = ml_engine.run_ml_pipeline(
                    feat_progress_cb=_feat_cb,
                    log_cb=_ml_log,
                )
            if ml_result.get("status") == "ok":
                st.success(
                    f"ML complete — {ml_result['tickers_scored']} stocks scored | "
                    f"STRONG BUY: {ml_result['strong_buy']} | "
                    f"BUY: {ml_result['buy']} | "
                    f"HOLD: {ml_result['hold']}"
                )
            else:
                st.warning(f"ML pipeline: {ml_result}")

        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — STOCK ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

with tab_stock:
    st.header("📊 Fundamental Stock Analysis")

    # ── Ticker selector ───────────────────────────────────────────────────
    companies_df = db.get_all_companies()
    all_tickers  = sorted(companies_df["ticker"].tolist()) if not companies_df.empty else []

    col_s1, col_s2 = st.columns([1, 3])
    with col_s1:
        if all_tickers:
            default_idx = all_tickers.index("AAPL") if "AAPL" in all_tickers else 0
            chosen_ticker = st.selectbox("Select Ticker", all_tickers, index=default_idx)
        else:
            chosen_ticker = st.text_input("Ticker (type to search)", "AAPL").upper()

        refresh_btn = st.button("Refresh from API", help="Force re-download even if cached")

    with col_s2:
        if chosen_ticker:
            comp_row = companies_df[companies_df["ticker"] == chosen_ticker] if not companies_df.empty else pd.DataFrame()
            if not comp_row.empty:
                r = comp_row.iloc[0]
                st.markdown(f"### {r.get('name', chosen_ticker)}")
                st.caption(f"{r.get('exchange','')} | {r.get('sector','')} | {r.get('industry','')}")
            else:
                st.markdown(f"### {chosen_ticker}")

    if not chosen_ticker:
        st.info("Select or type a ticker to begin analysis.")
        st.stop()

    # On-demand fetch if not in DB or stale
    if refresh_btn or not db.company_exists(chosen_ticker):
        with st.spinner(f"Fetching data for {chosen_ticker}..."):
            status = scraper.refresh_ticker(chosen_ticker, force=refresh_btn)
        col_s1.caption(f"Prices: {status['prices']} | Fundamentals: {status['fundamentals']}")

    # ── Data retrieval ────────────────────────────────────────────────────
    prices_df = db.get_prices(chosen_ticker)
    ann_df    = db.get_fundamentals(chosen_ticker, period_type="annual")
    ml_rec    = db.get_ml_recommendation(chosen_ticker)

    if prices_df.empty:
        st.warning(f"No price data found for {chosen_ticker}. Click 'Refresh from API'.")
        st.stop()

    prices_df = prices_df.sort_values("date")
    prices_df["date"] = pd.to_datetime(prices_df["date"])

    # ML badge
    if ml_rec:
        label = ml_rec["label"]
        score = ml_rec["score"]
        color = {"STRONG BUY / ALPHACLASS": "🟢", "BUY": "🔵", "HOLD / UNDERPERFORM": "🟡"}.get(label, "⚪")
        st.info(f"**ML Engine:** {color} {label} — Confidence Score: **{score:.3f}**  "
                f"_(compiled {ml_rec.get('compiled_at','')[:10]})_")

    # Quick price metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    latest_close  = float(prices_df["close"].iloc[-1])
    earliest_close= float(prices_df["close"].iloc[0])
    m1.metric("Latest Close",    f"${latest_close:,.2f}")
    m2.metric("10-yr Start",     f"${earliest_close:,.2f}")
    pct_10yr = (latest_close - earliest_close) / earliest_close * 100
    m3.metric("10-yr Return",    f"{pct_10yr:+.1f}%")
    m4.metric("Price Rows",      f"{len(prices_df):,}")
    m5.metric("Data From",       str(prices_df["date"].min())[:10])

    st.divider()

    # ── CHART 1: Revenue / Net Income / Operating Margin ─────────────────
    if not ann_df.empty:
        ann_df["data"] = ann_df["data_json"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )

        def get_annual_series(stmt_type: str, key: str) -> pd.Series:
            sub = ann_df[ann_df["statement_type"] == stmt_type]
            vals = {}
            for _, row in sub.iterrows():
                for k, v in row["data"].items():
                    if k.lower().replace(" ", "").replace("_","") == key.lower().replace(" ","").replace("_",""):
                        try:
                            vals[row["period_end"][:7]] = float(v)
                        except (TypeError, ValueError):
                            pass
                        break
            return pd.Series(vals).sort_index()

        revenue    = get_annual_series("income", "TotalRevenue")
        net_income = get_annual_series("income", "NetIncome")
        op_income  = get_annual_series("income", "OperatingIncome")
        total_debt = get_annual_series("balance", "TotalDebt")
        total_cash = get_annual_series("balance", "CashAndCashEquivalents")
        equity_s   = get_annual_series("balance", "StockholdersEquity")
        fcf        = get_annual_series("cashflow", "FreeCashFlow")
        capex      = get_annual_series("cashflow", "CapitalExpenditure")

        op_margin = (op_income / revenue.reindex(op_income.index)).dropna() * 100
        de_ratio  = (total_debt / equity_s.reindex(total_debt.index).replace(0, np.nan)).dropna()

        # Chart 1 — Revenue & Net Income + Operating Margin
        with st.expander("Chart 1 — Revenue, Net Income & Operating Margin", expanded=True):
            fig1 = sp.make_subplots(specs=[[{"secondary_y": True}]])
            if not revenue.empty:
                fig1.add_trace(go.Bar(
                    x=revenue.index, y=revenue.values / 1e9,
                    name="Revenue ($B)", marker_color="steelblue", opacity=0.75,
                ), secondary_y=False)
            if not net_income.empty:
                fig1.add_trace(go.Scatter(
                    x=net_income.index, y=net_income.values / 1e9,
                    name="Net Income ($B)", line=dict(color="limegreen", width=2),
                ), secondary_y=False)
            if not op_margin.empty:
                fig1.add_trace(go.Scatter(
                    x=op_margin.index, y=op_margin.values,
                    name="Operating Margin (%)", line=dict(color="orange", width=2, dash="dot"),
                ), secondary_y=True)
            fig1.update_layout(
                title=f"{chosen_ticker} — Revenue, Net Income & Operating Margin (10-year)",
                hovermode="x unified", height=420,
                legend=dict(orientation="h", y=-0.15),
            )
            fig1.update_yaxes(title_text="$ Billions", secondary_y=False)
            fig1.update_yaxes(title_text="Operating Margin (%)", secondary_y=True)
            st.plotly_chart(fig1, use_container_width=True)

        # Chart 2 — Debt-to-Equity & Cash Reserves
        with st.expander("Chart 2 — Debt/Equity Ratio vs. Cash Reserves", expanded=True):
            fig2 = sp.make_subplots(specs=[[{"secondary_y": True}]])
            if not de_ratio.empty:
                fig2.add_trace(go.Scatter(
                    x=de_ratio.index, y=de_ratio.values,
                    name="Debt-to-Equity", line=dict(color="tomato", width=2),
                    fill="tozeroy", fillcolor="rgba(255,99,71,0.1)",
                ), secondary_y=False)
            if not total_cash.empty:
                fig2.add_trace(go.Bar(
                    x=total_cash.index, y=total_cash.values / 1e9,
                    name="Cash & Equivalents ($B)", marker_color="mediumseagreen", opacity=0.7,
                ), secondary_y=True)
            fig2.update_layout(
                title=f"{chosen_ticker} — Debt/Equity Trajectory vs. Cash Reserves",
                hovermode="x unified", height=400,
                legend=dict(orientation="h", y=-0.15),
            )
            fig2.update_yaxes(title_text="D/E Ratio (x)", secondary_y=False)
            fig2.update_yaxes(title_text="Cash ($B)", secondary_y=True)
            st.plotly_chart(fig2, use_container_width=True)

        # Chart 3 — FCF vs. CapEx
        with st.expander("Chart 3 — Free Cash Flow vs. Capital Expenditures", expanded=True):
            fig3 = go.Figure()
            if not fcf.empty:
                fig3.add_trace(go.Bar(
                    x=fcf.index, y=fcf.values / 1e9,
                    name="Free Cash Flow ($B)", marker_color="mediumslateblue", opacity=0.8,
                ))
            if not capex.empty:
                capex_abs = capex.abs()
                fig3.add_trace(go.Bar(
                    x=capex_abs.index, y=-capex_abs.values / 1e9,
                    name="CapEx ($B, inverted)", marker_color="salmon", opacity=0.8,
                ))
            fig3.update_layout(
                title=f"{chosen_ticker} — Free Cash Flow vs. CapEx (10-year)",
                barmode="overlay", hovermode="x unified", height=400,
                yaxis_title="$ Billions",
                legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(fig3, use_container_width=True)

    else:
        st.warning("No annual fundamental data available. Run the Data Mine or click Refresh.")

    # Chart 4 — Historical Price + ML score overlay
    with st.expander("Chart 4 — Historical Price", expanded=True):
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=prices_df["date"], y=prices_df["close"],
            name="Adj. Close", line=dict(color="dodgerblue", width=1.5),
            fill="tozeroy", fillcolor="rgba(30,144,255,0.06)",
        ))
        if not prices_df["volume"].isna().all():
            fig4.add_trace(go.Bar(
                x=prices_df["date"], y=prices_df["volume"],
                name="Volume", marker_color="rgba(100,100,200,0.25)",
                yaxis="y2",
            ))
        fig4.update_layout(
            title=f"{chosen_ticker} — 10-year Adjusted Close Price",
            hovermode="x unified", height=450,
            xaxis=dict(rangeslider=dict(visible=True)),
            yaxis=dict(title="Price ($)"),
            yaxis2=dict(title="Volume", overlaying="y", side="right",
                        showgrid=False, tickformat=".2s"),
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── AI Deep-Dive ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("🤖 AI Fundamental Deep-Dive")

    cached_summary = db.get_ai_summary(chosen_ticker)
    if cached_summary:
        st.caption(f"Cached summary from {cached_summary.get('created_at','')[:10]} "
                   f"(model: {cached_summary.get('model','')})")
        st.markdown(cached_summary["markdown"])
        if st.button("Regenerate AI Summary", key="regen_ai"):
            if not api_key_input:
                st.error("Enter your Anthropic API key in the sidebar.")
            else:
                with st.spinner("Calling Claude API..."):
                    try:
                        text = ai_engine.generate_ai_summary(
                            chosen_ticker, api_key_input, force_refresh=True
                        )
                        st.markdown(text)
                        st.success("Summary regenerated and cached.")
                    except Exception as exc:
                        st.error(f"API error: {exc}")
    else:
        st.info("No cached summary. Click the button below to generate one.")
        if st.button("🤖 Generate AI Fundamental Deep-Dive", type="primary", key="gen_ai"):
            if not api_key_input:
                st.error("Enter your Anthropic API key in the sidebar.")
            else:
                with st.spinner(f"Asking Claude to analyse {chosen_ticker}..."):
                    try:
                        text = ai_engine.generate_ai_summary(chosen_ticker, api_key_input)
                        st.markdown(text)
                        st.success("Summary saved to local database.")
                    except Exception as exc:
                        st.error(f"API error: {exc}")

    # ── Raw data tables ───────────────────────────────────────────────────
    with st.expander("Raw fundamental data tables"):
        if not ann_df.empty:
            for stmt in ("income", "balance", "cashflow"):
                sub = ann_df[ann_df["statement_type"] == stmt][["period_end","data"]]
                if sub.empty:
                    continue
                st.markdown(f"**{stmt.title()} Statement (Annual)**")
                rows = []
                for _, row in sub.iterrows():
                    flat = {"period_end": row["period_end"]}
                    flat.update(row["data"] if isinstance(row["data"], dict)
                                else json.loads(row.get("data_json", "{}")))
                    rows.append(flat)
                st.dataframe(pd.DataFrame(rows).set_index("period_end"),
                             use_container_width=True)
        else:
            st.info("No fundamentals in database for this ticker.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — ML RANKINGS
# ═════════════════════════════════════════════════════════════════════════════

with tab_ml:
    st.header("🤖 ML Predictive Rankings")
    st.caption(
        "Random Forest + Gradient Boosting ensemble. "
        "Scores reflect probability of fundamental outperformance vs. market median. "
        "Updated each time the Data Mine completes."
    )

    recs_df = db.get_ml_recommendations()

    if recs_df.empty:
        st.info("No ML recommendations yet. Run the Data Mine (with ML pipeline enabled) first.")
    else:
        # ── Filters ──────────────────────────────────────────────────────
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            min_score = st.slider("Minimum Confidence Score", 0.0, 1.0, 0.0, 0.05)
        with col_f2:
            label_filter = st.multiselect(
                "Labels",
                ["STRONG BUY / ALPHACLASS", "BUY", "HOLD / UNDERPERFORM"],
                default=["STRONG BUY / ALPHACLASS", "BUY", "HOLD / UNDERPERFORM"],
            )
        with col_f3:
            top_n = st.selectbox("Top N to display", [25, 50, 100, 250, "All"], index=1)

        filtered = recs_df[recs_df["score"] >= min_score]
        if label_filter:
            filtered = filtered[filtered["label"].isin(label_filter)]
        if top_n != "All":
            filtered = filtered.head(int(top_n))

        # ── Summary metrics ───────────────────────────────────────────────
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Stocks Scored",     f"{len(recs_df):,}")
        mc2.metric("STRONG BUY",        f"{(recs_df['label']=='STRONG BUY / ALPHACLASS').sum():,}")
        mc3.metric("BUY",               f"{(recs_df['label']=='BUY').sum():,}")
        mc4.metric("HOLD/UNDERPERFORM", f"{(recs_df['label']=='HOLD / UNDERPERFORM').sum():,}")

        # ── Score distribution chart ──────────────────────────────────────
        with st.expander("Score Distribution", expanded=False):
            fig_dist = go.Figure(go.Histogram(
                x=recs_df["score"],
                nbinsx=40,
                marker_color="steelblue",
                opacity=0.8,
                name="Confidence Score",
            ))
            fig_dist.add_vline(x=0.72, line_dash="dash", line_color="limegreen",
                               annotation_text="STRONG BUY threshold (0.72)")
            fig_dist.add_vline(x=0.55, line_dash="dash", line_color="dodgerblue",
                               annotation_text="BUY threshold (0.55)")
            fig_dist.update_layout(
                title="ML Confidence Score Distribution",
                xaxis_title="Score", yaxis_title="Count", height=320,
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        # ── Rankings table ────────────────────────────────────────────────
        disp = filtered[["ticker", "score", "label", "compiled_at"]].copy()
        disp["score"] = disp["score"].round(4)
        disp["rank"]  = range(1, len(disp) + 1)
        disp = disp[["rank", "ticker", "score", "label", "compiled_at"]]

        def _color_label(v: str) -> str:
            if "STRONG" in v:
                return "color: limegreen; font-weight: bold"
            if v == "BUY":
                return "color: dodgerblue"
            return "color: orange"

        styled = disp.style.applymap(_color_label, subset=["label"])
        st.dataframe(styled, use_container_width=True, hide_index=True, height=500)

        # ── Top 20 bar chart ──────────────────────────────────────────────
        with st.expander("Top 20 by Confidence Score", expanded=True):
            top20 = filtered.head(20)
            colors = [
                "limegreen" if "STRONG" in lbl else
                "dodgerblue" if lbl == "BUY" else "orange"
                for lbl in top20["label"]
            ]
            fig_bar = go.Figure(go.Bar(
                x=top20["ticker"],
                y=top20["score"],
                marker_color=colors,
                text=[f"{s:.3f}" for s in top20["score"]],
                textposition="outside",
            ))
            fig_bar.update_layout(
                title="Top 20 ML-Ranked Equities",
                xaxis_title="Ticker", yaxis_title="Confidence Score",
                yaxis=dict(range=[0, 1.05]),
                height=420,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # ── Feature detail for selected ticker ───────────────────────────
        st.subheader("Feature Breakdown")
        ml_ticker_sel = st.selectbox("Inspect ticker features", filtered["ticker"].tolist(),
                                     key="ml_feat_sel")
        if ml_ticker_sel:
            rec = db.get_ml_recommendation(ml_ticker_sel)
            if rec and rec.get("features"):
                feat_df = pd.DataFrame(
                    [(k, v) for k, v in rec["features"].items() if v is not None],
                    columns=["Feature", "Value"],
                )
                feat_df["Value"] = feat_df["Value"].apply(
                    lambda v: f"{float(v):.4f}" if v is not None else "N/A"
                )
                st.dataframe(feat_df, use_container_width=True, hide_index=True)
            else:
                st.info("No feature data stored for this ticker.")

        # ── Run ML standalone button ──────────────────────────────────────
        st.divider()
        if st.button("Re-run ML Pipeline (uses existing DB data)", key="run_ml_standalone"):
            ml_log = st.empty()
            ml_prog2 = st.progress(0)
            ml_lines: list[str] = []
            feat_d = {"n": 0}
            feat_t = {"n": 1}

            def _fc2(ticker: str, done: int, total: int):
                feat_d["n"] = done; feat_t["n"] = total
                ml_prog2.progress(min(done / max(total, 1), 1.0),
                                  text=f"Features: {done}/{total}")

            def _ml2(msg: str):
                ml_lines.append(msg)
                ml_log.code("\n".join(ml_lines[-8:]))

            with st.spinner("Running ML pipeline..."):
                res = ml_engine.run_ml_pipeline(feat_progress_cb=_fc2, log_cb=_ml2)
            if res.get("status") == "ok":
                st.success(f"ML complete: {res['tickers_scored']} scored")
                st.rerun()
            else:
                st.warning(f"ML result: {res}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — HELP
# ═════════════════════════════════════════════════════════════════════════════

with tab_help:
    st.header("❓ FinMine Analytics Engine — User Guide")
    st.markdown("""
## Quick Start

1. **First run:** Go to the **⛏️ Data Mine** tab and click **Start Data Mine**.
   - Choose *S&P 500 only* for a ~10-minute initial run.
   - The full US equity universe (~7,000 tickers) takes several hours.
   - Mining is **checkpoint-based** — you can stop and restart at any time.

2. **Analyse a stock:** Go to **📊 Stock Analysis**, pick a ticker, and explore the 4 interactive charts.

3. **AI Deep-Dive:** In the Stock Analysis tab, enter your Anthropic API key in the sidebar, then click **Generate AI Fundamental Deep-Dive**. The response is cached permanently — no repeat API calls.

4. **ML Rankings:** Go to **🤖 ML Rankings** to view the full sorted leaderboard. Filter by label or minimum confidence score.

---

## Architecture

| Module | Purpose |
|--------|---------|
| `database.py` | SQLite WAL, schema, all read/write helpers |
| `scraper.py` | Universe fetch, yfinance OHLCV + fundamentals, checkpoints |
| `ml_engine.py` | Feature engineering, RF + GB ensemble, scoring |
| `ai_engine.py` | Claude API integration, payload assembly, caching |
| `finmine_app.py` | Streamlit UI (this file) |

---

## Data Sources

| Data | Source |
|------|--------|
| OHLCV prices (10 years, daily) | `yfinance` |
| Income / Balance / Cash Flow statements | `yfinance` (SEC EDGAR via yfinance) |
| Ticker universe (full) | NASDAQ Screener API |
| Ticker universe (S&P 500) | Wikipedia S&P 500 table |
| AI summaries | Anthropic Claude API |

---

## Database Schema

```
companies            — ticker, exchange, name, sector, industry
historical_prices    — ticker, date, OHLCV, dividends, splits
financial_fundamentals — ticker, period_type, period_end, statement_type, data_json
ml_recommendations   — ticker, score, label, features_json, compiled_at
ai_summaries         — ticker, markdown, model, created_at
mining_checkpoint    — ticker, status (pending/prices_done/done/error)
```

---

## ML Engine Details

- **Features:** Revenue CAGR, Net Income CAGR, FCF CAGR, Avg Operating Margin,
  Avg FCF Margin, Margin velocity, D/E ratio & trend, Cash stability, scale metrics
- **Model:** Ensemble of Random Forest (300 trees) + Gradient Boosting (200 trees),
  probabilities averaged
- **Target:** Binary — does the stock's 3-year return beat the universe median?
- **Fallback:** If <20 labeled stocks exist (new DB), a weighted rank-based heuristic
  is used instead of supervised ML

---

## Staleness Policy

| Data type | Refreshed when |
|-----------|---------------|
| Prices | Older than 24 hours |
| Fundamentals | Older than 90 days |
| AI Summary | Never (manual regenerate button) |
| ML Scores | After each Data Mine run |

---

## Troubleshooting

- **No data for ticker:** Click *Refresh from API* in the Stock Analysis tab.
- **AI button grayed out:** Enter your `sk-ant-...` key in the sidebar.
- **Mine stalls:** Mining is checkpoint-based. Just click *Start Data Mine* again.
- **DB corruption:** Delete `finmine.db` and re-mine. All data is re-fetchable.
""")


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "FinMine Analytics Engine — local financial data mining & AI research platform. "
    "For educational and research purposes only. Not investment advice."
)
