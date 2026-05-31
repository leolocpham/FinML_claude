"""
ml_engine.py — FinMine Analytics Engine
Scikit-Learn Random Forest + Gradient Boosting pipeline.
Builds feature matrix from 10-year fundamentals, trains on
historical cross-sectional data, outputs confidence scores
and recommendation labels stored in ml_recommendations.
"""

import json
import logging
from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import database as db

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# LABEL THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

LABEL_THRESHOLDS = {
    "STRONG BUY / ALPHACLASS": 0.72,
    "BUY":                     0.55,
    "HOLD / UNDERPERFORM":     0.0,
}

def _score_to_label(score: float) -> str:
    for label, threshold in LABEL_THRESHOLDS.items():
        if score >= threshold:
            return label
    return "HOLD / UNDERPERFORM"


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _safe_cagr(start_val: float, end_val: float, years: float) -> float | None:
    if years <= 0 or start_val is None or end_val is None:
        return None
    if start_val <= 0 or end_val <= 0:
        return None
    try:
        return (end_val / start_val) ** (1.0 / years) - 1.0
    except (ZeroDivisionError, OverflowError):
        return None


def _extract_series(fundamentals_df: pd.DataFrame, stmt_type: str,
                    key: str) -> pd.Series:
    """
    From fundamentals DataFrame returns a time-sorted Series of one metric.
    """
    sub = fundamentals_df[fundamentals_df["statement_type"] == stmt_type].copy()
    values = {}
    for _, row in sub.iterrows():
        data = row["data"] if isinstance(row["data"], dict) else json.loads(row.get("data_json", "{}"))
        # case-insensitive key lookup
        for k, v in data.items():
            if k.lower().replace(" ", "").replace("_", "") == key.lower().replace(" ", "").replace("_", ""):
                try:
                    values[row["period_end"]] = float(v)
                except (TypeError, ValueError):
                    pass
                break
    if not values:
        return pd.Series(dtype=float)
    return pd.Series(values).sort_index()


def build_feature_vector(ticker: str) -> dict | None:
    """
    Extracts quantitative features for a single ticker from the local database.
    Returns None if insufficient data exists.
    """
    ann_df = db.get_fundamentals(ticker, period_type="annual")
    if ann_df.empty or len(ann_df) < 2:
        return None

    ann_df["data"] = ann_df["data_json"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )

    def series(stmt: str, key: str) -> pd.Series:
        return _extract_series(ann_df, stmt, key)

    revenue       = series("income", "TotalRevenue")
    net_income    = series("income", "NetIncome")
    op_income     = series("income", "OperatingIncome")
    ebit          = series("income", "Ebit")
    total_assets  = series("balance", "TotalAssets")
    total_debt    = series("balance", "TotalDebt")
    total_cash    = series("balance", "CashAndCashEquivalents")
    equity        = series("balance", "StockholdersEquity")
    fcf           = series("cashflow", "FreeCashFlow")
    capex         = series("cashflow", "CapitalExpenditure")

    # Use at least 2 years for CAGR; use all available years
    def _cagr_from_series(s: pd.Series) -> float | None:
        s = s.dropna()
        if len(s) < 2:
            return None
        yrs = max(1.0, (pd.to_datetime(s.index[-1]) - pd.to_datetime(s.index[0])).days / 365.25)
        return _safe_cagr(s.iloc[0], s.iloc[-1], yrs)

    # Operating margin series
    op_margin_series = (op_income / revenue.reindex(op_income.index)).dropna()

    # FCF margin series
    fcf_margin_series = (fcf / revenue.reindex(fcf.index)).dropna()

    # Debt-to-equity series
    de_series = (total_debt / equity.reindex(total_debt.index).replace(0, np.nan)).dropna()

    # Cash-to-liabilities proxy
    cash_stability = (total_cash / total_debt.reindex(total_cash.index).replace(0, np.nan)).dropna()

    features: dict = {
        # Growth
        "revenue_cagr":       _cagr_from_series(revenue),
        "net_income_cagr":    _cagr_from_series(net_income),
        "fcf_cagr":           _cagr_from_series(fcf),
        # Profitability
        "avg_op_margin":      float(op_margin_series.mean())   if not op_margin_series.empty else None,
        "avg_fcf_margin":     float(fcf_margin_series.mean())  if not fcf_margin_series.empty else None,
        "op_margin_trend":    _cagr_from_series(op_margin_series.replace(0, np.nan).dropna()),
        "fcf_margin_velocity":_cagr_from_series(fcf_margin_series.replace(0, np.nan).dropna()),
        # Balance sheet safety
        "avg_de_ratio":       float(de_series.mean())          if not de_series.empty else None,
        "de_trend":           _cagr_from_series(de_series.replace(0, np.nan).dropna()),
        "avg_cash_stability": float(cash_stability.mean())     if not cash_stability.empty else None,
        # Latest snapshot values (scaled to billions for interpretability)
        "latest_revenue_B":   float(revenue.iloc[-1]) / 1e9   if not revenue.empty else None,
        "latest_fcf_B":       float(fcf.iloc[-1])    / 1e9    if not fcf.empty else None,
        "latest_net_income_B":float(net_income.iloc[-1]) / 1e9 if not net_income.empty else None,
        "years_of_data":      len(revenue.dropna()),
    }

    # Return None if we have fewer than 4 meaningful features
    valid = sum(1 for v in features.values() if v is not None and not np.isnan(v))
    if valid < 4:
        return None

    return features


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE MATRIX BUILDER
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "revenue_cagr", "net_income_cagr", "fcf_cagr",
    "avg_op_margin", "avg_fcf_margin", "op_margin_trend", "fcf_margin_velocity",
    "avg_de_ratio", "de_trend", "avg_cash_stability",
    "latest_revenue_B", "latest_fcf_B", "latest_net_income_B",
    "years_of_data",
]


def build_feature_matrix(
    tickers: list[str],
    progress_cb: Callable[[str, int, int], None] | None = None,
) -> pd.DataFrame:
    """
    Builds a feature matrix for all tickers. Returns DataFrame indexed by ticker.
    """
    rows = []
    for i, ticker in enumerate(tickers):
        fv = build_feature_vector(ticker)
        if fv is not None:
            fv["ticker"] = ticker
            rows.append(fv)
        if progress_cb:
            progress_cb(ticker, i + 1, len(tickers))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("ticker")
    # Ensure all expected columns exist
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan
    return df[FEATURE_COLS]


# ─────────────────────────────────────────────────────────────────────────────
# TARGET LABELING
# ─────────────────────────────────────────────────────────────────────────────

def _compute_price_return(ticker: str, years: int = 3) -> float | None:
    """3-year total price return for a ticker."""
    prices = db.get_prices(ticker)
    if prices.empty or len(prices) < 100:
        return None
    prices = prices.sort_values("date")
    cutoff_date = (
        pd.to_datetime(prices["date"].iloc[-1]) - pd.DateOffset(years=years)
    ).strftime("%Y-%m-%d")
    sub = prices[prices["date"] >= cutoff_date]
    if len(sub) < 50:
        return None
    start_p = sub["close"].iloc[0]
    end_p   = sub["close"].iloc[-1]
    if start_p <= 0:
        return None
    return (end_p - start_p) / start_p


def build_target_labels(tickers: list[str]) -> pd.Series:
    """
    Binary outperformance label: 1 if 3-year return > median of universe.
    Returns Series indexed by ticker.
    """
    returns: dict[str, float] = {}
    for ticker in tickers:
        r = _compute_price_return(ticker, years=3)
        if r is not None:
            returns[ticker] = r

    if len(returns) < 10:
        return pd.Series(dtype=int)

    s = pd.Series(returns)
    median_ret = s.median()
    return (s > median_ret).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def _make_pipeline(model_type: str = "rf") -> Pipeline:
    if model_type == "gb":
        clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.08,
            subsample=0.8, random_state=42,
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     clf),
    ])


def train_and_score(
    feature_matrix: pd.DataFrame,
    target_labels:  pd.Series,
    log_cb: Callable[[str], None] | None = None,
) -> pd.Series:
    """
    Trains the ensemble model on labeled data and returns a confidence
    score Series (probability of outperformance) for ALL tickers in
    feature_matrix (including those without labels, scored via predict_proba).
    """
    def _log(msg: str):
        log.info(msg)
        if log_cb:
            log_cb(msg)

    # Align labeled subset
    common = feature_matrix.index.intersection(target_labels.index)
    if len(common) < 20:
        _log(f"Insufficient labeled data ({len(common)} stocks). Scoring by heuristics.")
        return _heuristic_scores(feature_matrix)

    X_train = feature_matrix.loc[common]
    y_train = target_labels.loc[common]

    _log(f"Training on {len(X_train)} labeled stocks (pos={y_train.sum()}, neg={(~y_train.astype(bool)).sum()})")

    # Ensemble: RF + GB, average probabilities
    pipe_rf = _make_pipeline("rf")
    pipe_gb = _make_pipeline("gb")

    pipe_rf.fit(X_train, y_train)
    pipe_gb.fit(X_train, y_train)

    _log("Models trained. Scoring full universe...")

    X_all = feature_matrix
    imputer = SimpleImputer(strategy="median")
    X_all_imp = imputer.fit_transform(X_all)

    prob_rf = pipe_rf.predict_proba(X_all)[:, 1]
    prob_gb = pipe_gb.predict_proba(X_all)[:, 1]
    scores  = (prob_rf + prob_gb) / 2.0

    return pd.Series(scores, index=feature_matrix.index)


def _heuristic_scores(feature_matrix: pd.DataFrame) -> pd.Series:
    """
    Fallback when not enough labeled data: rank by a weighted composite.
    """
    df = feature_matrix.copy()
    imp = SimpleImputer(strategy="median")
    arr = imp.fit_transform(df)
    df_imp = pd.DataFrame(arr, index=df.index, columns=df.columns)

    weights = {
        "revenue_cagr":       0.20,
        "net_income_cagr":    0.15,
        "fcf_cagr":           0.15,
        "avg_op_margin":      0.15,
        "avg_fcf_margin":     0.10,
        "avg_de_ratio":      -0.10,   # lower is better
        "de_trend":          -0.05,
        "avg_cash_stability": 0.10,
    }
    score = pd.Series(0.0, index=df_imp.index)
    for col, w in weights.items():
        if col in df_imp.columns:
            # rank-normalize to [0, 1]
            ranked = df_imp[col].rank(pct=True)
            score += w * ranked

    # Normalize to [0, 1]
    mn, mx = score.min(), score.max()
    if mx > mn:
        score = (score - mn) / (mx - mn)
    return score


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_ml_pipeline(
    feat_progress_cb:  Callable | None = None,
    log_cb:            Callable[[str], None] | None = None,
) -> dict:
    """
    Full ML pipeline called after the data mine completes.
    1. Loads company tickers from DB
    2. Builds feature matrix
    3. Builds target labels from price history
    4. Trains model and scores all tickers
    5. Writes scores to ml_recommendations table
    Returns summary dict.
    """
    def _log(msg: str):
        log.info(msg)
        if log_cb:
            log_cb(msg)

    companies = db.get_all_companies()
    if companies.empty:
        _log("No companies in database — run data mine first.")
        return {"status": "no_data"}

    tickers = companies["ticker"].tolist()
    _log(f"Building feature matrix for {len(tickers)} tickers...")

    fm = build_feature_matrix(tickers, progress_cb=feat_progress_cb)
    if fm.empty:
        _log("Feature matrix is empty — insufficient fundamental data.")
        return {"status": "empty_features"}

    _log(f"Feature matrix: {fm.shape[0]} stocks × {fm.shape[1]} features")

    _log("Computing target labels from price returns...")
    targets = build_target_labels(fm.index.tolist())

    _log("Training ensemble model...")
    scores = train_and_score(fm, targets, log_cb=log_cb)

    _log("Writing recommendations to database...")
    written = 0
    for ticker, score in scores.items():
        if pd.isna(score):
            continue
        label    = _score_to_label(float(score))
        features = fm.loc[ticker].to_dict() if ticker in fm.index else {}
        # Clean NaN values from features dict
        clean_features = {
            k: (None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v))
            for k, v in features.items()
        }
        db.upsert_ml_recommendation(ticker, float(score), label, clean_features)
        written += 1

    result = {
        "status":      "ok",
        "tickers_scored": written,
        "strong_buy":  int((scores >= LABEL_THRESHOLDS["STRONG BUY / ALPHACLASS"]).sum()),
        "buy":         int(((scores >= LABEL_THRESHOLDS["BUY"]) &
                            (scores < LABEL_THRESHOLDS["STRONG BUY / ALPHACLASS"])).sum()),
        "hold":        int((scores < LABEL_THRESHOLDS["BUY"]).sum()),
    }
    _log(f"ML pipeline complete: {result}")
    return result
