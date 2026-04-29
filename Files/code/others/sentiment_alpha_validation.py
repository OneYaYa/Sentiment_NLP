"""
Validate whether the oil-news sentiment signal has predictive power for futures returns.

Phase 1: Predictive regression, information coefficient (correlation), portfolio sorts.
Phase 2: Bootstrap/permutation test, subperiod stability.

Uses signal at date t and next-day return ret_{t+1} (aligned with next_day_execution backtest).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import statsmodels.api as sm

from sentiment_strategy import (
    add_rolling_volatility,
    load_daily_sentiment_features,
    load_yahoo_oil_csv,
    merge_sentiment_features,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
FINAL_ANALYSIS = ROOT_DIR / "final_analysis"
SENTIMENT_FINBERT = FINAL_ANALYSIS / "daily_sentiment_scores_2017_2024.csv"
SENTIMENT_CRUDEBERT = FINAL_ANALYSIS / "daily_sentiment_scores_crudebert_2017_2024.csv"
WTI_CSV = DATA_DIR / "wti.csv"
BRENT_CSV = DATA_DIR / "brent.csv"
OUTPUT_DIR = DATA_DIR / "backtest_plots"

# Fixed params for Z (match strategy-style signal)
Z_WINDOW = 120
Z_MIN_PERIODS = 60
S_RAW_WEIGHTS = (0.2, 0.3, 0.5)  # w1, w3, w7

# Subperiods for Phase 2
SUBPERIODS = [
    ("2017-2019", "2017-01-01", "2019-12-31"),
    ("2020-2021", "2020-01-01", "2021-12-31"),
    ("2022-2024", "2022-01-01", "2024-12-31"),
]
N_PERM = 2000  # permutation iterations


def prepare_data(market_csv: Path, sentiment_csv: Path) -> pd.DataFrame:
    """Load market + sentiment, merge, forward-fill; one row per trading day."""
    mkt = load_yahoo_oil_csv(str(market_csv), recompute_returns=True)
    sent = load_daily_sentiment_features(sentiment_csv)
    df = merge_sentiment_features(mkt, sent)
    for col in ["oil_sent_1d", "oil_sent_3d", "oil_sent_7d", "oil_news_count_7d"]:
        if col in df.columns:
            df[col] = df[col].ffill().fillna(0.0)
    return df


def add_forward_return_and_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Add ret_next (next trading day return) and optional Z signal."""
    df = df.copy()
    df["ret_next"] = df["ret"].shift(-1)
    # S_raw and Z (strategy-style) for optional signal
    w1, w3, w7 = S_RAW_WEIGHTS
    df["S_raw"] = (
        w1 * df["oil_sent_1d"] + w3 * df["oil_sent_3d"] + w7 * df["oil_sent_7d"]
    )
    intensity = np.log1p(df["oil_news_count_7d"].clip(lower=0))
    df["S_intensity"] = df["S_raw"] * intensity
    mu = df["S_intensity"].rolling(Z_WINDOW, min_periods=Z_MIN_PERIODS).mean()
    sd = (
        df["S_intensity"]
        .rolling(Z_WINDOW, min_periods=Z_MIN_PERIODS)
        .std(ddof=0)
        .replace(0.0, np.nan)
    )
    df["Z"] = (df["S_intensity"] - mu) / sd
    return df


def ols_regression(y: np.ndarray, x: np.ndarray) -> dict[str, float]:
    """OLS: y = alpha + beta*x with Newey-West (HAC) standard errors. Returns alpha, beta, t_beta, R2, n, Shapiro-Wilk on residuals."""
    n = len(y)
    X = sm.add_constant(np.asarray(x, dtype=float))
    y_ = np.asarray(y, dtype=float)
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y_)
    if valid.sum() < 10:
        return {
            "alpha": np.nan, "beta": np.nan, "t_beta": np.nan, "R2": np.nan, "n": n,
            "shapiro_w": np.nan, "shapiro_p": np.nan,
        }
    X_v = X[valid]
    y_v = y_[valid]
    n_v = len(y_v)

    try:
        model = sm.OLS(y_v, X_v).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": int(4 * (n_v / 100) ** (2 / 9))},
        )
    except Exception:
        return {
            "alpha": np.nan, "beta": np.nan, "t_beta": np.nan, "R2": np.nan, "n": n_v,
            "shapiro_w": np.nan, "shapiro_p": np.nan,
        }

    b = model.params
    resid = model.resid
    r2 = float(model.rsquared)
    t_beta = float(model.tvalues[1]) if len(model.tvalues) > 1 else np.nan

    # Shapiro-Wilk on residuals (H0: normal)
    shapiro_w, shapiro_p = np.nan, np.nan
    if len(resid) >= 3:
        resid_clean = resid[np.isfinite(resid)]
        if len(resid_clean) >= 3:
            max_n = 5000
            if len(resid_clean) > max_n:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(resid_clean), size=max_n, replace=False)
                resid_sub = resid_clean[idx]
            else:
                resid_sub = resid_clean
            try:
                shapiro_w, shapiro_p = scipy_stats.shapiro(resid_sub)
            except Exception:
                pass

    return {
        "alpha": float(b[0]),
        "beta": float(b[1]),
        "t_beta": t_beta,
        "R2": r2,
        "n": n_v,
        "shapiro_w": float(shapiro_w),
        "shapiro_p": float(shapiro_p),
    }


def correlation_and_ttest(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Pearson correlation and t-stat for H0: rho = 0."""
    n = len(x)
    if n < 3:
        return {"corr": np.nan, "t_stat": np.nan, "n": n}
    r = np.corrcoef(x, y)[0, 1]
    if np.isnan(r):
        return {"corr": np.nan, "t_stat": np.nan, "n": n}
    t = r * np.sqrt(n - 2) / np.sqrt(1 - r * r) if abs(r) < 1 else np.nan
    return {"corr": float(r), "t_stat": float(t), "n": n}


def quintile_sorts(signal: np.ndarray, ret_next: np.ndarray) -> dict[str, Any]:
    """Sort by signal into quintiles; return avg ret_next per quintile and long-short."""
    valid = np.isfinite(signal) & np.isfinite(ret_next)
    s = signal[valid]
    r = ret_next[valid]
    n = len(s)
    if n < 10:
        return {"quintile_means": [], "long_short": np.nan, "t_ls": np.nan, "n": n}
    # Assign quintile by rank (0=lowest signal, 4=highest)
    rank = np.argsort(np.argsort(s))
    q = (rank * 5 // n).clip(0, 4)
    means = [float(r[q == i].mean()) for i in range(5)]
    top = r[q == 4]
    bot = r[q == 0]
    ls = np.mean(top) - np.mean(bot)
    se_top = np.var(top, ddof=1) / len(top) if len(top) > 1 else 0.0
    se_bot = np.var(bot, ddof=1) / len(bot) if len(bot) > 1 else 0.0
    se_ls = np.sqrt(se_top + se_bot) if (se_top + se_bot) > 0 else np.nan
    t_ls = ls / se_ls if se_ls and not np.isnan(se_ls) and se_ls > 0 else np.nan
    return {"quintile_means": means, "long_short": float(ls), "t_ls": float(t_ls) if not np.isnan(t_ls) else np.nan, "n": n}


def run_phase1(df: pd.DataFrame, instrument: str, signal_col: str) -> dict[str, Any]:
    """Phase 1: regression, IC, quintiles for one signal column."""
    valid = df["ret_next"].notna() & df[signal_col].notna()
    d = df.loc[valid].copy()
    if len(d) < 20:
        return {}
    y = d["ret_next"].values
    x = d[signal_col].values

    reg = ols_regression(y, x)
    ic = correlation_and_ttest(x, y)
    qq = quintile_sorts(x, y)

    return {
        "instrument": instrument,
        "signal": signal_col,
        "n": len(d),
        "beta": reg["beta"],
        "t_beta": reg["t_beta"],
        "R2": reg["R2"],
        "shapiro_w": reg.get("shapiro_w", np.nan),
        "shapiro_p": reg.get("shapiro_p", np.nan),
        "corr": ic["corr"],
        "t_corr": ic["t_stat"],
        "long_short": qq.get("long_short", np.nan),
        "t_ls": qq.get("t_ls", np.nan),
        "quintile_means": qq.get("quintile_means", []),
    }


def run_phase2_permutation(
    df: pd.DataFrame, signal_col: str, metric: str = "corr"
) -> dict[str, Any]:
    """Permutation test: shuffle ret_next, recompute metric, return p-value."""
    valid = df["ret_next"].notna() & df[signal_col].notna()
    d = df.loc[valid]
    if len(d) < 20:
        return {"p_value": np.nan, "actual": np.nan, "metric": metric}
    signal = d[signal_col].values
    ret_next = d["ret_next"].values

    if metric == "corr":
        actual = np.corrcoef(signal, ret_next)[0, 1]
        if np.isnan(actual):
            return {"p_value": np.nan, "actual": np.nan, "metric": metric}
        nulls = []
        rng = np.random.default_rng(42)
        for _ in range(N_PERM):
            shuf = rng.permutation(ret_next)
            r = np.corrcoef(signal, shuf)[0, 1]
            nulls.append(r)
        nulls = np.array(nulls)
        # Two-tailed: fraction as or more extreme than |actual|
        p = np.mean(np.abs(nulls) >= np.abs(actual))
    elif metric == "beta":
        reg = ols_regression(ret_next, signal)
        actual = reg["beta"]
        nulls = []
        rng = np.random.default_rng(42)
        for _ in range(N_PERM):
            shuf = rng.permutation(ret_next)
            r = ols_regression(shuf, signal)["beta"]
            nulls.append(r)
        nulls = np.array(nulls)
        p = np.mean(np.abs(nulls) >= np.abs(actual))
    else:
        return {"p_value": np.nan, "actual": np.nan, "metric": metric}

    return {"p_value": float(p), "actual": float(actual), "metric": metric, "n_perm": N_PERM}


def run_phase2_subperiods(
    df: pd.DataFrame, instrument: str, signal_col: str
) -> list[dict[str, Any]]:
    """Run correlation and regression in each subperiod."""
    results = []
    for name, start, end in SUBPERIODS:
        mask = (pd.to_datetime(df["date"]) >= start) & (pd.to_datetime(df["date"]) <= end)
        sub = df.loc[mask]
        if sub["ret_next"].notna().sum() < 20:
            continue
        r = run_phase1(sub, instrument, signal_col)
        if r:
            r["subperiod"] = name
            results.append(r)
    return results


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Validate sentiment signal alpha")
    parser.add_argument("--sentiment", "-s", choices=["finbert", "crudebert"], default="crudebert",
                        help="Sentiment source: finbert (oil_news) or crudebert (default: crudebert)")
    args = parser.parse_args()
    sentiment_csv = SENTIMENT_CRUDEBERT if args.sentiment == "crudebert" else SENTIMENT_FINBERT

    if not sentiment_csv.exists():
        print(f"Sentiment CSV not found: {sentiment_csv}")
        if args.sentiment == "crudebert":
            print("Run: python final_analysis/calculate_daily_sentiment.py --input crudebert")
        else:
            print("Run: python final_analysis/calculate_daily_sentiment.py")
        return

    instruments = []
    if WTI_CSV.exists():
        instruments.append(("WTI", WTI_CSV))
    if BRENT_CSV.exists():
        instruments.append(("Brent", BRENT_CSV))
    if not instruments:
        print("No market CSV (wti.csv or brent.csv) found.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    signal_cols = ["oil_sent_1d", "oil_sent_7d", "Z"]
    all_phase1: list[dict[str, Any]] = []
    all_subperiods: list[dict[str, Any]] = []
    perm_results: list[dict[str, Any]] = []

    print("=" * 60)
    print("Sentiment Alpha Validation")
    print("=" * 60)
    print(f"Sentiment: {args.sentiment} -> {sentiment_csv.name}")
    print("Signal at t vs ret_{t+1} (next trading day return)")
    print()

    for instrument, market_csv in instruments:
        df = prepare_data(market_csv, sentiment_csv)
        df = add_forward_return_and_signals(df)
        df = df.dropna(subset=["ret_next"]).reset_index(drop=True)
        print(f"--- {instrument} --- N = {len(df)} (with ret_next)")

        for signal_col in signal_cols:
            if signal_col not in df.columns:
                continue
            r1 = run_phase1(df, instrument, signal_col)
            if r1:
                all_phase1.append(r1)
                print(f"  Signal: {signal_col}")
                print(f"    Regression: beta = {r1['beta']:.6f}, t = {r1['t_beta']:.3f}, R2 = {r1['R2']:.6f}")
                if not np.isnan(r1.get("shapiro_w", np.nan)):
                    print(f"    Shapiro-Wilk (residuals): W = {r1['shapiro_w']:.4f}, p = {r1['shapiro_p']:.4f}  (H0: normal)")
                print(f"    Correlation: r = {r1['corr']:.4f}, t = {r1['t_corr']:.3f}")
                print(f"    Long-short (Q5-Q1): {r1['long_short']:.6f}, t = {r1['t_ls']:.3f}")
                if r1.get("quintile_means"):
                    print(f"    Quintile avg ret: {[f'{x:.6f}' for x in r1['quintile_means']]}")

            # Phase 2: subperiods
            for sp in run_phase2_subperiods(df, instrument, signal_col):
                all_subperiods.append(sp)

            # Phase 2: permutation (once per instrument, use oil_sent_7d)
            if signal_col == "oil_sent_7d":
                perm = run_phase2_permutation(df, signal_col, metric="corr")
                perm["instrument"] = instrument
                perm_results.append(perm)
                print(f"    Permutation p-value (corr): {perm['p_value']:.4f} (actual r = {perm['actual']:.4f})")

            # Contrarian: negated signal (long when sentiment low, short when high)
            df_neg = df.copy()
            df_neg["_neg_signal"] = -df_neg[signal_col]
            r_neg = run_phase1(df_neg, instrument, "_neg_signal")
            if r_neg:
                print(f"    Contrarian (-{signal_col}): beta = {r_neg['beta']:.6f}, t = {r_neg['t_beta']:.3f} | corr = {r_neg['corr']:.4f}, t = {r_neg['t_corr']:.3f} | long-short = {r_neg['long_short']:.6f}, t = {r_neg['t_ls']:.3f}")

    # Save report
    rows = []
    for r in all_phase1:
        rows.append({
            "instrument": r["instrument"],
            "signal": r["signal"],
            "n": r["n"],
            "beta": r["beta"],
            "t_beta": r["t_beta"],
            "R2": r["R2"],
            "shapiro_w": r.get("shapiro_w", np.nan),
            "shapiro_p": r.get("shapiro_p", np.nan),
            "corr": r["corr"],
            "t_corr": r["t_corr"],
            "long_short": r["long_short"],
            "t_ls": r["t_ls"],
        })
    if rows:
        report_csv = OUTPUT_DIR / f"alpha_validation_report_{args.sentiment}.csv"
        try:
            pd.DataFrame(rows).to_csv(report_csv, index=False)
            print(f"\nReport saved: {report_csv}")
        except (PermissionError, OSError) as e:
            # File may be open in Excel or another program
            alt = ROOT_DIR / "others" / f"alpha_validation_report_{args.sentiment}.csv"
            try:
                pd.DataFrame(rows).to_csv(alt, index=False)
                print(f"\nReport could not be written to {report_csv} ({e}). Saved instead to: {alt}")
            except Exception:
                print(f"\nReport could not be saved: {report_csv} ({e}). Close the file if open in Excel and re-run.")

    # Subperiod summary
    if all_subperiods:
        print("\n--- Subperiod stability (corr by period) ---")
        for r in all_subperiods:
            print(f"  {r['instrument']} {r['signal']} {r.get('subperiod','')}: corr = {r.get('corr', np.nan):.4f}, t = {r.get('t_corr', np.nan):.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
