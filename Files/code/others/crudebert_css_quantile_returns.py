"""
Cumulative sentiment (CSS) normalized, quintile assignment, and mean cumulative returns by quantile.

Steps:
  1. Normalize daily CSS: subtract previous 60-day mean, divide by previous 60-day std.
  2. Assign quintiles: Q1 = top 10%, Q2 = next 20%, Q3 = middle 40%, Q4 = next 20%, Q5 = bottom 10%.
  3. For each day, compute cumulative return of oil futures from 10 days before to 10 days after.
  4. Compute mean cumulative return per quantile.
  5. Test whether mean(Q1) - mean(Q5) is significant (t-test and permutation test).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from sentiment_strategy import load_yahoo_oil_csv


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
FINAL_ANALYSIS_DIR = ROOT_DIR / "final_analysis"
PLOTS_DIR = DATA_DIR / "backtest_plots"

CUM_SENTIMENT_CSV = FINAL_ANALYSIS_DIR / "daily_cumulative_sentiment_scores_crudebert_2017_2024.csv"
WTI_CSV = DATA_DIR / "wti.csv"
BRENT_CSV = DATA_DIR / "brent.csv"

# Quintile boundaries (cumulative): top 10%, next 20%, middle 40%, next 20%, bottom 10%
# So quantile cut points: 90, 70, 30, 10 (percentiles)
QUANTILE_PCT = (0.90, 0.70, 0.30, 0.10)  # Q1 >= p90, Q2 in [p70,p90), Q3 [p30,p70), Q4 [p10,p30), Q5 < p10
N_PERM = 5000


def load_and_merge(css_csv: Path, market_csv: Path) -> pd.DataFrame:
    """Load CSS and futures, merge on date, ensure sorted by date."""
    css = pd.read_csv(css_csv)
    css["date"] = pd.to_datetime(css["date"])
    mkt = load_yahoo_oil_csv(str(market_csv), recompute_returns=True)
    mkt = mkt[["date", "close"]].copy()
    mkt["date"] = pd.to_datetime(mkt["date"])
    df = css[["date", "css_7d_exp"]].merge(mkt, on="date", how="inner")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def normalize_previous_60d(series: pd.Series, window: int = 60) -> pd.Series:
    """
    Normalize by previous 60-day mean and std.
    At t: use mean and std of series at t-60,...,t-1; then (series_t - mean) / std.
    """
    prev = series.shift(1)
    mu = prev.rolling(window=window, min_periods=window).mean()
    sig = prev.rolling(window=window, min_periods=window).std(ddof=0)
    sig = sig.replace(0.0, np.nan)
    return (series - mu) / sig


def assign_quantiles(normalized: pd.Series) -> pd.Series:
    """
    Assign Q1 (top 10%), Q2 (next 20%), Q3 (middle 40%), Q4 (next 20%), Q5 (bottom 10%).
    Returns Series of 'Q1','Q2','Q3','Q4','Q5' (or NaN where normalized is NaN).
    """
    valid = normalized.dropna()
    if len(valid) == 0:
        return pd.Series(index=normalized.index, dtype=object)
    q90, q70, q30, q10 = np.percentile(valid, [90, 70, 30, 10])
    out = pd.Series(index=normalized.index, dtype=object)
    out[:] = np.nan
    x = normalized.values
    out.loc[normalized >= q90] = "Q1"
    out.loc[(normalized >= q70) & (normalized < q90)] = "Q2"
    out.loc[(normalized >= q30) & (normalized < q70)] = "Q3"
    out.loc[(normalized >= q10) & (normalized < q30)] = "Q4"
    out.loc[normalized < q10] = "Q5"
    return out


def cumulative_return_10_before_10_after(close: pd.Series) -> pd.Series:
    """
    For each day t: cumulative return from close at t-10 to close at t+10.
    Return = (close_{t+10} / close_{t-10}) - 1.
    """
    close_fwd = close.shift(-10)
    close_lag = close.shift(10)
    return (close_fwd / close_lag) - 1.0


def run_analysis(css_csv: Path, market_csv: Path, instrument: str) -> dict[str, Any]:
    df = load_and_merge(css_csv, market_csv)
    if len(df) < 80:
        return {"error": "Insufficient data"}

    # 1. Normalize CSS by previous 60-day mean and std
    df["css_norm"] = normalize_previous_60d(df["css_7d_exp"], window=60)

    # 2. Assign quintiles (only on valid normalized scores)
    df["quantile"] = assign_quantiles(df["css_norm"])

    # 3. Cumulative return from 10 days before to 10 days after
    df["cumret_10_10"] = cumulative_return_10_before_10_after(df["close"])

    # Drop rows missing quantile or cumret (need 60+ history for norm, 10 before/after for cumret)
    work = df.dropna(subset=["quantile", "cumret_10_10"]).copy()
    if work.empty:
        return {"error": "No rows with valid quantile and cumulative return"}

    # 4. Mean cumulative return per quantile
    mean_by_q = work.groupby("quantile")["cumret_10_10"].agg(["mean", "std", "count"])
    mean_by_q = mean_by_q.reindex(["Q1", "Q2", "Q3", "Q4", "Q5"]).dropna(how="all")

    q1_returns = work.loc[work["quantile"] == "Q1", "cumret_10_10"].values
    q5_returns = work.loc[work["quantile"] == "Q5", "cumret_10_10"].values

    # 5. Test: difference between Q1 and Q5 mean
    diff_mean = np.nan
    t_stat = np.nan
    t_pvalue = np.nan
    perm_pvalue = np.nan

    if len(q1_returns) >= 2 and len(q5_returns) >= 2:
        diff_mean = float(np.mean(q1_returns) - np.mean(q5_returns))
        # Welch's t-test (unequal variances)
        t_stat, t_pvalue = scipy_stats.ttest_ind(q1_returns, q5_returns, equal_var=False)
        # Permutation test: H0 distribution of (Q1, Q5) labels is irrelevant to returns
        rng = np.random.default_rng(42)
        combined = np.concatenate([q1_returns, q5_returns])
        n1 = len(q1_returns)
        null_diffs = []
        for _ in range(N_PERM):
            perm = rng.permutation(combined)
            null_diffs.append(np.mean(perm[:n1]) - np.mean(perm[n1:]))
        null_diffs = np.array(null_diffs)
        perm_pvalue = float(np.mean(np.abs(null_diffs) >= np.abs(diff_mean)))

    return {
        "instrument": instrument,
        "n_obs": len(work),
        "mean_by_quantile": mean_by_q,
        "q1_mean": float(np.mean(q1_returns)) if len(q1_returns) else np.nan,
        "q5_mean": float(np.mean(q5_returns)) if len(q5_returns) else np.nan,
        "q1_count": len(q1_returns),
        "q5_count": len(q5_returns),
        "diff_q1_q5": diff_mean,
        "t_stat": float(t_stat) if not np.isnan(t_stat) else np.nan,
        "t_pvalue": float(t_pvalue) if not np.isnan(t_pvalue) else np.nan,
        "perm_pvalue": perm_pvalue,
        "work_df": work,
    }


def main() -> None:
    if not CUM_SENTIMENT_CSV.exists():
        print(f"Cumulative sentiment CSV not found: {CUM_SENTIMENT_CSV}")
        return

    instruments: list[tuple[str, Path]] = []
    if WTI_CSV.exists():
        instruments.append(("WTI", WTI_CSV))
    if BRENT_CSV.exists():
        instruments.append(("Brent", BRENT_CSV))

    if not instruments:
        print("No futures CSV found.")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CSS normalized → quintiles → mean cumulative return (t-10 to t+10)")
    print("=" * 60)
    print(f"CSS: previous 60-day mean/std normalization")
    print(f"Quintiles: Q1=top 10%, Q2=next 20%, Q3=middle 40%, Q4=next 20%, Q5=bottom 10%")
    print("Cumulative return: (close_{t+10} / close_{t-10}) - 1")
    print()

    for name, mkt_path in instruments:
        print(f"--- {name} ---")
        res = run_analysis(CUM_SENTIMENT_CSV, mkt_path, name)
        if res.get("error"):
            print(f"  {res['error']}")
            continue

        print(f"  Observations (with valid norm, quantile, cumret): {res['n_obs']}")
        print()
        print("  Mean cumulative return by quantile:")
        print(res["mean_by_quantile"].to_string())
        print()
        print(f"  Q1 mean cum. return: {res['q1_mean']:.6f}  (n={res['q1_count']})")
        print(f"  Q5 mean cum. return: {res['q5_mean']:.6f}  (n={res['q5_count']})")
        print(f"  Difference (Q1 - Q5): {res['diff_q1_q5']:.6f}")
        print(f"  Welch t-test: t = {res['t_stat']:.4f}, p = {res['t_pvalue']:.4f}")
        print(f"  Permutation test (H0: no difference): p = {res['perm_pvalue']:.4f}  (n_perm={N_PERM})")
        print()

        # Box plot of cumulative returns by quantile
        work = res["work_df"]
        labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]
        data = [
            work.loc[work["quantile"] == q, "cumret_10_10"].dropna().values
            for q in labels
        ]
        # Skip plotting if all are empty for some reason
        if any(len(arr) > 0 for arr in data):
            plt.figure(figsize=(8, 5))
            plt.boxplot(
                data,
                labels=labels,
                showfliers=True,
                patch_artist=True,
            )
            plt.title(f"{name}: Cumulative return (t-10 to t+10) by CSS quantile")
            plt.ylabel("Cumulative return")
            plt.grid(axis="y", alpha=0.3)
            out_path = PLOTS_DIR / f"crudebert_css_quantile_boxplot_{name.lower()}.png"
            plt.tight_layout()
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved box plot: {out_path}")
            print()

    print("Done.")


if __name__ == "__main__":
    main()
