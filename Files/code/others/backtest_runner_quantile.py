"""
Walk-forward backtest for the quantile + medium-horizon strategy on Crudebert CSS.

Uses the same fold structure as backtest_runner.py (6-month train/test split,
warm-up at front). In each fold, p10/p90 are computed from the train period
(after warm-up); the strategy is run on train+test with those thresholds.
Outputs: data/wti_strategy_backtest_quantile.csv, data/brent_strategy_backtest_quantile.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sentiment_strategy_quantile import (
    QuantileStrategyConfig,
    compute_quantile_thresholds,
    load_css_and_market,
    normalize_css_previous_60d,
    performance_summary,
    run_quantile_strategy,
)

# Paths: data and final_analysis live under code/, not code/others/
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
FINAL_ANALYSIS = ROOT_DIR / "final_analysis"
CSS_CSV = FINAL_ANALYSIS / "daily_cumulative_sentiment_scores_crudebert_2017_2024.csv"
WTI_CSV = DATA_DIR / "wti.csv"
BRENT_CSV = DATA_DIR / "brent.csv"

# Same fold structure as backtest_runner.py (train_test_split.png)
FOLDS = [
    ("2017-01-01", "2022-12-31", "2023-01-01", "2023-06-30"),
    ("2017-07-01", "2023-06-30", "2023-07-01", "2023-12-31"),
    ("2018-01-01", "2023-12-31", "2024-01-01", "2024-06-30"),
    ("2018-07-01", "2024-06-30", "2024-07-01", "2024-12-31"),
]
WARMUP_DAYS = 126

# Strategy config (no grid; single quantile + medium-horizon specification)
STRATEGY_CONFIG = QuantileStrategyConfig(
    hold_days=21,
    target_vol=0.01,
    w_max=2.0,
    cost_per_unit=0.0002,
    next_day_execution=True,
    enable_vol_spike_filter=True,
    vol_spike_mult=2.5,
)


def date_range_mask(df: pd.DataFrame, start: str, end: str, date_col: str = "date") -> pd.Series:
    d = pd.to_datetime(df[date_col])
    return (d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))


def run_fold(
    df: pd.DataFrame,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    instrument: str = "WTI",
) -> dict[str, Any]:
    """
    Run one fold: compute p10/p90 from train (after warm-up), run quantile strategy
    on train+test, return train stats (after warm-up), test stats, and test backtest df.
    """
    mask_train = date_range_mask(df, train_start, train_end)
    mask_train_and_test = date_range_mask(df, train_start, test_end)

    df_train = df.loc[mask_train].copy().reset_index(drop=True)
    df_fold = df.loc[mask_train_and_test].copy().reset_index(drop=True)

    if len(df_train) == 0:
        return {
            "instrument": instrument,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "p10": None,
            "p90": None,
            "train_stats": {},
            "test_stats": {},
            "bt_test": None,
            "error": "No data in train period",
        }

    # Warm-up: first WARMUP_DAYS of train; score only after this
    first_train_date = pd.to_datetime(df_train["date"].min())
    warmup_end_date = first_train_date + pd.Timedelta(days=WARMUP_DAYS)

    # p10, p90 from train period after warm-up (use css_norm from df_train - we need to add it)
    df_train = df_train.copy()
    df_train["css_norm"] = normalize_css_previous_60d(df_train["css_7d_exp"].astype(float), window=60)
    mask_scoring = (pd.to_datetime(df_train["date"]) > warmup_end_date) & (
        pd.to_datetime(df_train["date"]) <= pd.Timestamp(train_end)
    )
    scoring_norm = df_train.loc[mask_scoring, "css_norm"].dropna()
    if len(scoring_norm) < 20:
        return {
            "instrument": instrument,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "p10": None,
            "p90": None,
            "train_stats": {},
            "test_stats": {},
            "bt_test": None,
            "error": "Insufficient train data after warm-up for quantile thresholds",
        }

    p10, p90 = compute_quantile_thresholds(scoring_norm, low_pct=10.0, high_pct=90.0)
    if not (np.isfinite(p10) and np.isfinite(p90)):
        return {
            "instrument": instrument,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "p10": p10,
            "p90": p90,
            "train_stats": {},
            "test_stats": {},
            "bt_test": None,
            "error": "Invalid p10/p90 from train",
        }

    # Run strategy on full fold (train + test) with train-derived p10, p90
    bt = run_quantile_strategy(df_fold, p10, p90, STRATEGY_CONFIG)

    # Train stats (after warm-up)
    mask_train_bt = (pd.to_datetime(bt["date"]) > warmup_end_date) & (
        pd.to_datetime(bt["date"]) <= pd.Timestamp(train_end)
    )
    bt_train = bt.loc[mask_train_bt].reset_index(drop=True)
    train_stats = performance_summary(bt_train) if len(bt_train) > 0 else {}

    # Test stats and test backtest df
    mask_test_bt = date_range_mask(bt, test_start, test_end)
    bt_test = bt.loc[mask_test_bt].reset_index(drop=True)
    test_stats = performance_summary(bt_test) if len(bt_test) > 0 else {}

    return {
        "instrument": instrument,
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "p10": p10,
        "p90": p90,
        "train_stats": train_stats,
        "test_stats": test_stats,
        "bt_test": bt_test,
    }


def main() -> None:
    if not CSS_CSV.exists():
        print(f"Crudebert CSS CSV not found: {CSS_CSV}")
        print(
            "Run: python final_analysis/calculate_cumulative_sentiment.py "
            "--input final_analysis/daily_sentiment_scores_crudebert_2017_2024.csv "
            "--output final_analysis/daily_cumulative_sentiment_scores_crudebert_2017_2024.csv"
        )
        return

    instruments = []
    if WTI_CSV.exists():
        instruments.append(("WTI", WTI_CSV))
    if BRENT_CSV.exists():
        instruments.append(("Brent", BRENT_CSV))

    if not instruments:
        print("No market CSV found. Expected data/wti.csv or data/brent.csv")
        return

    cfg = STRATEGY_CONFIG
    print("=" * 60)
    print("Quantile + medium-horizon backtest (Crudebert CSS)")
    print("=" * 60)
    print(f"CSS: {CSS_CSV}")
    print(f"Folds: train ~6y, test 6 months; first {WARMUP_DAYS} days of train = warm-up")
    print(f"Strategy: long when CSS_norm >= p90 (Q1), short when <= p10 (Q5); hold {cfg.hold_days} days")
    print(f"Config: target_vol={cfg.target_vol}, w_max={cfg.w_max}, next_day_execution={cfg.next_day_execution}")
    print()

    all_results: list[dict[str, Any]] = []
    instrument_bt_tests: dict[str, list[pd.DataFrame]] = {name: [] for name, _ in instruments}

    for instrument, market_csv in instruments:
        print(f"\n--- {instrument} ---")
        df = load_css_and_market(market_csv, CSS_CSV)
        print(f"Market + CSS: {len(df)} rows, {df['date'].min()} to {df['date'].max()}")

        for (train_start, train_end, test_start, test_end) in FOLDS:
            print(f"\n  Fold: Train {train_start} to {train_end} | Test {test_start} to {test_end}")
            result = run_fold(
                df,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                instrument=instrument,
            )
            all_results.append(result)

            if "error" in result:
                print(f"    Error: {result['error']}")
                continue

            if result.get("bt_test") is not None and len(result["bt_test"]) > 0:
                instrument_bt_tests[instrument].append(result["bt_test"])

            vs = result["train_stats"]
            ts = result["test_stats"]
            print(f"    p10 = {result['p10']:.4f}, p90 = {result['p90']:.4f}")
            print(f"    Train Sharpe: {vs.get('sharpe', 0):.4f}, MaxDD: {vs.get('max_drawdown', 0):.4f}")
            print(f"    Test  Sharpe: {ts.get('sharpe', 0):.4f}, MaxDD: {ts.get('max_drawdown', 0):.4f}, "
                  f"AnnRet: {ts.get('annualized_return', 0):.4f}")

    print("\n" + "=" * 60)
    print("Test Performance Summary (all folds)")
    print("=" * 60)
    for r in all_results:
        if "error" in r:
            continue
        ts = r["test_stats"]
        print(f"{r['instrument']} | Train {r['train_start']} to {r['train_end']} | Test {r['test_start']} to {r['test_end']} | "
              f"Sharpe: {ts.get('sharpe', 0):.4f} | MaxDD: {ts.get('max_drawdown', 0):.4f} | "
              f"AnnRet: {ts.get('annualized_return', 0):.4f}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for instrument, _ in instruments:
        frames = instrument_bt_tests.get(instrument, [])
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
        out_name = "wti_strategy_backtest_quantile.csv" if instrument == "WTI" else "brent_strategy_backtest_quantile.csv"
        out_path = DATA_DIR / out_name
        combined.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path} ({len(combined)} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()
