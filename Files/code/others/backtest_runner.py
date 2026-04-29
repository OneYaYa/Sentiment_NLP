"""
Walk-forward backtest with parameter grid search using daily or cumulative sentiment.

Supports --sentiment finbert | crudebert and --cumulative (Crudebert CSS only).
Fold structure matches train_test_split.png:
  - Four folds with 6-month rolling train/test split (~6y train, 6-month test per fold).
  - In each fold, the first WARMUP_DAYS of the train period are used only for rolling
    data setup (z-score, volatility); param search is scored on the remainder of the train period.
  - Overfitting reduction: coarser grid, turnover penalty, min activity, top-k then lowest turnover.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sentiment_strategy import (
    SentimentDirectionalConfig,
    add_rolling_volatility,
    load_cumulative_sentiment_features,
    load_daily_sentiment_features,
    load_yahoo_oil_csv,
    merge_sentiment_features,
    optimization_score,
    performance_summary,
    run_directional_strategy,
)

# Paths: repo uses code/data and code/final_analysis (this file lives in code/others/)
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
FINAL_ANALYSIS = ROOT_DIR / "final_analysis"
SENTIMENT_FINBERT = FINAL_ANALYSIS / "daily_sentiment_scores_2017_2024.csv"
SENTIMENT_CRUDEBERT = FINAL_ANALYSIS / "daily_sentiment_scores_crudebert_2017_2024.csv"
SENTIMENT_CRUDEBERT_CSS = FINAL_ANALYSIS / "daily_cumulative_sentiment_scores_crudebert_2017_2024.csv"
WTI_CSV = DATA_DIR / "wti.csv"
BRENT_CSV = DATA_DIR / "brent.csv"

# Parameter grid (coarser to reduce overfitting)
ENTRY_Z = [0.5, 0.7]
EXIT_Z = [0.0, 0.1, 0.2]
WEIGHTS = [(0.2, 0.3, 0.5), (0.1, 0.3, 0.6), (0.3, 0.3, 0.4), (0.5, 0.3, 0.2)]
TARGET_VOL = [0.005, 0.01]
W_MAX = [1.5, 2.0]
Z_WINDOW = [60, 120]

# Walk-forward folds: 6-month rolling train/test split (match train_test_split.png)
# Each fold: (train_start, train_end, test_start, test_end); test = 6 months
FOLDS = [
    ("2017-01-01", "2022-12-31", "2023-01-01", "2023-06-30"),  # Fold 1
    ("2017-07-01", "2023-06-30", "2023-07-01", "2023-12-31"),  # Fold 2
    ("2018-01-01", "2023-12-31", "2024-01-01", "2024-06-30"),  # Fold 3
    ("2018-07-01", "2024-06-30", "2024-07-01", "2024-12-31"),  # Fold 4
]
# Warm-up: first N calendar days of each train period used only for rolling stats (not for param scoring)
WARMUP_DAYS = 126  # ~6 months; covers z_window=120 and vol windows

# Overfitting-reduction: scoring and filters (applied to train period after warm-up)
TURNOVER_PENALTY = 0.4  # stiffer penalty on turnover (was 0.2)
PARAM_PENALTY = 0.05   # small penalty for extreme entry_z / z_window
MIN_TRAIN_SHARPE = 0.2  # prefer params with train Sharpe >= this; if none, use best by score
MIN_ACTIVE_DAYS = 20    # require at least this many non-flat days in train (after warm-up)
MIN_ACTIVE_FRAC = 0.05  # or at least this fraction of train days
TOP_K = 10             # among top K by score, pick by secondary criterion (lowest turnover)


def date_range_mask(df: pd.DataFrame, start: str, end: str, date_col: str = "date") -> pd.Series:
    """Boolean mask for rows where date is in [start, end] (inclusive)."""
    d = pd.to_datetime(df[date_col])
    return (d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))


def prepare_market_sentiment(
    market_csv: Path,
    sentiment_csv: Path,
    *,
    use_cumulative: bool = False,
) -> pd.DataFrame:
    """Load market + sentiment, merge, forward-fill sentiment on market dates."""
    mkt = load_yahoo_oil_csv(str(market_csv), recompute_returns=True)
    mkt = add_rolling_volatility(mkt)

    if use_cumulative:
        sent = load_cumulative_sentiment_features(sentiment_csv)
    else:
        sent = load_daily_sentiment_features(sentiment_csv)
    df = merge_sentiment_features(mkt, sent)

    for col in ["oil_sent_1d", "oil_sent_3d", "oil_sent_7d", "oil_news_count_7d"]:
        if col in df.columns:
            df[col] = df[col].ffill().fillna(0.0)

    return df


def run_fold(
    df: pd.DataFrame,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    instrument: str = "WTI",
) -> dict[str, Any]:
    """
    Run one fold: long training window (with warm-up at front), 6-month test.
    Rolling data is set up over the full train window; param search is scored only on
    train period after WARMUP_DAYS. Best params are then evaluated on the test period.
    """
    mask_train = date_range_mask(df, train_start, train_end)
    mask_train_and_test = date_range_mask(df, train_start, test_end)

    df_train = df.loc[mask_train].copy().reset_index(drop=True)
    df_train_and_test = df.loc[mask_train_and_test].copy().reset_index(drop=True)

    if len(df_train) == 0:
        return {
            "instrument": instrument,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "best_params": {},
            "train_score": -1e9,
            "train_stats": {},
            "test_stats": {},
            "bt_test": None,
            "error": "No data in train period",
        }

    # Warm-up: first WARMUP_DAYS of train used only for rolling stats; score params after this
    first_train_date = pd.to_datetime(df_train["date"].min())
    warmup_end_date = first_train_date + pd.Timedelta(days=WARMUP_DAYS)

    def collect_candidates() -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for entry_z, exit_z, (w1, w3, w7), target_vol, w_max, z_window in itertools.product(
            ENTRY_Z, EXIT_Z, WEIGHTS, TARGET_VOL, W_MAX, Z_WINDOW
        ):
            cfg = SentimentDirectionalConfig(
                w1=w1, w3=w3, w7=w7,
                z_window=z_window,
                z_min_periods=z_window // 2,
                entry_z=entry_z,
                exit_z=exit_z,
                target_vol=target_vol,
                w_max=w_max,
                contrarian=False,
            )
            bt = run_directional_strategy(df_train, cfg)
            # Score only on train period after warm-up
            mask_scoring = (pd.to_datetime(bt["date"]) > warmup_end_date) & (
                pd.to_datetime(bt["date"]) <= pd.Timestamp(train_end)
            )
            bt_scoring = bt.loc[mask_scoring].reset_index(drop=True)
            if len(bt_scoring) == 0:
                continue
            n_days = len(bt_scoring)
            active_days = int((bt_scoring["pos_state"] != 0).sum())
            min_active = max(MIN_ACTIVE_DAYS, int(n_days * MIN_ACTIVE_FRAC))
            if active_days < min_active:
                continue
            stats = performance_summary(bt_scoring)
            score = optimization_score(
                stats,
                turnover_penalty=TURNOVER_PENALTY,
                param_penalty=PARAM_PENALTY,
                entry_z=entry_z,
                z_window=z_window,
            )
            params = {
                "entry_z": entry_z,
                "exit_z": exit_z,
                "w1": w1, "w3": w3, "w7": w7,
                "target_vol": target_vol,
                "w_max": w_max,
                "z_window": z_window,
            }
            out.append({
                "score": score,
                "turnover": stats.get("avg_daily_turnover") or 0.0,
                "sharpe": stats.get("sharpe") or 0.0,
                "params": params,
                "cfg": cfg,
                "stats": stats,
            })
        return out

    candidates = collect_candidates()

    # Prefer candidates with train Sharpe >= MIN_TRAIN_SHARPE; if none, use all
    preferred = [c for c in candidates if c["sharpe"] >= MIN_TRAIN_SHARPE]
    pool = preferred if preferred else candidates

    if not pool:
        return {
            "instrument": instrument,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "best_params": {},
            "train_score": -1e9,
            "train_stats": {},
            "test_stats": {},
            "bt_test": None,
            "error": "No valid param combination",
        }

    # Top-k by score, then pick lowest turnover among top-k
    pool.sort(key=lambda c: c["score"], reverse=True)
    top_k = pool[:TOP_K]
    best_candidate = min(top_k, key=lambda c: c["turnover"])

    best_cfg = best_candidate["cfg"]
    best_params = best_candidate["params"]
    best_train_stats = best_candidate["stats"]
    best_score = best_candidate["score"]

    # Test with best params (run on train+test so rolling stats carry over into test)
    bt_test_full = run_directional_strategy(df_train_and_test, best_cfg)
    mask_test_bt = date_range_mask(bt_test_full, test_start, test_end)
    bt_test = bt_test_full.loc[mask_test_bt].reset_index(drop=True)
    test_stats = performance_summary(bt_test) if len(bt_test) > 0 else {}

    return {
        "instrument": instrument,
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "best_params": best_params,
        "train_score": best_score,
        "train_stats": best_train_stats,
        "test_stats": test_stats,
        "bt_test": bt_test,
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Walk-forward backtest with sentiment")
    parser.add_argument("--sentiment", "-s", choices=["finbert", "crudebert"], default="crudebert",
                        help="Sentiment source: finbert (oil_news) or crudebert (default: crudebert)")
    parser.add_argument("--cumulative", "-c", action="store_true",
                        help="Use Crudebert cumulative (CSS) sentiment; only valid with --sentiment crudebert")
    args = parser.parse_args()

    use_cumulative = args.cumulative
    if use_cumulative and args.sentiment != "crudebert":
        print("--cumulative is only valid with --sentiment crudebert. Ignoring --cumulative.")
        use_cumulative = False

    if use_cumulative:
        sentiment_csv = SENTIMENT_CRUDEBERT_CSS
    else:
        sentiment_csv = SENTIMENT_CRUDEBERT if args.sentiment == "crudebert" else SENTIMENT_FINBERT

    if not sentiment_csv.exists():
        print(f"Sentiment CSV not found: {sentiment_csv}")
        if use_cumulative:
            print("Run: python final_analysis/calculate_cumulative_sentiment.py "
                  "--input final_analysis/daily_sentiment_scores_crudebert_2017_2024.csv "
                  "--output final_analysis/daily_cumulative_sentiment_scores_crudebert_2017_2024.csv")
        elif args.sentiment == "crudebert":
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
        print("No market CSV found. Expected data/wti.csv or data/brent.csv")
        return

    print("=" * 60)
    print("Walk-Forward Backtest (6-month train/test split, warm-up at front)")
    print("=" * 60)
    print(f"Sentiment: {args.sentiment}" + (" (cumulative)" if use_cumulative else "") + f" -> {sentiment_csv}")
    print(f"Folds: train ~6y, test 6 months; first {WARMUP_DAYS} days of train = warm-up for rolling data")
    print(f"Parameter grid: {len(ENTRY_Z)*len(EXIT_Z)*len(WEIGHTS)*len(TARGET_VOL)*len(W_MAX)*len(Z_WINDOW)} combinations")
    print(f"Score = Sharpe - |MaxDD| - {TURNOVER_PENALTY}*turnover - param_penalty; top-{TOP_K} then lowest turnover")
    print()

    all_results: list[dict[str, Any]] = []
    # Collect test-set backtest DataFrames per instrument for CSV export
    instrument_bt_tests: dict[str, list[pd.DataFrame]] = {name: [] for name, _ in instruments}

    for instrument, market_csv in instruments:
        print(f"\n--- {instrument} ---")
        df = prepare_market_sentiment(market_csv, sentiment_csv, use_cumulative=use_cumulative)
        print(f"Market + sentiment: {len(df)} rows, {df['date'].min()} to {df['date'].max()}")

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

            bp = result["best_params"]
            vs = result["train_stats"]
            ts = result["test_stats"]
            print(f"    Best params: entry_z={bp['entry_z']}, exit_z={bp['exit_z']}, "
                  f"w=({bp['w1']},{bp['w3']},{bp['w7']}), target_vol={bp['target_vol']}, "
                  f"w_max={bp['w_max']}, z_window={bp['z_window']}")
            print(f"    Train Score: {result['train_score']:.4f} | "
                  f"Train Sharpe: {vs.get('sharpe', 0):.4f}, MaxDD: {vs.get('max_drawdown', 0):.4f}")
            print(f"    Test  Sharpe: {ts.get('sharpe', 0):.4f}, MaxDD: {ts.get('max_drawdown', 0):.4f}, "
                  f"AnnRet: {ts.get('annualized_return', 0):.4f}")

    # Summary table
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

    # Save concatenated test-set backtest summaries to data folder
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for instrument, market_csv in instruments:
        frames = instrument_bt_tests.get(instrument, [])
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
        out_name = "wti_strategy_backtest.csv" if instrument == "WTI" else "brent_strategy_backtest.csv"
        out_path = DATA_DIR / out_name
        combined.to_csv(out_path, index=False)
        print(f"\nSaved test-set backtest: {out_path} ({len(combined)} rows)")


if __name__ == "__main__":
    main()
