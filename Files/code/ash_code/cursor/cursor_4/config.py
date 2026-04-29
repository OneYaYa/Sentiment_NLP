"""Defaults for walk-forward random forest directional strategy and backtest."""

from __future__ import annotations

DEFAULT_TICKER = "USO"

DEFAULT_TRAIN_WINDOW = 252
DEFAULT_N_LAGS = 5
DEFAULT_PROB_THRESHOLD = 0.5

# Lower = faster walk-forward loop (~minutes with 200+ trees on long history). Increase via CLI.
DEFAULT_N_ESTIMATORS = 100
# Shallow trees help with small rolling windows. Use --max-depth 0 for unlimited depth.
DEFAULT_MAX_DEPTH = 8

DEFAULT_RANDOM_STATE = 42

DEFAULT_START_DATE: str | None = "2015-01-01"
DEFAULT_END_DATE: str | None = None

INITIAL_CASH = 100_000.0
COMMISSION_RATE = 0.0005
RISK_FREE_ANNUAL = 0.0
TRADING_DAYS_PER_YEAR = 252
