"""Defaults for logistic-regression directional strategy and backtest."""

from __future__ import annotations

DEFAULT_TICKER = "USO"

# Rolling training window (trading days) before each prediction.
DEFAULT_TRAIN_WINDOW = 252
# Number of lagged daily returns as features (uses ret.shift(1)..shift(n)).
DEFAULT_N_LAGS = 5
# Long when predicted P(next-day return > 0) exceeds this.
DEFAULT_PROB_THRESHOLD = 0.5

DEFAULT_START_DATE: str | None = "2015-01-01"
DEFAULT_END_DATE: str | None = None

INITIAL_CASH = 100_000.0
COMMISSION_RATE = 0.0005
RISK_FREE_ANNUAL = 0.0
TRADING_DAYS_PER_YEAR = 252
