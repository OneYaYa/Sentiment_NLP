"""Regime-switching strategy: ADX trend vs chop, SMA vs RSI."""

from __future__ import annotations

DEFAULT_TICKER = "USO"

# SMA (trend regime)
DEFAULT_FAST_WINDOW = 20
DEFAULT_SLOW_WINDOW = 50

# RSI (mean-reversion regime)
DEFAULT_RSI_WINDOW = 14
DEFAULT_RSI_OVERSOLD = 30.0
DEFAULT_RSI_OVERBOUGHT = 70.0

# ADX: period and threshold (trend if ADX >= threshold)
DEFAULT_ADX_PERIOD = 14
DEFAULT_ADX_THRESHOLD = 25.0

DEFAULT_START_DATE: str | None = "2015-01-01"
DEFAULT_END_DATE: str | None = None

INITIAL_CASH = 100_000.0
COMMISSION_RATE = 0.0005
RISK_FREE_ANNUAL = 0.0
TRADING_DAYS_PER_YEAR = 252
