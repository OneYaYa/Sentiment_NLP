"""Defaults for mean-reversion (RSI) oil strategy and backtest."""

from __future__ import annotations

# USO = US Oil Fund ETF. "CL=F" / "BZ=F" are futures — see data_loader / backtest for handling.
DEFAULT_TICKER = "USO"

# RSI length (trading days).
DEFAULT_RSI_WINDOW = 14
# Hysteresis: go long when RSI <= oversold; stay long until RSI >= overbought, then flat.
DEFAULT_RSI_OVERSOLD = 30.0
DEFAULT_RSI_OVERBOUGHT = 70.0

DEFAULT_START_DATE: str | None = "2015-01-01"
DEFAULT_END_DATE: str | None = None

INITIAL_CASH = 100_000.0
COMMISSION_RATE = 0.0005
RISK_FREE_ANNUAL = 0.0
TRADING_DAYS_PER_YEAR = 252
