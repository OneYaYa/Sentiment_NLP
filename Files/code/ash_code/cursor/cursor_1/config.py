"""Default parameters for the base oil trading strategy and backtest."""

from __future__ import annotations

# USO = US Oil Fund ETF (smooth series, good baseline). "CL=F" / "BZ=F" are futures; raw Yahoo
# closes can include extreme rolls (e.g. April 2020) — returns are sanitized in backtest.
DEFAULT_TICKER = "USO"

# Simple moving average crossover: fast / slow lookbacks (trading days).
DEFAULT_FAST_WINDOW = 20
DEFAULT_SLOW_WINDOW = 50

# Backtest window (None = use all data returned by yfinance).
DEFAULT_START_DATE: str | None = "2015-01-01"
DEFAULT_END_DATE: str | None = None

INITIAL_CASH = 100_000.0
# Proportional one-way transaction cost (e.g. 0.0005 = 5 bps per trade side).
COMMISSION_RATE = 0.0005

# Risk-free rate for Sharpe (annualized); set to 0 for simplicity unless you have a series.
RISK_FREE_ANNUAL = 0.0

# Trading days per year for annualization
TRADING_DAYS_PER_YEAR = 252
