"""Long-only backtest with optional proportional commission on turnover."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import RISK_FREE_ANNUAL, TRADING_DAYS_PER_YEAR


def _sanitize_market_returns(mkt_ret: pd.Series) -> pd.Series:
    """Clip pathological daily returns (e.g. futures roll / negative prints) so equity stays well-defined."""
    r = mkt_ret.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return r.clip(-0.99, 10.0)


def run_long_only_backtest(
    close: pd.Series,
    position: pd.Series,
    commission_rate: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    price = close.astype(float)
    mkt_ret = _sanitize_market_returns(price.pct_change().fillna(0.0))
    pos = position.astype(float).reindex(price.index).fillna(0.0)
    pos_lagged = pos.shift(1).fillna(0.0)

    turnover = (pos - pos_lagged).abs()
    costs = turnover * commission_rate
    strat_ret = pos_lagged * mkt_ret - costs
    equity = (1.0 + strat_ret).cumprod()
    return strat_ret, equity, pos


def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return float(dd.min())


def annualized_sharpe(daily_returns: pd.Series, rf_annual: float = RISK_FREE_ANNUAL) -> float:
    r = daily_returns.dropna()
    if len(r) < 2:
        return float("nan")
    rf_daily = (1.0 + rf_annual) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    excess = r - rf_daily
    std = excess.std()
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(np.sqrt(TRADING_DAYS_PER_YEAR) * excess.mean() / std)


def summarize(
    daily_returns: pd.Series,
    equity: pd.Series,
    years: float | None = None,
) -> dict[str, float]:
    r = daily_returns.dropna()
    n = len(r)
    if n == 0:
        return {
            "total_return": float("nan"),
            "cagr": float("nan"),
            "vol_annual": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
        }

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) else float("nan")

    if years is None or years <= 0:
        years = n / TRADING_DAYS_PER_YEAR
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0) if len(equity) > 1 else float("nan")

    vol_annual = float(r.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    sharpe = annualized_sharpe(r)
    mdd = max_drawdown(equity)

    return {
        "total_return": total_return,
        "cagr": cagr,
        "vol_annual": vol_annual,
        "sharpe": sharpe,
        "max_drawdown": mdd,
    }


def buy_and_hold_returns(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    price = close.astype(float)
    ones = pd.Series(1.0, index=price.index)
    strat_ret, equity, _ = run_long_only_backtest(price, ones, commission_rate=0.0)
    return strat_ret, equity
