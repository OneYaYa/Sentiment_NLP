"""Charts for base strategy backtest vs buy-and-hold."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_and_drawdown(
    strategy_equity: pd.Series,
    bh_equity: pd.Series,
    title: str,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    ax0 = axes[0]
    ax0.plot(strategy_equity.index, strategy_equity.values, label="SMA strategy", color="#1f77b4", linewidth=1.5)
    ax0.plot(bh_equity.index, bh_equity.values, label="Buy & hold", color="#888888", linewidth=1.2, alpha=0.9)
    ax0.set_ylabel("Equity (normalized)")
    ax0.set_title(title)
    ax0.legend(loc="upper left")
    ax0.grid(True, alpha=0.3)

    dd_s = strategy_equity / strategy_equity.cummax() - 1.0
    dd_b = bh_equity / bh_equity.cummax() - 1.0
    axes[1].fill_between(dd_s.index, dd_s.values, 0, color="#1f77b4", alpha=0.35, label="Strategy DD")
    axes[1].plot(dd_b.index, dd_b.values, color="#888888", linewidth=1.0, label="B&H DD")
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    axes[1].legend(loc="lower left", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_price_and_position(
    df: pd.DataFrame,
    title: str,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    """
    df must include: close, sma_fast, sma_slow, position (0/1) — position should reflect held exposure
    (e.g. signal lagged one day) for accurate shading.
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    y_min = float(df["close"].min())
    y_max = float(df["close"].max())
    pad = 0.02 * (y_max - y_min)
    long_mask = (df["position"] >= 0.5).to_numpy()
    ax.fill_between(
        df.index,
        y_min - pad,
        y_max + pad,
        where=long_mask,
        alpha=0.12,
        color="green",
        linewidth=0,
        label="Long (executed)",
    )
    ax.plot(df.index, df["close"], label="Close", color="black", linewidth=1.0, alpha=0.85)
    ax.plot(df.index, df["sma_fast"], label="Fast SMA", color="#2ca02c", linewidth=1.0)
    ax.plot(df.index, df["sma_slow"], label="Slow SMA", color="#d62728", linewidth=1.0)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
