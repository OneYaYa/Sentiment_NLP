"""Charts for random-forest backtest vs buy-and-hold."""

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
    ax0.plot(
        strategy_equity.index,
        strategy_equity.values,
        label="Random forest",
        color="#1f77b4",
        linewidth=1.5,
    )
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


def plot_price_prob_position(
    df: pd.DataFrame,
    prob_threshold: float,
    title: str,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    """
    df must include: close, prob_up, position (0/1 executed — typically signal lagged one day).
    """
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    ax_p = axes[0]
    y_min = float(df["close"].min())
    y_max = float(df["close"].max())
    pad = 0.02 * (y_max - y_min)
    long_mask = (df["position"] >= 0.5).to_numpy()
    ax_p.fill_between(
        df.index,
        y_min - pad,
        y_max + pad,
        where=long_mask,
        alpha=0.12,
        color="green",
        linewidth=0,
        label="Long (executed)",
    )
    ax_p.plot(df.index, df["close"], label="Close", color="black", linewidth=1.0, alpha=0.85)
    ax_p.set_ylabel("Price")
    ax_p.set_title(title)
    ax_p.legend(loc="upper left", fontsize=8)
    ax_p.grid(True, alpha=0.3)

    ax_r = axes[1]
    ax_r.plot(df.index, df["prob_up"], label=r"$P(\mathrm{up})$", color="#9467bd", linewidth=1.0)
    ax_r.axhline(prob_threshold, color="#d62728", linestyle="--", linewidth=0.9, label=f"Threshold ({prob_threshold:g})")
    ax_r.set_ylim(0, 1)
    ax_r.set_ylabel(r"$P(\mathrm{up})$")
    ax_r.set_xlabel("Date")
    ax_r.legend(loc="upper left", fontsize=8)
    ax_r.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
