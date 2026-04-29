"""
Visualize four folds with 6-month rolling train/test split.

Each fold: training and test windows roll forward by 6 months.
- Fold 1: Train 2017-01–2022-12, Test 2023 H1
- Fold 2: Train 2017-07–2023-06, Test 2023 H2
- Fold 3: Train 2018-01–2023-12, Test 2024 H1
- Fold 4: Train 2018-07–2024-06, Test 2024 H2

Output: data/backtest_plots/train_test_split.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
PLOTS_DIR = ROOT_DIR / "data" / "backtest_plots"
OUTPUT_PATH = PLOTS_DIR / "train_test_split.png"

# Four folds: (train_start, train_end, test_start, test_end) – 6-month roll each time
FOLDS = [
    ("Fold 1", "2017-01-01", "2022-12-31", "2023-01-01", "2023-06-30"),
    ("Fold 2", "2017-07-01", "2023-06-30", "2023-07-01", "2023-12-31"),
    ("Fold 3", "2018-01-01", "2023-12-31", "2024-01-01", "2024-06-30"),
    ("Fold 4", "2018-07-01", "2024-06-30", "2024-07-01", "2024-12-31"),
]


def main() -> None:
    def to_num(s: str) -> float:
        return mdates.datestr2num(s)

    n_folds = len(FOLDS)
    fig, ax = plt.subplots(figsize=(14, 4))
    bar_height = 0.18
    gap = 0.06

    # Y positions: one row per fold (top = Fold 1)
    y_centers = [1 - (i + 0.5) * (bar_height + gap) for i in range(n_folds)]

    for i, (label, train_s, train_e, test_s, test_e) in enumerate(FOLDS):
        y = y_centers[i]
        # Training block
        train_left = to_num(train_s)
        train_width = to_num(train_e) - train_left
        ax.barh(
            y,
            width=train_width,
            left=train_left,
            height=bar_height,
            color="steelblue",
            alpha=0.85,
            edgecolor="navy",
            linewidth=0.8,
        )
        ax.text(
            train_left + train_width / 2, y,
            "Train",
            ha="center", va="center", fontsize=9, color="white", fontweight="bold",
        )
        # Test block
        test_left = to_num(test_s)
        test_width = to_num(test_e) - test_left
        ax.barh(
            y,
            width=test_width,
            left=test_left,
            height=bar_height,
            color="#27ae60",
            alpha=0.9,
            edgecolor="darkgreen",
            linewidth=0.8,
        )
        ax.text(
            test_left + test_width / 2, y,
            "Test",
            ha="center", va="center", fontsize=9, color="white", fontweight="bold",
        )

    ax.set_xlim(to_num("2016-06-01"), to_num("2025-01-01"))
    ax.set_ylim(0, 1.02)
    ax.set_yticks(y_centers)
    ax.set_yticklabels([f[0] for f in FOLDS])
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlabel("Time")
    ax.set_title("Four folds: 6‑month rolling train/test split")
    legend_elements = [
        mpatches.Patch(facecolor="steelblue", alpha=0.85, edgecolor="navy", label="Training"),
        mpatches.Patch(facecolor="#27ae60", alpha=0.9, edgecolor="darkgreen", label="Test"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", framealpha=0.9)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
