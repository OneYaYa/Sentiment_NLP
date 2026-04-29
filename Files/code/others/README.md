# Backtest, validation, visualization, and Crudebert CSS

This folder (`code/others/`) holds the main **trading backtests**, **alpha validation**, **backtest plots**, **Crudebert CSS analyses**, and supporting strategy modules.

**Data layout:** Market CSVs and sentiment outputs stay under **`code/data/`** and **`code/final_analysis/`**. Scripts here resolve those paths with `ROOT_DIR = Path(__file__).resolve().parent.parent` (the `code/` directory).

## Prerequisites

From the repository root (where `requirements.txt` lives):

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
pip install -r requirements.txt
```

## Working directory

Run the commands below from **`code/others`**:

```powershell
cd "path\to\Campbell-B\code\others"
```

## Contents of this folder (overview)

| Files | Role |
|-------|------|
| `sentiment_strategy.py` | Z-score directional strategy (loads daily/cumulative sentiment features). |
| `sentiment_strategy_quantile.py` | Quantile + medium-horizon strategy on CSS. |
| `backtest_runner.py` | Walk-forward backtest (z-score strategy). |
| `backtest_runner_quantile.py` | Walk-forward backtest (quantile strategy). |
| `sentiment_alpha_validation.py` | Predictive tests: daily sentiment vs next-day returns. |
| `visualize_backtest.py` | Equity, weights, turnover, cost from backtest CSVs. |
| `plot_sentiment_and_prices.py` | Daily sentiment vs futures close. |
| `plot_train_test_split.py` | Diagram of train/test folds. |
| `crudebert_css_alpha_validation.py` | CSS vs next-day return (OLS, NW), CSS/price plots. |
| `crudebert_css_shocks_monthly.py` | Monthly positive/negative CSS shocks vs month-end price. |
| `crudebert_css_quantile_returns.py` | Quintile cumulative returns, tests, box plots. |

## 1. Backtest (z-score / directional strategy)

**File:** `backtest_runner.py`

Walk-forward folds (~6 years train, 6 months test, warm-up at start of train). Writes:

- `../data/wti_strategy_backtest.csv`
- `../data/brent_strategy_backtest.csv`

```powershell
python backtest_runner.py --sentiment crudebert
python backtest_runner.py --sentiment crudebert --cumulative
python backtest_runner.py --sentiment finbert
```

- `--sentiment finbert` | `crudebert`: daily sentiment source.
- `--cumulative` / `-c`: Crudebert **CSS** cumulative CSV (only with `crudebert`).

## 2. Backtest (quantile + medium-horizon strategy)

**Files:** `backtest_runner_quantile.py`, `sentiment_strategy_quantile.py`

Same folds as above. Writes:

- `../data/wti_strategy_backtest_quantile.csv`
- `../data/brent_strategy_backtest_quantile.csv`

```powershell
python backtest_runner_quantile.py
```

## 3. Alpha validation (daily sentiment → next-day returns)

**File:** `sentiment_alpha_validation.py`

OLS (Newey–West), correlation, quintiles, permutation, subperiods.

```powershell
python sentiment_alpha_validation.py --sentiment crudebert
python sentiment_alpha_validation.py --sentiment finbert
```

**Output:** `../data/backtest_plots/alpha_validation_report_<sentiment>.csv`

## 4. Visualization

### Backtest (equity, weights, turnover, cost)

**File:** `visualize_backtest.py`

```powershell
python visualize_backtest.py
python visualize_backtest.py --strategy zscore
python visualize_backtest.py --strategy quantile
```

**Reads:** `../data/wti_strategy_backtest.csv` / `brent_strategy_backtest.csv`, or `*_quantile.csv` when `--strategy quantile`.

**Writes:** `../data/backtest_plots/wti_backtest_plots.png` (and Brent), or `*_quantile.png`.

### Sentiment vs futures prices

**File:** `plot_sentiment_and_prices.py`

```powershell
python plot_sentiment_and_prices.py --sentiment crudebert
python plot_sentiment_and_prices.py --sentiment finbert
```

**Writes:** `../data/backtest_plots/sentiment_and_prices.png`

### Train / test fold diagram

**File:** `plot_train_test_split.py`

```powershell
python plot_train_test_split.py
```

**Writes:** `../data/backtest_plots/train_test_split.png`

## 5. Crudebert CSS analyses

All run from **`code/others`** (same as above).

| Script | Purpose | Example outputs under `../data/backtest_plots/` |
|--------|---------|-----------------------------------------------|
| `crudebert_css_alpha_validation.py` | Next-day return on CSS_t; CSS vs price time series | `crudebert_css_wti.png`, `crudebert_css_brent.png` |
| `crudebert_css_shocks_monthly.py` | Monthly shock counts vs month-end futures | `crudebert_css_shocks_all.png` |
| `crudebert_css_quantile_returns.py` | Quintile cumulative returns, Welch/permutation tests, box plots | `crudebert_css_quantile_boxplot_wti.png`, `crudebert_css_quantile_boxplot_brent.png` |

```powershell
python crudebert_css_alpha_validation.py
python crudebert_css_shocks_monthly.py
python crudebert_css_quantile_returns.py
```

**Input:** `../final_analysis/daily_cumulative_sentiment_scores_crudebert_2017_2024.csv` (and `../data/wti.csv`, `../data/brent.csv`).

## 6. Sentiment pipeline (not in `others`)

Daily and cumulative article-level sentiment CSVs are produced from **`code/final_analysis/`**:

```powershell
cd ..\final_analysis
python calculate_daily_sentiment.py --input crudebert
python calculate_cumulative_sentiment.py --input daily_sentiment_scores_crudebert_2017_2024.csv --output daily_cumulative_sentiment_scores_crudebert_2017_2024.csv
```

(Adjust `--input` / `--output` paths as needed.)

## Quick reference

| Task | Command (from `code/others`) |
|------|------------------------------|
| Z-score backtest (Crudebert CSS) | `python backtest_runner.py -s crudebert -c` |
| Quantile backtest | `python backtest_runner_quantile.py` |
| Alpha validation | `python sentiment_alpha_validation.py -s crudebert` |
| Plot backtest | `python visualize_backtest.py -s quantile` |
| CSS regression / shocks / quantiles | `python crudebert_css_alpha_validation.py` / `crudebert_css_shocks_monthly.py` / `crudebert_css_quantile_returns.py` |
| Fold diagram | `python plot_train_test_split.py` |
