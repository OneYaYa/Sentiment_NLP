# Base Strategies
_Exported on 3/30/2026 at 21:35:26 EDT from Cursor (2.6.20)_

---

**User**

The Campbell - B directory contains code and files related to the group project I am working on. The aim of the project is to develop and test llm strategies from scratch with the help of ai. For the past one month, we worked on a strategy based on news sentiment analysis but now the supervisors want a base strategy so that more complex strategies could be compared and developed from the base strategies.

Generate code to implement a base trading strategy, backtest and display the results. The thing to trade is oil (commodity trading). Generate the entire code in python (ex. use modules like yfinance for the data for backtesting).

Generate all the code inside code/ash_code directory inside Campbell-B.

---

**Cursor**

Exploring the project structure and implementing a base oil-trading strategy with backtesting in `code/ash_code`.



Creating a self-contained base strategy package: SMA crossover on oil (yfinance), a proper backtest with lookahead avoided, buy-and-hold benchmark, metrics, and plots.



Fixing incomplete plotting logic in `plot_results.py`, simplifying `main.py` (remove duplicate plotting), and correcting the trade-count metric.



Refactoring plotting: use `fill_between` for position shading, add a `show` parameter, set the matplotlib backend before importing pyplot when using `--no-show`, and simplify `main.py`.



Investigating extreme returns in CL=F data causing invalid buy-and-hold metrics:



Sanitizing extreme daily returns (April 2020 produced invalid negative-oil prints on CL=F, ~-306% daily returns). Adding return sanitization and defaulting to USO for a cleaner ETF baseline; keeping CL=F available.



Repairing non-positive futures prints (e.g. April 2020 WTI) in the data loader so buy-and-hold on CL=F stays economically interpretable.



Here is what was added under `code/ash_code/` and how to use it.

## What you have

**Base strategy:** long-only **20/50 simple moving average (SMA) crossover** on the close — stay long when fast SMA &gt; slow SMA, otherwise flat. Signals are applied **one day later** (no lookahead). **Buy-and-hold** on the same series is reported as the benchmark.

**Data:** `yfinance` daily OHLCV. Default symbol is **`USO`** (US Oil Fund ETF); you can use **`CL=F`** (WTI) or **`BZ=F`** (Brent). Futures rows with non-positive closes are forward-filled and **return clipping** is applied so paths stay numerically stable (see `data_loader.py` and `backtest.py`).

**Outputs:** Console metrics (total return, CAGR, annualized vol, Sharpe, max drawdown, long entries) and two PNGs in `code/ash_code/output/`:
- `equity_drawdown.png` — strategy vs buy-and-hold equity and drawdowns  
- `price_smas_position.png` — price, SMAs, green shading when the strategy is long (executed)

**Files**

| File | Role |
|------|------|
| `config.py` | Ticker, windows, cash, commission, risk-free |
| `data_loader.py` | Yahoo download + futures bar repair |
| `strategy.py` | SMAs and binary position |
| `backtest.py` | Long-only engine, fees on turnover, metrics |
| `plot_results.py` | Matplotlib charts |
| `main.py` | CLI + run + print + save/show plots |
| `requirements.txt` | `pandas`, `numpy`, `matplotlib`, `yfinance` |

## How to run

From the repo’s `code` folder:

```bash
pip install -r ash_code/requirements.txt
python ash_code/main.py
```

Headless (save PNGs only):

```bash
python ash_code/main.py --no-show
```

Examples:

```bash
python ash_code/main.py --ticker CL=F --fast 20 --slow 50 --start 2015-01-01
python ash_code/main.py --ticker BZ=F --no-show
```

This gives you a **simple, comparable baseline** (rules-based, no sentiment) so you can stack sentiment or other models on top and compare Sharpe, drawdown, and turnover against the same machinery.

---

**User**

create a directory inside ash_code called cursor. inside cursor create an other directory called cursor_1. move all the files generated into cursor_1.

---

**Cursor**

Creating `ash_code/cursor/cursor_1` and moving the generated project files into it.



Updating `main.py` run instructions now that the project lives in `cursor/cursor_1`.



Here is what was done:

### Layout

- **`code/ash_code/cursor/cursor_1/`** now contains everything that used to live directly under `ash_code`:
  - `config.py`, `data_loader.py`, `strategy.py`, `backtest.py`, `plot_results.py`, `main.py`
  - `requirements.txt`, `__init__.py`
  - If an **`output/`** folder existed under `ash_code`, it was moved into `cursor_1` as well (there was no `output/` in the tree after the move, so nothing extra was relocated).

### Docstring

`main.py` was updated so the documented commands match the new path:

```bash
python ash_code/cursor/cursor_1/main.py
python ash_code/cursor/cursor_1/main.py --no-show
```

Run from the **`code`** directory (or adjust paths). Imports are unchanged because `main.py` still adds its own directory (`cursor_1`) to `sys.path`.

---

**User**

now explain what each file does and how to run the code

---

**Cursor**

## Layout

Everything lives under **`Campbell-B/code/ash_code/cursor/cursor_1/`**.

---

## What each file does

| File | Purpose |
|------|--------|
| **`main.py`** | Entry point: parses CLI flags, loads data, runs the SMA strategy and buy-and-hold backtest, prints metrics (return, CAGR, vol, Sharpe, max drawdown, long entries), and saves/shows charts. |
| **`config.py`** | Central defaults: Yahoo ticker (default **USO**), SMA windows (20/50), backtest start/end dates, starting cash, commission rate, risk-free rate for Sharpe, trading days per year. |
| **`data_loader.py`** | Downloads daily OHLCV with **yfinance** for the chosen symbol. **Repairs** bad futures bars (e.g. non-positive closes) where needed so returns stay finite. |
| **`strategy.py`** | Computes **fast/slow SMAs** on close and a **long-only signal**: position = 1 when fast SMA &gt; slow SMA, else 0. |
| **`backtest.py`** | **Long-only backtest engine**: applies **one-day lag** on the signal (no lookahead), optional **commission on turnover**, **sanitizes** extreme daily returns, builds **equity** from daily returns, and computes **summary stats** (Sharpe, max drawdown, etc.). Includes **buy-and-hold** on the same close series. |
| **`plot_results.py`** | **Matplotlib** figures: **equity vs buy-and-hold** with drawdown panel, and **price + SMAs** with green shading for executed long periods. |
| **`requirements.txt`** | Lists **`pandas`**, **`numpy`**, **`matplotlib`**, **`yfinance`** for `pip install`. |
| **`__init__.py`** | Marks the folder as a Python package (minimal; optional for running `main.py` directly). |

---

## How to run

**1. Install dependencies** (from the repo root or any folder; point to this file):

```bash
pip install -r code/ash_code/cursor/cursor_1/requirements.txt
```

If you use the project venv:

```bash
source venv/bin/activate   # or your venv path
pip install -r code/ash_code/cursor/cursor_1/requirements.txt
```

**2. Run the backtest** from the **`Campbell-B/code`** directory (so paths match the docs):

```bash
cd /Users/aswathsuresh/Documents/Projects/Campbell-B/code
python ash_code/cursor/cursor_1/main.py
```

That opens plot windows and writes PNGs under **`ash_code/cursor/cursor_1/output/`** by default.

**Headless** (no GUI; only saves PNGs):

```bash
python ash_code/cursor/cursor_1/main.py --no-show
```

**Useful options:**

| Flag | Meaning |
|------|--------|
| `--ticker USO` | Oil ETF (default). |
| `--ticker CL=F` | WTI futures. |
| `--ticker BZ=F` | Brent futures. |
| `--fast 20 --slow 50` | SMA windows (defaults match `config.py`). |
| `--start 2015-01-01` | Sample start. |
| `--end 2024-12-31` | Optional end. |
| `--output-dir /path/to/folder` | Where to save PNGs. |

**Example:**

```bash
python ash_code/cursor/cursor_1/main.py --ticker CL=F --start 2015-01-01 --no-show
```

You need **network** on first run so **yfinance** can download price data.

---

**User**

Now with one base strategy done, give me possible ideas for the second base strategy. 

---

**Cursor**

Providing distinct, rules-based second baselines that pair well with your long-only SMA crossover and the same daily oil data.



---

**User**

I'm planning to do multiple strategies. For now, which strategy do you think I should go for as the second one. Moving averages have been covered as the first strategy.

---

**Cursor**

Given your first baseline is **moving averages** (short vs long trend / crossover), the strongest choice for a **second** rule-based strategy is **long-horizon time-series momentum** (often called **12–1 momentum** in the literature: e.g. long if the past ~12-month return is positive, otherwise flat).

**Why this one**

1. **Different idea from MAs** — SMA cross compares **two smoothed levels** of price. Momentum compares **today’s price to a single past price** (cumulative return over a long window). Same OHLCV data, but a different economic story (“persistent drift” vs “which smoother is on top”).
2. **Standard benchmark** — Very common in academic and practitioner work, so it is easy to justify to supervisors and to cite.
3. **Easy to implement** — One lagged return or sign rule on `Close`, fits your existing long-only, one-day-lag backtest.
4. **Good ladder for later** — When you add sentiment or news, you can contrast **price momentum** vs **text/sentiment momentum** in a clear way.

**What I would not pick as “the” second only**

- **Another MA variant** (triple cross, MACD, etc.) — overlaps the first story unless you explicitly want a family of technical rules.
- **Buy-and-hold** — Still essential as a **benchmark row** in tables and charts, but it is passive; it does not add a second *active* rule unless your goal is explicitly “SMA vs passive only.”

**If you want maximum contrast to trend-following** (instead of another trend rule), the alternative second baseline is **mean reversion** (e.g. RSI or z-score vs a moving average). That pairs well if your thesis is that oil or news **mean-reverts**; momentum pairs better if your thesis is **persistence**.

**Practical pick:** use **12–1 (or 6–1) momentum** as the second implemented strategy, and keep **buy-and-hold** as the always-on benchmark in the same report.

---

**User**

Give me options 

---

**Cursor**

Here are **clear options for a second base strategy** (you already have **moving-average crossover** as #1). Pick based on what story you want to tell and how different you want the rule to be.

---

### 1. **Long-horizon time-series momentum** (e.g. 12–1 or 6–1 month)  
- **Rule:** Long if past return over a long window (e.g. ~252 or ~126 trading days) is **positive**, else flat.  
- **vs MAs:** Trend signal from **cumulative return**, not two smoothers crossing.  
- **Best if:** You want a standard “momentum factor” baseline and room to compare with **news/sentiment momentum** later.

---

### 2. **Mean reversion** (RSI or price z-score)  
- **Rule:** Long when **oversold** (e.g. RSI(14) &lt; 30, or price below 20d mean by k·σ); flat or exit on “overbought.”  
- **vs MAs:** **Fade** extremes instead of **follow** trend.  
- **Best if:** You want a deliberate contrast to trend-following (MA + momentum both “ride”; this “buys dips”).

---

### 3. **Breakout / Donchian** (e.g. N-day high)  
- **Rule:** Long when close **breaks above** the highest high of the prior N days; exit on a lower rule (M-day low or stop).  
- **vs MAs:** **Level breakouts** vs **MA cross**; often fewer, clearer events.  
- **Best if:** You want a second **trend** rule that feels different from MA math (channel / highs).

---

### 4. **Buy-and-hold** (as a formal “strategy #2” row)  
- **Rule:** 100% long the whole sample.  
- **vs MAs:** **No timing** — pure exposure.  
- **Best if:** You want the simplest passive comparator; note it’s not an “active” rule like the others.

---

### 5. **Volatility targeting** (long-only, fractional size)  
- **Rule:** Same direction signal as one of the above (or always long), but **scale** exposure so rolling vol targets a level (e.g. 10% ann.), cap at 100% long.  
- **vs MAs:** Separates **risk sizing** from **direction**.  
- **Best if:** Fairness vs strategies that change size matters; **more implementation** than 1–4.

---

**Quick pairing**

| Want… | Consider |
|--------|----------|
| Standard second rule + distinct from MA | **1 Momentum** |
| Maximum contrast to trend | **2 Mean reversion** |
| Another trend rule, different mechanic | **3 Breakout** |
| Simplest passive benchmark | **4 Buy-and-hold** |
| Emphasize risk / sizing | **5 Vol targeting** |

If you say which column matches your project (e.g. “momentum vs mean reversion”), the next step is to lock one option and define exact parameters (windows, thresholds).

---

**User**

Implement 2. Mean reversion inside CAMPBELL-B/code/ash_code/cursor_2 similar to first strategy implemented in cursor_1.

---

**Cursor**

Implementing the mean-reversion baseline in `code/ash_code/cursor_2`, mirroring `cursor_1` and adapting the strategy layer and plots.



Creating `cursor_2` with RSI-based mean reversion (hysteresis: enter long when RSI ≤ oversold, exit when RSI ≥ overbought), mirroring `cursor_1` structure.



Retrying file creation after the previous writes were blocked.



Here is what was added under [`code/ash_code/cursor_2/`](code/ash_code/cursor_2/) (parallel to `cursor_1`).

### Strategy (mean reversion)
- **RSI(14)** on close (simple rolling gain/loss RSI).
- **Hysteresis (long-only):** start flat → **go long** when RSI ≤ **30** → **stay long** until RSI ≥ **70** → flat again. Repeat.
- Same **one-day lag** on positions and **commission on turnover** as in `cursor_1`.

### Files
| File | Role |
|------|------|
| `config.py` | Defaults: `USO`, RSI 14, oversold 30, overbought 70, dates, cash, fees |
| `data_loader.py` | yfinance OHLCV + non-positive bar repair (same idea as `cursor_1`) |
| `backtest.py` | Long-only backtest + return sanitization + metrics + buy & hold |
| `strategy.py` | `rsi()`, hysteresis state machine, `add_rsi_mean_reversion_signals()` |
| `plot_results.py` | Equity vs B&H + drawdowns; **price + RSI panel** with 30/70 lines |
| `main.py` | CLI, prints table, saves `output/equity_drawdown.png`, `output/price_rsi_position.png` |
| `requirements.txt` | Same stack as `cursor_1` |
| `__init__.py` | Package marker |

### How to run
From the **`code`** directory:

```bash
pip install -r ash_code/cursor_2/requirements.txt
python ash_code/cursor_2/main.py
```

Headless:

```bash
python ash_code/cursor_2/main.py --no-show
```

Useful flags: `--ticker CL=F`, `--rsi-window 14`, `--oversold 30`, `--overbought 70`, `--start`, `--end`, `--output-dir`.

A test run completed successfully; plots are under `cursor_2/output/`. On USO 2015–present this rule is **not** attractive (negative total return vs buy & hold), which is acceptable for a **baseline** you compare against trend/SMA or sentiment strategies—document that mean reversion can fail in strong trends.

---

**User**

made a mistake in the directory. move all the files inside CAMPBELL-B/code/ash_code/cursor/cursor_2

---

**Cursor**

Moving `cursor_2` from `ash_code/cursor_2` to `ash_code/cursor/cursor_2` to match `cursor_1`.



Updating `main.py` docstring to the correct path.



Here is what was done:

### Move
- Everything that was in **`code/ash_code/cursor_2/`** (Python modules, `requirements.txt`, `output/` with PNGs, `__pycache__`) is now under **`code/ash_code/cursor/cursor_2/`**, next to **`cursor_1`**.
- The empty old folder **`code/ash_code/cursor_2/`** was removed.

### Docstring
In [`main.py`](code/ash_code/cursor/cursor_2/main.py), the run instructions were updated to:

```bash
python ash_code/cursor/cursor_2/main.py
python ash_code/cursor/cursor_2/main.py --no-show
```

(from the `code` directory, same pattern as `cursor_1`).

---

**User**

now for the third strategy, give me strategies with basic machine learning 

---

**Cursor**

Here are **third-strategy ideas that use basic machine learning** (still compatible with daily OHLCV and your existing backtest pattern). Each assumes you **label** something (e.g. “next day return ≥ 0”) and **predict** it from **past-only** features, then trade **long-only** when the model says “up” (or probability > 0.5), with **one-day lag** on the signal like your other baselines.

---

### 1. **Logistic regression (directional classifier)**  
**Idea:** Features = last *k* days of returns, maybe SMA distance, RSI, realized vol; **target** = 1 if next-day return > 0 else 0.  
**Model:** `sklearn.linear_model.LogisticRegression` (with scaling).  
**Why “basic”:** One interpretable weight per feature; fast; easy to explain to supervisors.  
**Caveat:** Linear decision boundary; may underfit if effects are nonlinear.

---

### 2. **Random forest classifier**  
**Same features/target as above.**  
**Model:** `sklearn.ensemble.RandomForestClassifier`.  
**Why:** Handles nonlinearities and interactions; still “basic” ML in practice; feature importance plots.  
**Caveat:** Easier to overfit on small samples; use **time-based** train/test or **walk-forward** validation.

---

### 3. **Gradient boosting (XGBoost / LightGBM / sklearn HistGradientBoosting)**  
**Idea:** Same tabular setup, often stronger than RF on medium-sized panels.  
**Why:** Strong default for structured features; still a standard “ML baseline” in industry.  
**Caveat:** More hyperparameters; still need strict **no leakage** (fit only on past data).

---

### 4. **Regressing next-day return, then trade on sign**  
**Idea:** Predict **continuous** next-day return with **Ridge** or **RandomForestRegressor**; go long if predicted return > 0 (or > threshold).  
**Why:** Directly tied to returns; Ridge is very simple.  
**Caveat:** Noisy targets; thresholding predicted return can be unstable.

---

### 5. **Volatility or regime as a second target**  
**Idea:** Classify **high-vol vs low-vol** tomorrow (or “large move” vs not), then combine with a **direction** model or use only **long in predicted low-vol** regimes.  
**Why:** Separates **risk** from **direction**; still basic ML.  
**Caveat:** More moving parts; need clear rules for combining signals.

---

### 6. **(Optional) Simple feedforward network**  
**Idea:** Small MLP on a **fixed window** of past returns (scaled).  
**Why:** Still “basic ML” in the deep-learning sense but often unnecessary for daily oil unless you have lots of data.  
**Caveat:** Harder to tune, easier to overfit; logistic/Ridge/RF often enough for a course baseline.

---

## What you should standardize across all ML ideas

| Piece | Recommendation |
|--------|----------------|
| **Features** | Only use information **available at close of day t** to predict **t+1** (or use features up to t−1 if you execute at t open—pick one convention and stick to it). |
| **Labels** | Next-day return sign or return bucket; **no** same-day future mixed into features. |
| **Evaluation** | **Walk-forward** or **rolling train** (e.g. train on past 3–5 years, test on next year) — not a single random split. |
| **Trading rule** | Long-only: e.g. long if P(up) > 0.5 or predicted return > 0; same **lagged** execution as your SMA/RSI code. |

---

## Practical pick for a “third” strategy in a group project

- **Best balance of simple + credible:** **Logistic regression** or **Ridge regression** on a **small feature set** (lags + 1–2 technicals).  
- **If you want a stronger ML baseline without much complexity:** **Random forest** with the same features and **walk-forward** backtest.

If you tell me whether you prefer **classifier** (long vs flat) vs **regressor** (predict return then threshold), I can narrow this to one concrete spec (features list, label, horizon, and train/test scheme) that matches your `cursor_1` / `cursor_2` style.

---

**User**

Implement logistic_regression inside CAMPBELL-B/code/ash_code/cursor/cursor_3.

---

**Cursor**

Implementing the logistic-regression strategy in `cursor_3`, mirroring `cursor_1`/`cursor_2` and adding a rolling-window walk-forward predictor.



Creating `cursor_3` with rolling-window logistic regression: features = lagged returns; label = next-day return > 0; walk-forward training with no lookahead.



Fixing features: include returns through day `t` (lags `ret.shift(0)` … `ret.shift(n_lags-1)`).



Here’s what was added under [`code/ash_code/cursor/cursor_3/`](code/ash_code/cursor/cursor_3/).

### Strategy
- **Walk-forward logistic regression** on **lagged daily returns** (`ret`, `ret.shift(1)`, … `ret.shift(n_lags-1)`), all known at the close of day `t`.
- **Label:** `y_t = 1` if **next-day** return `ret[t+1] > 0` (no lookahead in features).
- **Training:** For each day `k`, fit on the previous **`train_window`** rows only (default **252**), with `StandardScaler` fit on that training block.
- **Trade:** Long **iff** `P(up) > prob_threshold` (default **0.5**). `LogisticRegression` uses `class_weight="balanced"`, `solver="lbfgs"`.
- Skips windows with **&lt;10** rows, **one class only**, or failed fits.

### Files (same layout as `cursor_1` / `cursor_2`)
| File | Role |
|------|------|
| `config.py` | `USO`, `train_window`, `n_lags`, `prob_threshold`, dates, fees |
| `data_loader.py` | yfinance OHLCV + futures bar repair |
| `backtest.py` | Same long-only engine + sanitized returns |
| `strategy.py` | `walk_forward_logistic_positions()` |
| `plot_results.py` | Equity vs B&H + drawdowns; **price** + **P(up)** with threshold |
| `main.py` | CLI + metrics + `output/equity_drawdown.png`, `output/price_prob_position.png` |
| `requirements.txt` | Adds **`scikit-learn>=1.0.0`** |
| `__init__.py` | Package marker |

### How to run
From the **`code`** directory:

```bash
pip install -r ash_code/cursor/cursor_3/requirements.txt
python ash_code/cursor/cursor_3/main.py
```

Headless:

```bash
python ash_code/cursor/cursor_3/main.py --no-show
```

Examples:

```bash
python ash_code/cursor/cursor_3/main.py --train-window 252 --n-lags 5 --prob-threshold 0.5 --ticker CL=F
```

A test run completed successfully; plots are in `cursor_3/output/`. On the default USO sample the rule is **not** attractive (negative total return vs buy & hold), which is acceptable for a **simple ML baseline** you compare to SMA/RSI and later sentiment models.

---

**User**

Implement random forest inside CAMPBELL-B/code/ash_code/cursor/cursor_4.

---

**Cursor**

Implementing `cursor_4` as a walk-forward Random Forest classifier, mirroring `cursor_3`'s feature pipeline and backtest.



Creating `cursor_4` with `RandomForestClassifier`, matching `cursor_3`'s walk-forward pipeline and scaling.



Lowering default tree count so runs finish faster; full accuracy remains available via `--n-estimators`.



Here is what was added under [`code/ash_code/cursor/cursor_4/`](code/ash_code/cursor/cursor_4/).

### Strategy
- Same setup as **`cursor_3`**: walk-forward **binary** target — next-day return &gt; 0 — using **lagged returns** `ret.shift(0)…ret.shift(n_lags-1)` (known at the close of day `t`).
- **Model:** `sklearn.ensemble.RandomForestClassifier` (`class_weight="balanced"`, `n_jobs=-1`, `random_state` from config).
- **Preprocessing:** `StandardScaler` fit on each rolling train window only.
- **Trade:** long if `P(up) > prob_threshold` (default 0.5).
- **Defaults:** `n_estimators=100`, `max_depth=8` (use **`--max-depth 0`** for `max_depth=None`).

### Files
| File | Role |
|------|------|
| `config.py` | Ticker, `train_window`, `n_lags`, thresholds, RF defaults |
| `data_loader.py` | yfinance + futures bar repair (same as other cursors) |
| `backtest.py` | Same long-only engine |
| `strategy.py` | `_feature_matrix`, `walk_forward_random_forest_positions()` |
| `plot_results.py` | Equity vs B&H + **Random forest** label; price + P(up) |
| `main.py` | CLI: `--n-estimators`, `--max-depth` (0 → unlimited), `--random-state`, … |
| `requirements.txt` | `pandas`, `numpy`, `matplotlib`, `yfinance`, `scikit-learn` |
| `__init__.py` | Package marker |

### Run
From the **`code`** directory:

```bash
pip install -r ash_code/cursor/cursor_4/requirements.txt
python ash_code/cursor/cursor_4/main.py
```

```bash
python ash_code/cursor/cursor_4/main.py --no-show
python ash_code/cursor/cursor_4/main.py --n-estimators 200 --max-depth 8 --ticker USO
```

**Note:** A full history with daily refits is **slow** (many minutes with 200 trees). Defaults use **100 trees** to shorten runs; raise `--n-estimators` when you need a heavier model.

A test run finished successfully and wrote `output/equity_drawdown.png` and `output/price_prob_position.png`.

---

**User**

from the results from all the strategies generated till now. which one do you think did well ? 

---

**Cursor**

Using the **runs we actually got** (mostly **USO**, **daily**, **from ~2015**, same long-only + lag + commission framing), the ranking is pretty clear.

### By the printed metrics

| Strategy | Approx. total return (vs B&H) | Notes |
|----------|----------------------------------|--------|
| **SMA crossover (`cursor_1`)** | **Strong** — strategy **well ahead** of buy & hold on USO in the run we saw (high total return, **best Sharpe** among the rules, **smaller drawdown** than B&H in that run). | **Best overall** in this set. |
| **Random forest (`cursor_4`)** | **Moderate** — **positive** total return while B&H was **negative** in that sample; Sharpe **positive** but **below** the SMA run. | Second-best here. |
| **RSI mean reversion (`cursor_2`)** | **Weak** — large **negative** total return vs B&H. | Poor on this sample. |
| **Logistic regression (`cursor_3`)** | **Weakest** — **very negative** return and **negative** Sharpe in the run. | Worst of the four. |

So **the moving-average crossover baseline did best** on these results: higher return and better risk-adjusted stats than the others in the same environment.

### Caveats (important)

- This is **one symbol / one window / one implementation**; another ticker (e.g. `CL=F`) or date range can change the order.
- **ML strategies** (`cursor_3`/`cursor_4`) can look better or worse with different features, thresholds, train window, or regularization — we didn’t do a full hyperparameter search.
- **Past backtests don’t guarantee** future performance.

**Bottom line:** Among the strategies generated so far, **SMA (`cursor_1`) looks like the one that “did well”** by a clear margin; **random forest** is the only other one that still looked **plausibly competitive** versus buy & hold in the USO run we saw.

---

**User**

think of a strategy that could be implemented next for better results than achieved till now. 

---

**Cursor**

Here are practical **next strategy directions** that are more likely to **beat a simple SMA on oil** than “more of the same” lag-only ML.

### 1. **Regime-conditioned rules (often the biggest easy win)**  
Oil spends long stretches **trending** and others **choppy**. Your **RSI** rule loses in trends; your **SMA** can lose in chop.  

**Idea:** Classify **regime** (e.g. high vs low **realized vol**, or **ADX**/trend strength), then:  
- in **trending** regimes → use **trend** rule (SMA or breakout),  
- in **mean-reverting** regimes → use **MR** (RSI/z-score) or stay flat.  

**Why it can beat SMA-only:** You stop applying the wrong rule in the wrong phase.  
**Effort:** Medium — mostly logic + one extra indicator series, same backtest shell.

---

### 2. **Meta-labeling / “filter the SMA” (strong ML use case)**  
Keep the **SMA** as the **primary** signal, train a **classifier** only to answer: *“When SMA says long, should we actually take the trade?”*  

**Features:** lags, vol, **drawdown from peak**, **time since last signal**, maybe **volume** if you trust it.  
**Label:** e.g. “was the **next 5-day** return positive?” or “did trade beat **costs**?”  

**Why it can beat raw ML on lags:** The model **specializes** on when the simple rule works, instead of learning direction from scratch.  
**Effort:** Medium — new labels + walk-forward, but reuses your `cursor_3`/`cursor_4` patterns.

---

### 3. **Use what you already have: news / sentiment (your original angle)**  
You already invested in **news sentiment**. A natural “next strategy” is **not** another price-only baseline, but **combine**:  
- e.g. **long only when** (SMA bullish **and** short-horizon sentiment &gt; 0), or  
- **size** or **gate** entries by sentiment **z-score**.  

**Why it can beat price-only rules:** Adds **orthogonal** information.  
**Effort:** Higher (alignment, leakage, data quality), but aligned with the project story.

---

### 4. **Volatility targeting on top of a good directional signal**  
Once direction is reasonable, **scale** exposure so **annual vol** is near a target (cap at 100% long).  

**Why it can improve metrics:** Often **raises Sharpe** and **smooths** drawdowns vs fixed 0/1 size.  
**Effort:** Medium — extend the backtest to **fractional** weights.

---

### Recommendation (single best “next” to try first)

If the goal is **better results than you’ve seen so far** with **reasonable scope**: implement **regime switching (1)** or **meta-labeling on the SMA (2)**.  

- **(1)** if you want **interpretable** rules and a clear supervisor story.  
- **(2)** if you want a **clear ML upgrade** that’s harder to dismiss as “random curve fit” than raw lag-only classification.

If your priority is **maximum edge relative to the rest of the project**, **(3) sentiment + SMA** is the most differentiated from what you’ve already backtested.

---

**User**

Implement regime switching (1) inside CAMPBELL-B/code/ash_code/cursor/cursor_5 and meta-labeling on the SMA (2) inside CAMPBELL-B/code/ash_code/cursor/cursor_6.

---

**Cursor**

Implementing `cursor_5` (ADX regime: SMA vs RSI) and `cursor_6` (SMA gated by walk-forward logistic meta-labeling). Checking existing strategy helpers and OHLC availability.



Implementing both packages: adding `cursor_5` (ADX regime + SMA/RSI) and `cursor_6` (SMA + walk-forward meta logistic).



Implementing `cursor_6` meta-labeling.



Here’s what was added and how it behaves.

### `cursor_5` — Regime switching (ADX)

- **Logic:** Compute **ADX** from **High/Low/Close** (Wilder-style smoothing).  
  - **Trend** if `ADX >= threshold` (default **25**): **SMA crossover** (long when fast &gt; slow).  
  - **Chop** if `ADX < threshold`: **RSI hysteresis** (same idea as `cursor_2`: enter ≤ oversold, exit ≥ overbought).
- **Outputs:** `output/equity_drawdown.png`, `output/price_regime_adx.png` (price with **orange = trend regime**, **green = executed long**, ADX + threshold).
- **Run:**  
  `python ash_code/cursor/cursor_5/main.py`  
  Flags: `--fast`, `--slow`, `--rsi-window`, `--oversold`, `--overbought`, `--adx-period`, `--adx-threshold`, `--ticker`, `--no-show`, etc.

**Smoke test (USO):** completed; metrics printed.

---

### `cursor_6` — Meta-labeling on SMA

- **Primary:** **SMA long** (fast &gt; slow), same as `cursor_1`.
- **Meta:** From index `train_window` onward, **only when primary is long**, fit a **walk-forward logistic regression** on lagged returns **using only past days where SMA was long** in the rolling window. Target = **next-day return &gt; 0**. Long if `P(up) > prob_threshold` (default **0.5**); if primary is flat → **flat**.
- **Warmup:** For `k < train_window`, position follows **primary SMA only** (no meta yet).
- **Fallback:** If meta can’t train (too few samples, one class, etc.) → **stay long** on primary (`1`).
- **Outputs:** `output/equity_drawdown.png`, `output/price_meta_prob.png` (price + SMAs, **P(up)** with threshold).
- **Run:**  
  `python ash_code/cursor/cursor_6/main.py`  
  Requires **`scikit-learn`** (see `requirements.txt`).

**Bugfix:** `np.isnan` on the feature matrix was corrected to `np.isnan(X_np[train_idx]).any(axis=1)`.

---

### Shared layout
Each folder has `config.py`, `data_loader.py`, `backtest.py`, `strategy.py`, `plot_results.py`, `main.py`, `requirements.txt`, `__init__.py` (same long-only + lag + commission pattern as your other cursors).

From the **`code`** directory:

```bash
pip install -r ash_code/cursor/cursor_5/requirements.txt
pip install -r ash_code/cursor/cursor_6/requirements.txt
python ash_code/cursor/cursor_5/main.py --no-show
python ash_code/cursor/cursor_6/main.py --no-show
```

