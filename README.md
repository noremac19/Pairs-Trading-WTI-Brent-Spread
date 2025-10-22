# WTI–Brent Pairs Trading Strategy

This repository implements a **pairs trading** strategy between **WTI** and **Brent** crude oil futures using spread mean reversion with **regime switching**. The pipeline builds a z‑scored spread, classifies market regimes with a **Gaussian Hidden Markov Model (HMM)**, and generates long/short signals with regime‑dependent thresholds. It then backtests PnL and reports performance (e.g., **Sharpe ratio**).

> **Files used by the script**
> - `finalproject.py` – main script
> - `WTIData.csv`, `BrentData.csv` – input price data (PX_LAST and active ticker columns expected)

---

## ## Project Overview

- **Goal**: Exploit mean reversion in the **WTI–Brent spread** while adapting to changing market conditions via **regimes** (high vs. low dependency).
- **Approach**:  
  1) Build spread = WTI − Brent,  
  2) Standardize to z‑scores,  
  3) Fit a 2‑state **GaussianHMM** on the spread z‑score,  
  4) Trade when the spread deviates from zero by regime‑dependent thresholds,  
  5) Backtest and compute performance metrics.

---

## ## Data Requirements

The script expects two CSVs in the working directory:

| File          | Required Columns                              | Notes                                   |
|---------------|-----------------------------------------------|-----------------------------------------|
| `WTIData.csv` | `Date`, `PX_LAST`, `FUT_CUR_GEN_TICKER`       | Daily WTI front (or active) contract    |
| `BrentData.csv` | `Date`, `PX_LAST`, `FUT_CUR_GEN_TICKER`     | Daily Brent front (or active) contract  |

- Dates must be parseable (e.g., `YYYY-MM-DD`).  
- The script **forward‑fills** missing values and aligns by date, then filters to dates **≥ 2021‑01‑01**.

---

## ## Methodology

### ### 1) Preprocessing
- Read WTI & Brent CSVs, select columns, convert `Date` to datetime.
- Forward‑fill prices, align indexes, and compute the **spread**:  
  \[`Spread` = `WTI` − `Brent`\]
- Optionally run stationarity checks (ADF) on returns / spread (import is present).

### ### 2) Linear Relationship (OLS)
- Fit `WTI ~ α + β × Brent` via **OLS** to understand the co‑movement.
- Plot price series and the raw spread over time for visual inspection.

### ### 3) Returns & Standardization
- Compute log returns (×100) for each leg and the **return spread**.  
- Standardize the spread to a **z‑score**:  
  \[`Spread_z` = zscore(`Spread`)\]

### ### 4) Regime Switching (GaussianHMM)
- Train a **2‑state** `GaussianHMM` on `Spread_z` (diag covariance).  
- Identify **High** and **Low** dependency regimes by comparing mean `Spread_z` per state.  
- Add a `Regime` column: `"High"` or `"Low"`.

### ### 5) Signal Generation
Regime‑dependent thresholds (from the script):

- **High dependency**:  
  - `Spread_z` > **+2** ⇒ **Sell Spread** (short WTI / long Brent)  
  - `Spread_z` < **−2** ⇒ **Buy Spread** (long WTI / short Brent)
- **Low dependency**:  
  - `Spread_z` > **+1** ⇒ **Sell Spread**  
  - `Spread_z` < **−1** ⇒ **Buy Spread**  
- Otherwise ⇒ **Hold**.

### ### 6) Backtesting Logic (Simplified)
- Iterate chronologically; open/close positions based on `Signal` transitions.  
- Includes hooks for **position sizing** (`calculate_contracts(...)`) and **transaction costs** (per‑leg constants).  
- Computes daily **PnL**, cumulative **Portfolio_Value**, **Daily_Return**, and **Sharpe Ratio** (with a default annual **risk‑free** of 3%).  
- Generates plots for prices, spread, and `Spread_z` time series.

> **Note:** The provided script includes clearly marked placeholders (`...`) for parts that you may want to complete or customize (e.g., exact PnL math, contract sizing function, and explicit transaction cost accounting).

---

## ## Installation

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# Install core dependencies
pip install numpy pandas matplotlib scipy statsmodels hmmlearn arch copulas
```

> If you encounter platform‑specific wheels for `arch`, install with:  
> `pip install arch==6.*` (or a compatible version for your OS/Python)

---

## ## How to Run

1. Place `WTIData.csv` and `BrentData.csv` in the repository root (or update paths in the script).  
2. Run the script:
   ```bash
   python finalproject.py
   ```
3. View generated plots and console output (e.g., **Sharpe ratio**).

---

## ## Expected Columns (after processing)

The script produces these key columns in the merged `data` DataFrame:

- `WTI`, `Brent` – aligned price series (FFill where needed)  
- `Spread` – price spread (`WTI − Brent`)  
- `Log_Returns_WTI`, `Log_Returns_Brent`, `Returns_Spread` – daily return series  
- `Spread_z` – z‑score of `Spread`  
- `Regime` – `"High"` or `"Low"` dependency regime from HMM  
- `Signal` – one of `{Buy Spread, Sell Spread, Hold}`  
- `PnL`, `Portfolio_Value`, `Daily_Return` – backtest outputs (ensure placeholders are completed)  

---

## ## Plots

- **WTI vs. Brent Price** time series  
- **WTI–Brent Spread** time series  
- **Spread z‑Score** time series (for visualizing signals / thresholds)

---

## ## Notes & To‑Dos

- Some parts of the backtest are **stubbed** with placeholders (`...`) in `finalproject.py`.  
  - Implement `calculate_contracts(price, cash)` to size positions.  
  - Confirm **contract size** (e.g., 1,000 barrels per contract) and **tick value** for WTI/Brent.  
  - Finish trade entry/exit PnL accounting (including transaction costs/slippage).  
- Libraries imported but not fully used in the current script version (e.g., `arch`, `copulas`):  
  - You can extend the model with **GARCH volatility** or **copula‑based dependence** if desired.  

---

## ## Disclaimer

This code is for **research and educational purposes** only and **not** investment advice. Futures trading involves significant risk of loss.

---

## ## License

**MIT License** — free to use, modify, and distribute with attribution.
