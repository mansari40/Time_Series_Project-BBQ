# BBQ Sauce Sales Forecasting: A Deep Learning Time Series Analysis (2020–2025)

Forecasting daily BBQ sauce demand using time-series EDA, decomposition, and an N-BEATS deep learning model. The project combines sales, climate, and pricing data to understand demand drivers and generate short-horizon business-day forecasts.

---

## Contents
- [Project Overview](#project-overview)
- [Data](#data)
- [Methodology](#methodology)
- [Results (High-Level)](#results-high-level)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [How to Run](#how-to-run)
- [Reproducibility Notes](#reproducibility-notes)
- [Limitations](#limitations)


---

## Project Overview

This notebook-based project analyzes daily BBQ sauce sales (2020–2025) to:
1. Identify trend and seasonality patterns (daily/weekly/monthly).
2. Check whether external covariates (weather, prices) explain demand.
3. Build a robust forecasting pipeline for operational planning.

The forecasting workflow emphasizes realistic evaluation using a held-out test period and walk-forward (rolling) backtesting.

---

## Data

The analysis uses **three CSV files**:

1. **Sales data** — `barbecue_sauce_sales.csv`  
   - Key fields (expected): `date`, `amount`  
   - Notes: the series reflects a **weekday-only operation** (weekends are not observed / structurally absent).

2. **Climate data** — `climate_data.csv`  
   - Key fields (expected): `date`, `temp`, `precip`, `sun_hours`  
   - Used to test short-term weather influence.

3. **Price data** — `bbq_sauce_prices_daily.csv`  
   - Key fields (expected): `date`, `avg_price` (and possibly `min_price`, `max_price`)  
   - Used to explore price dynamics and potential elasticity.

> **Important:** The notebook currently loads files via relative paths (e.g., `"barbecue_sauce_sales.csv"`). Place the CSVs in the same folder as the notebook, or adjust paths to a `data/` directory.

---

## Methodology

### 1) Data preparation & quality checks
- Parse dates, coerce numeric values, and validate data types.
- Identify missing business days and structural gaps.
- Aggregate to weekly/monthly levels to reveal patterns and reduce noise.

### 2) Exploratory analysis
- Distribution + outlier analysis (spikes and volatility).
- Day-of-week and year-over-year comparisons.
- Time-series decomposition (trend / seasonality / residual).
- Stationarity testing (ADF) where relevant.

### 3) Covariate analysis
- Correlation checks (Pearson/Spearman) between sales and:
  - temperature, precipitation, sunshine hours
  - average price and derived revenue measures
- Visual diagnostics to validate (or refute) expected relationships.

### 4) Forecasting with N-BEATS (Darts)
- Convert target series and covariates into `TimeSeries` objects.
- Engineer calendar covariates:
  - day-of-week and day-of-year encodings (including sin/cos seasonality features)
  - weekday indicators
- Train/validation/test split for honest evaluation.
- Hyperparameter search (random sampling over configurations).
- Walk-forward backtesting for operational realism.

---

## Results (High-Level)

The notebook reports:
- Weak seasonal structure (notably spring/summer peaks) and growth phases across years.
- Weekday-only demand behavior (weekends are structural zeros / absent in the data).
- Weather and price effects appear limited compared to calendar/seasonality signals.
- The tuned N-BEATS model captures trend + seasonality and provides stable short-horizon forecasts; errors increase on exceptional spike days (inherently hard to predict).

> For exact metrics and plots, run the notebook end-to-end. The final sections include test-set evaluation and walk-forward backtesting summaries.

---

## Repository Structure
```
.
├── Time series Project.ipynb
├── data/
│   ├── barbecue_sauce_sales.csv
│   ├── climate_data.csv
│   └── bbq_sauce_prices_daily.csv
└── README.md
```

If you keep CSVs in `data/`, update the notebook paths accordingly.

---

## Setup

### Option A — pip + venv (recommended)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### Option B — install directly (quick)
```bash
pip install pandas numpy matplotlib seaborn plotly scipy statsmodels scikit-learn
pip install torch pytorch-lightning
pip install "u8darts[torch]"
pip install jupyter ipykernel
```

**Note:** `u8darts[torch]` provides the darts module used in the notebook.

### Suggested requirements.txt
```
pandas
numpy
matplotlib
seaborn
plotly
scipy
statsmodels
scikit-learn
torch
pytorch-lightning
u8darts[torch]
jupyter
ipykernel
```

---

## How to Run

1. Place the CSV files either:
   - in the same directory as `Time series Project.ipynb`, or
   - inside `data/` and update file paths in the notebook.

2. Start Jupyter:
```bash
   jupyter lab
```

3. Open `Time series Project.ipynb` and run all cells top-to-bottom.

---

## Reproducibility Notes

The notebook uses a fixed random seed for parts of the modeling workflow.

Results can still vary slightly across machines due to:
- library versions
- CPU vs GPU execution
- nondeterministic deep learning ops (depending on configuration)

For fully reproducible runs, pin dependency versions and document your Python environment.

---

## Limitations

- Demand spikes (rare, high-magnitude days) are difficult for any model to predict reliably without external event signals (promotions, holidays, supply issues, campaigns).
- The dataset reflects a weekday-only process; forecasts are most meaningful for business-day planning.
- Covariate signals (weather, price) may be too weak, too aggregated, or lagged relative to purchasing behavior.

---

