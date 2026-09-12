# Core ML Time-Series Forecasting

Two end-to-end forecasting pipelines: a stacked-LSTM next-day price model for Apple stock, and a three-model comparison (Random Forest / LSTM / ARMA) on Bitcoin.

All splits are chronological (`shuffle=False`) — no random shuffling, no lookahead into the test window.

---

## Results

**Apple — `AppleStockAnalysis.ipynb`**

| Model | Data | Horizon | Test RMSE |
|---|---|---|---|
| 3-layer stacked LSTM (256 units) | 3y daily closes | next day | **6.51 USD** |

**Bitcoin — `BitcoinPredictionRNN.ipynb`**

| Model | Data | Horizon | Test RMSE |
|---|---|---|---|
| Random Forest (depth-tuned) | 1-min bars, 18 features | +24 min | 48.39 |
| ARMA (4,4)/(5,1) | daily bars | next day | 405.24 * |
| Stacked LSTM (256 units) | daily bars | next day | 586.17 |

> **These three Bitcoin numbers are not directly comparable** — see [Known issues](#known-issues). The Random Forest forecasts 24 minutes ahead on minute bars, while ARMA and the LSTM forecast a day ahead on daily bars. The RF's much lower error mostly reflects the easier horizon, not a better model.

\* Predates a fix to the rolling-forecast loop; needs a re-run.

---

## 1. Apple stock forecasting — `AppleStockAnalysis.ipynb`

Predicts next-day AAPL closing price from a 3-year window of daily closes.

* Ingest via `yfinance` (`Ticker("AAPL").history(period="3y")`); EDA over dividends, stock splits, and returns since inception.
* `MinMaxScaler` normalisation, then supervised framing with a **7-day lookback** window.
* **67/33 chronological split** — the test set is strictly the most recent third.
* Model: `LSTM(256) → LSTM(256) → LSTM(256) → Dense(1)`, Adam, MSE loss, 50 epochs, batch size 50, `shuffle=False`.
* Predictions are **inverse-transformed back to dollar prices** before scoring, so RMSE is in USD, not scaled units.

**Result:** test RMSE **6.51 USD**.

---

## 2. Bitcoin model comparison — `BitcoinPredictionRNN.ipynb`

**Data & features**
* Multi-source CSVs pulled from **S3** (Coinbase + Bitstamp exchange data, crypto news sets).
* Minute-level `Timestamp` → datetime index, forward-filled.
* `add_datepart` expands the date into calendar features, giving **18 predictors**: 7 market (`Open`, `High`, `Low`, `Close`, `Volume_(BTC)`, `Volume_(Currency)`, `Weighted_Price`) + 11 calendar (`Month`, `Week`, `Day`, `Dayofweek`, `Dayofyear`, `Is_month_end/start`, `Is_quarter_end/start`, `Is_year_end/start`).

**Models**

| | Setup | Split |
|---|---|---|
| Random Forest | `max_depth` swept 20→40, train/test error curves per depth; target `Close.shift(-24)` on minute bars | 90/10 chronological |
| Stacked LSTM | 3-day lookback, `LSTM(256) → LSTM(256) → Dense(1)`, MinMax-scaled, daily bars | 67/33 chronological |
| ARMA | Orders (4,4)/(5,1), rolling-origin loop appending each observation, daily bars | 76/24 chronological |

**Statistical checks**
* **ADF test** for stationarity on the daily series.
* **ACF plots** (48 lags) on the differenced series to motivate ARMA order selection.

**Event analysis:** predictions plotted across the Nov 2017 – Mar 2018 run-up and crash to inspect behaviour under regime change.

---

## Repo structure

```
├── AppleStockAnalysis.ipynb     # AAPL ingestion, EDA, stacked-LSTM forecasting
├── BitcoinPredictionRNN.ipynb   # BTC ingestion, feature engineering, RF/LSTM/ARMA models
├── requirements.txt             # Pinned deps (Python 3.8)
└── README.md
```

---

## Setup

The notebooks depend on two APIs later removed upstream (`fastai.structured.add_datepart` and `statsmodels.tsa.arima_model.ARMA`), so versions are pinned and **Python 3.8 is required**:

```bash
conda create -n tsml python=3.8 -y && conda activate tsml
pip install -r requirements.txt
jupyter lab
```

`AppleStockAnalysis.ipynb` runs top-to-bottom as-is — it pulls its own data from `yfinance`.

---

## Known issues

Recorded rather than quietly patched, since they affect how the numbers should be read.

1. **The Bitcoin comparison is not like-for-like.** The intended daily resample (`gold = coinbase.resample('D')…`) is commented out, so the Random Forest trains on raw 1-minute bars with `shift(-24)` — a 24-minute horizon, not the 2 days the inline comment claims. ARMA and the LSTM run on daily bars at a next-day horizon. Making this a genuine benchmark requires resampling to daily before the RF and re-running all three on one target.
2. **ARMA rolling forecast (fixed, needs re-run).** The loop built a fresh `ARMA(history, …)` per step but then called `mod.fit()` — a model fitted on the *full* series, test window included. That is lookahead leakage, and it made 405.24 optimistic. Now corrected to `model.fit()`; the reported figure is the pre-fix number.
3. **Bitcoin data files are not in the repo.** The notebook reads `Bitcoin2015Daily.csv` and `cryptonewscleanedallmagic.csv`, intermediate files not produced by the S3 `wget` cells, so the BTC notebook is not reproducible end-to-end without them.
4. **Low `n_estimators`.** The RF sweep uses `n_estimators=5` to keep the depth comparison fast; results would likely improve with a realistic forest size.
