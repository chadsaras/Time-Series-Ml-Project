# Core ML Time-series Project

This repository consolidates two complementary time-series forecasting pipelines:

1. **Stock Price Forecasting**: An end-to-end Jupyter notebook (`AppleStockAnalysis.ipynb`) that retrieves historical Apple (AAPL) stock data, performs exploratory analysis, and trains an LSTM/GRU model to predict next-day closing prices.
2. **Cryptocurrency Price Comparison**: A second notebook (`BitcoinPredictionRNN.ipynb`) that ingests multi-source Bitcoin (BTC) price data, engineers temporal and market features, and compares three forecasting approaches (Random Forest, LSTM/GRU, ARMA) under a unified evaluation framework.

---

## Project Structure

```
├── AppleStockAnalysis.ipynb     # Stock data download, EDA, LSTM/GRU forecasting
├── BitcoinPredictionRNN.ipynb    # Bitcoin data ingestion, feature engineering, RF/LSTM/ARMA models
└── README.md                     # This overview
```

---

## Requirements

* Python 3.8+
* `yfinance`, `numpy`, `pandas`, `matplotlib`, `seaborn`
* `tensorflow` or `keras` for RNNs
* `scikit-learn` for Random Forest and scaling
* `statsmodels` for ARMA modeling
* (Optional) `fastai`, `pytorch-nightly` if using GPU-enabled variants

Install all dependencies via:

```bash
pip install -r requirements.txt
```

---

## Usage

1. **Apple Stock Forecast**

   * Open `AppleStockAnalysis.ipynb` in JupyterLab/Notebook.
   * Run all cells to fetch data, visualize price/dividend history, and train/evaluate the LSTM/GRU model.
   * Outputs include RMSE metrics and a combined DataFrame of actual vs. predicted prices.

2. **Bitcoin Price Comparison**

   * Open `BitcoinPredictionRNN.ipynb`.
   * The notebook downloads Bitcoin CSVs, performs EDA, and engineers calendar/market features for a Random Forest baseline.
   * It then frames sequences for a stacked LSTM/GRU and fits an ARMA model.
   * Compare test-set RMSE for all three models and visualize performance over key market events.

---

## Key Results

* **Apple LSTM/GRU**: Achieved a test RMSE of approximately 6.506423928549112, demonstrating non-linear pattern learning over a 3-year window.
* **Bitcoin Models**: Random Forest baseline RMSE: 48.39190001714336; LSTM/GRU RMSE: 586.1710704755135; ARMA RMSE: 405.2444219277155 over the same test period.

---

## Contributing

Contributions and enhancements (e.g., feature expansion, hyperparameter tuning, cross-validation modules) are welcome via pull requests.

---
