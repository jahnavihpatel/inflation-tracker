# US Inflation Tracker 📈

A data science capstone project tracking 70 consumer items using BLS CPI data and Facebook Prophet forecasting.

## Tech Stack
- **Data**: BLS (Bureau of Labor Statistics) public API
- **Forecasting**: Facebook Prophet (time-series ML)
- **Dashboard**: Streamlit + Plotly
- **Deployment**: Streamlit Community Cloud (free)

## Skills Demonstrated
- Time series analysis & decomposition (trend + seasonality)
- ML forecasting with uncertainty quantification (Prophet)
- REST API data ingestion (BLS API)
- Interactive dashboard development (Streamlit + Plotly)
- Model evaluation (MAE, RMSE, MAPE on holdout set)
- End-to-end data pipeline (fetch → model → visualize → deploy)

---

## Local Setup

### 1. Clone & install
```bash
git clone https://github.com/YOUR_USERNAME/inflation-tracker.git
cd inflation-tracker
pip install -r requirements.txt
```

### 2. Fetch real CPI data from BLS
```bash
python data_pipeline.py
```
This saves `data/prices.csv` with monthly prices for 70 items (2015–2025).

### 3. Run the Prophet forecast model
```bash
python forecast_model.py
```
This saves `data/forecasts.csv` and `data/model_metrics.csv`.

### 4. Launch the dashboard
```bash
streamlit run app.py
```
Opens at http://localhost:8501

---

## Deploy to Streamlit Community Cloud (free)

1. Push your project to a **public GitHub repo**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set `app.py` as the main file
4. Click **Deploy**

> **Note**: The BLS data files (`data/prices.csv`, `data/forecasts.csv`) must be committed to your repo, OR you can add a `startup.py` that fetches data on first run.

### Committing data files (recommended for capstone)
```bash
git add data/prices.csv data/forecasts.csv data/model_metrics.csv
git commit -m "add pre-fetched CPI data and forecasts"
git push
```

---

## Project Structure
```
inflation-tracker/
├── app.py               # Streamlit dashboard
├── data_pipeline.py     # BLS API data fetching
├── forecast_model.py    # Prophet forecasting model
├── requirements.txt     # Python dependencies
├── .streamlit/
│   └── config.toml      # Dark theme config
├── data/
│   ├── prices.csv       # Raw CPI prices (generated)
│   ├── forecasts.csv    # Prophet forecasts (generated)
│   └── model_metrics.csv# MAE, RMSE, MAPE per item (generated)
└── README.md
```

---

## Data Source
All price data sourced from the **Bureau of Labor Statistics (BLS) Consumer Price Index (CPI-U)**.
- Website: https://www.bls.gov/data/
- API docs: https://www.bls.gov/developers/api_signature_v2.htm
- No API key required for basic access (≤25 series per request)

## Model Details
Forecasting uses **Facebook Prophet** with:
- Multiplicative seasonality mode (better for prices that trend upward)
- Yearly seasonality enabled
- 95% confidence intervals
- Holdout evaluation on last 12 months (MAE, RMSE, MAPE reported)
- Changepoint prior scale: 0.05 (regularized, avoids overfitting)
