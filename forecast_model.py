"""
Inflation Tracker — Forecasting Model (ARIMA via statsmodels)
No pystan / Prophet dependency — works on all platforms.
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

os.makedirs("data", exist_ok=True)


def load_data(path: str = "data/prices.csv") -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def compute_metrics(actual, predicted) -> dict:
    actual, predicted = np.array(actual), np.array(predicted)
    mae  = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-9))) * 100
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE_%": round(mape, 2)}


def forecast_item(df_item: pd.DataFrame, forecast_months: int = 12):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    df_item = df_item.sort_values("date").copy()
    series  = df_item.set_index("date")["value"].asfreq("MS")
    series  = series.interpolate()

    metrics = {"MAE": None, "RMSE": None, "MAPE_%": None}
    if len(series) > 24:
        train, test = series.iloc[:-12], series.iloc[-12:]
        try:
            m = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,0,12),
                        enforce_stationarity=False, enforce_invertibility=False)
            fit = m.fit(disp=False)
            preds = fit.forecast(12)
            metrics = compute_metrics(test.values, preds.values)
        except Exception:
            pass

    m2   = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,0,12),
                   enforce_stationarity=False, enforce_invertibility=False)
    fit2 = m2.fit(disp=False)
    fc   = fit2.get_forecast(forecast_months)
    fc_mean = fc.predicted_mean
    ci      = fc.conf_int(alpha=0.05)
    fitted  = fit2.fittedvalues

    hist_df = pd.DataFrame({
        "ds": series.index, "actual": series.values, "yhat": fitted.values,
        "yhat_lower": np.nan, "yhat_upper": np.nan,
        "trend": fitted.values, "yearly": series.values - fitted.values,
        "is_forecast": False,
    })
    fc_df = pd.DataFrame({
        "ds": fc_mean.index, "actual": np.nan, "yhat": fc_mean.values,
        "yhat_lower": ci.iloc[:, 0].values, "yhat_upper": ci.iloc[:, 1].values,
        "trend": fc_mean.values, "yearly": 0.0, "is_forecast": True,
    })
    return pd.concat([hist_df, fc_df], ignore_index=True), metrics


def run(input_path="data/prices.csv", output_path="data/forecasts.csv"):
    df    = load_data(input_path)
    items = df["item"].unique()
    print(f"Forecasting {len(items)} items with ARIMA...")

    all_forecasts, all_metrics = [], []
    for item in items:
        print(f"  → {item}")
        df_item = df[df["item"] == item].copy()
        meta    = df_item[["item", "category", "unit"]].iloc[0]
        try:
            fc, metrics = forecast_item(df_item)
            fc["item"] = meta["item"]; fc["category"] = meta["category"]; fc["unit"] = meta["unit"]
            all_forecasts.append(fc)
            metrics["item"] = item
            all_metrics.append(metrics)
        except Exception as e:
            print(f"    FAILED: {e}")

    if not all_forecasts:
        print("No forecasts generated."); return

    pd.concat(all_forecasts, ignore_index=True).to_csv(output_path, index=False)
    print(f"\nSaved forecasts → {output_path}")
    metrics_df = pd.DataFrame(all_metrics).set_index("item")
    metrics_df.to_csv("data/model_metrics.csv")
    print(metrics_df.to_string())


if __name__ == "__main__":
    run()
