from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))
else:
    # Also supports placing app.py in the project root with src on PYTHONPATH.
    sys.path.insert(0, str(BASE_DIR / "src"))

from src.data_cleaning import clean_df
from src.feature_engineering import create_feature

app = Flask(__name__)

DATA_PATH = BASE_DIR / "data" / "proceed" / "clean_data.csv"
MODEL_PATH = BASE_DIR / "models" / "model.pkl"

def load_resources():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    raw_df = clean_df(str(DATA_PATH))
    feature_df = create_feature(raw_df)
    feature_df["Arrival_Date"] = pd.to_datetime(
        feature_df["Arrival_Date"], errors="coerce"
    )
    feature_df = feature_df.dropna(subset=["Arrival_Date", "Modal_Price"])
    fitted_model = joblib.load(MODEL_PATH)
    return feature_df, fitted_model

df, model = load_resources()

def get_season(date):
    month = date.month
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Summer"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    return "Autumn"

def forecast_data(dataframe, fitted_model, commodity, market,
                  start_date, end_date, last_data_date, target="Modal_Price"):
    data = dataframe[
        (dataframe["Commodity"] == commodity) &
        (dataframe["Market"] == market)
    ].copy()

    data["Arrival_Date"] = pd.to_datetime(data["Arrival_Date"], errors="coerce")
    last_data_date = pd.Timestamp(last_data_date)
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    # Only use observations available on or before the chosen cutoff.
    data = data[data["Arrival_Date"] <= last_data_date]
    data = data.sort_values("Arrival_Date").dropna(subset=[target])

    if len(data) < 30:
        raise ValueError("Is Commodity aur Market ke liye cutoff date tak kam se kam 30 records chahiye.")
    if start_date <= last_data_date:
        raise ValueError("Start date, last data date ke baad honi chahiye.")
    if end_date < start_date:
        raise ValueError("End date, start date ke barabar ya uske baad honi chahiye.")

    history = data[target].astype(float).tolist()
    last_row = data.iloc[-1].copy()
    results = []

    all_dates = pd.date_range(
        start=last_data_date + pd.Timedelta(days=1),
        end=end_date,
        freq="D"
    )

    features = [
        "Commodity", "Grade", "Market", "Variety",
        "Year", "Month", "Day_of_week", "Season",
        "lag1", "lag7", "lag_30", "rolling_mean7",
        "rolling_mean30", "rolling_std7"
    ]

    for future_date in all_dates:
        row = last_row.copy()
        row["Year"] = future_date.year
        row["Month"] = future_date.month
        row["Day_of_week"] = future_date.dayofweek
        row["Season"] = get_season(future_date)

        row["lag1"] = history[-1]
        row["lag7"] = history[-7]
        row["lag_30"] = history[-30]
        row["rolling_mean7"] = float(np.mean(history[-7:]))
        row["rolling_mean30"] = float(np.mean(history[-30:]))
        row["rolling_std7"] = float(np.std(history[-7:]))

        X_future = pd.DataFrame([row[features].to_dict()])
        prediction = float(fitted_model.predict(X_future)[0])
        history.append(prediction)

        if future_date >= start_date:
            results.append({
                "forecast_date": future_date.strftime("%d %b %Y"),
                "commodity": commodity,
                "market": market,
                "season": row["Season"],
                "pred_price": round(prediction, 2)
            })

    return pd.DataFrame(results)

def available_values(column):
    return sorted(df[column].dropna().astype(str).unique().tolist())

@app.route("/", methods=["GET", "POST"])
def index():
    forecast_rows = None
    single_result = None
    error = None
    active_tab = request.values.get("tab", "forecast")

    commodities = available_values("Commodity")
    markets = available_values("Market")
    default_commodity = commodities[0] if commodities else ""
    default_market = markets[0] if markets else ""

    if request.method == "POST":
        try:
            action = request.form.get("action")

            if action == "forecast":
                commodity = request.form["commodity"]
                market = request.form["market"]
                last_date = request.form["last_data_date"]
                start = request.form["start_date"]
                end = request.form["end_date"]

                result = forecast_data(
                    df, model, commodity, market, start, end, last_date
                )
                forecast_rows = result.to_dict(orient="records")
                active_tab = "forecast"

            elif action == "single":
                commodity = request.form["single_commodity"]
                market = request.form["single_market"]
                prediction_date = pd.Timestamp(request.form["prediction_date"])

                subset = df[
                    (df["Commodity"] == commodity) &
                    (df["Market"] == market)
                ].copy()
                subset["Arrival_Date"] = pd.to_datetime(subset["Arrival_Date"], errors="coerce")
                subset = subset.dropna(subset=["Arrival_Date"])
                if subset.empty:
                    raise ValueError("Selected Commodity aur Market ka data nahi mila.")

                cutoff = subset["Arrival_Date"].max()
                if prediction_date <= cutoff:
                    raise ValueError(
                        f"Prediction date {cutoff.strftime('%d %b %Y')} ke baad honi chahiye. "
                        "Single prediction yahan future date ke liye hai."
                    )

                result = forecast_data(
                    df, model, commodity, market,
                    prediction_date, prediction_date, cutoff
                )
                if result.empty:
                    raise ValueError("Is date ke liye prediction generate nahi hui.")
                single_result = result.iloc[0].to_dict()
                active_tab = "single"

        except Exception as exc:
            error = str(exc)

    latest_date = df["Arrival_Date"].max()
    total_records = len(df)
    commodity_count = df["Commodity"].nunique()
    market_count = df["Market"].nunique()

    return render_template(
        "index.html",
        commodities=commodities,
        markets=markets,
        default_commodity=default_commodity,
        default_market=default_market,
        forecast_rows=forecast_rows,
        single_result=single_result,
        error=error,
        active_tab=active_tab,
        latest_date=latest_date.strftime("%d %b %Y") if pd.notna(latest_date) else "N/A",
        total_records=f"{total_records:,}",
        commodity_count=commodity_count,
        market_count=market_count
    )

if __name__ == "__main__":
    app.run(debug=True)
