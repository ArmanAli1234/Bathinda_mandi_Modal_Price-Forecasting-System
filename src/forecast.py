import pandas as pd
import numpy as np
from data_cleaning import clean_df
from feature_engineering import create_feature
from data_preprocessing import preprocess_data
from sklearn.pipeline import Pipeline
import joblib
import os
import warnings
warnings.filterwarnings("ignore")


# load the data
PATH="./data/proceed/clean_data.csv"
df=clean_df(PATH)
make_feature=create_feature(df)

X_train,y_train,X_test,y_test=preprocess_data(make_feature)

# load the final pipeline
model=joblib.load("models/model.pkl")

# make a function for get season

def get_season(date):
    month=date.month

    if month in [12,1,2]:
        return "Winter"
    elif month in [3,4,5]:
        return "Summer"
    elif month in [6,7,8,9]:
        return "Monsoon"
    else:
        return "Autumn"


# forecasting 

def forecast_data(
    df,
    model,
    commodity,
    market,
    start_date,
    end_date,
    last_data_date,
    target="Modal_Price"
):

    # filter the data
    data = df[
        (df["Commodity"] == commodity) &
        (df["Market"] == market)
    ].copy()

    # change datatype
    data["Arrival_Date"] = pd.to_datetime(
        data["Arrival_Date"], errors="coerce"
    )

    data = data.sort_values("Arrival_Date", ascending=True)

    if len(data) < 30:
        raise ValueError("At least 30 data records required!")

    # historical prices
    history = data[target].astype(float).tolist()

    # data validation
    last_data_date = pd.Timestamp(last_data_date)
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    if start_date <= last_data_date:
        raise ValueError("Start date must be after last_data_date")

    if end_date < start_date:
        raise ValueError("End date must be after the start_date. fix it!")

    # future dates
    all_dates = pd.date_range(
        start=last_data_date + pd.Timedelta(days=1),
        end=end_date,
        freq="D"
    )

    last_row = data.iloc[-1].copy()
    result = []

    for future_date in all_dates:

        row = last_row.copy()

        # calendar features
        row["Year"] = future_date.year
        row["Month"] = future_date.month
        row["Day_of_week"] = future_date.dayofweek

        # season feature
        row["Season"] = get_season(future_date)

        # lag features
        row["lag1"] = history[-1]
        row["lag7"] = history[-7]
        row["lag_30"] = history[-30]

        # rolling features
        row["rolling_mean7"] = np.mean(history[-7:])
        row["rolling_mean30"] = np.mean(history[-30:])
        row["rolling_std7"] = np.std(history[-7:])

        # data features
        features = [
            "Commodity", "Grade", "Market", "Variety",
            "Year", "Month", "Day_of_week", "Season",
            "lag1", "lag7", "lag_30", "rolling_mean7",
            "rolling_mean30", "rolling_std7"
        ]

        X_future = pd.DataFrame([row[features].to_dict()])

        # prediction
        pred = float(model.predict(X_future)[0])

        # updated history
        history.append(pred)

        if future_date >= start_date:
            result.append({
                "forecast_date": future_date,
                "Commodity": commodity,
                "market": market,
                "Season": row["Season"],
                "pred_price": pred
            })

    return pd.DataFrame(result)


forecast = forecast_data(
    df=make_feature,
    model=model,
    commodity="Apple",
    market="Bathinda",
    start_date="2026-11-08",
    end_date="2026-11-22",
    last_data_date="2026-11-07"
)

print(forecast)