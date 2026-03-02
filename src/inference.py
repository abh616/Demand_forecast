import pandas as pd
import joblib
from datetime import datetime
import os
import numpy as np
import holidays

# ---------------- PATHS ----------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "xgboost_v3.pkl")
FEATURE_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "feature_columns_v3.pkl")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "clean_sales_data.csv")

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)

history_df = pd.read_csv(DATA_PATH)
history_df["Date"] = pd.to_datetime(history_df["Date"])

# ---------------- HOLIDAY SETUP ----------------
us_holidays = holidays.US()
uk_holidays = holidays.UK()
ca_holidays = holidays.CA()

# ---------------- TEMPERATURE MAP ----------------
seasonal_temp = {
    "USA": {1: 2, 2: 5, 3: 10, 4: 16, 5: 22, 6: 27, 7: 30, 8: 29, 9: 24, 10: 17, 11: 10, 12: 4},
    "Canada": {1: -5, 2: -3, 3: 2, 4: 10, 5: 18, 6: 23, 7: 26, 8: 25, 9: 19, 10: 10, 11: 3, 12: -2},
    "UK": {1: 4, 2: 5, 3: 8, 4: 12, 5: 16, 6: 19, 7: 21, 8: 20, 9: 17, 10: 13, 11: 8, 12: 5}
}

def predict(request):

    product = request["product_name"]
    location = request["location"]
    platform = request["platform"]
    category = request["category"]

    product_history = history_df[
        (history_df["Product Name"] == product) &
        (history_df["Location"] == location) &
        (history_df["Platform"] == platform)
    ].sort_values("Date")

    if product_history.empty:
        raise ValueError("No historical data for selected combination.")

    if len(product_history) < 8:
        raise ValueError("Not enough historical data (need at least 8 records).")

    # ---------------- LAG FEATURES ----------------
    lag_1 = product_history["Units Sold"].iloc[-1]
    lag_2 = product_history["Units Sold"].iloc[-2]
    lag_3 = product_history["Units Sold"].iloc[-3]
    lag_4 = product_history["Units Sold"].iloc[-4]
    lag_6 = product_history["Units Sold"].iloc[-6]
    lag_8 = product_history["Units Sold"].iloc[-8]

    # ---------------- ROLLING FEATURES ----------------
    rolling_mean_4 = product_history["Units Sold"].tail(4).mean()
    rolling_std_4 = product_history["Units Sold"].tail(4).std()

    rolling_mean_8 = product_history["Units Sold"].tail(8).mean()
    rolling_std_8 = product_history["Units Sold"].tail(8).std()

    # ---------------- PRICE FEATURES ----------------
    latest_price = product_history["Price"].iloc[-1]
    price_lag_1 = product_history["Price"].iloc[-2]
    price_change = (latest_price - price_lag_1) / price_lag_1 if price_lag_1 != 0 else 0

    # ---------------- TIME FEATURES ----------------
    now = datetime.now()
    year = now.year
    month = now.month
    week = now.isocalendar().week

    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    # ---------------- HOLIDAY FEATURE ----------------
    today = now.date()

    if location == "USA":
        is_holiday = int(today in us_holidays)
    elif location == "UK":
        is_holiday = int(today in uk_holidays)
    elif location == "Canada":
        is_holiday = int(today in ca_holidays)
    else:
        is_holiday = 0

    # ---------------- TEMPERATURE ----------------
    temperature = seasonal_temp[location][month]

    # ---------------- FUEL INDEX ----------------
    fuel_index = history_df["Date"].rank(pct=True).iloc[-1]

    # ---------------- BUILD INPUT ----------------
    input_dict = {
        "lag_1": lag_1,
        "lag_2": lag_2,
        "lag_3": lag_3,
        "lag_4": lag_4,
        "lag_6": lag_6,
        "lag_8": lag_8,
        "rolling_mean_4": rolling_mean_4,
        "rolling_std_4": rolling_std_4,
        "rolling_mean_8": rolling_mean_8,
        "rolling_std_8": rolling_std_8,
        "Price": latest_price,
        "price_lag_1": price_lag_1,
        "price_change": price_change,
        "year": year,
        "month": month,
        "week": week,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "IsHoliday": is_holiday,
        "Temperature": temperature,
        "Fuel_Index": fuel_index
    }

    input_df = pd.DataFrame([input_dict])

    # ---------------- HANDLE ENCODING ----------------
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df[f"Product Name_{product}"] = 1
    input_df[f"Location_{location}"] = 1
    input_df[f"Platform_{platform}"] = 1
    input_df[f"Category_{category}"] = 1

    input_df = input_df[feature_columns]

    prediction = model.predict(input_df)[0]

    return int(round(prediction))