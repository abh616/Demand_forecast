import pandas as pd
import joblib
from datetime import datetime

MODEL_PATH = "models/random_forest_v1.pkl"
FEATURE_PATH = "models/feature_columns.pkl"
DATA_PATH = "data/processed/clean_sales_data.csv"

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)

history_df = pd.read_csv(DATA_PATH)
history_df["Date"] = pd.to_datetime(history_df["Date"])


def predict(request):

    product = request["product_name"]
    location = request["location"]
    platform = request["platform"]
    category = request["category"]


    # Filter historical data
    product_history = history_df[
        (history_df["Product Name"] == product) &
        (history_df["Location"] == location) &
        (history_df["Platform"] == platform)
    ].sort_values("Date")

    if product_history.empty:
        raise ValueError("No historical data for selected combination.")


    # Compute lag features
    lag_1 = product_history["Units Sold"].iloc[-1]
    lag_2 = product_history["Units Sold"].iloc[-2]
    lag_4 = product_history["Units Sold"].iloc[-4]

    rolling_mean_4 = product_history["Units Sold"].tail(4).mean()
    rolling_std_4 = product_history["Units Sold"].tail(4).std()

    now = datetime.now()
    year = now.year
    month = now.month
    week = now.isocalendar().week
    latest_price = product_history["Price"].iloc[-1]


    # Build base feature dict
    input_dict = {
    "lag_1": lag_1,
    "lag_2": lag_2,
    "lag_4": lag_4,
    "rolling_mean_4": rolling_mean_4,
    "rolling_std_4": rolling_std_4,
    "year": year,
    "month": month,
    "week": week,
    "Price": latest_price,
    "Discount": 0,
    "Units Returned": 0
    }


    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Add encoded categorical columns
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Activate selected categorical features
    input_df[f"Product Name_{product}"] = 1
    input_df[f"Location_{location}"] = 1
    input_df[f"Platform_{platform}"] = 1
    input_df[f"Category_{category}"] = 1

    # Ensure column order
    input_df = input_df[feature_columns]

    prediction = model.predict(input_df)[0]
    
    if len(product_history) < 4:
        raise ValueError("Not enough historical data for selected combination.")


    return int(round(prediction))

