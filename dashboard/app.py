import streamlit as st
import requests
from datetime import datetime
import pandas as pd

DATA_PATH = "data/processed/clean_sales_data.csv"
history_df = pd.read_csv(DATA_PATH)
history_df["Date"] = pd.to_datetime(history_df["Date"])


API_URL = "http://127.0.0.1:8000/predict"

st.title("Demand Forecasting Dashboard")

st.subheader("Enter Forecast Inputs")

import pandas as pd

DATA_PATH = "data/processed/clean_sales_data.csv"
df = pd.read_csv(DATA_PATH)

product_list = sorted(df["Product Name"].unique())
location_list = sorted(df["Location"].unique())
platform_list = sorted(df["Platform"].unique())
category_list = sorted(df["Category"].unique())

product_name = st.selectbox("Product", product_list)
location = st.selectbox("Location", location_list)
platform = st.selectbox("Platform", platform_list)
category = st.selectbox("Category", category_list)

today = datetime.now()
current_week = today.isocalendar().week
current_month = today.month
current_year = today.year

today = datetime.now()
current_week = today.isocalendar().week
current_year = today.year

st.info(f"Forecasting Demand for Week {current_week}, {current_year}")



if st.button("Predict Demand"):
    product_history = history_df[
        (history_df["Product Name"] == product_name) &
        (history_df["Location"] == location) &
        (history_df["Platform"] == platform)
    ].sort_values("Date")

    if not product_history.empty:
        last_actual = product_history["Units Sold"].iloc[-1]
        last_date = product_history["Date"].iloc[-1]

        st.write(f"Last Recorded Date: {last_date.date()}")
        st.write(f"Last Actual Units Sold: {last_actual}")


    

    payload = {
    "product_name": product_name,
    "location": location,
    "platform": platform,
    "category": category,

    
    
}


    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        result = response.json()
        predicted = result["predicted_units_sold"]

        st.success(f"Predicted Units Sold: {predicted}")

        if not product_history.empty:
            change = predicted - last_actual
            st.write(f"Change from Last Week: {change}")

    else:
        st.error("Prediction failed.")
