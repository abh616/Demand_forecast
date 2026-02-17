import streamlit as st
import requests

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


if st.button("Predict Demand"):

    payload = {
    "product_name": product_name,
    "location": location,
    "platform": platform,
    "category": category,
    
}


    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        result = response.json()
        st.success(f"Predicted sales: {result['predicted_units_sold']:.2f}")
    else:
        st.error("Prediction failed.")
