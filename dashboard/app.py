import streamlit as st

from datetime import datetime
import pandas as pd

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference import predict


import streamlit as st
import pandas as pd
from datetime import datetime
import sys
import os

# Get project root directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Allow importing from src
sys.path.append(BASE_DIR)

from src.inference import predict

# Absolute path to CSV
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "clean_sales_data.csv")

df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])

df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])






st.title("Demand Forecasting Dashboard")

st.subheader("Enter Forecast Inputs")

import pandas as pd



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



st.info(f"Forecasting Demand for Week {current_week}, {current_year}")



if st.button("Predict Demand"):

    try:
        product_history = df[
            (df["Product Name"] == product_name) &
            (df["Location"] == location) &
            (df["Platform"] == platform)
        ].sort_values("Date")

        if not product_history.empty:
            last_actual = product_history["Units Sold"].iloc[-1]
            last_date = product_history["Date"].iloc[-1]

            st.write(f"Last Recorded Date: {last_date.date()}")
            st.write(f"Last Actual Units Sold: {last_actual}")

        prediction = predict({
            "product_name": product_name,
            "location": location,
            "platform": platform,
            "category": category
        })

        st.success(f"Predicted Units Sold: {prediction}")

        if not product_history.empty:
            change = prediction - last_actual
            st.write(f"Change from Last Week: {change}")

    except Exception as e:
        st.error(str(e))
