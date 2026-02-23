

import streamlit as st
from datetime import datetime
import pandas as pd
import sys
import os
import holidays


st.set_page_config(layout="wide")


# ---------------------------
# Project Paths
# ---------------------------

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from src.inference import predict

DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "clean_sales_data.csv")

# ---------------------------
# Load Data
# ---------------------------

df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])

# ---------------------------
# Holiday Integration (Context Feature)
# ---------------------------

us_holidays = holidays.US()
uk_holidays = holidays.UK()
ca_holidays = holidays.CA()

df["IsHoliday"] = False

df.loc[df["Location"] == "USA", "IsHoliday"] = (
    df.loc[df["Location"] == "USA", "Date"].isin(us_holidays)
)
df.loc[df["Location"] == "UK", "IsHoliday"] = (
    df.loc[df["Location"] == "UK", "Date"].isin(uk_holidays)
)
df.loc[df["Location"] == "Canada", "IsHoliday"] = (
    df.loc[df["Location"] == "Canada", "Date"].isin(ca_holidays)
)


st.markdown(
    """
    <style>
    .metric-box {
        background-color: #111827;
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #1f2937;
    }
    .metric-title {
        font-size: 14px;
        color: #9ca3af;
    }
    .metric-value {
        font-size: 20px;
        font-weight: bold;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

today = datetime.now()
current_week = today.isocalendar().week
current_month = today.month
current_year = today.year

today = datetime.now().date()

holiday_countries = []

if today in us_holidays:
    holiday_countries.append("USA")

if today in uk_holidays:
    holiday_countries.append("UK")

if today in ca_holidays:
    holiday_countries.append("Canada")

if holiday_countries:
    holiday_display = "Yes – " + ", ".join(holiday_countries)
else:
    holiday_display = "No"

cols = st.columns(5)

with cols[0]:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-title">📅 Date</div>
            <div class="metric-value">{today}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with cols[1]:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-title">🗓 Year</div>
            <div class="metric-value">{current_year}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with cols[2]:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-title">📊 Week</div>
            <div class="metric-value">{current_week}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with cols[3]:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-title">🌡 Month</div>
            <div class="metric-value">{current_month}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with cols[4]:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-title">🎉 Holiday Today</div>
            <div class="metric-value">{holiday_display}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    
# ---------------------------
# UI Title
# ---------------------------

st.title("Retail Demand Forecasting System")

tab1, tab2 = st.tabs(["Forecast Dashboard", "Ask Insights"])

# ======================================================
# TAB 1: FORECAST DASHBOARD
# ======================================================

with tab1:

    col_left, col_right = st.columns([1, 2])

    # ---------------------------
    # LEFT PANEL – Controls & Prediction
    # ---------------------------
    with col_left:

        st.subheader("Forecast Controls")

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

        

        st.markdown("---")

        if st.button("Predict Demand"):

            try:
                product_history = df[
                    (df["Product Name"] == product_name) &
                    (df["Location"] == location) &
                    (df["Platform"] == platform)
                ].sort_values("Date")

                if product_history.empty:
                    st.error("No historical data for selected combination.")
                else:
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

                    change = prediction - last_actual
                    st.write(f"Change from Last Week: {change}")

            except Exception as e:
                st.error(str(e))

    # ---------------------------
    # RIGHT PANEL – Charts
    # ---------------------------
    with col_right:

        st.subheader("Historical Demand Trend")

        product_history = df[
            (df["Product Name"] == product_name) &
            (df["Location"] == location) &
            (df["Platform"] == platform)
        ].sort_values("Date")

        if not product_history.empty:
            st.line_chart(product_history.set_index("Date")["Units Sold"])

            st.subheader("Discount vs Units Sold")
            st.scatter_chart(product_history[["Discount", "Units Sold"]])

        st.subheader("Holiday vs Non-Holiday Demand")

        holiday_summary = df.groupby("IsHoliday")["Units Sold"].mean()
        st.bar_chart(holiday_summary)

        if True in holiday_summary.index and False in holiday_summary.index:
            lift = (
                (holiday_summary[True] - holiday_summary[False]) /
                holiday_summary[False]
            ) * 100
            st.write(f"Demand Increase During Holidays: {lift:.2f}%")

        st.subheader("Category Sales Distribution")
        category_sales = df.groupby("Category")["Units Sold"].sum()
        st.bar_chart(category_sales)

# ======================================================
# TAB 2: INSIGHTS CHATBOT
# ======================================================

with tab2:

    st.subheader("Sales Intelligence Assistant")

    query = st.text_input("Ask something about sales insights:")

    if query:
        q = query.lower()

        if "highest product" in q:
            top = df.groupby("Product Name")["Units Sold"].sum().idxmax()
            st.write(f"Highest Selling Product: {top}")

        elif "best platform" in q:
            top = df.groupby("Platform")["Units Sold"].sum().idxmax()
            st.write(f"Best Performing Platform: {top}")

        elif "peak month" in q:
            df["Month"] = df["Date"].dt.month
            peak = df.groupby("Month")["Units Sold"].sum().idxmax()
            st.write(f"Peak Demand Month: {peak}")

        else:
            st.write("Question not recognized. Try asking about highest product, best platform, or peak month.")