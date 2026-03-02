import streamlit as st
from datetime import datetime
import pandas as pd
import sys
import os
import holidays

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Retail Demand Forecasting System",
    layout="wide"
)

# ---------------- IMPORT PREDICT ----------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from src.inference import predict

DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "clean_sales_data.csv")

df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])

# ---------------- HOLIDAYS ----------------
us_holidays = holidays.US()
uk_holidays = holidays.UK()
ca_holidays = holidays.CA()

# ---------------- CONTEXT SNAPSHOT ----------------
today = datetime.now().date()
current_week = datetime.now().isocalendar().week
current_month = datetime.now().month
current_year = datetime.now().year

holiday_countries = []
if today in us_holidays:
    holiday_countries.append("USA")
if today in uk_holidays:
    holiday_countries.append("UK")
if today in ca_holidays:
    holiday_countries.append("Canada")

holiday_display = "Yes – " + ", ".join(holiday_countries) if holiday_countries else "No"

# ---------------- SNAPSHOT STYLE ----------------
st.markdown("""
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
.prediction-card {
    background-color: #1f2937;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #374151;
}
</style>
""", unsafe_allow_html=True)

# ---------------- KPI STRIP ----------------
kpi_cols = st.columns(5)

kpis = [
    ("📅 Date", today),
    ("🗓 Year", current_year),
    ("📊 Week", current_week),
    ("🌡 Month", current_month),
    ("🎉 Holiday", holiday_display)
]

for col, (title, value) in zip(kpi_cols, kpis):
    with col:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("## Retail Demand Forecasting System")

# ---------------- SIDEBAR CONTROLS ----------------
with st.sidebar:
    st.header("Forecast Controls")

    product_list = sorted(df["Product Name"].unique())
    location_list = sorted(df["Location"].unique())
    platform_list = sorted(df["Platform"].unique())
    category_list = sorted(df["Category"].unique())

    product_name = st.selectbox("Product", product_list)
    location = st.selectbox("Location", location_list)
    platform = st.selectbox("Platform", platform_list)
    category = st.selectbox("Category", category_list)

    predict_button = st.button("Predict Demand")

# ---------------- TABS ----------------
tab1, tab2 = st.tabs(["Forecast Dashboard", "Ask Insights"])

# =========================================================
# TAB 1: DASHBOARD
# =========================================================
with tab1:

    product_history = df[
        (df["Product Name"] == product_name) &
        (df["Location"] == location) &
        (df["Platform"] == platform)
    ].sort_values("Date")

    col1, col2 = st.columns([1, 2])

    # -------- Prediction Card --------
    with col1:
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        st.subheader("Prediction Result")

        if predict_button:
            try:
                prediction = predict({
                    "product_name": product_name,
                    "location": location,
                    "platform": platform,
                    "category": category
                })

                st.success(f"Predicted Units Sold: {prediction}")

                if not product_history.empty:
                    last_actual = product_history["Units Sold"].iloc[-1]
                    change = prediction - last_actual
                    st.write(f"Last Actual: {last_actual}")
                    st.write(f"Change from Last Period: {change}")

            except Exception as e:
                st.error(str(e))

        st.markdown('</div>', unsafe_allow_html=True)

    # -------- Historical Trend --------
    with col2:
        st.subheader("Historical Demand Trend")
        if not product_history.empty:
            st.line_chart(
                product_history.set_index("Date")["Units Sold"],
                height=250
            )

    st.markdown("---")

    # -------- Compact Chart Grid --------
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        st.subheader("Discount vs Units Sold")
        st.scatter_chart(
            product_history[["Discount", "Units Sold"]],
            height=220
        )

    with row1_col2:
        st.subheader("Category Distribution")
        category_sales = df.groupby("Category")["Units Sold"].sum()
        st.bar_chart(category_sales, height=220)

# =========================================================
# TAB 2: CHATBOT
# =========================================================
with tab2:

    st.subheader("Ask About Sales Insights")

    query = st.text_input("Ask something about sales:")

    if query:
        q = query.lower()

        if "highest product" in q:
            top = df.groupby("Product Name")["Units Sold"].sum().idxmax()
            st.write(f"Highest selling product: {top}")

        elif "best platform" in q:
            top = df.groupby("Platform")["Units Sold"].sum().idxmax()
            st.write(f"Best performing platform: {top}")

        elif "peak month" in q:
            df["Month"] = df["Date"].dt.month
            peak = df.groupby("Month")["Units Sold"].sum().idxmax()
            st.write(f"Peak demand month: {peak}")

        else:
            st.write("Question not recognized.")