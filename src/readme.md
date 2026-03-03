# Retail Demand Forecasting System

## Overview

This project builds an end-to-end retail demand forecasting system to predict weekly units sold for products across different locations and platforms.

The system includes:

- Structured feature engineering for time-series modeling  
- Model comparison and optimization  
- Leakage-aware evaluation  
- Interactive dashboard deployment using Streamlit  

The objective is to generate accurate short-term demand forecasts to support inventory and operational planning.

---

## Problem Statement

Retail demand is influenced by historical trends, seasonality, pricing, and contextual factors.

Accurate demand forecasting helps businesses:

- Optimize inventory levels  
- Reduce stockouts and overstock  
- Improve procurement decisions  
- Support pricing strategy  

This project focuses on weekly demand prediction using supervised machine learning techniques.

---

## Dataset

The dataset contains weekly sales records with the following fields:

- Product Name  
- Location  
- Platform  
- Category  
- Date  
- Units Sold  
- Price  
- Discount  
- Units Returned  

The data spans approximately 13 months of historical sales.

---

## Feature Engineering

Retail demand is strongly autoregressive. To capture temporal structure and contextual effects, the following features were engineered:

### Lag Features

- lag_1  
- lag_2  
- lag_3  
- lag_4  
- lag_6  
- lag_8  

These capture short-term and mid-term demand memory.

### Rolling Statistics

- rolling_mean_4  
- rolling_mean_8  
- rolling_std_4  
- rolling_std_8  

These capture trend stability and momentum.

### Seasonality Encoding

Month was encoded using sine and cosine transformation to represent cyclical patterns:

- month_sin  
- month_cos  

### Pricing Dynamics

- price_lag_1  
- price_change  

These capture price elasticity effects.

### External Context

- Holiday indicator (location-aware)  
- Temperature proxy (season-based)  
- Fuel index trend proxy  

---

## Model Development

Two models were evaluated:

1. Random Forest Regressor  
2. XGBoost Regressor  

A strict time-based train-test split was used to prevent future data leakage.

Revenue was initially included as a feature but removed after identifying target leakage, ensuring realistic performance evaluation.

---

## Evaluation Metrics

Model performance was evaluated using:

- MAE (Mean Absolute Error)  
- RMSE (Root Mean Squared Error)  
- MAPE (Mean Absolute Percentage Error)  

### Final Model Performance (XGBoost with enhanced features)

- MAE: 7.56  
- RMSE: 9.63  
- MAPE: ~5.0%  

### Baseline Random Forest

- MAE: ~9.55  
- MAPE: ~6.4%  

Feature engineering reduced MAE by approximately 20 percent compared to the baseline model.

---

## Key Observations

- Rolling averages were the most influential predictors.  
- Demand behavior is trend-driven and strongly autoregressive.  
- Seasonality contributed meaningfully through cyclical encoding.  
- External contextual features had moderate impact relative to temporal features.  

---

## Deployment

The system is deployed using Streamlit and includes:

- Sidebar-based forecast controls  
- Context snapshot (date, week, holiday status)  
- Historical demand visualization  
- Compact analytical dashboard layout  

To run locally:

streamlit run dashboard/app.py


---

## Limitations

- Requires historical sales data (cold-start limitation for new products)  
- External shock events are not modeled  
- Temperature and fuel features are proxy-based  
- Limited historical time span (13 months)  

---

## Future Improvements

- Multi-step forecasting (predicting multiple future weeks)  
- Hierarchical modeling for new product fallback  
- Automated retraining pipeline  
- Integration with real weather and macroeconomic APIs  
- Drift monitoring and performance tracking  

---

## Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- XGBoost  
- Streamlit  
- Joblib  