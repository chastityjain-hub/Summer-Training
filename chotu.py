# streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np

# =========================
# Load Data from GitHub
# =========================
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/YOUR_GITHUB_USERNAME/YOUR_REPO/main/demand.csv"
    data = pd.read_csv(url)
    data['week'] = pd.to_datetime(data['week'], format='%d/%m/%y')
    return data

data = load_data()

# =========================
# Sidebar Menu
# =========================
menu = st.sidebar.radio("📊 Select Section", 
                        ["Data Cleaning", "Demand Data Analysis", "Forecasting"])

# =========================
# 1. Data Cleaning
# =========================
if menu == "Data Cleaning":
    st.title("🧹 Data Cleaning")
    st.write("Checking for missing values...")

    # Fill missing total_price by SKU median
    median_price_per_sku = data.groupby('sku_id')['total_price'].median()
    data['total_price'].fillna(data['sku_id'].map(median_price_per_sku), inplace=True)

    st.write("✅ Missing values handled successfully.")
    st.write(data.head())

    st.write("Remaining missing values per column:")
    st.write(data.isnull().sum())

# =========================
# 2. Demand Data Analysis
# =========================
elif menu == "Demand Data Analysis":
    st.title("📈 Demand Data Analysis")

    # Combine SKUs → Weekly Aggregated Data
    weekly_data = data.groupby('week')['units_sold'].sum().reset_index()
    weekly_data.set_index('week', inplace=True)

    st.subheader("Time Series of Units Sold")
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(weekly_data.index, weekly_data['units_sold'])
    ax.set_title("Weekly Units Sold (All SKUs Combined)")
    st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("Feature Correlation Heatmap")
    correlation = data.corr()
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Decomposition
    st.subheader("Seasonal Decomposition")
    decomposition = seasonal_decompose(weekly_data['units_sold'], period=52)
    fig = decomposition.plot()
    fig.set_size_inches(10,8)
    st.pyplot(fig)

# =========================
# 3. Forecasting
# =========================
elif menu == "Forecasting":
    st.title("🔮 Forecasting Models (All SKUs Combined)")

    # Prepare Data
    weekly_data = data.groupby('week')['units_sold'].sum().resample('W').sum()
    train = weekly_data[:int(0.8*len(weekly_data))]
    test = weekly_data[int(0.8*len(weekly_data)):]

    results = {}

    # Holt-Winters
    try:
        hw_model = ExponentialSmoothing(train, seasonal='add', seasonal_periods=52).fit()
        hw_pred = hw_model.predict(start=test.index[0], end=test.index[-1])
        hw_rmse = sqrt(mean_squared_error(test, hw_pred))
        results['Holt-Winters'] = hw_rmse
        st.write(f"**Holt-Winters RMSE:** {hw_rmse:.2f}")
    except Exception as e:
        st.error(f"Holt-Winters failed: {e}")

    # ARIMA
    try:
        arima_model = ARIMA(train, order=(1,1,1)).fit()
        arima_pred = arima_model.predict(start=test.index[0], end=test.index[-1])
        arima_rmse = sqrt(mean_squared_error(test, arima_pred))
        results['ARIMA'] = arima_rmse
        st.write(f"**ARIMA RMSE:** {arima_rmse:.2f}")
    except Exception as e:
        st.error(f"ARIMA failed: {e}")

    # Prophet
    try:
        prophet_data = weekly_data.reset_index()
        prophet_data.columns = ['ds','y']
        train_p = prophet_data[:int(0.8*len(prophet_data))]
        test_p = prophet_data[int(0.8*len(prophet_data)):]
        
        prophet = Prophet(yearly_seasonality=True)
        prophet.fit(train_p)
        future = prophet.make_future_dataframe(periods=len(test_p), freq='W')
        forecast = prophet.predict(future)
        
        prophet_rmse = sqrt(mean_squared_error(test_p['y'], forecast['yhat'][-len(test_p):]))
        results['Prophet'] = prophet_rmse
        st.write(f"**Prophet RMSE:** {prophet_rmse:.2f}")

        # Plot Prophet forecast
        st.subheader("Prophet Forecast")
        fig = prophet.plot(forecast)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Prophet failed: {e}")

    # Best Model
    if results:
        best_model = min(results, key=results.get)
        st.success(f"🏆 Best Model: **{best_model}** with RMSE {results[best_model]:.2f}")
