# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from math import sqrt
import plotly.graph_objs as go

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(page_title="Demand Forecasting Dashboard", layout="wide")
st.title("📊 Demand Forecasting Dashboard")

# ---------------------------
# Load Data (direct from repo demand.csv)
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("demand.csv")
    return df

data = load_data()

# Sidebar navigation
menu = st.sidebar.selectbox(
    "Choose Section",
    ["📌 Data Cleaning & Preprocessing", "🔎 Exploratory Data Analysis", "🤖 Forecasting Models"]
)

# ---------------------------
# Data Cleaning & Preprocessing
# ---------------------------
if menu == "📌 Data Cleaning & Preprocessing":
    st.header("📌 Data Cleaning & Preprocessing")

    st.subheader("Missing Values Before Cleaning")
    st.write(data.isnull().sum())

    # Fill missing values for total_price
    median_price_per_sku = data.groupby('sku_id')['total_price'].median()
    data['total_price'].fillna(data['sku_id'].map(median_price_per_sku), inplace=True)

    st.subheader("✅ Missing Values After Cleaning")
    st.write(data.isnull().sum())

    st.success("Data has been cleaned! Missing values handled using median per SKU.")

    st.subheader("Preview of Cleaned Data")
    st.dataframe(data.head())

# ---------------------------
# EDA Section
# ---------------------------
elif menu == "🔎 Exploratory Data Analysis":
    st.header("🔎 Exploratory Data Analysis (EDA)")

    # Convert week column
    data['week'] = pd.to_datetime(data['week'], format='%d/%m/%y')
    data.set_index('week', inplace=True)

    # Weekly aggregation
    weekly_data = data['units_sold'].resample('W').sum()

    st.subheader("📈 Weekly Units Sold Over Time")
    st.line_chart(weekly_data)

    st.subheader("📊 Distribution of Total Price")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(x=data['total_price'], ax=ax, color="skyblue")
    st.pyplot(fig)

    st.subheader("🔥 Correlation Heatmap")
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.info("Observations: Featured or displayed SKUs sell more. Lower-priced items tend to sell in higher quantities.")

# ---------------------------
# Forecasting Models Section
# ---------------------------
elif menu == "🤖 Forecasting Models":
    st.header("🤖 Forecasting Models")

    # Convert week column if not already
    data['week'] = pd.to_datetime(data['week'], format='%d/%m/%y')
    data.set_index('week', inplace=True)

    # Weekly aggregation
    weekly_data = data['units_sold'].resample('W').sum()

    # Train-test split
    train_data = weekly_data[:int(0.8*len(weekly_data))]
    test_data = weekly_data[int(0.8*len(weekly_data)):]

    # Holt-Winters
    hw_model = ExponentialSmoothing(train_data, seasonal='add', seasonal_periods=52).fit()
    hw_predictions = hw_model.predict(start=test_data.index[0], end=test_data.index[-1])
    hw_rmse = sqrt(mean_squared_error(test_data, hw_predictions))

    # ARIMA
    arima_model = ARIMA(train_data, order=(1, 0, 0)).fit()
    arima_predictions = arima_model.predict(start=test_data.index[0], end=test_data.index[-1])
    arima_rmse = sqrt(mean_squared_error(test_data, arima_predictions))

    # Prophet
    prophet_data = weekly_data.reset_index()
    prophet_data.columns = ['ds', 'y']
    train_prophet = prophet_data[:int(0.8*len(prophet_data))]
    test_prophet = prophet_data[int(0.8*len(prophet_data)):]
    prophet_model = Prophet(yearly_seasonality=True)
    prophet_model.fit(train_prophet)
    future = prophet_model.make_future_dataframe(periods=len(test_prophet))
    prophet_predictions = prophet_model.predict(future)
    prophet_rmse = sqrt(mean_squared_error(test_prophet['y'], prophet_predictions['yhat'][-len(test_prophet):]))

    # Show RMSEs
    st.subheader("📊 Model Evaluation (RMSE)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Holt-Winters RMSE", f"{hw_rmse:.2f}")
    col2.metric("ARIMA RMSE", f"{arima_rmse:.2f}")
    col3.metric("Prophet RMSE", f"{prophet_rmse:.2f}")

    # Best Model
    rmse_values = [hw_rmse, arima_rmse, prophet_rmse]
    model_names = ['Holt-Winters', 'ARIMA', 'Prophet']
    best_model = model_names[rmse_values.index(min(rmse_values))]
    st.success(f"🏆 The best model is: **{best_model}**")

    # Forecast Comparison Plot
    st.subheader("🔮 Forecast Comparison")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly_data.index, y=weekly_data, mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(x=hw_predictions.index, y=hw_predictions, mode="lines", name="Holt-Winters"))
    fig.add_trace(go.Scatter(x=arima_predictions.index, y=arima_predictions, mode="lines", name="ARIMA"))
    fig.add_trace(go.Scatter(x=prophet_predictions['ds'], y=prophet_predictions['yhat'], mode="lines", name="Prophet"))
    st.plotly_chart(fig, use_container_width=True)
