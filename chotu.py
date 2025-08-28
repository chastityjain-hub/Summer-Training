
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
# Streamlit App
# ---------------------------
st.set_page_config(page_title="Demand Forecasting", layout="wide")
st.title("📊 Demand Forecasting Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload your demand.csv file", type=["csv"])

if uploaded_file:
    # Load dataset
    data = pd.read_csv(uploaded_file)
    st.subheader("Raw Data Preview")
    st.dataframe(data.head())

    # Handle missing values
    if data['total_price'].isnull().sum() > 0:
        median_price_per_sku = data.groupby('sku_id')['total_price'].median()
        data['total_price'].fillna(data['sku_id'].map(median_price_per_sku), inplace=True)

    # Convert week column
    data['week'] = pd.to_datetime(data['week'], format='%d/%m/%y')
    data.set_index('week', inplace=True)

    # Weekly aggregation
    weekly_data = data['units_sold'].resample('W').sum()
    st.subheader("📈 Weekly Units Sold")
    st.line_chart(weekly_data)

    # ----------- EDA -----------
    st.subheader("🔎 Exploratory Data Analysis")

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(x=data['total_price'], ax=ax)
    st.pyplot(fig)

    corr = data.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # ----------- Forecasting Models -----------
    st.subheader("🤖 Forecasting Models")

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

    # Display RMSE
    st.write(f"**Holt-Winters RMSE:** {hw_rmse:.2f}")
    st.write(f"**ARIMA RMSE:** {arima_rmse:.2f}")
    st.write(f"**Prophet RMSE:** {prophet_rmse:.2f}")

    # Best Model
    rmse_values = [hw_rmse, arima_rmse, prophet_rmse]
    model_names = ['Holt-Winters', 'ARIMA', 'Prophet']
    best_model = model_names[rmse_values.index(min(rmse_values))]
    st.success(f"✅ The best model is: {best_model}")

    # ----------- Plot Forecasts -----------
    st.subheader("📊 Forecast Comparison")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly_data.index, y=weekly_data, mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(x=hw_predictions.index, y=hw_predictions, mode="lines", name="Holt-Winters"))
    fig.add_trace(go.Scatter(x=arima_predictions.index, y=arima_predictions, mode="lines", name="ARIMA"))
    fig.add_trace(go.Scatter(x=prophet_predictions['ds'], y=prophet_predictions['yhat'], mode="lines", name="Prophet"))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("👆 Please upload a `demand.csv` file to start analysis.")
