# ---------------------------
# Demand Forecasting Dashboard
# ---------------------------

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
# App Config
# ---------------------------
st.set_page_config(page_title="Demand Forecasting", layout="wide")
st.title("📊 Demand Forecasting Dashboard (All SKUs Combined)")

# ---------------------------
# Load Data
# ---------------------------
uploaded_file = st.file_uploader("Upload your demand.csv file (optional)", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
else:
    url = "https://raw.githubusercontent.com/username/repo/main/demand.csv"  # 🔹 replace with your GitHub raw file URL
    data = pd.read_csv(url)

st.sidebar.subheader("Navigation")
menu = st.sidebar.radio("Go to section:", ["Data Cleaning", "Demand Data Analysis", "Forecasting"])

# ---------------------------
# Preprocess (combine SKUs)
# ---------------------------
data['week'] = pd.to_datetime(data['week'], format='%d/%m/%y')

# Aggregate units_sold & price across all SKUs
combined_data = data.groupby('week').agg({
    'units_sold': 'sum',
    'total_price': 'mean'   # using avg price across SKUs
}).reset_index()

combined_data.set_index('week', inplace=True)

# ---------------------------
# 1️⃣ Data Cleaning
# ---------------------------
if menu == "Data Cleaning":
    st.header("🧹 Data Cleaning")
    
    missing_count = combined_data.isnull().sum().sum()
    st.write(f"Total Missing Values: {missing_count}")
    
    if combined_data['total_price'].isnull().sum() > 0:
        median_price = combined_data['total_price'].median()
        combined_data['total_price'].fillna(median_price, inplace=True)
        st.success("Missing 'total_price' values filled with median.")
    else:
        st.info("No missing 'total_price' values found.")
    
    st.write("✅ Data cleaned and ready for analysis.")
    st.dataframe(combined_data.head())

# ---------------------------
# 2️⃣ Demand Data Analysis
# ---------------------------
elif menu == "Demand Data Analysis":
    st.header("📊 Demand Data Analysis (All SKUs Combined)")
    
    # Weekly Demand Plot
    st.subheader("Weekly Units Sold")
    st.line_chart(combined_data['units_sold'])
    
    # Boxplot of Price
    st.subheader("Price Distribution")
    fig, ax = plt.subplots(figsize=(6,3))
    sns.boxplot(x=combined_data['total_price'], ax=ax)
    st.pyplot(fig)
    
    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    corr = combined_data.corr()
    fig, ax = plt.subplots(figsize=(6,3))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ---------------------------
# 3️⃣ Forecasting
# ---------------------------
elif menu == "Forecasting":
    st.header("🤖 Forecasting Models (All SKUs Combined)")
    
    weekly_data = combined_data['units_sold']
    
    # Train/Test Split
    train_data = weekly_data[:int(0.8*len(weekly_data))]
    test_data = weekly_data[int(0.8*len(weekly_data)):]
    
    # Holt-Winters
    hw_model = ExponentialSmoothing(train_data, seasonal='add', seasonal_periods=52).fit()
    hw_pred = hw_model.predict(start=test_data.index[0], end=test_data.index[-1])
    hw_rmse = sqrt(mean_squared_error(test_data, hw_pred))
    
    # ARIMA
    arima_model = ARIMA(train_data, order=(1,0,0)).fit()
    arima_pred = arima_model.predict(start=test_data.index[0], end=test_data.index[-1])
    arima_rmse = sqrt(mean_squared_error(test_data, arima_pred))
    
    # Prophet
    prophet_data = weekly_data.reset_index()
    prophet_data.columns = ['ds','y']
    train_prophet = prophet_data[:int(0.8*len(prophet_data))]
    test_prophet = prophet_data[int(0.8*len(prophet_data)):]
    prophet_model = Prophet(yearly_seasonality=True)
    prophet_model.fit(train_prophet)
    future = prophet_model.make_future_dataframe(periods=len(test_prophet))
    prophet_pred = prophet_model.predict(future)
    prophet_rmse = sqrt(mean_squared_error(test_prophet['y'], prophet_pred['yhat'][-len(test_prophet):]))
    
    # Display RMSE
    st.write(f"Holt-Winters RMSE: {hw_rmse:.2f}")
    st.write(f"ARIMA RMSE: {arima_rmse:.2f}")
    st.write(f"Prophet RMSE: {prophet_rmse:.2f}")
    
    # Best Model
    rmse_values = [hw_rmse, arima_rmse, prophet_rmse]
    model_names = ['Holt-Winters','ARIMA','Prophet']
    best_model = model_names[rmse_values.index(min(rmse_values))]
    st.success(f"✅ Best Model: {best_model}")
    
    # Forecast Comparison Plot
    st.subheader("Forecast Comparison")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly_data.index, y=weekly_data, mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(x=hw_pred.index, y=hw_pred, mode="lines", name="Holt-Winters"))
    fig.add_trace(go.Scatter(x=arima_pred.index, y=arima_pred, mode="lines", name="ARIMA"))
    fig.add_trace(go.Scatter(x=prophet_pred['ds'], y=prophet_pred['yhat'], mode="lines", name="Prophet"))
    st.plotly_chart(fig, use_container_width=True)
    
    # Download Forecast
    forecast_df = pd.DataFrame({
        "Date": prophet_pred['ds'],
        "Forecasted_Units": prophet_pred['yhat']
    })
    st.download_button(
        "📥 Download Forecast CSV",
        forecast_df.to_csv(index=False).encode('utf-8'),
        "forecast.csv",
        "text/csv"
    )
