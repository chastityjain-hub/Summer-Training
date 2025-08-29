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
# Streamlit App Configuration
# ---------------------------
st.set_page_config(page_title="Demand Forecasting", layout="wide")
st.title("📊 Demand Forecasting Dashboard")

# ---------------------------
# Load Data (directly from repo demand.csv)
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("demand.csv")
    return df

data = load_data()

# Sidebar menu for navigation
menu = ["Data Cleaning / Preprocessing", "Data Analysis", "Forecasting"]
choice = st.sidebar.selectbox("Choose Page", menu)

# ---------------------------
# Data Cleaning & Preprocessing
# ---------------------------
if choice == "Data Cleaning / Preprocessing":
    st.subheader("🧹 Data Cleaning / Preprocessing")
    st.write("Preview of raw data:")
    st.dataframe(data.head())

    # Handle missing values in 'total_price'
    if data['total_price'].isnull().sum() > 0:
        st.write("Filling missing `total_price` values with median per SKU...")
        median_price_per_sku = data.groupby('sku_id')['total_price'].median()
        data['total_price'].fillna(data['sku_id'].map(median_price_per_sku), inplace=True)
    
    # Convert 'week' column to datetime
    st.write("Converting `week` column to datetime format and setting it as index...")
    data['week'] = pd.to_datetime(data['week'], format='%d/%m/%y', errors='coerce')
    data.set_index('week', inplace=True)

    st.success("✅ Data cleaning and preprocessing completed.")
    st.write("Preview of cleaned data:")
    st.dataframe(data.head())

# ---------------------------
# EDA Section
# ---------------------------
elif choice == "Data Analysis":
    st.subheader("🔎 Exploratory Data Analysis (EDA)")

    # Ensure 'week' is datetime
    data['week'] = pd.to_datetime(data['week'], format='%d/%m/%y', errors='coerce')
    data.set_index('week', inplace=True)

    # Weekly aggregation
    weekly_data = data['units_sold'].resample('W').sum()
    st.write("Weekly aggregated units sold:")
    st.line_chart(weekly_data)

    # ✅ ADD DECOMPOSITION CHART AFTER 1ST CHART
    from statsmodels.tsa.seasonal import seasonal_decompose

    st.write("Seasonal Decomposition of Weekly Units Sold:")
    decomposition = seasonal_decompose(weekly_data, model='additive', period=52)

    fig, axs = plt.subplots(4, 1, figsize=(12, 10))

    decomposition.trend.plot(ax=axs[0], legend=True, title="Trend")
    decomposition.seasonal.plot(ax=axs[1], legend=True, title="Seasonality")
    decomposition.resid.plot(ax=axs[2], legend=True, title="Residuals")
    weekly_data.plot(ax=axs[3], legend=True, title="Original")

    plt.tight_layout()
    st.pyplot(fig)

    # Boxplot for total_price
    st.write("Distribution of `total_price`:")
    data['total_price'] = pd.to_numeric(data['total_price'], errors='coerce')
    clean_prices = data['total_price'].dropna()

    if clean_prices.empty:
        st.warning("⚠️ No valid `total_price` values available for plotting.")
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.boxplot(x=clean_prices, ax=ax, color="skyblue")
        st.pyplot(fig)

    # Correlation heatmap
    st.write("Correlation Heatmap:")
    corr = data.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.info("💡 Observations: Featured or displayed SKUs sell more. Lower-priced items tend to sell in higher quantities.")


# ---------------------------
# Forecasting Section
# ---------------------------
elif choice == "Forecasting":
    st.subheader("🤖 Forecasting Models")

    # Prepare weekly data
    data['week'] = pd.to_datetime(data['week'], format='%d/%m/%y', errors='coerce')
    data.set_index('week', inplace=True)
    weekly_data = data['units_sold'].resample('W').sum()

    # Train-test split (80-20)
    train_data = weekly_data[:int(0.8*len(weekly_data))]
    test_data = weekly_data[int(0.8*len(weekly_data)):]

    # ---------------- Holt-Winters Forecast ----------------
    hw_model = ExponentialSmoothing(train_data, seasonal='add', seasonal_periods=52).fit()
    hw_predictions = hw_model.predict(start=test_data.index[0], end=test_data.index[-1])
    hw_rmse = sqrt(mean_squared_error(test_data, hw_predictions))

    # ---------------- ARIMA Forecast ----------------
    arima_model = ARIMA(train_data, order=(1, 0, 0)).fit()
    arima_predictions = arima_model.predict(start=test_data.index[0], end=test_data.index[-1])
    arima_rmse = sqrt(mean_squared_error(test_data, arima_predictions))

    # ---------------- Prophet Forecast ----------------
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
    st.subheader("📊 Model Performance (RMSE)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Holt-Winters RMSE", f"{hw_rmse:.2f}")
    col2.metric("ARIMA RMSE", f"{arima_rmse:.2f}")
    col3.metric("Prophet RMSE", f"{prophet_rmse:.2f}")

    # Determine Best Model
    rmse_values = [hw_rmse, arima_rmse, prophet_rmse]
    model_names = ['Holt-Winters', 'ARIMA', 'Prophet']
    best_model = model_names[rmse_values.index(min(rmse_values))]
    st.success(f"🏆 The best model is: **{best_model}**")

    # ---------------- Plot Forecasts ----------------
    st.subheader("📊 Forecast Comparison")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly_data.index, y=weekly_data, mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(x=hw_predictions.index, y=hw_predictions, mode="lines", name="Holt-Winters"))
    fig.add_trace(go.Scatter(x=arima_predictions.index, y=arima_predictions, mode="lines", name="ARIMA"))
    fig.add_trace(go.Scatter(x=prophet_predictions['ds'], y=prophet_predictions['yhat'], mode="lines", name="Prophet"))
    st.plotly_chart(fig, use_container_width=True)
