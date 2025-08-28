# ---------------------------
# Demand Forecasting Dashboard
# ---------------------------

# Import libraries
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
# Streamlit App Config
# ---------------------------
st.set_page_config(page_title="Demand Forecasting", layout="wide")
st.title("📊 Demand Forecasting Dashboard")

# ---------------------------
# Data Load
# ---------------------------
uploaded_file = st.file_uploader("Upload your demand.csv file (optional)", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
else:
    # Read directly from GitHub
    url = "https://raw.githubusercontent.com/username/repo/main/demand.csv"  # <-- Replace with your GitHub raw URL
    data = pd.read_csv(url)

st.subheader("Raw Data Preview")
st.dataframe(data.head())

# ---------------------------
# Data Cleaning
# ---------------------------
if data['total_price'].isnull().sum() > 0:
    median_price_per_sku = data.groupby('sku_id')['total_price'].median()
    data['total_price'].fillna(data['sku_id'].map(median_price_per_sku), inplace=True)

# Convert 'week' to datetime and set as index
data['week'] = pd.to_datetime(data['week'], format='%d/%m/%y')
data.set_index('week', inplace=True)

# ---------------------------
# SKU Filter
# ---------------------------
sku_list = data['sku_id'].unique()
selected_sku = st.selectbox("Select SKU to forecast:", sku_list)
filtered_data = data[data['sku_id'] == selected_sku]

# Weekly aggregation
weekly_data = filtered_data['units_sold'].resample('W').sum()
st.subheader(f"📈 Weekly Units Sold for SKU {selected_sku}")
st.line_chart(weekly_data)

# ---------------------------
# Exploratory Data Analysis (EDA)
# ---------------------------
st.subheader("🔎 Exploratory Data Analysis")

# Boxplot for price distribution
fig, ax = plt.subplots(figsize=(10, 4))
sns.boxplot(x=filtered_data['total_price'], ax=ax)
st.pyplot(fig)

# Correlation heatmap
corr = filtered_data.corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# ---------------------------
# Forecasting Models
# ---------------------------
st.subheader("🤖 Forecasting Models")

# Split data into train/test (80/20)
train_data = weekly_data[:int(0.8*len(weekly_data))]
test_data = weekly_data[int(0.8*len(weekly_data)):]

# 1. Holt-Winters
hw_model = ExponentialSmoothing(train_data, seasonal='add', seasonal_periods=52).fit()
hw_predictions = hw_model.predict(start=test_data.index[0], end=test_data.index[-1])
hw_rmse = sqrt(mean_squared_error(test_data, hw_predictions))

# 2. ARIMA
arima_model = ARIMA(train_data, order=(1, 0, 0)).fit()
arima_predictions = arima_model.predict(start=test_data.index[0], end=test_data.index[-1])
arima_rmse = sqrt(mean_squared_error(test_data, arima_predictions))

# 3. Prophet
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

# ---------------------------
# Forecast Comparison Plot
# ---------------------------
st.subheader("📊 Forecast Comparison")
fig = go.Figure()
fig.add_trace(go.Scatter(x=weekly_data.index, y=weekly_data, mode="lines", name="Actual"))
fig.add_trace(go.Scatter(x=hw_predictions.index, y=hw_predictions, mode="lines", name="Holt-Winters"))
fig.add_trace(go.Scatter(x=arima_predictions.index, y=arima_predictions, mode="lines", name="ARIMA"))
fig.add_trace(go.Scatter(x=prophet_predictions['ds'], y=prophet_predictions['yhat'], mode="lines", name="Prophet"))
st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Download Forecast
# ---------------------------
forecast_df = pd.DataFrame({
    "Date": prophet_predictions['ds'],
    "Forecasted_Units": prophet_predictions['yhat']
})
st.download_button(
    "📥 Download Forecast as CSV",
    forecast_df.to_csv(index=False).encode('utf-8'),
    "forecast.csv",
    "text/csv"
)
