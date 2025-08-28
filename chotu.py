import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

# ----------------------------
# Load Data
# ----------------------------
@st.cache_data
def load_data():
    # Make sure "aemana.csv" is in the same repo/folder as your Streamlit app
    data = pd.read_csv("demand.csv")
    return data

# ----------------------------
# Main App
# ----------------------------
def main():
    st.title("📊 Demand Forecasting App")

    menu = ["Data Cleaning", "Demand Analysis", "Forecasting"]
    choice = st.sidebar.radio("Choose Section", menu)

    # Load data
    data = load_data()

    # ========================
    # Data Cleaning Section
    # ========================
    if choice == "Data Cleaning":
        st.subheader("🧹 Data Cleaning")

        st.write("Raw Data Preview:")
        st.dataframe(data.head())

        # Basic cleaning
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        data = data.dropna(subset=["Date"])
        
        st.write("✅ After cleaning (Date column fixed & NaN removed):")
        st.dataframe(data.head())

    # ========================
    # Demand Analysis Section
    # ========================
    elif choice == "Demand Analysis":
        st.subheader("📈 Demand Data Analysis")

        # Convert to datetime
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        data = data.dropna(subset=["Date"])

        # Combine all SKUs by summing Demand
        demand_data = data.groupby("Date")["Demand"].sum().reset_index()

        st.line_chart(demand_data.set_index("Date")["Demand"])

        st.write("Total demand over time (all SKUs combined).")

    # ========================
    # Forecasting Section
    # ========================
    elif choice == "Forecasting":
        st.subheader("🔮 Forecasting Demand")

        # Prepare data
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        demand_data = data.groupby("Date")["Demand"].sum().reset_index()
        demand_data = demand_data.set_index("Date")

        # Holt-Winters
        model_hw = ExponentialSmoothing(demand_data["Demand"], trend="add", seasonal=None).fit()
        forecast_hw = model_hw.forecast(10)

        # ARIMA
        model_arima = ARIMA(demand_data["Demand"], order=(2,1,2)).fit()
        forecast_arima = model_arima.forecast(10)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(demand_data.index, demand_data["Demand"], label="Actual")
        ax.plot(forecast_hw.index, forecast_hw, label="Holt-Winters Forecast")
        ax.plot(forecast_arima.index, forecast_arima, label="ARIMA Forecast")
        ax.legend()
        st.pyplot(fig)

        st.write("Forecast Results:")
        st.write(pd.DataFrame({
            "Holt-Winters": forecast_hw,
            "ARIMA": forecast_arima
        }))

# Run app
if __name__ == "__main__":
    main()
