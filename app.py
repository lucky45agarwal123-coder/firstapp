import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime

st.set_page_config(page_title="ARIMA Stock Forecast", layout="wide")

st.title("Stock Price Forecast using ARIMA")

ticker = st.text_input(
    "Enter Stock Ticker",
    "RELIANCE.NS"
)

if st.button("Generate Forecast"):

    end_date = datetime.today()
    start_date = end_date - pd.DateOffset(years=5)

    data = yf.download(
        ticker,
        start=start_date,
        end=end_date
    )

    if len(data) == 0:
        st.error("No data found.")
    else:

        close_prices = data["Close"]

        st.subheader("Historical Stock Price")

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(close_prices.index,
                close_prices.values,
                label="Close Price")
        ax.set_title(f"{ticker} Closing Price (Last 5 Years)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()

        st.pyplot(fig)

        st.subheader("ARIMA Forecast")

        model = ARIMA(close_prices, order=(5,1,0))
        model_fit = model.fit()

        target_date = pd.Timestamp("2027-06-30")

        forecast_steps = (
            target_date -
            close_prices.index[-1]
        ).days

        forecast = model_fit.forecast(
            steps=forecast_steps
        )

        june_2027_price = forecast.iloc[-1]

        st.success(
            f"Predicted Price on June 30, 2027: ₹{june_2027_price:.2f}"
        )

        forecast_dates = pd.date_range(
            start=close_prices.index[-1] + pd.Timedelta(days=1),
            periods=forecast_steps
        )

        forecast_df = pd.DataFrame({
            "Date": forecast_dates,
            "Forecast": forecast
        })

        fig2, ax2 = plt.subplots(figsize=(12,6))

        ax2.plot(
            close_prices.index,
            close_prices.values,
            label="Historical"
        )

        ax2.plot(
            forecast_df["Date"],
            forecast_df["Forecast"],
            label="Forecast"
        )

        ax2.set_title(
            f"{ticker} Forecast till June 2027"
        )

        ax2.legend()

        st.pyplot(fig2)
