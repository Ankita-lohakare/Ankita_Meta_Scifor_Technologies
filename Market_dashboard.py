import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Fetching data for a specific stock
ticker = "AAPL"
stock_data = yf.download(ticker, period="1d", interval="1m")


plt.figure(figsize=(12, 6))
plt.plot(stock_data['Close'], label='Closing Price')
plt.title('AAPL Closing Prices Over Time')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

stock_data['50_MA'] = stock_data['Close'].rolling(window=50).mean()
stock_data['200_MA'] = stock_data['Close'].rolling(window=200).mean()

st.title("Stock Market Dashboard")

# User input for selecting stocks
ticker = st.text_input("Enter Stock Ticker", "AAPL")

# Fetch and display data
if ticker:
    data = yf.download(ticker)
    st.line_chart(data['Close'])
    st.write(data)

    # Moving averages
    st.line_chart(data[['Close', '50_MA', '200_MA']])

    csv = data.to_csv().encode('utf-8')

st.download_button(
    label="Download CSV",
    data=csv,
    file_name=f"{ticker}_data.csv",
    mime='text/csv',
)
