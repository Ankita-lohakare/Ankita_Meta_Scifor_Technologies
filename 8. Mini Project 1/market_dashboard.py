import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob

st.sidebar.title("Stock Market Dashboard")
st.sidebar.markdown("Select Stocks, Time Frame, and Options")

# Stock Selection
stocks = ["AAPL", "GOOGL", "META"]
selected_stocks = st.sidebar.multiselect("Select Stocks", stocks, default=["AAPL", "GOOGL"])

# Date Range Selection
timeframe = st.sidebar.selectbox("Select Timeframe", ["1d", "5d", "1mo", "6mo", "1y", "5y", "max"])
show_ma_50 = st.sidebar.checkbox("Show 50-Day Moving Average", value=True)
show_ma_200 = st.sidebar.checkbox("Show 200-Day Moving Average", value=True)
download_data = st.sidebar.checkbox("Enable Downloadable Reports")

def get_stock_data(ticker, period="1mo"):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    return data

st.title("Real-Time Stock Price Updates")
for stock in selected_stocks:
    st.subheader(f"{stock} Stock Price")
    data = get_stock_data(stock, timeframe)
    st.write(f"**Latest Price for {stock}:** ${data['Close'][-1]:.2f}")

    if show_ma_50:
        data["MA50"] = data["Close"].rolling(window=50).mean()
    if show_ma_200:
        data["MA200"] = data["Close"].rolling(window=200).mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data["Close"], label=f"{stock} Close Price")
    if show_ma_50:
        ax.plot(data["MA50"], label="50-Day MA", linestyle="--")
    if show_ma_200:
        ax.plot(data["MA200"], label="200-Day MA", linestyle="--")
    ax.set_title(f"{stock} Stock Price and Moving Averages")
    ax.legend()
    st.pyplot(fig)

st.header("Comparative Analysis")
comp_data = pd.DataFrame()
for stock in selected_stocks:
    data = get_stock_data(stock, timeframe)
    data["Ticker"] = stock
    comp_data = pd.concat([comp_data, data])
st.line_chart(comp_data.pivot_table(index="Date", columns="Ticker", values="Close"))

def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

news_data = {
    "AAPL": ["Apple is launching new products soon.", "Apple faces supply chain challenges."],
    "GOOGL": ["Google reports strong ad revenue.", "Regulation poses new challenges for Google."],
    "META": ["Meta invests heavily in the Metaverse.", "Privacy concerns continue to affect Meta."]
}

st.header("Sentiment Analysis")
for stock in selected_stocks:
    st.subheader(f"{stock} News Sentiment")
    sentiments = [get_sentiment(news) for news in news_data.get(stock, [])]
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
    sentiment_label = "Positive" if avg_sentiment > 0 else "Negative" if avg_sentiment < 0 else "Neutral"
    st.write(f"Average Sentiment for {stock}: {sentiment_label}")
    for i, news in enumerate(news_data.get(stock, [])):
        st.write(f"{i + 1}. {news}")

if download_data:
    for stock in selected_stocks:
        data = get_stock_data(stock, timeframe)
        csv = data.to_csv().encode()
        st.download_button(
            label=f"Download {stock} Data as CSV",
            data=csv,
            file_name=f"{stock}_data.csv",
            mime="text/csv",
        )