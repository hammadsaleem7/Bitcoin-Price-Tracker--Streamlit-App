import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import datetime
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# --------------------- HEADER ---------------------
st.set_page_config(page_title="Crypto Predictor", layout="wide")
st.title("📈 Cryptocurrency Price Prediction & News Sentiment")

# --------------------- USER INPUT ---------------------
crypto = st.selectbox("Choose Cryptocurrency", ["BTC-USD", "ETH-USD", "BNB-USD"])
start_date = st.date_input("Start Date", datetime.date(2022, 1, 1))
end_date = st.date_input("End Date", datetime.date.today())

# --------------------- DATA FETCHING ---------------------
@st.cache_data
def get_crypto_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    return df

data = get_crypto_data(crypto, start_date, end_date)

# --------------------- LSTM MODEL PLACEHOLDER ---------------------
def load_pretrained_lstm():
    # Simulate predictions with random walk for demo
    forecast = data['Close'].values[-1] + np.random.randn(7)
    return forecast

# --------------------- NEWS SCRAPING ---------------------
def fetch_news_sentiment(keyword="bitcoin"):
    url = f"https://news.google.com/rss/search?q={keyword}+cryptocurrency"
    res = requests.get(url)
    soup = BeautifulSoup(res.content, "xml")
    articles = soup.findAll('item')[:5]

    analyzer = SentimentIntensityAnalyzer()
    news_summary = []
    for article in articles:
        title = article.title.text
        link = article.link.text
        score = analyzer.polarity_scores(title)['compound']
        news_summary.append({"title": title, "score": score, "link": link})
    return news_summary

# --------------------- DISPLAY DATA ---------------------
st.subheader("📊 Historical Price Data")
st.line_chart(data['Close'])

# --------------------- PRICE PREDICTION ---------------------
st.subheader("🔮 Price Prediction (Next 7 Days)")
predicted = load_pretrained_lstm()
for i, val in enumerate(predicted, 1):
    st.write(f"Day {i}: ${val:.2f}")

# --------------------- NEWS & SENTIMENT ---------------------
st.subheader("📰 Trending News & Sentiment")
news = fetch_news_sentiment(crypto.split('-')[0].lower())
for item in news:
    st.markdown(f"- [{item['title']}]({item['link']}) — Sentiment Score: `{item['score']}`")

# --------------------- FOOTER ---------------------
st.markdown("---")
st.caption("This is a demo. For educational use only.")

