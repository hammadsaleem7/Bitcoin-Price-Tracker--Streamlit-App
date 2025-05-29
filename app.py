import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="Crypto Tracker", layout="wide")
st.title("📈 Crypto Price Tracker with News Sentiment")

# User input
crypto = st.selectbox("Choose Cryptocurrency", ["BTC-USD", "ETH-USD", "BNB-USD"])
start_date = st.date_input("Start Date", datetime.date(2022, 1, 1))
end_date = st.date_input("End Date", datetime.date.today())

# Get data
@st.cache_data
def get_crypto_data(symbol, start, end):
    return yf.download(symbol, start=start, end=end)

data = get_crypto_data(crypto, start_date, end_date)

# Chart
st.subheader("📊 Price Chart")
st.line_chart(data['Close'])

# Trending news
def fetch_news_sentiment(query="bitcoin"):
    url = f"https://news.google.com/rss/search?q={query}+cryptocurrency"
    res = requests.get(url)
    soup = BeautifulSoup(res.content, "xml")
    articles = soup.findAll('item')[:5]

    analyzer = SentimentIntensityAnalyzer()
    results = []
    for article in articles:
        title = article.title.text
        score = analyzer.polarity_scores(title)['compound']
        results.append((title, score, article.link.text))
    return results

st.subheader("📰 Trending News & Sentiment")
news = fetch_news_sentiment(crypto.split("-")[0].lower())
for title, score, link in news:
    st.markdown(f"- [{title}]({link}) — Sentiment: `{score}`")

st.markdown("---")
st.caption("This version excludes ML prediction to remain Streamlit Cloud compatible.")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.subheader("📉 Bitcoin Movement Prediction (Up/Down)")

def prepare_features(data):
    df = data.copy()
    df['Return'] = df['Close'].pct_change()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df.dropna(inplace=True)

    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    X = df[['Return', 'MA5', 'MA10']]
    y = df['Target']
    return train_test_split(X, y, test_size=0.2, shuffle=False)

try:
    X_train, X_test, y_train, y_test = prepare_features(data)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    latest_data = data.copy().tail(10)
    latest_return = latest_data['Close'].pct_change().iloc[-1]
    ma5 = latest_data['Close'].rolling(window=5).mean().iloc[-1]
    ma10 = latest_data['Close'].rolling(window=10).mean().iloc[-1]

    pred = model.predict([[latest_return, ma5, ma10]])[0]
    prediction_text = "🔼 Bitcoin will likely go UP tomorrow." if pred == 1 else "🔽 Bitcoin will likely go DOWN tomorrow."
    st.success(prediction_text)

except Exception as e:
    st.warning("Not enough data for prediction. Try expanding the date range.")

