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
