import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Crypto Tracker", layout="wide")
st.title("📈 Bitcoin Price Tracker with Prediction & News Sentiment")

# -------------------- DATE RANGE --------------------
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=365 * 3)  # last 3 years

st.sidebar.header("Date Range")
start_date = st.sidebar.date_input("Start Date", start_date)
end_date = st.sidebar.date_input("End Date", end_date)

# -------------------- GET DATA --------------------
@st.cache_data
def get_crypto_data(symbol, start, end):
    return yf.download(symbol, start=start, end=end)

crypto = "BTC-USD"
data = get_crypto_data(crypto, start_date, end_date)

# -------------------- DATA CHECK --------------------
if data.empty or len(data.dropna()) < 20:
    st.error("❌ Not enough data returned from yfinance. Try again later or check your date range.")
    st.stop()

st.write("✅ Latest BTC data preview:")
st.dataframe(data.tail())

# -------------------- PRICE CHART --------------------
st.subheader("📊 Bitcoin Price Chart")
st.line_chart(data['Close'])

# -------------------- NEWS + SENTIMENT --------------------
def fetch_news_sentiment(query="bitcoin"):
    url = f"https://news.google.com/rss/search?q={query}+cryptocurrency"
    res = requests.get(url)
    soup = BeautifulSoup(res.content, "xml")
    articles = soup.findAll('item')[:5]

    analyzer = SentimentIntensityAnalyzer()
    results = []
    for article in articles:
        title = article.title.text
        link = article.link.text
        score = analyzer.polarity_scores(title)['compound']
        results.append((title, score, link))
    return results

st.subheader("📰 Trending News & Sentiment")
try:
    news = fetch_news_sentiment("bitcoin")
    for title, score, link in news:
        sentiment = "🔼 Positive" if score > 0 else "🔽 Negative" if score < 0 else "⚖️ Neutral"
        st.markdown(f"- [{title}]({link}) — **{sentiment}** (`{score:.2f}`)")
except Exception as e:
    st.error(f"News fetching failed: {e}")

# -------------------- PREDICT UP/DOWN --------------------
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

    if len(X_train) == 0:
        st.warning("Not enough features after processing. Try a broader date range.")
    else:
        model = LogisticRegression()
        model.fit(X_train, y_train)

        latest_data = data.copy().tail(10)
        latest_return = latest_data['Close'].pct_change().iloc[-1]
        ma5 = latest_data['Close'].rolling(window=5).mean().iloc[-1]
        ma10 = latest_data['Close'].rolling(window=10).mean().iloc[-1]

        if np.isnan([latest_return, ma5, ma10]).any():
            st.warning("Latest data not sufficient for prediction.")
        else:
            pred = model.predict([[latest_return, ma5, ma10]])[0]
            prediction_text = "🔼 Bitcoin will likely go **UP** tomorrow." if pred == 1 else "🔽 Bitcoin will likely go **DOWN** tomorrow."
            st.success(prediction_text)

except Exception as e:
    st.error(f"Prediction failed: {e}")
