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

# Page config
st.set_page_config(page_title="Bitcoin Price Predictor", layout="wide")
st.title("📈 Bitcoin Price Tracker & Tomorrow's Movement Prediction")

# Date range selector with default 3 years
end_date = datetime.date.today()
start_date_default = end_date - datetime.timedelta(days=365*3)

st.sidebar.header("Select Date Range")
start_date = st.sidebar.date_input("Start Date", start_date_default)
end_date = st.sidebar.date_input("End Date", end_date)

if start_date >= end_date:
    st.error("Error: Start date must be before End date.")
    st.stop()

@st.cache_data
def load_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    return df

# Load BTC data
crypto_symbol = "BTC-USD"
data = load_data(crypto_symbol, start_date, end_date)

if data.empty or len(data.dropna()) < 20:
    st.error("Not enough data to perform prediction. Please select a broader date range.")
    st.stop()

st.subheader("Bitcoin Price Data Preview")
st.dataframe(data.tail())

st.subheader("Bitcoin Closing Price Chart")
st.line_chart(data['Close'])

# Fetch news and analyze sentiment
def fetch_news_sentiment(query="bitcoin"):
    url = f"https://news.google.com/rss/search?q={query}+cryptocurrency"
    res = requests.get(url)
    soup = BeautifulSoup(res.content, "xml")
    articles = soup.find_all('item')[:5]
    analyzer = SentimentIntensityAnalyzer()
    results = []
    for article in articles:
        title = article.title.text
        link = article.link.text
        score = analyzer.polarity_scores(title)['compound']
        results.append((title, score, link))
    return results

st.subheader("Trending News & Sentiment for Bitcoin")
try:
    news_items = fetch_news_sentiment("bitcoin")
    for title, score, link in news_items:
        sentiment = "🔼 Positive" if score > 0 else "🔽 Negative" if score < 0 else "⚖️ Neutral"
        st.markdown(f"- [{title}]({link}) — **{sentiment}** (`{score:.2f}`)")
except Exception as e:
    st.error(f"Failed to fetch news: {e}")

# Prepare features for prediction
def prepare_features(df):
    df = df.copy()
    df['Return'] = df['Close'].pct_change()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df.dropna(inplace=True)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    X = df[['Return', 'MA5', 'MA10']]
    y = df['Target']
    return train_test_split(X, y, test_size=0.2, shuffle=False)

st.subheader("Bitcoin Price Movement Prediction for Tomorrow")

try:
    X_train, X_test, y_train, y_test = prepare_features(data)
    if len(X_train) == 0:
        st.warning("Insufficient data after feature preparation. Try increasing date range.")
    else:
        model = LogisticRegression()
        model.fit(X_train, y_train)

        latest_data = data.tail(10)
        latest_return = latest_data['Close'].pct_change().iloc[-1]
        ma5 = latest_data['Close'].rolling(window=5).mean().iloc[-1]
        ma10 = latest_data['Close'].rolling(window=10).mean().iloc[-1]

        features = np.array([latest_return, ma5, ma10])
        if np.isnan(features).any():
            st.warning("Latest data incomplete, cannot predict.")
        else:
            prediction = model.predict(features.reshape(1, -1))[0]
            if prediction == 1:
                st.success("🔼 Bitcoin is predicted to go UP tomorrow.")
            else:
                st.success("🔽 Bitcoin is predicted to go DOWN tomorrow.")

except Exception as e:
    st.error(f"Prediction error: {e}")

