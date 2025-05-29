# Bitcoin Price Tracker & Movement Prediction

![Streamlit](https://img.shields.io/badge/Streamlit-App-blue) ![Python](https://img.shields.io/badge/Python-3.8+-yellow) ![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

This is a simple **Streamlit** web application that tracks Bitcoin price data, analyzes trending cryptocurrency news sentiment, and predicts whether the price of Bitcoin will go **up or down** the next day using basic machine learning techniques.

---

## Features

* 📈 Fetches and displays historical Bitcoin (BTC-USD) price data with customizable date ranges.
* 📰 Scrapes latest Bitcoin-related news headlines and evaluates sentiment using VADER sentiment analysis.
* 🔮 Predicts Bitcoin’s next-day price movement (up or down) based on past price returns and moving averages using Logistic Regression.
* ✅ Interactive and user-friendly interface for quick insights.

---

## How It Works

1. **Data Collection:** Uses the `yfinance` library to download historical Bitcoin price data.
2. **Feature Engineering:** Computes daily returns and moving averages (5-day and 10-day) as features.
3. **News Sentiment:** Scrapes Google News RSS feeds related to Bitcoin and scores headlines with VADER sentiment analyzer.
4. **Model Training:** Trains a Logistic Regression classifier to predict if the price will rise or fall the following day.
5. **Prediction:** Uses the latest available data to predict tomorrow’s movement.
6. **Visualization:** Shows price charts and sentiment insights directly in the web app.

---

## Installation

Make sure you have Python 3.8+ installed.

```bash
git clone https://github.com/your-username/bitcoin-price-predictor.git
cd bitcoin-price-predictor
pip install -r requirements.txt
streamlit run app.py
```

---

## Usage

* Select the desired date range for historical data in the sidebar.
* View the Bitcoin closing price chart.
* See recent news headlines with sentiment indicators.
* Get a prediction on whether Bitcoin’s price will go up or down tomorrow.

---

## Dependencies

* streamlit
* yfinance
* pandas
* numpy
* vaderSentiment
* requests
* beautifulsoup4
* lxml
* scikit-learn

---

## Notes

* Prediction is based on simple logistic regression and basic features — for educational/demo purposes only.
* News sentiment is based on headline analysis, which may not fully reflect market sentiment.
* Accuracy depends on quality and quantity of data; ensure a broad date range for better results.

---

## Screenshots

![Uploading image.png…]()

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

* [yfinance](https://github.com/ranaroussi/yfinance) for financial data
* [Streamlit](https://streamlit.io/) for easy app deployment
* [VADER Sentiment](https://github.com/cjhutto/vaderSentiment) for sentiment analysis

---

If you want, I can help you generate this file in markdown format or tailor it to your exact repo details!
