# Predicting Tesla Stock Prices Using X (Twitter) Sentiment Analysis

## Project Overview
This project examines how public sentiment on social media impacts Tesla’s stock performance, given the significant online presence of its CEO, Elon Musk. We integrate sentiment analysis of Tesla-related tweets with traditional financial indicators to predict both next-day stock movements and future price levels. By combining behavioral data from X (formerly Twitter) with historical market data, we apply multiple machine learning models, including LSTM, SVM, Random Forest, and Linear Regression, to explore how real-time sentiment contributes to improved accuracy in financial forecasting.

## Methods
!(methods)[stock_prediction_methods.png]

### Data Preprocessing
Given the large volume and complexity of the raw datasets, extensive preprocessing was performed using Python libraries such as pandas, matplotlib, and spaCy. The steps are summarized as follows:

- Time Alignment: Matched tweet sentiment with daily closing prices (2018–2020).
- Volatility Filtering: Removed extreme outliers (black-swan events) beyond 2× standard deviation of daily returns.
- Sentiment Cleaning:

    1. Processed text using spaCy for tokenization and lemmatization.

    2. Filtered out neutral tweets using a curated list of emotional keywords.
    
- Feature Engineering:

    1. Created lag features for past 3-day sentiment trends.

    2. Generated technical indicators (rolling mean, EMA, volatility).

    3. Normalized data via z-score and min-max scaling.
 

### Sentiment Analysis
To understand how people felt about Tesla, we analyzed tweets using two tools: TextBlob and VADER. TextBlob helped capture the general tone of each tweet, while VADER picked up on stronger emotions such as excitement or frustration by recognizing punctuation and wording. We then calculated the average daily sentiment score to show how public mood toward Tesla changed over time. Since the stock market is closed on weekends, we combined weekend tweets with Monday’s data so that the overall mood from those days was still reflected in our analysis.

### Stock Price Prediction
We used several machine learning models to see how well sentiment and stock data could predict Tesla’s stock movements. For short-term direction (up or down), we tested models such as Logistic Regression, Support Vector Machine (SVM), and Random Forest, which are good at finding patterns in data. To predict the actual price levels, we used Linear Regression (refined by feature selection) and a deep learning model called LSTM (Long Short-Term Memory), which can recognize trends that unfold over time. Each model was trained on historical data and then tested on future data to check how accurately it could capture both the general trend and day-to-day changes in Tesla’s stock price.

## Results

| Model               | Task           | Metric               | Score      |
| ------------------- | -------------- | -------------------- | ---------- |
| Logistic Regression | Classification | Accuracy             | 54.46%     |
| SVC (Linear)        | Classification | Accuracy             | 52.48%     |
| SVC (RBF)           | Classification | Accuracy             | 55.45%     |
| Random Forest       | Classification | Accuracy             | 58.42%     |
| Linear Regression   | Regression     | MAE                  | 0.362      |
| LSTM Hybrid         | Regression     | Directional Accuracy | 56.57%     |

Among all the models tested, the Random Forest performed best in predicting short-term stock movement, showing the strongest ability to capture daily direction changes. The LSTM model was more effective at recognizing long-term patterns and overall trends, though it struggled with the high volatility of Tesla’s stock. Overall, incorporating sentiment features from tweets provided some improvement—helping the models better understand general market mood and reinforce trend predictions, even if it didn’t lead to highly precise forecasts.

## Discussion

This study demonstrates that social sentiment can meaningfully complement financial indicators in stock forecasting. While models like Random Forest and LSTM modestly outperform traditional baselines, results highlight that social media sentiment alone cannot fully predict price movements, especially during volatile periods. Moving forward, the model could be improved by integrating more advanced sentiment tools like FinBERT, news headlines, or macroeconomic indicators such as inflation and interest rates. Including these broader data sources would likely provide a more complete picture of investor behavior and improve prediction accuracy during volatile market periods.

## Tools & Libraries

Python: pandas, numpy, scikit-learn, matplotlib, seaborn, keras, tensorflow

NLP: spaCy, VADER, TextBlob

Data: (Kaggle Tesla Tweets)[https://www.kaggle.com/datasets/hindy51/tesla-tweets/data] & Yahoo Finance Stock Data
