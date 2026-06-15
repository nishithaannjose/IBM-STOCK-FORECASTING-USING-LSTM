# IBM STOCK FORECASTING USING LSTM
This project aims to forecast the stock prices of IBM using various machine learning techniques. The primary focus is on predicting the closing prices for the next ten days based on historical data.


## Table of Contents

- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Data](#data)
- [Preprocessing](#preprocessing)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Predictions](#Predictions)
- [Conclusion](#conclusion)
- [Tools Used](#tools-used)

## Introduction

Stock price forecasting is a challenging task due to the volatile nature of financial markets. This project leverages LSTM (Long Short-Term Memory) networks, a type of recurrent neural network (RNN), to predict IBM's stock prices.

## Project Overview

1. **Data Collection**: The project utilizes historical stock price data for IBM.
2. **Preprocessing**: The data is normalized and split into training and testing sets.
3. **Modeling**: An LSTM model is trained on the preprocessed data.
4. **Evaluation**: The model's performance is evaluated on the test data.
5. **Deployment**: The trained model is deployed to make future predictions.

## Data

The dataset contains historical stock prices for IBM, including the following columns:

- Date
- Open
- High
- Low
- Close
- Adj Close
- Volume

## Preprocessing

1. **Normalization**: The features (`Open`, `High`, `Low`, `Close`, `Volume`) are scaled using MinMaxScaler.
2. **Data Splitting**: The normalized data is split into training (80%) and testing (20%) sets without shuffling.
3. **Sequence Generation**: Sequences of 50 time steps are created to serve as input for the LSTM model.

```python
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(stock_data[['open', 'high', 'low', 'volume', 'close']])

train_data, test_data = train_test_split(normalized_data, test_size=0.2, shuffle=False)
train_df = pd.DataFrame(train_data, columns=['open', 'high', 'low', 'volume', 'close'])
test_df = pd.DataFrame(test_data, columns=['open', 'high', 'low', 'volume', 'close']) 
```
## Modeling
An LSTM model is built and trained on the training sequences.

```python
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(50, 5)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=5)
])
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
history = model.fit(train_sequences, train_labels, epochs=200, batch_size=32, validation_data=(test_sequences, test_labels), verbose=1)
```
## Evaluation
The model's predictions are compared against actual values for both training and testing data. Various metrics such as Mean Squared Error (MSE) and Mean Absolute Error (MAE) are used to evaluate the performance.

## Predictions
The trained model is used to predict the closing prices for the next ten days.

```python
latest_prediction = []
last_seq = test_sequences[-1:]

for _ in range(10):
    prediction = model.predict(last_seq)
    latest_prediction.append(prediction)
    last_seq = np.append(last_seq[:, 1:, :], [prediction], axis=1)
```
## Visualizations


