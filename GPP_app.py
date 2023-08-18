import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import boxcox
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Load the data
gold = pd.read_csv('Gold_data.csv', header=0)
gold.columns = ['date', 'price']
gold['date'] = pd.to_datetime(gold['date'], format='%Y-%m-%d')
gold = gold.set_index('date')

# Apply Box-Cox transformation
data_boxcox = pd.Series(boxcox(gold['price'], lmbda=0), gold.index)

# Take the difference of the transformed data
data_boxcox_diff = pd.Series(data_boxcox - data_boxcox.shift(), index=gold.index)
data_boxcox_diff.dropna(inplace=True)

# Split data into train and test sets
train_len = 1746
train_data_boxcox = data_boxcox.iloc[:train_len]
test_data_boxcox = data_boxcox.iloc[train_len:]

# Fit the SARIMAX model
model = SARIMAX(train_data_boxcox, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
model_fit = model.fit()

# Forecast for the next 30 days after the test data
forecast_start = test_data_boxcox.index.min()
forecast_end = test_data_boxcox.index.max() + pd.DateOffset(days=30)
exog_forecast = gold.loc[forecast_start:forecast_end]

y_hat_sarimax = data_boxcox_diff.copy()
y_hat_sarimax['sarimax_forecast_boxcox'] = model_fit.predict(forecast_start, forecast_end, exog=exog_forecast)
y_hat_sarimax['sarimax_forecast'] = np.exp(y_hat_sarimax['sarimax_forecast_boxcox'])

# Retrieve the forecast for the next 30 days
forecast = y_hat_sarimax['sarimax_forecast'][-30:]

# Streamlit app
st.set_page_config(page_title="Gold Price Forecast", layout="wide")

# Page Title and Description
st.title("Gold Price Forecast")
st.markdown("This app provides a forecast of the gold price for the next 30 days.")

# Display Forecast
st.header("Forecast for the next 30 days")
st.dataframe(forecast.reset_index().rename(columns={"index": "Date", "sarimax_forecast": "Predicted Gold Price"}))



# Get Gold Price for a Specific Date
st.header("Get Gold Price for a Specific Date")
user_date = st.date_input("Select a date")
user_date = pd.to_datetime(user_date)

if user_date in gold.index:
    gold_price = gold.loc[user_date, 'price']
    st.success(f"The gold price on {user_date.strftime('%Y-%m-%d')} is {gold_price}")
elif user_date > forecast_end:
    st.warning("Gold price data is not available for the selected date.")
else:
    forecast_price = y_hat_sarimax['sarimax_forecast'].loc[user_date]
    st.info(f"The forecasted gold price on {user_date.strftime('%Y-%m-%d')} is {forecast_price}")
