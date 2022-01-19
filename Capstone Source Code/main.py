import streamlit as st
from datetime import date

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as data

START = "2012-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('WGU Capstone:Stock Prediction App')

user_input = st.text_input('Enter Stock Ticker (Must be in all CAPS)', 'TSLA')
df = data.DataReader(user_input, 'yahoo', START)
#stocks = ('FB', 'AAPL', 'AMZN', 'NFLX', 'GOOG', 'TSLA')
#selected_stock = st.selectbox('Select a stock for prediction', stocks)

predict_years = st.slider('Years of prediction:', 1, 10)
period = predict_years * 365

#Describing the data
st.subheader(f'Historical Data for : {user_input} from 2012-2021')
st.write(df.describe())


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text('Loading data.. Please Wait.....')
data = load_data(user_input)
data_load_state.text('Loading data... Successfully loaded!')

st.subheader(f'Raw data for : {user_input}')
st.write(data.head())
st.write(data.tail())


# Plotting the raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Graph of data with Range-slider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


plot_raw_data()

# Predicting the data forecast with Prophet.
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

model = Prophet()
model.fit(df_train)
future = model.make_future_dataframe(periods=period)
forecast = model.predict(future)

# Show and plot forecast
st.subheader('Prediction data')
st.write(forecast.head())
st.write(forecast.tail())


st.subheader(f'Blue line is the predictive line for the chart below:')
st.write(f'Prediction plot for {predict_years} year(s)')
fig1 = plot_plotly(model, forecast)
st.plotly_chart(fig1)


st.write("Prediction components based on trend and time")
fig2 = model.plot_components(forecast)
st.write(fig2)

#Streamlit Visualization
st.subheader("Closing Price vs Time Chart")
fig = plt.figure(figsize = (10,5))
plt.plot(df.Close)
st.pyplot(fig)

#For 100days moving average
st.subheader("Closing Price vs Time Chart with 100 days Moving Average")
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (10,5))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

#For 200days moving average
st.subheader("Closing Price vs Time Chart with 100 Days Moving Average & 200 days Moving Average")
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (10,5))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)

# Splitting the Data into Training and Testing
#75% for training and 25% for testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.75)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.75): int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)


#Load the predictive model
model = load_model('keras_model.h5')

#Making the predictions for testing part

past_100_days = data_training.tail(100) #data for the past 100 days
final_df = past_100_days.append(data_testing, ignore_index=True)

#scaling down the data

input_data = scaler.fit_transform(final_df)

#Testing for the data
x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

#Converting into arrays
x_test, y_test = np.array(x_test), np.array(y_test)

#For making predictions
y_predicted = model.predict(x_test)

#scaler
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#Final Visualization
st.subheader("Predicted Price vs Original")
fig2 = plt.figure(figsize=(10,5))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Prediction Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)