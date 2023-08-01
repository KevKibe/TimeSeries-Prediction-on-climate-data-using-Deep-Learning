import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import tensorflow as tf
from data_prep import DataProcessor, DataLoading
from model import TimeSeriesModel

def main():
    st.title("Climate Time Series Prediction")
    preprocessor = DataLoading('jena_climate_2009_2016.csv')
    climate_df = preprocessor.preprocess_data()
    preprocessor.rename_columns()
    model = tf.keras.models.load_model("rnn_model.h5")

    # Plot the  column against the index (time)
    plt.figure(figsize=(10, 6))
    plt.plot(climate_df['Date Time'], climate_df["Temperature (degC)"], color='salmon')
    plt.xlabel('Year')
    plt.ylabel('Temperature (degC)')
    plt.title(f'Jena Climate Temperature (degC) Data')
    plt.grid(True)

    # display the plot
    st.pyplot(plt)

    st.title("Prediction")

    split_time = 294000
    window_size = 64
    batch_size = 256
    shuffle_buffer_size = 1000
    data_processor = DataProcessor(window_size, batch_size, shuffle_buffer_size)
    times, temperatures = data_processor.parse_data_from_dataframe(climate_df, 'Temperature (degC)')
    time = np.array(times)
    series = np.array(temperatures)
    time_train, series_trainset, time_valid, series_validset = data_processor.train_val_split(time, series, split_time)

    last_timestamp = time_valid[-1]

    # selecting the time duration for future predictions
    time_duration = st.selectbox("Select Time Duration for Predictions", ["6 months", "1 year", "2 years", "3 years"])
    if time_duration == "6 months":
        future_time_steps = int(6 * 30 * 24)  # 6 months (approx. 30 days * 24 hours)
    elif time_duration == "1 year":
        future_time_steps = int(1 * 365 * 24)  # 1 year (365 days * 24 hours)
    elif time_duration == "2 years":
        future_time_steps = int(2 * 365 * 24)  # 2 years (365 days * 24 hours * 2)
    else:
        future_time_steps = int(3 * 365 * 24)  # 3 years (365 days * 24 hours * 3)

    future_time = np.arange(last_timestamp + 1, last_timestamp + 1 + future_time_steps)

    climate_df.set_index('Date Time', inplace=True)

    
    with st.spinner("Forecasting..."):

        timeseries = TimeSeriesModel(window_size=window_size, learning_rate=1e-3)
        future_forecast = timeseries.model_forecast(model, series, window_size).squeeze()
        future_forecast = future_forecast[-future_time_steps:]

    # Display the plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_valid, series_validset, format='-', label='Actual Data', color='blue')
    plt.plot(future_time, future_forecast, format='-', label='Predicted Data (Future)', color='green')
    plt.legend()
    plt.title(f'Actual vs. Predicted Data (Validation and Future) - {time_duration} Forecast')

    st.pyplot(plt)

if __name__ == "__main__":
    main()
