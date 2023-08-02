import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import tensorflow as tf
from data_prep import DataProcessor, DataLoading
from model import TimeSeriesModel, ModelEval
import plotly.graph_objects as go

# @st.cache
def fetch_data():
    preprocessor = DataLoading('jena_climate_2009_2016.csv')
    climate_df = preprocessor.preprocess_data()
    preprocessor.rename_columns()
    model = tf.keras.models.load_model("rnn_model.h5")
    return climate_df, model

def plot_climate_data(climate_df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=climate_df['Date Time'], y=climate_df["Temperature (degC)"],
                             mode='lines', name='Temperature (degC)', line=dict(color='salmon')))

    fig.update_layout(title='Jena Climate Temperature (degC) Data',
                      xaxis_title='Year', yaxis_title='Temperature (degC)',
                      width=1000, height=500, showlegend=True)

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')

    # Display the plot
    st.plotly_chart(fig)




# Main Streamlit app
def streamlit_app():
    
    climate_df, model = fetch_data()
    # Plot climate data
    plot_climate_data(climate_df)

    #Preprocess data
    window_size = 64
    data_processor = DataProcessor(window_size, batch_size=256, shuffle_buffer_size=1000)
    times, temperatures = data_processor.parse_data_from_dataframe(climate_df, 'Temperature (degC)')
    time = np.array(times)
    series = np.array(temperatures)
    time_train, series_trainset, time_valid, series_validset = data_processor.train_val_split(climate_df.index, series, time_step=294000)

    # Select time duration for future predictions
    # time_duration = st.selectbox("Select Time Duration for Predictions", ["6 months", "1 year", "2 years", "3 years"])

    # Perform forecasting
    # with st.spinner("Forecasting..."):
    def plot_future_forecast(model, series, time_valid, window_size, future_months):
        last_timestamp = time_valid[-1]
        future_time_steps = future_months * 30 * 24 * 6  # Assuming 30 days per month (24 hours * 6 10-minute intervals per hour)
        future_time = pd.date_range(start=last_timestamp, periods=future_time_steps+1, freq='10T')[1:]
        window_size = 64
        data_processor = DataProcessor(window_size, batch_size=256, shuffle_buffer_size=1000)
        times, temperatures = data_processor.parse_data_from_dataframe(climate_df, 'Temperature (degC)')
        time = np.array(times)
        series = np.array(temperatures)
        time_train, series_trainset, time_valid, series_validset = data_processor.train_val_split(climate_df.index, series, time_step=294000)
        future_forecast = ModelEval.model_forecast(model, series, window_size = 64).squeeze()
        future_forecast = future_forecast[-future_time_steps:]

        fig2 = go.Figure()

        fig2.add_trace(go.Scatter(x=time_valid, y=series, mode='lines', name='Actual Data', line=dict(color='salmon')))

        fig2.add_trace(go.Scatter(x=future_time, y=future_forecast, mode='lines', name='Predicted Data (Future)', line=dict(color='green')))

        fig2.update_layout(title='Actual vs. Predicted Data', xaxis_title='Time', yaxis_title='Value', width=1000, height=500)

        y_range = [min(min(series), min(future_forecast)) - 1, max(max(series), max(future_forecast)) + 1]
        fig2.update_yaxes(range=y_range)

        st.plotly_chart(fig2)

    future_months = 36
    plot_future_forecast(model, series_validset, time_valid, window_size, future_months)

# Run the Streamlit app
if __name__ == "__main__":
    streamlit_app()
