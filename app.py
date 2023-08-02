import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf

from data_prep import DataLoading, DataProcessor
from model import ModelEval


@st.cache_data
def fetch_data():
    preprocessor = DataLoading('jena_climate_2009_2016.csv')
    climate_df = preprocessor.preprocess_data()
    preprocessor.rename_columns()
    model = tf.keras.models.load_model("rnn_model.h5")
    return climate_df, model


def plot_climate_data(climate_df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=climate_df.index, y=climate_df["Temperature (degC)"],
                             mode='lines', name='Temperature (degC)', line=dict(color='salmon')))

    fig.update_layout(title='Jena Climate Temperature (degC) Data',
                      xaxis_title='Year', yaxis_title='Temperature (degC)',
                      width=1000, height=500, showlegend=True)

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')

    st.plotly_chart(fig)


@st.cache_data
def preprocess_data():
    climate_df, model = fetch_data()
    window_size = 64
    data_processor = DataProcessor(window_size, batch_size=256, shuffle_buffer_size=1000)
    times, temperatures = data_processor.parse_data_from_dataframe(climate_df, 'Temperature (degC)')
    time = np.array(times)
    series = np.array(temperatures)
    time_train, series_trainset, time_valid, series_validset = data_processor.train_val_split(climate_df.index,
                                                                                               temperatures,
                                                                                               time_step=294000)
    return model, climate_df, series_validset, time_valid


def plot_future_forecast(model, series, time_valid, future_months):
    last_timestamp = time_valid[-1]
    future_time_steps = future_months * 30 * 24 * 6  # Assuming 30 days per month (24 hours * 6 10-minute intervals per hour)
    future_time = pd.date_range(start=last_timestamp, periods=future_time_steps+1, freq='10T')[1:]

    model_eval = ModelEval()
    future_forecast = model_eval.model_forecast(model, series, window_size=64).squeeze()
    future_forecast = future_forecast[-future_time_steps:]

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(x=time_valid, y=series,
                              mode='lines', name='Actual Data', line=dict(color='salmon')))

    fig2.add_trace(go.Scatter(x=future_time, y=future_forecast,
                              mode='lines', name='Predicted Data (Future)', line=dict(color='green')))

    fig2.update_layout(title='Actual vs. Predicted Data', xaxis_title='Time',
                       yaxis_title='Value', width=1000, height=500)

    y_range_padding = (max(future_forecast) - min(future_forecast)) * 0.1
    fig2.update_yaxes(range=[min(future_forecast) - y_range_padding, max(future_forecast) + y_range_padding])

    st.plotly_chart(fig2)


def streamlit_app():
    model, climate_df, series_validset, time_valid = preprocess_data()

    plot_climate_data(climate_df)

    # Add a slider to select the number of years into the future for forecasting
    future_years = st.slider("Select Years into the Future for Forecasting", 0, 1, 10)
    future_months = future_years * 12

    plot_future_forecast(model=model, series=series_validset, time_valid=time_valid,
                         future_months=future_months)


if __name__ == "__main__":
    streamlit_app()
