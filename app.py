
import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.express as px


def main():
    st.title("Climate Time Series Prediction")
    preprocessor = ClimateDataPreprocessor('jena_climate_2009_2016.csv')

    # Preprocess the data and rename columns
    climate_df = preprocessor.preprocess_data()
    preprocessor.rename_columns()


    y_axis_column = st.selectbox("Select Y-axis column:", climate_df.columns)

    # Plot the selected column against the index (time)
    plt.figure(figsize=(10, 6))
    plt.plot(climate_df['Date Time'], climate_df[y_axis_column], color='salmon')
    plt.xlabel('Year')
    plt.ylabel(y_axis_column)
    plt.title(f'Jena Climate {y_axis_column} Data')
    plt.grid(True)

    # Display the plot in Streamlit
    st.pyplot(plt)

    st.title("Prediction")




if __name__ == "__main__":
    main()
