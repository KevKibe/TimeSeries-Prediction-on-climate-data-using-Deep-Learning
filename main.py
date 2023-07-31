from data_prep import DataProcessor, DataLoading
import numpy as np
from model import TimeSeriesModel
import matplotlib.pyplot as plt


preprocessor = DataLoading('jena_climate_2009_2016.csv')
climate_df = preprocessor.preprocess_data()
preprocessor.rename_columns()

split_time = 294000
window_size = 64
batch_size = 256
shuffle_buffer_size = 1000
data_processor = DataProcessor(window_size, batch_size, shuffle_buffer_size)

# Parse the data and split the dataset
times, temperatures = data_processor.parse_data_from_dataframe(climate_df, 'T (degC)')
time = np.array(times)
series = np.array(temperatures)
time_train, series_trainset, time_valid, series_validset = data_processor.train_val_split(time, series, split_time)

# Create the training dataset
training_data = data_processor.windowed_dataset(series_trainset)

model = TimeSeriesModel(window_size=64, learning_rate=1e-3)
history = model.train(training_data, epochs=30)
model.test(training_data)

forecast = model.model_forecast(series, window_size).squeeze()
forecast = forecast[split_time - window_size:-1]

plt.figure(figsize=(10, 6))
model.plot_series(time_valid, series_validset, label='Actual Data', color='blue')
model.plot_series(time_valid, forecast, label='Predicted Data', color='red')
plt.legend()
plt.title('Actual vs. Predicted Data (Validation)')
plt.show()