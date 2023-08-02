from data_prep import DataProcessor, DataLoading
import numpy as np
from model import TimeSeriesModel, ModelEval
import matplotlib.pyplot as plt

#Data Loading 
file_path = 'jena_climate_2009_2016.csv'
preprocessor = DataLoading(file_path)
climate_df = preprocessor.preprocess_data()
preprocessor.rename_columns()

#setting up variables for proprocessing
split_time = 294000
window_size = 64
batch_size = 256
shuffle_buffer_size = 1000
data_processor = DataProcessor(window_size, batch_size, shuffle_buffer_size)

# Parse the data and split the dataset
times, temperatures = data_processor.parse_data_from_dataframe(climate_df, 'Temperature (degC)')
time = np.array(times)
series = np.array(temperatures)
time_train, series_trainset, time_valid, series_validset = data_processor.train_val_split(climate_df.index, series, split_time)

# Create the training dataset
training_data = data_processor.windowed_dataset(series_trainset)

#model training and testing
model = TimeSeriesModel(window_size=64, learning_rate=1e-3)
history = model.train(training_data, epochs=30)
model.test(training_data)

#forecasting and evaluation on validation set
forecast = ModelEval.model_forecast(series, window_size).squeeze()
forecast = forecast[split_time - window_size:-1]
mse, mae = ModelEval.compute_metrics(series_validset, forecast)
print(f"mse: {mse:.2f}, mae: {mae:.2f} for forecast")



#THIS IS AN ADDITIONAL SCRIPT TO PLOT THE ACTUAL DATA VS PREDICTED DATA ON THE VALIDSET 
#YOU CAN RUN THIS ON A NOTEBOOK
# plt.figure(figsize=(10, 6))
# ModelEval.plot_series(time_valid, series_validset, label='Actual Data', color='blue')
# ModelEval.plot_series(time_valid, forecast, label='Predicted Data', color='red')
# plt.legend()
# plt.title('Actual vs. Predicted Data (Validation)')
# plt.show()