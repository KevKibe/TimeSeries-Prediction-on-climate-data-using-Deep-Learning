## TimeSeries-Prediction-on-climate-data-using-Deep-Learning

![image](https://github.com/KevKibe/TimeSeries-Prediction-on-climate-data-using-Deep-Learning/assets/86055894/1bddae41-aefe-4f53-9191-58c547895786)


## Description


- This is a time-series forecasting application on climate data specifically temperature data using a deep learning model
- The application has a myriad of uses such as getting climate data forecasts.
- You can try out this interactive [streamlt application](https://time-series-using-rnn.streamlit.app/) for forecasting future temperature data.

## Dataset
- The dataset I used is the [Jena Climate dataset](https://www.kaggle.com/datasets/mnassrib/jena-climate) is made up of 14 different quantities (such air temperature, atmospheric pressure, humidity, wind direction, and so on) were recorded every 10 minutes, over several years. This dataset covers data from January 1st 2009 to December 31st 2016.

## Usage
- This is how you can train the model as I have in the notebook or on an IDE on any of the columns in the dataset.
- Clone the repository: `git clone https://github.com/KevKibe/KevKibe/TimeSeries-Prediction-on-climate-data-using-Deep-Learning`
- Navigate to the project directory: `cd TimeSeries-Prediction-on-climate-data-using-Deep-Learning`
- Install the dependencies: `pip install -r requirements.txt`
- In the main.py file you can change the column name `Temperature (degC)` in this line `times, temperatures = data_processor.parse_data_from_dataframe(climate_df, 'Temperature (degC)')` 
  to train a model on any column you wish.
- Run the command `python main.py` which will train the model and then return the MAE and MSE.
- You can go ahead and plot the predicted vs the actual validation data

## Model
-This is the model consists of a 1D convolutional layer, followed by three LSTM (Long Short-Term Memory) layers, and three dense layers, with a single unit output layer for prediction.

```
model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1,
                               activation='relu', input_shape=[64,1]),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
model.summary()
```

## Results

- The model achieved a Mean Squared Error of `0.06` and Mean Absolute Error of `0.18`.

**:zap: I'm currently open for roles in Data Science, Machine Learning, NLP and Computer Vision.**

