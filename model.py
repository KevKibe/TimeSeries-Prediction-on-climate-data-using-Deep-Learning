import tensorflow as tf
import matplotlib as plt

class TimeSeriesModel:
    def __init__(self, window_size=64, learning_rate=1e-3):
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.model = self._create_model()

    def _create_uncompiled_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1,
                                   activation='relu', input_shape=[self.window_size, 1]),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1),
        ])
        return model

    def _create_model(self):
        model = self._create_uncompiled_model()
        model.compile(loss=tf.keras.losses.Huber(),
                      optimizer=tf.keras.optimizers.SGD(momentum=0.9, learning_rate=self.learning_rate),
                      metrics=["mae"])
        return model

    def train(self, training_data, epochs=30):
        history = self.model.fit(training_data, epochs=epochs)
        return history

    def test(self, test_data):
        for X_batch, y_batch in test_data.take(1):  # Take one batch
            y_batch = y_batch.numpy().squeeze()

            print("Input (X) Batch:")
            print(X_batch)

            # Generate a prediction
            print(f'Testing model prediction with input of shape {X_batch.shape}...')
            y_pred = self.model.predict(X_batch)

            # Compare the shape of the prediction and the label y (remove dimensions of size 1)
            y_pred_shape = y_pred.squeeze().shape

            assert y_pred_shape == y_batch.shape, (f'Squeezed predicted y shape = {y_pred_shape} '
                                                   f'whereas actual y shape = {y_batch.shape}.')

            # Exit the loop after displaying one sample
            break

        print("Your current architecture is compatible with the windowed dataset! :)")

class ModelEval():
    def compute_metrics(self, true_series, forecast):
        mse = tf.keras.metrics.mean_squared_error(true_series, forecast).numpy()
        mae = tf.keras.metrics.mean_absolute_error(true_series, forecast).numpy()
        return mse, mae

    def model_forecast(self,model, series, window_size):
        ds = tf.data.Dataset.from_tensor_slices(series)
        ds = ds.window(window_size, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(window_size))
        ds = ds.batch(128).prefetch(1)
        forecast = model.predict(ds)
        return forecast

    def plot_series(self, time, series, format="-", start=0, end=None, label=None, color=None):
        plt.plot(time[start:end], series[start:end], format, label=label, color=color)
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.grid(True)
