import numpy as np
import tensorflow as tf


class DataLoading:
    def __init__(self, filename):
        self.filename = filename
        self.climate_df = None

    def preprocess_data(self):
        if self.climate_df is not None:
            return self.climate_df

        self.climate_df = pd.read_csv(self.filename)
        self.climate_df['Date Time'] = pd.to_datetime(self.climate_df['Date Time'], format="%d.%m.%Y %H:%M:%S")
        self.climate_df['Year'] = self.climate_df['Date Time'].dt.year
        self.climate_df['Month'] = self.climate_df['Date Time'].dt.month
        self.climate_df = self.climate_df.drop_duplicates().reset_index(drop=True)
        return self.climate_df

    def rename_columns(self):
        new_column_names = {
            'p (mbar)': 'Pressure (mbar)',
            'T (degC)': 'Temperature (degC)',
            'Tpot (K)': 'Potential Temperature (K)',
            'Tdew (degC)': 'Dew Point Temperature (degC)',
            'rh (%)': 'Relative Humidity (%)',
            'VPmax (mbar)': 'Maximum Vapor Pressure (mbar)',
            'VPact (mbar)': 'Actual Vapor Pressure (mbar)',
            'VPdef (mbar)': 'Vapor Pressure Deficit (mbar)',
            'sh (g/kg)': 'Specific Humidity (g/kg)',
            'H2OC (mmol/mol)': 'Water Vapor Concentration (mmol/mol)',
            'rho (g/m**3)': 'Air Density (g/m^3)',
            'wv (m/s)': 'Wind Speed (m/s)',
            'max. wv (m/s)': 'Maximum Wind Speed (m/s)',
            'wd (deg)': 'Wind Direction (deg)'
        }

        self.climate_df.rename(columns=new_column_names, inplace=True)


class DataProcessor:
    def __init__(self, window_size, batch_size, shuffle_buffer_size):
        self.window_size = window_size
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size

    def parse_data_from_dataframe(self, df, column):
        times = []
        col_values = []

        count = 0
        for value in df[column]:
            value_float = float(value)
            col_values.append(value_float)
            times.append(int(count))
            count += 1

        return times, col_values

    def train_val_split(self, time, series, time_step):
        time_train = time[:time_step]
        series_train = series[:time_step]
        time_valid = time[time_step:]
        series_valid = series[time_step:]

        return time_train, series_train, time_valid, series_valid

    def windowed_dataset(self, series):
        data = []
        for i in range(len(series) - self.window_size):
            data.append(series[i:i + self.window_size + 1])
        data = np.array(data)

        X = data[:, :-1]
        y = data[:, -1]

        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.shuffle(self.shuffle_buffer_size).batch(self.batch_size).prefetch(1)

        return dataset



