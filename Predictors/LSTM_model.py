from datetime import datetime
import pandas as pd
import numpy as np
import datetime as datetime
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  
from keras.layers import Dense,Flatten,Dropout,SimpleRNN,LSTM
from keras.models import Sequential
from keras.metrics import MeanAbsoluteError, MeanAbsolutePercentageError, RootMeanSquaredError   

from keras.optimizers import Adam
from keras.losses import MeanSquaredError, MeanAbsoluteError
from keras.callbacks import EarlyStopping

from skforecast.ForecasterRnn import ForecasterRnn
from skforecast.ForecasterRnn.utils import create_and_compile_model
from skforecast.model_selection_multiseries import backtesting_forecaster_multiseries

import warnings
warnings.filterwarnings("ignore")
from Predictors.Predictor import Predictor

import ephem
import pytz


class LSTM_Predictor(Predictor):
    """
    A class used to predict time series data using Long Short-Term Memory (LSTM) networks.
    """

    def __init__(self, run_mode, target_column=None, 
                 verbose=False, input_len=None, output_len=None, seasonal_model=False, period=24):
        """
        Constructs all the necessary attributes for the LSTM_Predictor object.

        :param run_mode: The mode in which the predictor runs
        :param target_column: The target column of the DataFrame to predict
        :param verbose: If True, prints detailed outputs during the execution of methods
        :param input_len: Number of past observations to consider for each input sequence
        :param output_len: Number of future observations to predict
        :param seasonal_model: Boolean, if true include seasonal adjustments like Fourier features
        :param set_fourier: Boolean, if true use Fourier transformation on the data
        """
        super().__init__(verbose=verbose)  

        self.run_mode = run_mode
        self.verbose = verbose
        self.target_column = target_column
        self.input_len = input_len
        self.output_len = output_len
        self.seasonal_model = seasonal_model
        self.period = period
        

    def train_model(self):
        """
        Trains an LSTM model using the training and validation datasets.

        :param X_train: Input data for training
        :param y_train: Target variable for training
        :param X_valid: Input data for validation
        :param y_valid: Target variable for validation
        :return: A tuple containing the trained LSTM model and validation metrics
        """
        try:

            print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

            # EXOGENOUS VARIABLES
            

            # Define the location and timezones
            latitude = '32.7157'
            longitude = '-117.1611'
            local_tz = pytz.timezone('America/Los_Angeles')

            # Initialize the observer and sun
            obs = ephem.Observer()
            obs.lat = latitude
            obs.lon = longitude
            sun = ephem.Sun()

            def get_sun_times(date):
                # Convert the pandas Timestamp to the correct format for ephem
                obs.date = date.to_pydatetime()  # Convert to Python datetime object without timezone
                sunrise_utc = obs.next_rising(sun, use_center=True).datetime()
                sunset_utc = obs.next_setting(sun, use_center=True).datetime()
                
                # Convert to local timezone with DST
                sunrise_local = pytz.UTC.localize(sunrise_utc).astimezone(local_tz)
                sunset_local = pytz.UTC.localize(sunset_utc).astimezone(local_tz)
                return sunrise_local.hour + sunrise_local.minute/60, sunset_local.hour + sunset_local.minute/60
        
            for df in (self.train, self.valid, self.test):
            
                df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
                df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
                df['week_of_year_sin'] = np.sin(2 * np.pi * df.index.isocalendar().week / 52).astype('float64')
                df['week_of_year_cos'] = np.cos(2 * np.pi * df.index.isocalendar().week / 52).astype('float64')
                df['week_day_sin'] = np.sin(2 * np.pi * df.index.weekday / 7)
                df['week_day_cos'] = np.cos(2 * np.pi * df.index.weekday / 7)
                df['hour_day_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
                df['hour_day_cos'] = np.cos(2 * np.pi * df.index.hour / 24)


                df['roll_mean_1_day'] = df[self.target_column].rolling(window=self.period, min_periods=1).mean()
                df['roll_mean_7_day'] = df[self.target_column].rolling(window=self.period*7, min_periods=1).mean()
                df['roll_max_1_day'] = df[self.target_column].rolling(window=self.period, min_periods=1).max()
                df['roll_min_1_day'] = df[self.target_column].rolling(window=self.period, min_periods=1).min()
                df['roll_max_7_day'] = df[self.target_column].rolling(window=self.period*7, min_periods=1).max()
                df['roll_min_7_day'] = df[self.target_column].rolling(window=self.period*7, min_periods=1).min()

                df[['sunrise_hour', 'sunset_hour']] = df.index.to_series().apply(get_sun_times).tolist()
                df['sunrise_hour'] = df['sunrise_hour'].astype(float)
                df['sunset_hour'] = df['sunset_hour'].astype(float)


                df['daylight_hours'] = df['sunset_hour'] - df['sunrise_hour']



            exog_features = [
                            self.target_column,
                            'month_sin', 
                            'month_cos',
                            'week_of_year_sin',
                            'week_of_year_cos',
                            'week_day_sin',
                            'week_day_cos',
                            'hour_day_sin',
                            'hour_day_cos',

                            'roll_mean_1_day',
                            'roll_mean_7_day',
                            'roll_max_1_day',
                            'roll_min_1_day',
                            'roll_max_7_day',
                            'roll_min_7_day',
                            'sunrise_hour',
                            'sunset_hour',
                            'daylight_hours'
                        ]

            
            model = create_and_compile_model(
                        series = self.train[exog_features], # Series used as predictors
                        levels = self.target_column,               # Target column to predict
                        lags = self.input_len,
                        steps = self.output_len,
                        recurrent_layer = "LSTM",
                        activation = "tanh",
                        recurrent_units = [96,96,96],
                        dense_units = 96,
                        optimizer = Adam(learning_rate=0.01), 
                        loss = MeanSquaredError()
                                            )
            
            model.summary()


            forecaster = ForecasterRnn(
                                regressor = model,
                                levels = self.target_column,
                                transformer_series = None, #scaling already written in preprocesing file
                                fit_kwargs={
                                    "epochs": 200,  # Number of epochs to train the model.
                                    "batch_size": 1000,  # Batch size to train the model.
                                   
                                },
                                    )    
            
            #forecaster.fit(self.train[[self.target_column]]) not necessary if backtest is done (backtest includes training)

            return forecaster
        
        except Exception as e:
            print(f"An error occurred during the model training: {e}")
            return None
        
    def test_model(self, forecaster):
        
        try:

            exog_features = [
                            self.target_column,
                            'month_sin', 
                            'month_cos',
                            'week_of_year_sin',
                            'week_of_year_cos',
                            'week_day_sin',
                            'week_day_cos',
                            'hour_day_sin',
                            'hour_day_cos',

                            'roll_mean_1_day',
                            'roll_mean_7_day',
                            'roll_max_1_day',
                            'roll_min_1_day',
                            'roll_max_7_day',
                            'roll_min_7_day',
                            'sunrise_hour',
                            'sunset_hour',
                            'daylight_hours'
                        ]

            full_data = pd.concat([self.train, self.valid, self.test])

            _, predictions = backtesting_forecaster_multiseries(
                                    forecaster = forecaster,
                                    steps = self.output_len,
                                    series=full_data[exog_features],
                                    levels=forecaster.levels,
                                    initial_train_size=len(self.train) + len(self.valid), # Training + Validation Data
                                    metric="mean_absolute_error",
                                    verbose=False, # Set to True for detailed information
                                    refit=False,
                                )
            return predictions
        
        except Exception as e:
            print(f"An error occurred during the model test: {e}")
            return None
        

    def unscale_data(self, predictions, y_test, folder_path):
        
        """
        Unscales the predictions and test data using the scaler saved during model training.

        :param predictions: The scaled predictions that need to be unscaled
        :param y_test: The scaled test data that needs to be unscaled
        :param folder_path: Path to the folder containing the scaler object
        """
        # Load scaler for unscaling data
        with open(f"{folder_path}/scaler.pkl", "rb") as file:
            scaler = pickle.load(file)
        
        # Unscale predictions
        predictions = predictions.to_numpy().reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions) 
        predictions = predictions.flatten() 
        # Unscale test data
        y_test = pd.DataFrame(y_test)
        y_test = scaler.inverse_transform(y_test)
        y_test = pd.Series(y_test.flatten())

        return predictions, y_test                                
           

    def plot_predictions(self, predictions):
        """
        Plots the LSTM model predictions against the test data.

        :param predictions: The predictions made by the LSTM model
        """
        test = self.test[self.target_column]
        plt.plot(test.index, test, 'b-', label='Test Set')
        plt.plot(test.index, predictions, 'k--', label='LSTM')
        plt.title(f'LSTM prediction for feature: {self.target_column}')
        plt.xlabel('Time series index')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()