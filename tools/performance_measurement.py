import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, r2_score
from sktime.performance_metrics.forecasting import mean_squared_percentage_error

class PerfMeasure:

    def __init__(self, model_type, model, test, target_column, forecast_type):

        self.model_type = model_type
        self.model = model
        self.test = test
        self.target_column = target_column
        self.predictions = list()
        self.forecast_type = forecast_type            
        self.steps_ahead = self.test.shape[0]
    
    def get_performance_metrics(self, test, predictions):
        """
        Calculates a set of performance metrics for model evaluation.

        :param test: The actual test data.
        :param predictions: Predicted values by the model.
        :param naive: Boolean flag to indicate if the naive predictions should be considered.
        :return: A dictionary of performance metrics including MSE, RMSE, MAPE, MSPE, MAE, and R-squared.
        """
        try:
            match self.model_type:
                
                case 'ARIMA'|'SARIMA'|'SARIMAX'|'NAIVE':
                    
                    # Handle zero values in test_data for MAPE and MSPE calculations
                    test_zero_indices = np.where(test == 0)
                    test.iloc[test_zero_indices] = 0.00000001

                    pred_zero_indices = np.where(predictions == 0)
                    predictions.iloc[pred_zero_indices] = 0.00000001
                    
                case 'LSTM'|'XGB':
                    test.replace(0, 0.00000001, inplace=True)
                    predictions.replace(0, 0.00000001, inplace=True)
                    
              

            performance_metrics = {}
            
            rmse = root_mean_squared_error(test, predictions)

            mse = rmse ** 2

            performance_metrics['MSE'] = mse
            performance_metrics['RMSE'] = rmse
            performance_metrics['MAPE'] = mean_absolute_percentage_error(test, predictions)
            performance_metrics['MSPE'] = mean_squared_percentage_error(test, predictions)
            performance_metrics['MAE'] = mean_absolute_error(test, predictions)
            performance_metrics['R_2'] = r2_score(test, predictions)
            return performance_metrics
        
        except Exception as e:
            print(f"An error occurred during performance measurement: {e}")
            return None
