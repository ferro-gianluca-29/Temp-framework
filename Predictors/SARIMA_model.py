import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.deterministic import Fourier
from tools.time_series_analysis import  ljung_box_test

from skforecast.ForecasterSarimax import ForecasterSarimax
from skforecast.Sarimax import Sarimax
from skforecast.model_selection_sarimax import backtesting_sarimax
from skforecast.model_selection_sarimax import grid_search_sarimax
import pmdarima
from pmdarima import auto_arima


from tqdm import tqdm
import pickle
from Predictors.Predictor import Predictor

class SARIMA_Predictor(Predictor):
    """
    A class used to predict time series data using Seasonal ARIMA (SARIMA) models.
    """

    def __init__(self, run_mode, target_column=None, period = 24,
                 verbose=False, set_fourier=False):
        """
        Constructs all the necessary attributes for the SARIMA_Predictor object.

        :param run_mode: The mode in which the predictor runs
        :param target_column: The target column of the DataFrame to predict
        :param period: Seasonal period of the SARIMA model
        :param verbose: If True, prints detailed outputs during the execution of methods
        :param set_fourier: Boolean, if true use Fourier transformation on the data
        """

        super().__init__(verbose=verbose)  

        self.run_mode = run_mode
        self.verbose = verbose
        self.target_column = target_column
        self.set_fourier = set_fourier
        self.period = period
        self.SARIMA_order = []
        

    def train_model(self):
        """
        Trains a SARIMAX model using the training dataset and exogenous variables, if specified.

        :return: A tuple containing the trained model, validation metrics, and the index of the last training/validation timestep
        """
        try:    

            d = 0
            D = 0

            # Selection of the model with best AIC score
            """model = auto_arima(
                        y=self.train[self.target_column],
                        start_p=0,
                        start_q=0,
                        max_p=4,
                        max_q=4,
                        seasonal=True,
                        m = self.period,
                        test='adf',
                        d=None,  # Let auto_arima determine the optimal 'd'
                        D=None,
                        trace=True,
                        error_action='warn',  # Show warnings for troubleshooting
                        suppress_warnings=False,
                        stepwise=True
                        )"""

            period = self.period    
            target_train = self.train[self.target_column]


            """order = model.order
            seasonal_order = model.seasonal_order"""

            # for debug
            order = (1,0,1)
            seasonal_order = (1,0,1, 24)
            
            best_order = (order, seasonal_order)
            print(f"Best order found: {best_order}")
            

            self.SARIMA_order = best_order
            print("\nTraining the SARIMAX model...")

            model = Sarimax( order = order,
                                        seasonal_order=seasonal_order,
                                        #maxiter = 500
                                        )

            forecaster = ForecasterSarimax(
                 regressor=model,
             )
            forecaster.fit(y=target_train)    

            residuals = forecaster.regressor.sarimax_res.resid    

            
               
            valid_metrics = None
            
            last_index = self.train.index[-1]
            # Running the LJUNG-BOX test for residual correlation
            #residuals = model.resid()
            #ljung_box_test(residuals)
            print("Model successfully trained.")

            return forecaster, valid_metrics, last_index
        
        except Exception as e:
                print(f"An error occurred during the model training: {e}")
                return None
        

    def test_model(self, forecaster, last_index, forecast_type, output_len, ol_refit = False, period = 24): 
        """
        Tests a SARIMAX model by performing one-step or multi-step ahead predictions, optionally using exogenous variables or applying refitting.

        :param model: The SARIMAX model to be tested
        :param last_index: Index of the last training/validation timestep
        :param forecast_type: Type of forecasting ('ol-one' for open-loop one-step ahead, 'cl-multi' for closed-loop multi-step)
        :param ol_refit: Boolean indicating whether to refit the model after each forecast
        :param period: The period for Fourier terms if set_fourier is true
        :return: A pandas Series of the predictions
        """
        try:    
            print("\nTesting SARIMA model...\n")
            
            self.forecast_type = forecast_type
            test = self.test
            self.steps_ahead = self.test.shape[0]
            full_data = pd.concat([self.train, self.test])
            

            if self.forecast_type == 'ol-one':
                steps = 1
            elif self.forecast_type == 'ol-multi':
                steps = output_len

            predictions = []
                           
            _, predictions = backtesting_sarimax(
                    forecaster            = forecaster,
                    y                     = full_data[self.target_column],
                    initial_train_size    = len(self.train),
                    steps                 = steps,
                    metric                = 'mean_absolute_error',
                    refit                 = False,
                    n_jobs                = "auto",
                    verbose               = True,
                    show_progress         = True
                )

            predictions.rename(columns={'pred': self.target_column}, inplace=True)
            print("Model testing successful.")
            return predictions
                
                
                
        except Exception as e:
            print(f"An error occurred during the model test: {e}")
            return None 
        

    def plot_predictions(self, predictions):
        """
        Plots the SARIMA model predictions against the test data.

        :param predictions: The predictions made by the SARIMA model
        """
        test = self.test[:self.steps_ahead][self.target_column]
        plt.plot(test.index, test, 'b-', label='Test Set')
        plt.plot(test.index, predictions, 'k--', label='ARIMA')
        plt.title(f'SARIMA prediction for feature: {self.target_column}')
        plt.xlabel('Time series index')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

    
