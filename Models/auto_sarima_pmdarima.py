import warnings
from itertools import product
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from scipy.fft import fft
from statsmodels.tools.eval_measures import mse
import pmdarima as pm
from ConfidentIntervals.ConfidentIntervals import ConfidentIntervals
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, root_mean_squared_error, r2_score


class AutoArima:
    def __init__(self):
        self.stepwise_mode = None
        self.model_df = pd.DataFrame()
        # Time | Rw_Data | Y_Forecasted| Lower_CI | Upper_CI | Anomalies |

        self.model_quality_df = pd.DataFrame()
        # mape| rmse | mse

        self.train = None
        self.test = None
        self.test_index = None
        self.dominant_period = None

    def SARIMA_grid(self, endog, seasonal_order):
        warnings.simplefilter("ignore")

        # create an empty list to store values
        model_info = []

        # fit the model
        stepwise_model = pm.auto_arima(endog,
                                       start_p=0, start_q=0,
                                       start_P=0, start_Q=0,
                                       max_p=3, max_q=3,
                                       max_P=3, max_Q=3,
                                       m=seasonal_order,
                                       seasonal=True,
                                       trace=True,
                                       error_action='ignore',
                                       suppress_warnings=True,
                                       stepwise=True,
                                       information_criterion='aic',
                                       )
        print(stepwise_model.aic())
        self.stepwise_mode = stepwise_model

    def fft_analysis(self, signal):
        # Linear detrending
        slope, intercept = np.polyfit(np.arange(len(signal)), signal, 1)
        trend = np.arange(len(signal)) * slope + intercept
        detrended = signal - trend
        fft_values = fft(detrended)
        frequencies = np.fft.fftfreq(len(fft_values))
        # Remove negative frequencies and sort
        positive_frequencies = frequencies[frequencies > 0]
        magnitudes = np.abs(fft_values)[frequencies > 0]
        # Identify dominant frequency
        dominant_frequency = positive_frequencies[np.argmax(magnitudes)]
        # Convert frequency to period (e.g., days, weeks, months, etc.)
        dominant_period = round(1 / dominant_frequency)
        return dominant_period, positive_frequencies, magnitudes

    def train_test_split(self, data, k=0.9):
        # train = data['Raw_Data'].iloc[:int(len(data) * k)]
        # test = data['Raw_Data'].iloc[int(len(data) * k):]
        train = data['Raw_Data'].iloc[:int(len(data) * k)]
        test = data['Raw_Data'].iloc[int(len(data) * k):]
        return train, test

    def anomalies(self):
        self._conf_intervals()
        anomalies = pd.DataFrame()
        self.model_df['Anomalies'] = False

        self.model_df['Anomalies'] = (self.model_df['Raw_Data'] < self.model_df['Lower_CI']) | (
                self.model_df['Raw_Data'] > self.model_df['Upper_CI'])
        # self.model_df['Anomalies'] = self.model_df['Raw_Data'] > self.model_df['Upper_CI']

        anomalies = self.model_df['Anomalies'][self.model_df['Anomalies'] == True]
        # anomalies = self.model_df_least_MAPE['Anomalies'][self.model_df_least_MAPE['Anomalies'] == True]
        # anomalies = anomalies[anomalies['Anomalies'] == True]
        return anomalies

    def fit_predict(self, data):
        # 1. fill up model result dataframe
        self.model_df = data.copy()

        # 2. Train test split
        train, test = self.train_test_split(data, k=0.7)
        print('train/test', train, test)
        self.train, self.test = train, test

        # 3. Find dominant period = main season = m

        # 4. Find pdq and PDQ

        self.dominant_period, _, _ = self.fft_analysis(self.model_df['Raw_Data'].values)

        # 5. Execute Grid search
        self.SARIMA_grid(endog=train, seasonal_order=self.dominant_period)

        # Take the configurations of the best models (!!! AFTER GS execution !!!)

        # Fit the models and compute the forecasts
        preds, conf_int = self.stepwise_mode.predict(n_periods=test.shape[0], return_conf_int=True)
        print('preds', preds)

        self.model_df['Y_Predicted'] = preds
    def _conf_intervals(self):
        ci = ConfidentIntervals()
        true_data = self.test
        model_data = self.model_df['Y_Predicted'][self.test.index]
        lower_bond, upper_bound = ci.stats_ci(true_data=true_data,
                                              model_data=model_data)
        self.model_df['Upper_CI'] = self.model_df['Y_Predicted'] + upper_bound
        self.model_df['Lower_CI'] = self.model_df['Y_Predicted'] - lower_bond

    def _model_quality(self):
        # self.model_df = self.model_df_least_MAPE.copy()
        r2 = r2_score(self.test,
                      self.model_df['Y_Predicted'][self.test.index])
        mape = mean_absolute_percentage_error(self.test,
                                              self.model_df['Y_Predicted'][self.test.index])
        print('mape data', self.test, self.model_df['Y_Predicted'][self.test.index])
        rmse = root_mean_squared_error(self.test,
                                       self.model_df['Y_Predicted'][self.test.index])
        mse = mean_squared_error(self.test,
                                 self.model_df['Y_Predicted'][self.test.index])
        print('mape', mape)
        dict_data = {
            'MAPE': mape,
            'RMSE': rmse,
            'MSE': mse,
            'R2': r2
        }
        # self.model_quality_df = pd.DataFrame(list(dict_data.items()), columns=['METRIC', 'Value'])
        self.model_quality_df = pd.DataFrame([[mape, rmse, mse, r2]], columns=['MAPE', 'RMSE', 'MSE', 'R2'])
