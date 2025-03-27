import numpy as np
import pandas as pd
import itertools
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error, \
    mean_squared_log_error
from confident_intervals.confident_interavls import ConfidentIntervals
import warnings
from warnings import catch_warnings
from warnings import filterwarnings
from tqdm.notebook import tqdm
from scipy.fft import fft
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, root_mean_squared_error


class HoltWinters:

    def __init__(self):
        self.dominant_period = None
        self.best_gamma = 0.01
        self.best_beta = 0.01
        self.best_alpha = 0.01
        self.model_df = pd.DataFrame()
        # Time | Rw_Data | Y_Forecasted| Lower_CI | Upper_CI | Anomalies |

        self.model_quality_df = pd.DataFrame()
        # mape| rmse | mse

        self.test = None
        self.train = None

    def _train_test_split(self, data, k=0.9):
        # train = data['Raw_Data'].iloc[:int(len(data) * k)]
        # test = data['Raw_Data'].iloc[int(len(data) * k):]
        train = data['Raw_Data'].iloc[:int(len(data) * k)]
        test = data['Raw_Data'].iloc[int(len(data) * k):]
        # train = data['Raw_Data'][:'1997-12-01']
        # test = data['Raw_Data']['1998-01-01':]
        return train, test

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

    def _tes_optimizer(self, train, test, seasonal_periods=12):
        best_alpha, best_beta, best_gamma, best_mae = None, None, None, float("inf")
        alphas = betas = gammas = np.arange(0.10, 1, 0.10)
        abg = list(itertools.product(alphas, betas, gammas))
        for comb in abg:
            try:
                warnings.filterwarnings('ignore')
                tes_model = ExponentialSmoothing(train,
                                                 trend="add",
                                                 seasonal="add",
                                                 seasonal_periods=seasonal_periods).fit(smoothing_level=comb[0],
                                                                                        smoothing_slope=comb[1],
                                                                                        smoothing_seasonal=comb[2])
                y_pred = tes_model.forecast(len(test))
                # mae = mean_absolute_error(test, y_pred)
                mae = mean_absolute_percentage_error(test, y_pred)
                if mae < best_mae:
                    best_alpha, best_beta, best_gamma, best_mae = comb[0], comb[1], comb[2], mae
                print([round(comb[0], 2), round(comb[1], 2), round(comb[2], 2), round(mae, 2)])
            except:
                continue

        print("best_alpha:", round(best_alpha, 2), "best_beta:", round(best_beta, 2), "best_gamma:",
              round(best_gamma, 2),
              "best_mae:", round(best_mae, 4))

        return best_alpha, best_beta, best_gamma, best_mae

    def fit_predict(self, data):
        # 1. fill up model result dataframe
        self.model_df = data.copy()
        print('self.model_df', self.model_df)

        # 2. Train test split
        train, test = self._train_test_split(data, k=0.7)
        print('train/test', train, test)
        self.train, self.test = train, test

        # 3. Seasonal period
        self.dominant_period, _, _ = self.fft_analysis(signal=data['Raw_Data'].values)
        print('dominant_period', self.dominant_period)
        seasonal_periods = 7

        # 5. Optimization
        self.best_alpha, self.best_beta, self.best_gamma, best_mae = self._tes_optimizer(train,
                                                                                         test,
                                                                                         seasonal_periods=seasonal_periods)

        best_model = ExponentialSmoothing(train,
                                          trend="add",
                                          seasonal="add",
                                          seasonal_periods=seasonal_periods).fit(smoothing_level=self.best_alpha,
                                                                                 smoothing_slope=self.best_beta,
                                                                                 smoothing_seasonal=self.best_gamma)
        print('len(self.test)', len(self.test))
        self.model_df['Y_Predicted'] = best_model.forecast(len(test))

        # Calculate Model quality
        self._model_quality()

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

    def _conf_intervals(self):
        ci = ConfidentIntervals()
        true_data = self.test
        model_data = self.model_df['Y_Predicted'][self.test.index]
        bound = ci.confidence_intervals_v1_1(true_data=true_data,
                                             model_data=model_data)
        self.model_df['Upper_CI'] = self.model_df['Y_Predicted'] + bound
        self.model_df['Lower_CI'] = self.model_df['Y_Predicted'] - bound

    def _model_quality(self):
        mape = mean_absolute_percentage_error(self.test,
                                              self.model_df['Y_Predicted'][self.test.index])
        rmse = root_mean_squared_error(self.test,
                                       self.model_df['Y_Predicted'][self.test.index])
        mse = mean_squared_error(self.test,
                                 self.model_df['Y_Predicted'][self.test.index])
        dict_data = {
            'MAPE': mape,
            'RMSE': rmse,
            'MSE': mse
        }
        self.model_quality_df = pd.DataFrame(list(dict_data.items()), columns=['METRIC', 'Value'])
