import itertools
import numpy as np
import pandas as pd
import itertools
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error, mean_squared_log_error

from utils import timeseries_train_test_split
from confident_intervals.confident_interavls import ConfidentIntervals


class HoltWinters:

    def __init__(self):
        self.model_df = pd.DataFrame()
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.test = None
        self.train = None
        self.test_index = None
        self.fitted_data = None

    def fit_predict(self, data):
        """
            series - dataset with timeseries
            alpha - float [0.0, 1.0], smoothing parameter for level
            beta - float [0.0, 1.0], smoothing parameter for trend
        """
        self.model_df['Time'] = data['Time']
        self.model_df['Raw_Data'] = data['Raw_Data']
        raw_data = data['Raw_Data'].values.tolist()
        print('len(raw_data)', len(self.model_df['Raw_Data']))
        print('data', data)

        train, test, test_index = timeseries_train_test_split(data['Raw_Data'], 0.3)
        # print('test', test)
        self.train, self.test, self.test_index = train, test, test_index

        self._optimize(train=train, test=test, trend_mode='add', seasonal_mode='add', seasonal_period=12)

        tes_model = ExponentialSmoothing(train,
                                         trend="add",  # add || mul
                                         seasonal="add",  # add || mul
                                         seasonal_periods=24
                                         # we set 12. It represents that 12 step (month for our case) equals a seasonal period
                                         ).fit(smoothing_level=self.alpha,  # alpha
                                               smoothing_trend=self.beta,  # beta
                                               smoothing_seasonal=self.gamma  # gamma
                                               )

        # self.model_df['Holt_Winters'] = pd.concat([tes_model.fittedvalues, tes_model.forecast(len(test))])
        self.model_df['Holt_Winters'] = tes_model.forecast(len(test))
        self.fitted_data = tes_model.fittedvalues

        print('fitted_values', tes_model.fittedvalues)
        print('forecast', tes_model.forecast(len(test)))
        print(pd.concat([tes_model.fittedvalues, tes_model.forecast(len(test))]))

    def _optimize(self, train, test, trend_mode='add', seasonal_mode='add', seasonal_period=24):

        best_alpha, best_beta, best_gamma, best_mae = None, None, None, 100
        alphas = gammas = [x / 100 for x in range(0, 101, 5)]
        betas = [0 for x in range(0, 101, 5)]
        abg = list(itertools.product(alphas, betas, gammas))  # Creating combinations of the 3 lists

        for comb in abg:  # visit the each combination
            tes_model = ExponentialSmoothing(train, trend=trend_mode, seasonal=seasonal_mode,
                                             seasonal_periods=seasonal_period). \
                fit(smoothing_level=comb[0],
                    smoothing_trend=comb[1],
                    smoothing_seasonal=comb[
                        2])  # 0: alpha, 1: beta, 2: gamma. Creates a new TES instance by using each combination
            y_pred = tes_model.forecast(len(test))  # forecast the `step` step later by using the TES instance
            # print('y_pred', y_pred)
            # print('TEST', test)
            # mae = mean_absolute_error(test, y_pred)  # calculate the MAE (mean absolute error)
            mae = mean_absolute_percentage_error(test, y_pred)
            if mae < best_mae:  # mark the best parameters
                best_alpha, best_beta, best_gamma, best_mae = comb[0], comb[1], comb[2], mae
            print([round(comb[0], 2), round(comb[1], 2), round(comb[2], 2), round(mae, 2)])

        print("best_alpha:", round(best_alpha, 2), "best_beta:", round(best_beta, 2), "best_gamma:",
              round(best_gamma, 2),
              "best_mae:", round(best_mae, 4))

        self.alpha = best_alpha
        self.beta = best_beta
        self.gamma = best_gamma

        return best_alpha, best_beta, best_gamma, best_mae

    def anomalies(self):
        self._conf_intervals()
        anomalies = pd.DataFrame()
        self.model_df['Anomalies'] = False

        self.model_df['Anomalies'] = (self.model_df['Raw_Data'] < self.model_df['Lower_CI']) | (
                self.model_df['Raw_Data'] > self.model_df['Upper_CI'])
        # self.model_df['Anomalies'] = self.model_df['Raw_Data'] > self.model_df['Upper_CI']

        anomalies = self.model_df[['Time', 'Anomalies']]
        anomalies = anomalies[anomalies['Anomalies'] == True]
        return anomalies

    def _conf_intervals(self):
        ci = ConfidentIntervals()
        true_data = self.train
        model_data = self.fitted_data
        bound = ci.confidence_intervals_v1_1(true_data=true_data,
                                             model_data=model_data)
        self.model_df['Upper_CI'] = self.model_df['Holt_Winters'] + bound
        self.model_df['Lower_CI'] = self.model_df['Holt_Winters'] - bound
