import itertools
import numpy as np
import pandas as pd
import itertools
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from sklearn.metrics import mean_absolute_error,\
    root_mean_squared_error,\
    mean_absolute_percentage_error,\
    mean_squared_log_error, \
    mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from utils import timeseries_train_test_split
from ConfidentIntervals.ConfidentIntervals import ConfidentIntervals
from scipy.optimize import minimize


class HoltWinters2:

    def __init__(self):
        self.model_df = pd.DataFrame()
        self.best_alpha = None
        self.best_beta = None
        self.best_gamma = None
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

        self._optimize()

        tes_model = ExponentialSmoothing(train,
                                         trend="add",  # add || mul
                                         seasonal="add",  # add || mul
                                         seasonal_periods=24
                                         # we set 12. It represents that 12 step (month for our case) equals a seasonal period
                                         ).fit(smoothing_level=self.best_alpha,  # alpha
                                               smoothing_trend=self.best_beta,  # beta
                                               smoothing_seasonal=self.best_gamma  # gamma
                                               )

        # self.model_df['Holt_Winters'] = pd.concat([tes_model.fittedvalues, tes_model.forecast(len(test))])
        self.model_df['Holt_Winters'] = tes_model.forecast(len(test))
        self.fitted_data = tes_model.fittedvalues

        print('fitted_values', tes_model.fittedvalues)
        print('forecast', tes_model.forecast(len(test)))
        print(pd.concat([tes_model.fittedvalues, tes_model.forecast(len(test))]))

    def _timeseriesCVscore(self, params, series, loss_function=mean_squared_error, slen=24):
        # вектор ошибок
        errors = []

        values = series.values

        alpha, beta, gamma = params

        # задаём число фолдов для кросс-валидации
        tscv = TimeSeriesSplit(n_splits=2)

        # идем по фолдам, на каждом обучаем модель, строим прогноз на отложенной выборке и считаем ошибку
        for train, test in tscv.split(values):
            tes_model = ExponentialSmoothing(values[train],
                                             trend="add",  # add || mul
                                             seasonal="add",  # add || mul
                                             seasonal_periods=slen
                                             # we set 12. It represents that 12 step (month for our case) equals a seasonal period
                                             ).fit(smoothing_level=alpha,  # alpha
                                                   smoothing_trend=beta,  # beta
                                                   smoothing_seasonal=gamma  # gamma
                                                   )

            predictions = tes_model.forecast(len(test))
            actual = values[test]
            error = loss_function(predictions, actual)
            errors.append(error)

        return np.mean(np.array(errors))

    def _optimize(self):
        data = self.train  # отложим часть данных для тестирования

        # инициализируем значения параметров
        x = [0, 0, 0]

        # Минимизируем функцию потерь с ограничениями на параметры
        opt = minimize(self._timeseriesCVscore, x0=np.array([0, 0, 0]),
                       args=(data, mean_squared_log_error),
                       method="TNC", bounds=((0, 1), (0, 1), (0, 1))
                       )

        # Из оптимизатора берем оптимальное значение параметров
        self.best_alpha, self.best_beta, self.best_gamma = opt.x
        print("best_param", self.best_alpha, self.best_beta, self.best_gamma)
        # и подставляем их в модель



    # def anomalies(self):
    #     self._conf_intervals()
    #     anomalies = pd.DataFrame()
    #     self.model_df['Anomalies'] = False
    #
    #     self.model_df['Anomalies'] = (self.model_df['Raw_Data'] < self.model_df['Lower_CI']) | (
    #             self.model_df['Raw_Data'] > self.model_df['Upper_CI'])
    #     # self.model_df['Anomalies'] = self.model_df['Raw_Data'] > self.model_df['Upper_CI']
    #
    #     anomalies = self.model_df[['Time', 'Anomalies']]
    #     anomalies = anomalies[anomalies['Anomalies'] == True]
    #     return anomalies
    #
    # def _conf_intervals(self):
    #     ci = ConfidentIntervals()
    #     true_data = self.train
    #     model_data = self.fitted_data
    #     bound = ci.confidence_intervals_v1_1(true_data=true_data,
    #                                          model_data=model_data)
    #     self.model_df['Upper_CI'] = self.model_df['Holt_Winters'] + bound
    #     self.model_df['Lower_CI'] = self.model_df['Holt_Winters'] - bound
