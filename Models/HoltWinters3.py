# import itertools
# import numpy as np
# import pandas as pd
# import itertools
# import statsmodels.api as sm
# from statsmodels.tsa.holtwinters import ExponentialSmoothing
# from statsmodels.tsa.exponential_smoothing.ets import ETSModel
# from sklearn.metrics import mean_absolute_error,\
#     root_mean_squared_error,\
#     mean_absolute_percentage_error,\
#     mean_squared_log_error, \
#     mean_squared_error
# from sklearn.model_selection import TimeSeriesSplit
# from utils import timeseries_train_test_split
# from ConfidentIntervals.confident_interavls import ConfidentIntervals
# from scipy.optimize import minimize
#
#
# class HoltWinters3:
#
#         """
#         Holt-Winters model with the anomalies detection using Brutlag method
#
#         # series - initial time series
#         # slen - length of a season
#         # alpha, beta, gamma - Holt-Winters model coefficients
#         # n_preds - predictions horizon
#         # scaling_factor - sets the width of the confidence interval by Brutlag (usually takes values from 2 to 3)
#
#         """
#
#         def __init__(self, series, slen, alpha, beta, gamma, n_preds, scaling_factor=1.96):
#             self.series = series
#             self.slen = slen
#             self.alpha = alpha
#             self.beta = beta
#             self.gamma = gamma
#             self.n_preds = n_preds
#             self.scaling_factor = scaling_factor
#
#         def initial_trend(self):
#             sum = 0.0
#             for i in range(self.slen):
#                 sum += float(self.series[i + self.slen] - self.series[i]) / self.slen
#             return sum / self.slen
#
#         def initial_seasonal_components(self):
#             seasonals = {}
#             season_averages = []
#             n_seasons = int(len(self.series) / self.slen)
#             # let's calculate season averages
#             for j in range(n_seasons):
#                 season_averages.append(sum(self.series[self.slen * j:self.slen * j + self.slen]) / float(self.slen))
#             # let's calculate initial values
#             for i in range(self.slen):
#                 sum_of_vals_over_avg = 0.0
#                 for j in range(n_seasons):
#                     sum_of_vals_over_avg += self.series[self.slen * j + i] - season_averages[j]
#                 seasonals[i] = sum_of_vals_over_avg / n_seasons
#             return seasonals
#
#         def triple_exponential_smoothing(self):
#             self.result = []
#             self.Smooth = []
#             self.Season = []
#             self.Trend = []
#             self.PredictedDeviation = []
#             self.UpperBond = []
#             self.LowerBond = []
#
#             seasonals = self.initial_seasonal_components()
#
#             for i in range(len(self.series) + self.n_preds):
#                 if i == 0:  # components initialization
#                     smooth = self.series[0]
#                     trend = self.initial_trend()
#                     self.result.append(self.series[0])
#                     self.Smooth.append(smooth)
#                     self.Trend.append(trend)
#                     self.Season.append(seasonals[i % self.slen])
#
#                     self.PredictedDeviation.append(0)
#
#                     self.UpperBond.append(self.result[0] +
#                                           self.scaling_factor *
#                                           self.PredictedDeviation[0])
#
#                     self.LowerBond.append(self.result[0] -
#                                           self.scaling_factor *
#                                           self.PredictedDeviation[0])
#                     continue
#
#                 if i >= len(self.series):  # predicting
#                     m = i - len(self.series) + 1
#                     self.result.append((smooth + m * trend) + seasonals[i % self.slen])
#
#                     # when predicting we increase uncertainty on each step
#                     self.PredictedDeviation.append(self.PredictedDeviation[-1] * 1.01)
#
#                 else:
#                     val = self.series[i]
#                     last_smooth, smooth = smooth, self.alpha * (val - seasonals[i % self.slen]) + (1 - self.alpha) * (
#                                 smooth + trend)
#                     trend = self.beta * (smooth - last_smooth) + (1 - self.beta) * trend
#                     seasonals[i % self.slen] = self.gamma * (val - smooth) + (1 - self.gamma) * seasonals[i % self.slen]
#                     self.result.append(smooth + trend + seasonals[i % self.slen])
#
#                     # Deviation is calculated according to Brutlag algorithm.
#                     self.PredictedDeviation.append(self.gamma * np.abs(self.series[i] - self.result[i])
#                                                    + (1 - self.gamma) * self.PredictedDeviation[-1])
#
#                 self.UpperBond.append(self.result[-1] +
#                                       self.scaling_factor *
#                                       self.PredictedDeviation[-1])
#
#                 self.LowerBond.append(self.result[-1] -
#                                       self.scaling_factor *
#                                       self.PredictedDeviation[-1])
#
#                 self.Smooth.append(smooth)
#                 self.Trend.append(trend)
#                 self.Season.append(seasonals[i % self.slen])
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#     def fit_predict(self, data):
#         """
#             series - dataset with timeseries
#             alpha - float [0.0, 1.0], smoothing parameter for level
#             beta - float [0.0, 1.0], smoothing parameter for trend
#         """
#         self.model_df['Time'] = data['Time']
#         self.model_df['Raw_Data'] = data['Raw_Data']
#         raw_data = data['Raw_Data'].values.tolist()
#         print('len(raw_data)', len(self.model_df['Raw_Data']))
#         print('data', data)
#
#         train, test, test_index = timeseries_train_test_split(data['Raw_Data'], 0.3)
#         # print('test', test)
#         self.train, self.test, self.test_index = train, test, test_index
#
#         self._optimize()
#
#         tes_model = ExponentialSmoothing(train,
#                                          trend="add",  # add || mul
#                                          seasonal="add",  # add || mul
#                                          seasonal_periods=24
#                                          # we set 12. It represents that 12 step (month for our case) equals a seasonal period
#                                          ).fit(smoothing_level=self.best_alpha,  # alpha
#                                                smoothing_trend=self.best_beta,  # beta
#                                                smoothing_seasonal=self.best_gamma  # gamma
#                                                )
#
#         # self.model_df['Holt_Winters'] = pd.concat([tes_model.fittedvalues, tes_model.forecast(len(test))])
#         self.model_df['Holt_Winters'] = tes_model.forecast(len(test))
#         self.fitted_data = tes_model.fittedvalues
#
#         print('fitted_values', tes_model.fittedvalues)
#         print('forecast', tes_model.forecast(len(test)))
#         print(pd.concat([tes_model.fittedvalues, tes_model.forecast(len(test))]))
#
#     def _timeseriesCVscore(self, params, series, loss_function=mean_squared_error, slen=24):
#         # вектор ошибок
#         errors = []
#
#         values = series.values
#
#         alpha, beta, gamma = params
#
#         # задаём число фолдов для кросс-валидации
#         tscv = TimeSeriesSplit(n_splits=2)
#
#         # идем по фолдам, на каждом обучаем модель, строим прогноз на отложенной выборке и считаем ошибку
#         for train, test in tscv.split(values):
#             tes_model = ExponentialSmoothing(values[train],
#                                              trend="add",  # add || mul
#                                              seasonal="add",  # add || mul
#                                              seasonal_periods=slen
#                                              # we set 12. It represents that 12 step (month for our case) equals a seasonal period
#                                              ).fit(smoothing_level=alpha,  # alpha
#                                                    smoothing_trend=beta,  # beta
#                                                    smoothing_seasonal=gamma  # gamma
#                                                    )
#
#             predictions = tes_model.forecast(len(test))
#             actual = values[test]
#             error = loss_function(predictions, actual)
#             errors.append(error)
#
#         return np.mean(np.array(errors))
#
#     def _optimize(self):
#         data = self.train  # отложим часть данных для тестирования
#
#         # инициализируем значения параметров
#         x = [0, 0, 0]
#
#         # Минимизируем функцию потерь с ограничениями на параметры
#         opt = minimize(self._timeseriesCVscore, x0=np.array([0, 0, 0]),
#                        args=(data, mean_squared_log_error),
#                        method="TNC", bounds=((0, 1), (0, 1), (0, 1))
#                        )
#
#         # Из оптимизатора берем оптимальное значение параметров
#         self.best_alpha, self.best_beta, self.best_gamma = opt.x
#         print("best_param", self.best_alpha, self.best_beta, self.best_gamma)
#         # и подставляем их в модель
#
#
#
#     # def anomalies(self):
#     #     self._conf_intervals()
#     #     anomalies = pd.DataFrame()
#     #     self.model_df['Anomalies'] = False
#     #
#     #     self.model_df['Anomalies'] = (self.model_df['Raw_Data'] < self.model_df['Lower_CI']) | (
#     #             self.model_df['Raw_Data'] > self.model_df['Upper_CI'])
#     #     # self.model_df['Anomalies'] = self.model_df['Raw_Data'] > self.model_df['Upper_CI']
#     #
#     #     anomalies = self.model_df[['Time', 'Anomalies']]
#     #     anomalies = anomalies[anomalies['Anomalies'] == True]
#     #     return anomalies
#     #
#     # def _conf_intervals(self):
#     #     ci = ConfidentIntervals()
#     #     true_data = self.train
#     #     model_data = self.fitted_data
#     #     bound = ci.confidence_intervals_v1_1(true_data=true_data,
#     #                                          model_data=model_data)
#     #     self.model_df['Upper_CI'] = self.model_df['Holt_Winters'] + bound
#     #     self.model_df['Lower_CI'] = self.model_df['Holt_Winters'] - bound
