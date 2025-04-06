import sys
import scipy.stats as st
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, root_mean_squared_error
from tqdm import tqdm
import pandas as pd
from itertools import product
import numpy as np
from utils import timeseries_train_test_split
from confident_intervals.ConfidentIntervals import ConfidentIntervals


class SARIMAX:
    def __init__(self):
        self.result_table = None
        self.parameters_list = None
        self.param_mini = None
        self.param_seasonal_mini = None
        self.model_results = None
        self.best_model = None
        self.model_df = pd.DataFrame()
        self.model_quality_df = pd.DataFrame()

        self.train = None
        self.test = None
        self.test_index = None

    def _sarima_grid_search(self, y, seasonal_period):  # optimize
        best_mape = 100
        # p = d = q = [x / 10 for x in range(0, 3)]
        d = q = range(0, 4)
        p = range(0, 4)
        pdq = list(product(p, d, q))
        seasonal_pdq = [(x[0], x[1], x[2], seasonal_period) for x in list(product(p, d, q))]

        mini = float('+inf')

        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    mod = sm.tsa.statespace.SARIMAX(y,
                                                    order=param,
                                                    seasonal_order=param_seasonal,
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False)

                    results = mod.fit()

                    y_pred = results.forecast(len(self.test))  # forecast the `step` step later by using the TES instance
                    print('y_pred', y_pred)
                    print('TEST', self.test)
                    # mae = mean_absolute_error(test, y_pred)  # calculate the MAE (mean absolute error)
                    mape = mean_absolute_percentage_error(self.test, y_pred)
                    if mape < best_mape:
                        best_mape = mape# mark the best parameters
                        param_mini = param
                        param_seasonal_mini = param_seasonal
                    #
                    #
                    # if results.aic < mini:
                    #     mini = results.aic
                    #     param_mini = param
                    #     param_seasonal_mini = param_seasonal
                    # print('SARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
                    print('SARIMA{}x{} - mae:{}'.format(param, param_seasonal, mape))
                except:
                    print(param, param_seasonal, "Unexpected error:", sys.exc_info()[1])
        print('The set of parameters with the minimum MAPE is: SARIMA{}x{} - MAPE:{}'.format(param_mini,
                                                                                           param_seasonal_mini, best_mape))
        self.param_mini = param_mini
        self.param_seasonal_mini = param_seasonal_mini
        print("best_param", self.param_mini, self.param_seasonal_mini)
        return param_mini, param_seasonal_mini

    def fit(self, data):
        self.model_df['Time'] = data['Time']
        self.model_df['Raw_Data'] = data['Raw_Data']
        raw_data = data['Raw_Data'].values.tolist()

        train, test, test_index = timeseries_train_test_split(data['Raw_Data'], 0.3)
        # print('test', test)
        self.train, self.test, self.test_index = train, test, test_index

        # self._sarima_grid_search(train, seasonal_period=7)

        # best_model = sm.tsa.statespace.SARIMAX(train, order=self.param_mini,
        #                                        seasonal_order=self.param_seasonal_mini,
        #                                        enforce_stationarity=True,
        #                                        enforce_invertibility=False
        #                                        ).fit(disp=-1)
        best_model = sm.tsa.statespace.SARIMAX(data['Raw_Data'],  # self.model_df['Raw_Data']
                                               order=(1, 1, 1),
                                               seasonal_order=(1, 1, 1, 7),
                                               enforce_stationarity=True,
                                               enforce_invertibility=False
                                               ).fit(disp=-1)
        # self.model_results = best_model.fit()
        self.best_model = best_model
        # print('best_model', best_model.summary)
        # print(best_model.summary().tables[1])
        pred = self.best_model.get_prediction(start=self.test.index[0],
                                              dynamic=False)

        pred_ci = pred.conf_int(alpha=0.1)
        y_forecasted = pred.predicted_mean
        pred = self.best_model.get_prediction(start=self.test.index[0],
                                              end=self.test.index[-1],
                                              dynamic=False)
        self.model_df['SARIMAX'] = y_forecasted
        self._model_quality()
        return best_model

    # def plot_result(self):
    #     self.best_model.plot_diagnostics(figsize=(16, 8))
    #     plt.show()
    #
    def plot_model(self, y_test, y, seasonal_period, pred_start_time, pred_end_time, plot_anomalies=False):
        # A better representation of our true predictive power can be obtained using dynamic forecasts.
        # In this case, we only use information from the time series up to a certain point,
        # and after that, forecasts are generated using values from previous forecasted time points.
        pred = self.best_model.get_prediction(start=pred_start_time,
                                              dynamic=False)

        pred_ci = pred.conf_int(alpha=0.1)
        y_forecasted = pred.predicted_mean
        print(y_forecasted)
        print('y_test', y_test)
        y_fc = self.best_model.fittedvalues
        # mape_dynamic = mean_absolute_percentage_error(y_test.values, y_fc_dynamic.values[-len(y_test.values):])
        mape = mean_absolute_percentage_error(y_test, y_forecasted)
        mse = ((y_forecasted - y_test) ** 2).mean()

        print(
            f'The MSE Error of SARIMA with season_length={seasonal_period} {mse}')
        plt.figure(figsize=(14, 7))
        plt.plot(y, label='actual')
        plt.plot(y_forecasted, label='Forecast', color='green')
        # plt.plot(y_fc_dynamic, label='model')
        plt.plot(pred_ci.index, pred_ci.iloc[:, 0], "r--", alpha=0.5, label="Up/Low confidence")
        plt.plot(pred_ci.index, pred_ci.iloc[:, 1], "r--", alpha=0.5)

        plt.title(f'The MSE={round(mse, 2)} error of SARIMA with season_length={seasonal_period}')
        plt.xlabel("Date")
        plt.ylabel("Temperature, С")
        plt.grid(True)
        plt.legend()
        plt.show()

    #
    # def forecast(self, predict_steps, y):
    #     pred_uc = self.best_model.get_forecast(steps=predict_steps)
    #
    #     # SARIMAXResults.conf_int, can change alpha,the default alpha = .05 returns a 95% confidence interval.
    #     pred_ci = pred_uc.conf_int(alpha=0.1)
    #
    #     plt.figure(figsize=(14, 7))
    #     plt.plot(y, label='actual')
    #     #     print(pred_uc.predicted_mean)
    #     plt.plot(pred_uc.predicted_mean, label='Prediction', color='green')
    #     # ax.fill_between(pred_ci.index,
    #     #                 pred_ci.iloc[:, 0],
    #     #                 pred_ci.iloc[:, 1], color='k', alpha=.25, label='Conf.Interval')
    #     plt.plot(pred_ci.index, pred_ci.iloc[:, 0], "r--", alpha=0.5, label="Up/Low confidence")
    #     plt.plot(pred_ci.index, pred_ci.iloc[:, 1], "r--", alpha=0.5)
    #     plt.xlabel("Date")
    #     plt.ylabel("Temperature, С")
    #     plt.grid(True)
    #     plt.title(f'Model forecast for next {predict_steps} steps with CI')
    #     plt.legend()
    #     plt.show()
    #
    #     # Produce the forcasted tables
    #     pm = pred_uc.predicted_mean.reset_index()
    #     pm.columns = ['Date', 'Predicted_Mean']
    #     pci = pred_ci.reset_index()
    #     pci.columns = ['Date', 'Lower Bound', 'Upper Bound']
    #     final_table = pm.join(pci.set_index('Date'), on='Date')
    #
    #     return final_table

    def anomalies(self):
        self._conf_intervals()
        anomalies = pd.DataFrame()
        self.model_df['Anomalies'] = False
        self.model_df['Anomalies'] = (self.model_df['Raw_Data'] < self.model_df['Lower_CI']) | (
                self.model_df['Raw_Data'] > self.model_df['Upper_CI'])
        anomalies = self.model_df[['Time', 'Anomalies']]
        anomalies = anomalies[anomalies['Anomalies'] == True]
        return anomalies

    def _conf_intervals(self):
        print('self.test.index[0]', self.test.index[0])
        pred = self.best_model.get_prediction(start=self.test.index[0],
                                              end=self.test.index[-1],
                                              dynamic=False)
        # pred = self.best_model.forecast(len(self.test))
        pred_ci = pred.conf_int(alpha=0.1)
        print('pred_ci', pred_ci)
        self.model_df['Upper_CI'] = pred_ci.iloc[:, 1]
        self.model_df['Lower_CI'] = pred_ci.iloc[:, 0]

    def _model_quality(self):
        mape = mean_absolute_percentage_error(self.test, self.model_df.SARIMAX.iloc[self.test_index:])
        rmse = root_mean_squared_error(self.test, self.model_df.SARIMAX.iloc[self.test_index:])
        mse = mean_squared_error(self.test, self.model_df.SARIMAX.iloc[self.test_index:])
        print('mape', mape)
        dict_data = {
            'MAPE': mape,
            'RMSE': rmse,
            'MSE': mse
        }
        self.model_quality_df = df = pd.DataFrame(list(dict_data.items()), columns=['METRIC', 'Value'])


