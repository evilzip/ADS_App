from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import datetime as dt

import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
from confident_intervals.ConfidentIntervals import ConfidentIntervals
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, root_mean_squared_error, r2_score


class Model(ABC):
    def __init__(self):
        self.model_df = pd.DataFrame()
        # Time | Rw_Data | Y_Forecasted| Lower_CI | Upper_CI | Anomalies |

        self.model_quality_df = pd.DataFrame()
        # mape| rmse | mse

    def _train_test_split(self, data, k=0.9):
        pass

    def _optimizer(self, train, test, seasonal_periods=12):
        pass

    def fit_predict(self, data):
        pass

    def anomalies(self):
        self._conf_intervals()
        anomalies = pd.DataFrame()
        self.model_df['Anomalies'] = False

        self.model_df['Anomalies'] = (self.model_df['Raw_Data'] < self.model_df['Lower_CI']) \
                                     | \
                                     (self.model_df['Raw_Data'] > self.model_df['Upper_CI'])
        anomalies = self.model_df['Anomalies'][self.model_df['Anomalies'] == True]

        return anomalies

    def _conf_intervals(self):
        ci = ConfidentIntervals()
        true_data = self.y_test
        model_data = self.model_df['Y_Predicted'][self.y_test.index]
        bound = ci.confidence_intervals_v1_1(true_data=true_data,
                                             model_data=model_data)
        self.model_df['Upper_CI'] = self.model_df['Y_Predicted'] + bound
        self.model_df['Lower_CI'] = self.model_df['Y_Predicted'] - bound

    def _model_quality(self):
        r2 = r2_score(self.y_test,
                      self.model_df['Y_Predicted'][self.y_test.index])
        mape = mean_absolute_percentage_error(self.y_test,
                                              self.model_df['Y_Predicted'][self.y_test.index])
        rmse = root_mean_squared_error(self.y_test,
                                       self.model_df['Y_Predicted'][self.y_test.index])
        mse = mean_squared_error(self.y_test,
                                 self.model_df['Y_Predicted'][self.y_test.index])
        dict_data = {
            'MAPE': mape,
            'RMSE': rmse,
            'MSE': mse,
            'R2': r2
        }
        self.model_quality_df = pd.DataFrame(list(dict_data.items()), columns=['METRIC', 'Value'])
