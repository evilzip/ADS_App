import numpy as np
import pandas as pd
import datetime as dt

import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
from confident_intervals.confident_interavls import ConfidentIntervals
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, root_mean_squared_error


class XGBoostRegressor:
    def __init__(self):
        self.y_test = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.model_df = pd.DataFrame()
        # Time | Rw_Data | Y_Forecasted| Lower_CI | Upper_CI | Anomalies |

        self.model_quality_df = pd.DataFrame()
        # mape| rmse | mse

    def _create_features(self, data, label=None):
        """
        Creates time series features from datetime index
        """
        data['date'] = data.index
        print("data.index", data.index)
        data['hour'] = data['date'].dt.hour
        data['dayofweek'] = data['date'].dt.dayofweek
        data['quarter'] = data['date'].dt.quarter
        data['month'] = data['date'].dt.month
        data['year'] = data['date'].dt.year
        data['dayofyear'] = data['date'].dt.dayofyear
        data['dayofmonth'] = data['date'].dt.day
        data['weekofyear'] = data['date'].dt.isocalendar().week
        x = data[['hour', 'dayofweek', 'quarter', 'month', 'year',
                  'dayofyear', 'dayofmonth', 'weekofyear']]
        y = data['Raw_Data']
        return x, y

    def _train_test_split(self, data, k=0.9):
        train = data.iloc[:int(len(data) * k)]
        test = data.iloc[int(len(data) * k):]
        x_train, y_train = self._create_features(train)
        x_test, y_test = self._create_features(test)
        return x_train, y_train, x_test, y_test

    def _tes_optimizer(self, train, test, seasonal_periods=12):
        pass

    def fit_predict(self, data):
        # 1. fill up model result dataframe
        self.model_df = data.copy()

        # 2. Train test split
        self.x_train, self.y_train, self.x_test, self.y_test = self._train_test_split(self.model_df, k=0.7)

        reg = xgb.XGBRegressor(n_estimators=1000, early_stopping_rounds=50, )
        reg.fit(self.x_train, self.y_train,
                eval_set=[(self.x_train, self.y_train), (self.x_test, self.y_test)],
                verbose=False)  # Change verbose to True if you want to see it train

        # 6. Prediction
        self.model_df['Y_Predicted'] = np.nan
        self.model_df['Y_Predicted'].loc[self.y_test.index] = reg.predict(self.x_test)

        # 7. Calculate Model quality
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
        true_data = self.y_test
        model_data = self.model_df['Y_Predicted'][self.y_test.index]
        bound = ci.confidence_intervals_v1_1(true_data=true_data,
                                             model_data=model_data)
        self.model_df['Upper_CI'] = self.model_df['Y_Predicted'] + bound
        self.model_df['Lower_CI'] = self.model_df['Y_Predicted'] - bound

    def _model_quality(self):
        mape = mean_absolute_percentage_error(self.y_test,
                                              self.model_df['Y_Predicted'][self.y_test.index])
        rmse = root_mean_squared_error(self.y_test,
                                       self.model_df['Y_Predicted'][self.y_test.index])
        mse = mean_squared_error(self.y_test,
                                 self.model_df['Y_Predicted'][self.y_test.index])
        dict_data = {
            'MAPE': mape,
            'RMSE': rmse,
            'MSE': mse
        }
        self.model_quality_df = pd.DataFrame(list(dict_data.items()), columns=['METRIC', 'Value'])
