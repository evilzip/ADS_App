import numpy as np
import pandas as pd
import datetime as dt

import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
from ConfidentIntervals.ConfidentIntervals import ConfidentIntervals
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


class XGBoostRegressor2:
    def __init__(self):
        self.y_test = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.model_df = pd.DataFrame()
        # Time | Rw_Data | Y_Forecasted| Lower_CI | Upper_CI | Anomalies |

        self.model_quality_df = pd.DataFrame()
        # mape| rmse | mse

    def _create_features(self, data, lag_start=6, lag_end=25):
        """
        Creates time series features from datetime index
        """
        # Create lag features
        lag_columns = []
        for i in range(lag_start, lag_end):
            lag_columns.append(f'lag_{i}')
            data[f'lag_{i}'] = data['Raw_Data'].shift(i)

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
        calendar_columns = ['hour', 'dayofweek', 'quarter', 'month', 'year',
                            'dayofyear', 'dayofmonth', 'weekofyear']
        columns = lag_columns + calendar_columns
        x = data[columns].dropna()
        y = data['Raw_Data'].loc[x.index]
        return x, y

    def _create_lags(self, data):
        for i in range(6, 25):
            data["lag_{}".format(i)] = data['Raw_Data'].shift(i)

    def _train_test_split(self, x, y, k=0.9):
        print('x', x.shape)
        print('y', y.shape)
        x_train = x.iloc[:int(len(x) * k)]
        x_test = x.iloc[int(len(x) * k):]
        y_train = y.iloc[:int(len(x) * k)]
        y_test = y.iloc[int(len(x) * k):]
        print('x_train', x_train.shape)
        print('x_test', x_test.shape)
        print('y_train', y_train.shape)
        print('y_test', y_test.shape)
        return x_train, y_train, x_test, y_test

    def _tes_optimizer(self, train, test, seasonal_periods=12):
        pass

    def _prepare_data(self, data, lag_start, lag_end, k=0.7):
        x, y = self._create_features(data=data, lag_start=lag_start, lag_end=lag_end)
        x_train, y_train, x_test, y_test = self._train_test_split(x, y, k=k)

        return x_train, y_train, x_test, y_test

    def fit_predict(self, data):
        # 1. fill up model result dataframe
        self.model_df = data.copy()

        # 2. Train test split
        self.x_train, self.y_train, self.x_test, self.y_test =\
            self._prepare_data(self.model_df, lag_start=6, lag_end=25, k=0.7)

        # 3.
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(self.x_train)
        x_test_scaled = scaler.transform(self.x_test)

        print('x_train_scaled', x_train_scaled.shape)
        print('self.y_train', self.y_train.shape)
        print('x_test_scaled', x_test_scaled.shape)
        print('self.y_test', self.y_test.shape)


        reg = xgb.XGBRegressor(n_estimators=1000, early_stopping_rounds=50)
        reg.fit(x_train_scaled, self.y_train,
                eval_set=[(x_train_scaled, self.y_train), (x_test_scaled, self.y_test)],
                verbose=False)  # Change verbose to True if you want to see it train

        # 6. Prediction
        self.model_df['Y_Predicted'] = np.nan
        self.model_df['Y_Predicted'].loc[self.y_test.index] = reg.predict(x_test_scaled)

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
