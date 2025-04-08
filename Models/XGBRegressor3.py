# **********************************************************************************************************************
# *                                             Final Version of XGBoost                                               *
# **********************************************************************************************************************
import numpy as np
import pandas as pd
import datetime as dt
import optuna
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
from ConfidentIntervals.ConfidentIntervals import ConfidentIntervals
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


class XGBoostRegressor3:
    def __init__(self):
        self.y_test = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.test_index = None
        self.model_df = pd.DataFrame()
        # Model_df columns
        # Time | Rw_Data | Y_Forecasted| Lower_CI | Upper_CI | Anomalies |

        self.model_quality_df = pd.DataFrame()
        # Model quality column
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

    def _train_test_split(self, x, y, k=0.9):
        """

        :param x: train series
        :param y: test series
        :param k: proportion test/train
        :return: x_train, y_train, x_test, y_test - timeseries
        """
        test_index = int(len(x) * (1 - k))
        self.test_index = test_index
        print('x', x.shape)
        print('y', y.shape)
        x_train = x.iloc[:test_index]
        x_test = x.iloc[test_index:]
        y_train = y.iloc[:test_index]
        y_test = y.iloc[test_index:]
        print('x_train', x_train.shape)
        print('x_test', x_test.shape)
        print('y_train', y_train.shape)
        print('y_test', y_test.shape)
        return x_train, y_train, x_test, y_test

    def _optimizer(self, x_train, y_train, x_test, y_test):
        """
        Function to find the best XGBoost params with Optuna
        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        :return: study.best_param
        """

        def objective(trial):
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-8, 1.0),
                'max_depth': trial.suggest_int('max_depth', 1, 10),
                'subsample': trial.suggest_uniform('subsample', 0.1, 1.0),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1.0),
                'n_estimators': 100
            }
            model = xgb.XGBRegressor(**params)
            model.fit(x_train, y_train)
            predict = model.predict(x_test)
            mse = mean_squared_error(predict, y_test)
            return mse

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=50, timeout=600)

        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
        return study.best_params

    def _prepare_data(self, data, lag_start, lag_end, k=0.3):
        x, y = self._create_features(data=data, lag_start=lag_start, lag_end=lag_end)
        x_train, y_train, x_test, y_test = self._train_test_split(x, y, k=k)

        return x_train, y_train, x_test, y_test

    def fit_predict(self, data):
        # 1. fill up model result dataframe
        self.model_df = data.copy()

        # 2. Train test split
        self.x_train, self.y_train, self.x_test, self.y_test = \
            self._prepare_data(self.model_df, lag_start=6, lag_end=25, k=0.3)

        # 3. Optimization
        best_param = self._optimizer(x_train=self.x_train, y_train=self.y_train,
                                     x_test=self.x_test, y_test=self.y_test)

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(self.x_train)
        x_test_scaled = scaler.transform(self.x_test)

        print('x_train_scaled', x_train_scaled.shape)
        print('self.y_train', self.y_train.shape)
        print('x_test_scaled', x_test_scaled.shape)
        print('self.y_test', self.y_test.shape)

        reg = xgb.XGBRegressor(**best_param)
        reg.fit(x_train_scaled, self.y_train,
                eval_set=[(x_train_scaled, self.y_train), (x_test_scaled, self.y_test)],
                verbose=False)

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
        anomalies = self.model_df['Anomalies'][self.model_df['Anomalies'] == True]
        return anomalies

    def _conf_intervals(self):
        ci = ConfidentIntervals()
        true_data = self.y_test
        model_data = self.model_df['Y_Predicted'][self.y_test.index]
        lower_bond, upper_bound = ci.stats_ci(true_data=true_data,
                                              model_data=model_data)
        self.model_df['Upper_CI'] = self.model_df['Y_Predicted'] + upper_bound
        self.model_df['Lower_CI'] = self.model_df['Y_Predicted'] - lower_bond
        ci.print_all(true_data=true_data, model_data=model_data)

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
        self.model_quality_df = pd.DataFrame([[mape, rmse, mse, r2]], columns=['MAPE', 'RMSE', 'MSE', 'R2'])
