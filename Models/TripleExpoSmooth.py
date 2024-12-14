import itertools
import numpy as np
import pandas as pd
import itertools
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from sklearn.metrics import mean_absolute_error

from utils import timeseries_train_test_split


class TripleExpoSmooth:

    def __init__(self):
        self.model_df = pd.DataFrame()

    def fit_predict(self, data):
        """
            series - dataset with timeseries
            alpha - float [0.0, 1.0], smoothing parameter for level
            beta - float [0.0, 1.0], smoothing parameter for trend
        """
        self.model_df['Time'] = data['Time']
        self.model_df['Raw_Data'] = data['Raw_Data']
        raw_data = data['Raw_Data'].values.tolist()

        train, test, test_index = timeseries_train_test_split(data['Raw_Data'], 0.3)
        # print('test', test)

        # self._optimize(train=train, test=test,  trend_mode='add', seasonal_mode='add', seasonal_period=12)

        ets_model = ETSModel(train,
                             error='add',
                             trend="add",  # add || mul
                             seasonal="add",  # add || mul
                             seasonal_periods=30
                             )
        ets_result = ets_model.fit()
        pred = ets_result.get_prediction(start=0, end=len(train) + len(test), dynamic=False)
        df_pred = pred.summary_frame(alpha=0.1)
        self.model_df['Triple_Expo_Smooth'] = df_pred['mean']
        self.model_df['Upper_CI'] = df_pred.pi_upper
        self.model_df['Lower_CI'] = df_pred.pi_lower
        self.model_df['Anomalies'] = 0
        print('model_df', self.model_df)
        print(df_pred)
        print('model_result', ets_result.forecast(len(test)))
        # Simulate predictions.
        n_steps_prediction = len(test)
        n_repetitions = 500

        df_simul = ets_result.simulate(
            nsimulations=n_steps_prediction,
            repetitions=n_repetitions,
            anchor='start',
        )

        # Calculate confidence intervals.
        upper_ci = df_simul.quantile(q=0.9, axis='columns')
        lower_ci = df_simul.quantile(q=0.1, axis='columns')
        print("upper_ci", upper_ci)
        print("lower_ci", lower_ci)

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

    # def _conf_intervals(self):
    #     ci = ConfidentIntervals()
    #     true_data = self.model_df['Raw_Data']
    #     model_data = self.model_df['Double_Expo_Smooth']
    #     lower_bound, upper_bond = ci.confidence_intervals_v1(true_data=true_data,
    #                                                          model_data=model_data)
    #     self.model_df['Upper_CI'] = upper_bond
    #     self.model_df['Lower_CI'] = lower_bound
