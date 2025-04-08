import pandas as pd
from ConfidentIntervals.ConfidentIntervals import ConfidentIntervals


class MovingAverage:

    def __init__(self):
        self.model_df = pd.DataFrame()

    def fit(self):
        pass

    def predict(self):
        pass

    def fit_predict(self, data, window):
        self.model_df['Time'] = data['Time']
        self.model_df['Raw_Data'] = data['Raw_Data']
        rolling_mean = data['Raw_Data'].rolling(window=window).mean()
        data['Moving_Average'] = data['Raw_Data'].rolling(window=window).mean()
        self.model_df['Moving_Average'] = data['Moving_Average']
        self.model_df.dropna(inplace=True)
        data.dropna(inplace=True)
        return data

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
        true_data = self.model_df['Raw_Data']
        model_data = self.model_df['Moving_Average']
        lower_bound, upper_bond = ci.confidence_intervals_v1(true_data=true_data,
                                                             model_data=model_data)
        self.model_df['Upper_CI'] = upper_bond
        self.model_df['Lower_CI'] = lower_bound

    def plot_model(self):
        # должна возвращать:
        # 1. список всех доступных curves = все колонки без Time
        # 2.
        pass
