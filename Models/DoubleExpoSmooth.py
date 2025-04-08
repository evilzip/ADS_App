import pandas as pd
from ConfidentIntervals.ConfidentIntervals import ConfidentIntervals


class DoubleExpoSmooth:

    def __init__(self):
        self.model_df = pd.DataFrame()

    def fit_predict(self, data, alpha, beta):
        """
            series - dataset with timeseries
            alpha - float [0.0, 1.0], smoothing parameter for level
            beta - float [0.0, 1.0], smoothing parameter for trend
        """
        self.model_df['Time'] = data['Time']
        self.model_df['Raw_Data'] = data['Raw_Data']

        raw_data = data['Raw_Data'].values.tolist()
        result = [raw_data[0]]  # первое значение такое же, как в исходном ряде
        for n in range(1, len(raw_data)):
            if n == 1:
                level, trend = raw_data[0], raw_data[1] - raw_data[0]
            if n >= len(raw_data)+100:  # прогнозируем
                value = result[-1]
            else:
                value = raw_data[n]
            last_level, level = level, alpha * value + (1 - alpha) * (level + trend)
            trend = beta * (level - last_level) + (1 - beta) * trend
            result.append(level + trend)
        self.model_df['Double_Expo_Smooth'] = result


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
        model_data = self.model_df['Double_Expo_Smooth']
        lower_bound, upper_bond = ci.confidence_intervals_v1(true_data=true_data,
                                                             model_data=model_data)
        self.model_df['Upper_CI'] = upper_bond
        self.model_df['Lower_CI'] = lower_bound
