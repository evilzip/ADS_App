import pandas as pd
from ConfidentIntervals.ConfidentIntervals import ConfidentIntervals


class ExpoSmooth:

    def __init__(self):
        self.model_df = pd.DataFrame()

    def fit_predict(self, data, alpha):
        """
            series - dataset with timestamps
            alpha - float [0.0, 1.0], smoothing parameter
        """
        self.model_df['Time'] = data['Time']
        self.model_df['Raw_Data'] = data['Raw_Data']

        raw_data = data['Raw_Data'].values.tolist()
        result = [raw_data[0]]  # первое значение такое же, как в исходном ряде
        for n in range(1, len(raw_data)):
            result.append(alpha * raw_data[n] + (1 - alpha) * result[n - 1])
        self.model_df['Expo_Smooth'] = result

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
        model_data = self.model_df['Expo_Smooth']
        lower_bound, upper_bond = ci.confidence_intervals_v1(true_data=true_data,
                                                             model_data=model_data)
        self.model_df['Upper_CI'] = upper_bond
        self.model_df['Lower_CI'] = lower_bound
