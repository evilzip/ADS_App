from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error
import numpy as np


class ConfidentIntervals:

    def _init__(self):
        pass

    def confidence_intervals_v1(self, true_data, model_data, scale=1.96):
        mae = mean_absolute_error(true_data, model_data)
        deviation = np.std(true_data - model_data)
        lower_bond = model_data - (mae + scale * deviation)
        upper_bond = model_data + (mae + scale * deviation)
        return lower_bond, upper_bond

    def confidence_intervals_v1_1(self, true_data, model_data, scale=1.96):
        mae = mean_absolute_error(true_data, model_data)
        deviation = np.std(true_data - model_data)
        bond = (mae + scale * deviation)
        return bond
