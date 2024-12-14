import pandas as pd
import numpy as np


class IncomeDataProcessor:
    def __init__(self):
        self.time_labels = None
        self.x_points = None
        self.y_values = None
        self.data_frame = None

    def prep(self, df: pd.DataFrame, time_col: str = None, value_col: str = None) -> \
            (list, np.int64, np.float64, pd.tseries):
        if time_col is not None:
            if time_col == 'index':
                self.time_labels = df.index.strftime('%Y-%m-%d %H:%M:%S').tolist()
            else:
                self.time_labels = df[time_col].values.tolist()
            self.x_points = np.arange(df.shape[0] + 1)
        if value_col is not None:
            self.y_values = df[value_col].values
            self.data_frame = df[value_col]
        return self.time_labels, self.x_points, self.y_values, self.data_frame
