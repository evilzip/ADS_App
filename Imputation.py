from sklearn.impute import KNNImputer
from sklearn.datasets import load_diabetes
import pandas as pd
import numpy as np


class Imputation:
    def __init__(self, data):
        self.imputation_df = data
        self.missing_data_indexes = None
        self.missing_indexes = None
        self.missing_df = pd.DataFrame()

    def find_missing_data(self):
        self.find_missing_index()
        # self.imputation_df = data.copy()
        self.imputation_df['Missing'] = self.imputation_df['Raw_Data'].isnull()
        self.missing_data_indexes = self.imputation_df[self.imputation_df['Raw_Data'].isnull()].index
        # self.missing_df = data.loc[self.missing_indexes]
        self.missing_df = self.imputation_df['Missing'][self.imputation_df['Missing'] == True]

    def find_missing_index(self):
        # df_result = data.copy()
        # print('df_fixed_index', df_result)
        self.imputation_df.sort_index(inplace=True)
        # print('data_indexes', data_indexes)
        # index_time_frequency = pd.infer_freq(data.index)
        index_time_frequency = self.imputation_df.index.inferred_freq
        self.imputation_df.index.freq = index_time_frequency
        print('index_time_frequency', index_time_frequency)
        print('data.index', self.imputation_df.index)
        new_date_range = pd.date_range(start=self.imputation_df.index[0],
                                       end=self.imputation_df.index[-1],
                                       freq=index_time_frequency)
        print('new_date_range', new_date_range)
        self.imputation_df = self.imputation_df.reindex(new_date_range, fill_value=np.nan)
        self.missing_indexes = self.imputation_df[self.imputation_df['Raw_Data'].isnull()].index
        print("reindexed", self.imputation_df)

    def KNNImputation(self, data):
        imputed_df = data.copy()
        imputer = KNNImputer(n_neighbors=5)
        imputed_df['Raw_Data'] = imputer.fit_transform(imputed_df)
        print(data)
        return imputed_df

    def rolling_mean(self, data, window=5):
        imputed_df = data.copy()
        return imputed_df

    def impute_spline(self):
        # imputed_df = data.copy()
        self.imputation_df['Raw_Data'] = self.imputation_df['Raw_Data'].interpolate(method='polynomial', order=3)
        self.imputation_df['spline'] = self.imputation_df['Raw_Data'].loc[self.missing_data_indexes]
        # return imputed_df
