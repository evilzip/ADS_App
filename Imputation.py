from sklearn.impute import KNNImputer
from sklearn.datasets import load_diabetes
import pandas as pd


class Imputation:
    def __int__(self):
        self.imputation_df = pd.DataFrame()
        self.missing_indexes = None
        self.missing_df = pd.DataFrame()

    def find_missing_data(self, data):
        self.imputation_df = data.copy()
        self.imputation_df['Missing'] = data['Raw_Data'].isnull()
        self.missing_indexes = data[data['Raw_Data'].isnull()].index
        # self.missing_df = data.loc[self.missing_indexes]
        self.missing_df = self.imputation_df['Missing'][self.imputation_df['Missing']==True]

    def find_missing_index(self, data):
        data.sort_index(inplace=True)
        data_indexes = data.index
        # print('data_indexes', data_indexes)
        index_time_frequency = pd.infer_freq(data_indexes)
        data.index.freq = index_time_frequency
        print('index_time_frequency', index_time_frequency)
        print('data.index', data.index)


    def KNNImputation(self, data):
        imputed_df = data.copy()
        imputer = KNNImputer(n_neighbors=5)
        imputed_df['Raw_Data'] = imputer.fit_transform(imputed_df)
        print(data)
        return imputed_df

    def rolling_mean(self, data, window=5):
        imputed_df = data.copy()
        return imputed_df

    def spline(self, data):
        imputed_df = data.copy()
        imputed_df['Raw_Data'] = imputed_df['Raw_Data'].interpolate(method='polynomial', order=3)
        self.imputation_df['spline'] = imputed_df['Raw_Data'].loc[self.missing_indexes]
        return imputed_df
