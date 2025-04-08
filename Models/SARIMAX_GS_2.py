import warnings
from itertools import product
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from scipy.fft import fft
from statsmodels.tools.eval_measures import mse
from ConfidentIntervals.ConfidentIntervals import ConfidentIntervals
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, root_mean_squared_error, r2_score


class SARIMAX_GS_2:
    def __init__(self):

        self.model_df = pd.DataFrame()
        # Time | Rw_Data | Y_Forecasted| Lower_CI | Upper_CI | Anomalies |

        self.model_quality_df = pd.DataFrame()
        # mape| rmse | mse

        self.model_info = pd.DataFrame()
        self.model_df_least_MAPE = pd.DataFrame()
        self.model_df_least_MSE = pd.DataFrame()
        self.model_df_least_AIC = pd.DataFrame()
        self.model_df_least_BIC = pd.DataFrame()
        self.best_models = pd.DataFrame()

        self.train = None
        self.test = None
        self.test_index = None
        self.dominant_period = None

    def SARIMA_grid(self, endog, order, seasonal_order):
        warnings.simplefilter("ignore")

        # create an empty list to store values
        model_info = []

        # fit the model
        for i in order:
            for j in seasonal_order:
                try:
                    print(f'Current fitting orders: {i}, seasonal_order: {j}')
                    model_fit = SARIMAX(endog=endog, order=i, seasonal_order=j).fit(disp=False)
                    predict = model_fit.predict()

                    # calculate evaluation metrics: MAPE, RMSE, AIC & BIC
                    MAPE = (abs((endog - predict)[1:]) / (endog[1:])).mean()
                    MSE = mse(endog[1:], predict[1:])
                    AIC = model_fit.aic
                    BIC = model_fit.bic

                    # save order, seasonal order & evaluation metrics
                    model_info.append([i, j, MAPE, MSE, AIC, BIC])
                except:
                    continue

        # create a dataframe to store info of all models
        columns = ["order", "seasonal_order", "MAPE", "MSE", "AIC", "BIC"]
        model_info = pd.DataFrame(data=model_info, columns=columns)
        print('model_info' + '\n', model_info.head())
        L1 = model_info[model_info.MAPE == model_info.MAPE.min()]
        L2 = model_info[model_info.MSE == model_info.MSE.min()]
        L3 = model_info[model_info.AIC == model_info.AIC.min()]
        L4 = model_info[model_info.BIC == model_info.BIC.min()]
        self.best_models = pd.concat((L1, L2, L3, L4))
        print('best_models' + '\n', self.best_models.head())
        self.model_info = model_info

    def pdq_PDQm_finder(self):
        # p = [1, 2]
        # d = [0, 1]
        # q = [1, 2]
        # P = [0, 1]
        # D = [0, 1, 2]
        # Q = [0, 1]
        p = [1,2]
        d = [0,1]
        q = [1,2]
        P = [0,1]
        D = [0]
        Q = [0]
        self.dominant_period, _, _ = self.fft_analysis(self.model_df['Raw_Data'].values)
        s = [self.dominant_period]
        pdq = list(product(p, d, q))
        PDQm = list(product(P, D, Q, s))
        return pdq, PDQm

    def fft_analysis(self, signal):
        # Linear detrending
        slope, intercept = np.polyfit(np.arange(len(signal)), signal, 1)
        trend = np.arange(len(signal)) * slope + intercept
        detrended = signal - trend
        fft_values = fft(detrended)
        frequencies = np.fft.fftfreq(len(fft_values))
        # Remove negative frequencies and sort
        positive_frequencies = frequencies[frequencies > 0]
        magnitudes = np.abs(fft_values)[frequencies > 0]
        # Identify dominant frequency
        dominant_frequency = positive_frequencies[np.argmax(magnitudes)]
        # Convert frequency to period (e.g., days, weeks, months, etc.)
        dominant_period = round(1 / dominant_frequency)
        return dominant_period, positive_frequencies, magnitudes

    def train_test_split(self, data, k=0.9):

        # train = data['Raw_Data'].iloc[:int(len(data) * k)]
        # test = data['Raw_Data'].iloc[int(len(data) * k):]
        train = data['Raw_Data'].iloc[:int(len(data) * k)]
        test = data['Raw_Data'].iloc[int(len(data) * k):]
        return train, test

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

    def fit_predict(self, data):
        # 1. fill up model result dataframe
        self.model_df_least_MAPE = \
            self.model_df_least_MSE = \
            self.model_df_least_AIC = \
            self.model_df_least_BIC = \
            self.model_df = data.copy()

        # 2. Train test split
        train, test = self.train_test_split(data, k=0.7)
        print('train/test', train, test)
        self.train, self.test = train, test

        # 3. Find dominant period = main season = m
        print("data['Raw_Data'].values", type(data['Raw_Data'].values[0]))
        print('dominant_period', self.dominant_period)

        # 4. Find pdq and PDQ
        pdq, PDQm = self.pdq_PDQm_finder()

        # 5. Execute Grid search
        self.SARIMA_grid(endog=train, order=pdq, seasonal_order=PDQm)

        # Take the configurations of the best models (!!! AFTER GS execution !!!)
        model_dfs = [self.model_df_least_MAPE,
                     self.model_df_least_MSE,
                     self.model_df_least_AIC,
                     self.model_df_least_BIC]
        ord_list = [tuple(self.best_models.iloc[i, 0]) for i in range(self.best_models.shape[0])]
        s_ord_list = [tuple(self.best_models.iloc[i, 1]) for i in range(self.best_models.shape[0])]
        preds, ci_low, ci_up, MAPE_test = [], [], [], []

        # Fit the models and compute the forecasts
        for i in range(4):
            # print('ord_list[i]', ord_list[i])
            # print('s_ord_list[i]', s_ord_list[i])
            model_fit = SARIMAX(endog=train, order=ord_list[i],
                                seasonal_order=s_ord_list[i]).fit(disp=False)  # Fit the model
            # model_fit = SARIMAX(endog=train, order=ord_list[i],
            #                     seasonal_order=s_ord_list[i]).fit(disp=False)  # Fit the model
            # Compute preds
            pred_summary = model_fit.get_prediction(test.index[0],
                                                    test.index[-1]).summary_frame()
            # pred_summary = model_fit.get_prediction(start=test.index[0]).summary_frame()
            # Store results
            # preds.append(pred_summary['mean'])
            # ci_low.append(pred_summary['mean_ci_lower'][test.index])
            # ci_up.append(pred_summary['mean_ci_upper'][test.index])
            # MAPE_test.append((abs((test - pred_summary['mean']) / (test)).mean()))

            model_dfs[i]['Y_Predicted'] = pred_summary['mean']
            # model_dfs[i]['Lower_CI'] = pred_summary['mean_ci_lower'][test.index]
            # model_dfs[i]['Upper_CI'] = pred_summary['mean_ci_upper'][test.index]

        # Calculate Model quality
        self._model_quality()
        self.model_df = self.model_df_least_MAPE.copy()

        # print(model_dfs[i])
        # Mark anomalies which is outside of CI
        # self.anomalies(self.model_df_least_MAPE)

    def _conf_intervals(self):
        ci = ConfidentIntervals()
        true_data = self.test
        model_data = self.model_df['Y_Predicted'][self.test.index]
        lower_bond, upper_bound = ci.stats_ci(true_data=true_data,
                                              model_data=model_data)
        self.model_df['Upper_CI'] = self.model_df['Y_Predicted'] + upper_bound
        self.model_df['Lower_CI'] = self.model_df['Y_Predicted'] - lower_bond

    def _model_quality(self):
        # self.model_df = self.model_df_least_MAPE.copy()
        r2 = r2_score(self.test,
                      self.model_df_least_MAPE['Y_Predicted'][self.test.index])
        mape = mean_absolute_percentage_error(self.test,
                                              self.model_df_least_MAPE['Y_Predicted'][self.test.index])
        print('mape data', self.test, self.model_df_least_MAPE['Y_Predicted'][self.test.index])
        rmse = root_mean_squared_error(self.test,
                                       self.model_df_least_MAPE['Y_Predicted'][self.test.index])
        mse = mean_squared_error(self.test,
                                 self.model_df_least_MAPE['Y_Predicted'][self.test.index])
        print('mape', mape)
        dict_data = {
            'MAPE': mape,
            'RMSE': rmse,
            'MSE': mse,
            'R2': r2
        }
        # self.model_quality_df = pd.DataFrame(list(dict_data.items()), columns=['METRIC', 'Value'])
        self.model_quality_df = pd.DataFrame([[mape, rmse, mse, r2]], columns=['MAPE', 'RMSE', 'MSE', 'R2'])
