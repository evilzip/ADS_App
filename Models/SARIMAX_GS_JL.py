from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.fft import fft

class SARIMAX_GS_JL:
    def __init__(self):
        self.result_table = None
        self.parameters_list = None
        self.param_mini = None
        self.param_seasonal_mini = None
        self.model_results = None
        self.best_model = None
        self.model_df = pd.DataFrame()
        self.model_quality_df = pd.DataFrame()
        self.N_CORES = cpu_count() - 2  # -2 to avoid freezing station

        self.train = None
        self.test = None
        self.test_index = None

    def sarima_forecast(self, history, config):
        order, sorder, trend = config
        # define model
        model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False,
                        enforce_invertibility=False)
        # fit model
        model_fit = model.fit(disp=False)
        # make one step forecast
        yhat = model_fit.predict(len(history), len(history))
        return yhat[0]

    def measure_rmse(self, actual, predicted):
        return sqrt(mean_squared_error(actual, predicted))

    # split a univariate dataset into train/test sets
    def train_test_split(self, data, n_test):
        return data[:-n_test], data[-n_test:]

    # walk-forward validation for univariate data
    def walk_forward_validation(self, data, n_test, cfg):
        predictions = list()
        # split dataset
        train, test = self.train_test_split(data, n_test)
        # seed history with training dataset
        history = [x for x in train]
        # step over each time-step in the test set
        for i in range(len(test)):
            # fit model and make forecast for history
            yhat = self.sarima_forecast(history, cfg)
            # store forecast in list of predictions
            predictions.append(yhat)
            # add actual observation to history for the next loop
            history.append(test[i])
        # estimate prediction error
        error = self.measure_rmse(test, predictions)
        return error

    # score a model, return None on failure
    def score_model(self, data, n_test, cfg, debug=False):
        result = None
        # convert config to a key
        key = str(cfg)
        # show all warnings and fail on exception if debugging
        if debug:
            result = self.walk_forward_validation(data, n_test, cfg)
        else:
            # one failure during model validation suggests an unstable config
            try:
                # never show warnings when grid searching, too noisy
                with catch_warnings():
                    filterwarnings("ignore")
                    result = self.walk_forward_validation(data, n_test, cfg)
            except:
                error = None
        # check for an interesting result
        if result is not None:
            print(' > Model[%s] %.3f' % (key, result))
        return key, result

    def grid_search(self, data, cfg_list, n_test, parallel=True):
        scores = None
        if parallel:
            # execute configs in parallel
            executor = Parallel(n_jobs=self.N_CORES, backend='multiprocessing')
            tasks = (delayed(self.score_model)(data, n_test, cfg) for cfg in cfg_list)
            scores = executor(tasks)
        else:
            scores = [self.score_model(data, n_test, cfg) for cfg in cfg_list]
        # remove empty results
        scores = [r for r in scores if r[1] != None]
        # sort configs by error, asc
        scores.sort(key=lambda tup: tup[1])
        return scores



    def sarima_configs(self, seasonal=[0]):
        models = list()
        # define config lists
        p_params = [0, 1, 2]
        d_params = [0, 1]
        q_params = [0, 1, 2]
        t_params = ['n', 'c', 't', 'ct']
        P_params = [0, 1, 2]
        D_params = [0, 1]
        Q_params = [0, 1, 2]
        m_params = seasonal
        # create config instances
        for p in p_params:
            for d in d_params:
                for q in q_params:
                    for t in t_params:
                        for P in P_params:
                            for D in D_params:
                                for Q in Q_params:
                                    for m in m_params:
                                        cfg = [(p, d, q), (P, D, Q, m), t]
                                        models.append(cfg)
        return models


if __name__ == '__main__':
    sarimax_gs_jl = SARIMAX_GS_JL()
    # load dataset
    series = pd.read_csv('C:\DATA\ADS_App\Data\daily-total-female-births-in-cal.csv', header=0, index_col=0)
    data = series.values
    print(series)
    print(data.shape)
    plt.plot(series.index, series['Count'])
    plt.show()
    # data split
    n_test = 165
    # # model configs
    # cfg_list = sarima_configs()
    # model configs
    cfg_list = sarimax_gs_jl.sarima_configs()
    # grid search
    scores = sarimax_gs_jl.grid_search(data, cfg_list, n_test)
    print('done')
    # list top 3 configs
    for cfg, error in scores[:3]:
        print(cfg, error)