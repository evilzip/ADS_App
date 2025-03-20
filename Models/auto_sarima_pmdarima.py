import numpy as np
import pandas as pd
from sktime.utils.plotting import plot_series
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.base import ForecastingHorizon
from scipy import stats
import statsmodels.api as sm

import matplotlib.pyplot as plt


import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

ads = pd.read_csv('C:\DATA\ADS_App\Data\pdv_ads.csv', index_col=['Time'], parse_dates=['Time']).asfreq('H')
print(ads.shape)
# print(ads)
# ads.head()
#
#
#
y = ads.Ads/10**3  # в тысячах :)

plt.plot(ads.index, ads.Ads)
plt.title('Просмотры рекламы (тыс. часов за час)', fontsize=20, color='black')
plt.tick_params(axis = 'both', which = 'major', labelsize = 14)
plt.show()



y_train, y_test = temporal_train_test_split(y, test_size=48)
fh = ForecastingHorizon(y_test.index, is_relative=False)

plot_series(y_train, y_test, labels=["y_train", "y_test"])
plt.show
print(y_train.shape[0], y_test.shape[0])

import pmdarima as pm
from pmdarima import model_selection

arima_model = pm.auto_arima(

    y_train,
    start_p=1, start_q=1,
    max_p=5, max_q=5,

    seasonal=True, m=24,
    start_P=0, start_Q=0,
    max_P=2, max_Q=2,

    max_D=2, max_d=2,
    alpha=0.05,
    test='kpss',
    seasonal_test='ocsb',

    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=False,
    n_fits=100,
    information_criterion='bic',
    out_of_sample_size=7

    # Можно делать перебор гипер-параметров
    # на основе метрики на тестовой выборке
    # scoring='mae',
)

y_pred, pred_ci = arima_model.predict(
    n_periods=48,
    return_conf_int=True,
    alpha=0.05
)

mape(y_pred, y_test)

df_forecast = pd.DataFrame({'y_pred': y_pred, 'ci_lower': pred_ci[:,0], 'ci_upper': pred_ci[:,1]})
df_forecast.index = fh.to_absolute_index()
df_forecast.head()

fig, ax = plot_series(y_train, y_test, df_forecast.y_pred, labels=["y_train", "y_test", "y_pred"]);
ax.fill_between(
    ax.get_lines()[-1].get_xdata(),
    df_forecast["ci_lower"],
    df_forecast["ci_upper"],
    alpha=0.2,
    color=ax.get_lines()[-1].get_c(),
    label=f"95% prediction intervals",
)
ax.legend(loc='lower left')
plt.show()