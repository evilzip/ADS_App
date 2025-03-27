import pandas as pd
import random
import numpy as np


def timeseries_train_test_split(X, test_size):
    """
        Perform train-test split with respect to time series structure
    """

    # считаем индекс в датафрейме, после которого начинается тестовый отрезок
    # print('X_index', X.index)
    test_index = int(len(X) * (1 - test_size))  # + X.index[0]
    # print("test_index", test_index)

    # разбиваем весь датасет на тренировочную и тестовую выборку
    X_train = X.iloc[:test_index]
    # y_train = y.iloc[:test_index]
    X_test = X.iloc[test_index:]
    # y_test = y.iloc[test_index:]
    # print('X_Train', X_train)
    # print('X_test', X_test)

    return X_train, X_test, test_index


def generate_missing_data(data):
    random.seed(77)
    # посчитаем количество наблюдений
    n_samples = len(data['Raw_Data'])
    # вычислим 20 процентов от этого числа,
    # это будет количество пропусков
    how_many = int(0.01 * n_samples)
    # случайным образом выберем 20 процентов значений индекса
    mask_target = random.sample(list(data.index.values), how_many)
    # и заполним их значением NaN в столбце target
    data.loc[mask_target] = np.nan
    return data
