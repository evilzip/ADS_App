import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, root_mean_squared_error, r2_score

from ConfidentIntervals.ConfidentIntervals import ConfidentIntervals

# LSTM
import tensorflow as tf
from keras.layers import Input, Dense, LSTM, Dropout
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping


class LSTM_1:
    def __init__(self):
        self.test = None
        self.train = None
        self.y_test = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.window_size = 60
        self.model_df = pd.DataFrame()
        # Time | Rw_Data | Y_Forecasted| Lower_CI | Upper_CI | Anomalies |

        self.model_quality_df = pd.DataFrame()
        # mape| rmse | mse

    def _train_test_split(self, data, k=0.9):
        train = data['Raw_Data'].iloc[:int(len(data) * k)]
        test = data['Raw_Data'].iloc[int(len(data) * k):]
        return train, test

    def _tes_optimizer(self, train, test, seasonal_periods=12):
        pass

    def define_model(self, x_train):
        input1 = Input(shape=(self.window_size, 1))
        x = LSTM(units=64, return_sequences=True)(input1)
        x = Dropout(0.2)(x)
        x = LSTM(units=64, return_sequences=True)(x)
        x = Dropout(0.2)(x)
        x = LSTM(units=64)(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='softmax')(x)
        dnn_output = Dense(1)(x)

        model = Model(inputs=input1, outputs=[dnn_output])
        model.compile(loss='mean_squared_error', optimizer='Nadam')
        model.summary()

        return model

    def fit_predict(self, data):
        # 1. fill up model result dataframe
        self.model_df = data.copy()

        # 2. Data scaling
        scaler = MinMaxScaler()
        scaler.fit(self.model_df['Raw_Data'].values.reshape(-1, 1))

        # 3. Train test split
        self.train, self.test = self._train_test_split(self.model_df, k=0.7)
        test_size = len(self.test)

        train_data = self.model_df['Raw_Data'][:-test_size]
        train_data = scaler.transform(self.train.values.reshape(-1, 1))

        x_train = []
        y_train = []

        for i in range(self.window_size, len(train_data)):
            x_train.append(train_data[i - self.window_size:i, 0])
            y_train.append(train_data[i, 0])

        test_data = self.model_df['Raw_Data'][-test_size - self.window_size:]
        test_data = scaler.transform(test_data.values.reshape(-1, 1))

        x_test = []
        y_test = []

        for i in range(self.window_size, len(test_data)):
            x_test.append(test_data[i - self.window_size:i, 0])
            y_test.append(test_data[i, 0])

        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        y_train = np.reshape(y_train, (-1, 1))
        y_test = np.reshape(y_test, (-1, 1))

        print('X_train Shape: ', x_train.shape)
        print('y_train Shape: ', y_train.shape)
        print('X_test Shape:  ', x_test.shape)
        print('y_test Shape:  ', y_test.shape)

        model = self.define_model(x_train=x_train)
        # history = model.fit(x_train, y_train, epochs=150, batch_size=32, validation_split=0.1, verbose=1)
        # Fitting the LSTM to the Training set
        callbacks = [EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)]
        history = model.fit(x_train, y_train, epochs=100, batch_size=32, callbacks=callbacks)

        result = model.evaluate(x_test, y_test)
        y_pred = model.predict(x_test)

        y_test_true = scaler.inverse_transform(y_test)
        y_test_pred = scaler.inverse_transform(y_pred)

        # 6. Prediction
        self.model_df['Y_Predicted'] = np.nan
        self.model_df['Y_Predicted'][self.test.index] = y_test_pred.reshape((-1))

        # 7. Calculate Model quality
        self._model_quality()

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

    def _conf_intervals(self):
        ci = ConfidentIntervals()
        true_data = self.test
        model_data = self.model_df['Y_Predicted'][self.test.index]
        lower_bond, upper_bound = ci.bootstrap_ci_2(true_data=true_data,
                                                    model_data=model_data)
        self.model_df['Upper_CI'] = self.model_df['Y_Predicted'] + upper_bound
        self.model_df['Lower_CI'] = self.model_df['Y_Predicted'] - lower_bond

    def _model_quality(self):
        r2 = r2_score(self.test,
                      self.model_df['Y_Predicted'][self.test.index])
        mape = mean_absolute_percentage_error(self.test,
                                              self.model_df['Y_Predicted'][self.test.index])
        rmse = root_mean_squared_error(self.test,
                                       self.model_df['Y_Predicted'][self.test.index])
        mse = mean_squared_error(self.test,
                                 self.model_df['Y_Predicted'][self.test.index])
        dict_data = {
            'MAPE': mape,
            'RMSE': rmse,
            'MSE': mse,
            'R2': r2
        }
        self.model_quality_df = pd.DataFrame(list(dict_data.items()), columns=['METRIC', 'Value'])
