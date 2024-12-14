from abc import ABC, abstractmethod


class Model(ABC):

    def fit(self):
        pass

    def predict(self):
        pass

    def fit_predict(self):
        pass

    def anomalies(self):
        pass

    def plot_model(self):
        pass

    def plot_conf_intervals(self):
        pass

    def plot_anomalies(self):
        pass
