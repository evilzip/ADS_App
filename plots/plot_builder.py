import plotly.graph_objects as go
from plotly.subplots import make_subplots
import views.upload_page
from IncomeDataProcessor import IncomeDataProcessor
from plotly.subplots import make_subplots
from views import upload_page
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import numpy as np


class PlotBuilder:
    # Предполагается что после обработки на входе все методы построения графиков будут
    # рассчитывать на структуру данных датафрэйм со столбцами: ['time', 'value']
    # возможно появятся столбцы после работы моделей.
    def __int__(self):
        pass

    def _scatter_single(self, data: pd.DataFrame, column_name):
        name = data.columns.values[1]
        # curve = go.Scatter(x=data['Time'], y=data[column_name], name=name, mode='lines+markers')
        curve = go.Scatter(x=data.index.values, y=data[column_name], name=name, mode='lines+markers')
        return curve

    def _scatter_list(self, data: pd.DataFrame, columns_list: list):
        result_list = []
        for curve in columns_list:
            # curve = go.Scatter(x=data['Time'], y=data[curve], name=curve, mode='lines+markers')
            curve = go.Scatter(x=data.index.values, y=data[curve], name=curve, mode='lines+markers')
            result_list.append(curve)
        return result_list

    def _scatter_ci_list(self, data: pd.DataFrame):
        columns_list = ['Lower_CI', 'Upper_CI']
        result_list = []
        # for curve in columns_list:
        #     # curve = go.Scatter(x=data['Time'],
        #     #                    y=data[curve],
        #     #                    name=curve,
        #     #                    mode='lines+markers',
        #     #                    line=dict(color='green', dash='dash')
        #     #                    )
        #     curve = go.Scatter(x=data.index,
        #                        y=data[curve],
        #                        name=curve,
        #                        mode='lines+markers',
        #                        line=dict(color='green', dash='dash')
        #                        )
        curve = go.Scatter(  # Confidence bounds
            x=np.concatenate((data.index, data.index[::-1]), axis=0),
            y=np.concatenate((data['Lower_CI'], data['Upper_CI'][::-1]), axis=0),
            line=dict(color='rgba(255,255,255,0)'),
            fill='toself',
            fillcolor='rgba(0,0,0,0.2)',
            hoverinfo='skip',
            name='95% Confidence interval',
        )
        # curve_u = go.Scatter(x=data.index,
        #                      y=data['Upper_CI'],
        #                      fill='tozeroy',
        #                      name='Upper_CI',
        #                      mode='lines',
        #                      line=dict(color='green')
        #                      )
        # curve_l = go.Scatter(x=data.index,
        #                      y=data['Lower_CI'],
        #                      fill='tozeroy',
        #                      name='Lower_CI',
        #                      mode='lines',
        #                      line=dict(color='green')
        #                      )
        result_list.append(curve)
        # result_list.append(curve_l)
        return result_list

    def _anomalies_scatter(self, data: pd.DataFrame):
        anomalies_df = data[data['Anomalies'] == True]
        print('anomalies_df', anomalies_df)
        # curve = go.Scatter(x=anomalies_df['Time'], y=anomalies_df['Raw_Data'], name='Anomalies', mode='markers')
        curve = go.Scatter(x=anomalies_df.index.values, y=anomalies_df['Raw_Data'], name='Anomalies', mode='markers')
        return curve

    def _anomalies_v_lines(self, data: pd.DataFrame, fig):
        anomalies_df = data[data['Anomalies'] == True]
        print('anomalies_df', anomalies_df)
        for index in anomalies_df.index:
            fig.add_vline(x=index, line_color="red", line_width=1)

    def plot_model_scatter(self, data: pd.DataFrame, columns_list: list):
        curve = self._scatter_list(data, columns_list)
        curve_ci = self._scatter_ci_list(data)
        anomalies = self._anomalies_scatter(data)
        fig = go.Figure(curve)
        fig.add_traces(curve_ci)
        fig.add_trace(anomalies)
        self._anomalies_v_lines(data, fig=fig)
        fig.update_layout(plot_bgcolor='white')
        title = ', '.join(columns_list)
        fig.update_layout(title=title, title_x=0.5)
        fig.update_xaxes(
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            gridcolor='lightgrey',
        )
        fig.update_yaxes(
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            gridcolor='lightgrey'
        )
        fig.update_yaxes(title_text=data.columns.values[0])
        fig.update_xaxes(title_text='Time')
        return fig

    def plot_scatter(self, data: pd.DataFrame, columns_list: list):
        curve = self._scatter_list(data, columns_list)
        # curve_ci = self._scatter_ci_list(data)
        # anomalies = self._anomalies_scatter(data)
        fig = go.Figure(curve)
        # fig.add_traces(curve_ci)
        # fig.add_trace(anomalies)
        fig.update_layout(plot_bgcolor='white')
        title = ', '.join(columns_list)
        fig.update_layout(title=title, title_x=0.5)
        fig.update_xaxes(
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            gridcolor='lightgrey',
        )
        fig.update_yaxes(
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            gridcolor='lightgrey'
        )
        fig.update_yaxes(title_text=data.columns.values[0])
        fig.update_xaxes(title_text='Time')
        return fig

    def anomalies_pie(self, data):
        labels = ['Anomalies', 'Data Points']
        anomalies_amount = data['Anomalies'][data['Anomalies'] == True].count()
        points_amount = data.shape[0] - anomalies_amount
        values = [anomalies_amount, points_amount]
        # print('values', values)
        # fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
        # fig.update_traces(textinfo='label+value')
        fig = go.Figure()
        fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'pie'}, {'type': 'pie'}]])
        fig.add_trace(go.Pie(labels=labels, values=values, textinfo='label+value'), 1, 1)
        fig.add_trace(go.Pie(labels=labels, values=values, textinfo='label+percent'), 1, 2)
        fig.update_layout(showlegend=False)
        return fig
