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
    def __init__(self):
        self.TICKFONTSIZE = 16
        self.AXISTITLEFONT = 18

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
        curve = go.Scatter(  # Confidence bounds
            x=np.concatenate((data.index, data.index[::-1]), axis=0),
            y=np.concatenate((data['Lower_CI'], data['Upper_CI'][::-1]), axis=0),
            line=dict(color='rgba(255,255,255,0)'),
            fill='toself',
            fillcolor='rgba(0,0,0,0.2)',
            hoverinfo='skip',
            name='95% Confidence interval',
        )
        result_list.append(curve)
        # result_list.append(curve_l)
        return result_list

    def _anomalies_scatter(self, data: pd.DataFrame):
        anomalies_df = data[data['Anomalies'] == True]
        # curve = go.Scatter(x=anomalies_df['Time'], y=anomalies_df['Raw_Data'], name='Anomalies', mode='markers')
        curve = go.Scatter(x=anomalies_df.index.values, y=anomalies_df['Raw_Data'], name='Anomalies', mode='markers')
        return curve

    def _anomalies_v_lines(self, data: pd.DataFrame, fig):
        anomalies_df = data[data['Anomalies'] == True]
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
            tickfont=dict(size=self.TICKFONTSIZE),
            showline=True,
            linecolor='black',
            gridcolor='lightgrey',
        )
        fig.update_yaxes(
            mirror=True,
            ticks='outside',
            tickfont=dict(size=self.TICKFONTSIZE),
            showline=True,
            linecolor='black',
            gridcolor='lightgrey'
        )
        fig.update_yaxes(title_text=data.columns.values[0], title_font=dict(size=self.AXISTITLEFONT))
        fig.update_xaxes(title_text='Time', title_font=dict(size=self.AXISTITLEFONT))
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
            tickfont=dict(size=self.TICKFONTSIZE),
            showline=True,
            linecolor='black',
            gridcolor='lightgrey',
        )
        fig.update_yaxes(
            mirror=True,
            ticks='outside',
            tickfont=dict(size=self.TICKFONTSIZE),
            showline=True,
            linecolor='black',
            gridcolor='lightgrey'
        )
        fig.update_yaxes(title_text=data.columns.values[0], title_font=dict(size=self.AXISTITLEFONT))
        fig.update_xaxes(title_text='Time', title_font=dict(size=self.AXISTITLEFONT))
        return fig

    def anomalies_pie(self, data):
        labels = ['Anomalies', 'Data Points']
        total_points_amount = data.shape[0]
        anomalies_amount = data['Anomalies'][data['Anomalies'] == True].count()
        points_amount = data.shape[0] - anomalies_amount
        values = [anomalies_amount, points_amount]
        # fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
        # fig.update_traces(textinfo='label+value')
        fig = go.Figure()
        # fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'pie'}, {'type': 'pie'}]])
        fig.add_trace(go.Pie(labels=labels,
                             values=values,
                             textinfo='percent+value',
                             hole=0.5,
                             pull=[0, 0.1]))
        # fig.add_trace(go.Pie(labels=labels, values=values, textinfo='label+percent'), 1, 2)
        fig.update_layout(autosize=True),
                          # width=500,
                          # height=500)
        fig.update_layout(showlegend=True)
        fig.update_layout(annotations=[dict(text=f"Total: {total_points_amount}", x=0.5, y=0.5,
                                            font_size=16, showarrow=False, xanchor="center")])
        return fig

    def plot_missing_data(self, data, fig):
        missing_data_index = data[data['Raw_Data'].isnull()].index
        for index in missing_data_index:
            fig.add_vline(x=index, line_color='red', line_width=0.8)
        return fig

    def plot_imputed_data(self, data, fig):
        imputed_data = data[data.notnull()]
        curve = go.Scatter(x=imputed_data.index.values,
                           y=imputed_data.values,
                           mode='markers',
                           marker=dict(color='red'),
                           name='imputed data')
        fig.add_trace(curve)
        fig.update_layout(title='Raw_Data with Imputed data', title_x=0.5)
        return fig

    def missing_pie(self, data):
        labels = ['Missing Data', 'Data Points']
        total_points_amount = data.shape[0]
        missing_amount = data['Missing'][data['Missing'] == True].count()
        points_amount = data.shape[0] - missing_amount
        values = [missing_amount, points_amount]
        fig = go.Figure()
        # fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'pie'}, {'type': 'pie'}]])
        fig.add_trace(go.Pie(labels=labels,
                             values=values,
                             textinfo='percent+value',
                             hole=0.5,
                             pull=[0, 0.1]))
        # fig.add_trace(go.Pie(labels=labels, values=values, textinfo='label+percent'), 1, 2)
        fig.update_layout(autosize=True)
        fig.update_layout(showlegend=True)
        fig.update_layout(annotations=[dict(text=f"Total: {total_points_amount}", x=0.5, y=0.5,
                                            font_size=16, showarrow=False, xanchor="center")])
        return fig
