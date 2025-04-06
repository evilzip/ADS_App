from plotly.subplots import make_subplots
import views.upload_page
from views import upload_page
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import numpy as np
import datetime


class PlotBuilder:
    # Предполагается что после обработки на входе все методы построения графиков будут
    # рассчитывать на структуру данных датафрэйм со столбцами: ['time', 'value']
    # возможно появятся столбцы после работы моделей.
    def __init__(self):
        self.TICKFONTSIZE = 16 + 18
        self.AXISTITLEFONT = 18 + 18
        self.TITLEFONT = 20 + 18
        self.LEGENDFONT = 12 + 18

    def _scatter_single(self, data: pd.DataFrame, column_name):
        name = data.columns.values[1]
        # curve = go.Scatter(x=data['Time'], y=data[column_name], name=name, mode='lines+markers')
        curve = go.Scatter(x=data.index.values, y=data[column_name], name=name, mode='lines+markers')
        return curve

    def _scatter_list(self, data: pd.DataFrame, columns_list: list):
        result_list = []
        for curve in columns_list:
            # curve = go.Scatter(x=data['Time'], y=data[curve], name=curve, mode='lines+markers')
            curve = go.Scatter(x=data.index.values,
                               y=data[curve],
                               name=curve,
                               mode='lines+markers',
                               marker=dict(
                                   size=4,
                                   # opacity=0.5,
                                   # line=dict(
                                   #     color='MediumPurple',
                                   #     width=8
                                   # )
                               )
                               )
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
        print('data', data)
        anomalies_df = data[data['Anomalies'] == True]
        # curve = go.Scatter(x=anomalies_df['Time'], y=anomalies_df['Raw_Data'], name='Anomalies', mode='markers')
        curve = go.Scatter(x=anomalies_df.index.values,
                           y=anomalies_df['Raw_Data'],
                           name='Anomalies',
                           mode='markers',
                           marker=dict(
                               size=8,
                               symbol="diamond-open"),
                           )
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
        # self._anomalies_v_lines(data, fig=fig)
        fig.update_layout(plot_bgcolor='white')
        title = ', '.join(columns_list)
        fig.update_layout(title=dict(text=title, font=dict(size=self.TITLEFONT)), title_x=0.5, )
        fig.update_layout(legend=dict(font=dict(size=self.LEGENDFONT)))
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
        fig.update_layout(title=dict(text=title, font=dict(size=self.TITLEFONT)), title_x=0.5, )
        fig.update_layout(legend_font_size=self.LEGENDFONT)
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
        fig.update_layout(title=dict(text='Raw_Data with Imputed data', font=dict(size=self.TITLEFONT)), title_x=0.5, )
        fig.update_layout(legend=dict(font=dict(size=self.LEGENDFONT)))

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

    def heatmap_missing(self, data):
        fig = go.Figure()
        z = [data['Missing'].astype(int)]
        heat_map = go.Heatmap(z=z,
                              x=data['Missing'].index,
                              y=[''],
                              showscale=False,
                              colorscale='Blues'
                              )
        fig.add_trace(heat_map)
        fig.update_layout(title='Missing data', title_x=0.5)
        # fig.update_layout(coloraxis=False)
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
        # fig.update_yaxes(title_text=data.columns.values[0], title_font=dict(size=self.AXISTITLEFONT))
        fig.update_layout(
            autosize=False,
            height=250)
        fig.update_xaxes(title_text='Time', title_font=dict(size=self.AXISTITLEFONT))
        return fig

    def heatmap_anomalies(self, data):
        fig = go.Figure()
        z = [data['Anomalies'].astype(int)]
        heat_map = go.Heatmap(z=z,
                              x=data['Anomalies'].index,
                              y=[''],
                              showscale=False,
                              colorscale='Blues'
                              )
        fig.add_trace(heat_map)
        fig.update_layout(title='Anomalies', title_x=0.5)
        # fig.update_layout(coloraxis=False)
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
        # fig.update_yaxes(title_text=data.columns.values[0], title_font=dict(size=self.AXISTITLEFONT))
        fig.update_layout(
            autosize=False,
            height=250)
        fig.update_xaxes(title_text='Time', title_font=dict(size=self.AXISTITLEFONT))
        return fig

    def bar_missing(self, data):
        fig = go.Figure()
        bars = go.Bar(x=data.index, y=data['Missing'].astype(int), name='Missing Data')
        fig.add_trace(bars)
        fig.update_layout(plot_bgcolor='white')
        fig.update_layout(title=dict(text='Missing data', font=dict(size=self.TITLEFONT)), title_x=0.5, )
        fig.update_layout(showlegend=True)
        fig.update_layout(legend=dict(font=dict(size=self.LEGENDFONT)))
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
        # fig.update_yaxes(title_text=data.columns.values[0], title_font=dict(size=self.AXISTITLEFONT))
        fig.update_layout(
            autosize=False,
            height=400)
        fig.update_xaxes(title_text='Time', title_font=dict(size=self.AXISTITLEFONT))
        fig.update_traces(marker_color='red',
                          opacity=0.6)  # transparency

        return fig

    def bar_anomalies(self, data):
        fig = go.Figure()
        bars = go.Bar(x=data.index, y=data['Anomalies'].astype(int), name='Detected Anomalies')
        fig.add_trace(bars)
        fig.update_layout(plot_bgcolor='white')
        fig.update_layout(title=dict(text='Detected Anomalies', font=dict(size=self.TITLEFONT)), title_x=0.5, )
        fig.update_layout(showlegend=True)
        fig.update_layout(legend=dict(font=dict(size=self.LEGENDFONT)))
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
        # fig.update_yaxes(title_text=data.columns.values[0], title_font=dict(size=self.AXISTITLEFONT))
        fig.update_layout(
            autosize=False,
            height=400)
        fig.update_xaxes(title_text='Time', title_font=dict(size=self.AXISTITLEFONT))
        fig.update_traces(marker_color='red',
                          opacity=0.6)  # transparency

        return fig

    def missing_subplot(self, data):
        fig = make_subplots(rows=2, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.1,
                            subplot_titles=['Raw Data with Imputed Data',
                                            'Detected missing data and indexes'])

        raw_data = self._scatter_list(data, columns_list=['Raw_Data'])
        imputed_data = data['spline'][data['spline'].notnull()]
        imputed_curve = go.Scatter(x=imputed_data.index.values,
                                   y=imputed_data.values,
                                   mode='markers',
                                   marker=dict(color='red'),
                                   name='Imputed data')
        fig.add_traces(raw_data, 1, 1)
        fig.add_trace(imputed_curve, 1, 1)

        bars = go.Bar(x=data.index, y=data['Missing'].astype(int), name='Missing Data')

        fig.add_trace(bars, 2, 1)
        fig.update_traces(marker_color='red',
                          opacity=0.6, row=2, col=1)  # transparency
        fig.update_layout(
            autosize=False,
            height=1000)
        fig.update_layout(plot_bgcolor='white')
        fig.update_layout(title='Missing data', title_x=0.5)
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
