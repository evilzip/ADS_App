import views.upload_page
from IncomeDataProcessor import IncomeDataProcessor
from plotly.subplots import make_subplots
from views import upload_page
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


def plot_income_configured_data(data):
    curve = go.Scatter(x=data.iloc[:, 0], y=data.iloc[:, 1], name='test name', mode='lines+markers')
    fig = go.Figure(curve)
    fig.update_layout(plot_bgcolor='white')
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    return fig
# # #
# # # Plot!


# print(upload_page.cda)
# #
# # df = upload_page.df
# # idp = IncomeDataProcessor()
# # time_labels, x_points, y_values, time_series = idp.prep(df, value_col='close', time_col='index')
# #
# # # Create distplot with custom bin_size
# # curve = go.Scatter(x=df['timestamp'], y=df['close'], name='test name', mode='lines+markers')
