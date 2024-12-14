import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from _plot_builder import plot_income_configured_data
from plots.plot_builder import PlotBuilder
from Models.MovingAverage import MovingAverage
from Models.ExpoSmooth import ExpoSmooth
from Models.DoubleExpoSmooth import DoubleExpoSmooth
from Models.TripleExpoSmooth import TripleExpoSmooth
from Models.HoltWinters import HoltWinters
from Models.HoltWinters2 import HoltWinters2
from Models.SARIMAX import SARIMAX

pb = PlotBuilder()
moving_average = MovingAverage()
expo_smooth = ExpoSmooth()
dbl_expo_smooth = DoubleExpoSmooth()
tes = TripleExpoSmooth()
hw = HoltWinters()
hw2 = HoltWinters2()
sarimax = SARIMAX()


data = st.session_state.df_for_import

# global income_configured_data_frame
st.set_page_config(layout="wide")
# data = pd.read_csv('Data/configured_csv.csv')
if data is not None:
    # Raw data
    fig = pb.plot_scatter(data, columns_list=['Raw_Data'])
    st.plotly_chart(fig, theme=None, use_container_width=True, key='raw')
    # # Moving Average
    # moving_average.fit_predict(data=data, window=30)
    # moving_average.anomalies()
    # fig = pb.plot_model_scatter(data=moving_average.model_df,
    #                             columns_list=['Raw_Data', 'Moving_Average'])
    # st.plotly_chart(fig, theme=None, use_container_width=True, key='mov_avg')
    # st.write(moving_average.anomalies())
    # print(moving_average.model_df)
    #
    # # Expo Smoothing
    # expo_smooth.fit_predict(data=data, alpha=0.2)
    # expo_smooth.anomalies()
    # fig = pb.plot_model_scatter(data=expo_smooth.model_df,
    #                             columns_list=['Raw_Data', 'Expo_Smooth'])
    # st.plotly_chart(fig, theme=None, use_container_width=True, key='exp_smth')
    # st.write(expo_smooth.anomalies())
    # print(expo_smooth.model_df)
    #
    # # Double Expo Smooth
    # dbl_expo_smooth.fit_predict(data=data, alpha=0.02, beta=0.9)
    # dbl_expo_smooth.anomalies()
    # fig = pb.plot_model_scatter(data=dbl_expo_smooth.model_df,
    #                             columns_list=['Raw_Data', 'Double_Expo_Smooth'])
    # st.plotly_chart(fig, theme=None, use_container_width=True, key='dbl_exp_smth')
    # st.write(dbl_expo_smooth.anomalies())
    # print(dbl_expo_smooth.model_df)

    # HoltWinters1
    # hw.fit_predict(data=data)
    # hw.anomalies()
    # fig = pb.plot_model_scatter(data=hw.model_df,
    #                             columns_list=['Raw_Data', 'Holt_Winters'])
    # st.plotly_chart(fig, theme=None, use_container_width=True, key='hw1')
    # # st.write(tes.anomalies())
    # print(expo_smooth.model_df)

    # Holt Winters2
    # hw2.fit_predict(data=data)
    # hw2.anomalies()
    # fig = pb.plot_model_scatter(data=hw2.model_df,
    #                             columns_list=['Raw_Data', 'Holt_Winters'])
    # st.plotly_chart(fig, theme=None, use_container_width=True, key='hw2')

    # SARIMAX
    sarimax.fit(data=data)
    sarimax.anomalies()
    print(sarimax.model_df)
    fig = pb.plot_model_scatter(data=sarimax.model_df,
                                columns_list=['Raw_Data', 'SARIMAX'])
    st.plotly_chart(fig, theme=None, use_container_width=True, key='sarimax')
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Anomalies")
        st.write(sarimax.anomalies())

    with col2:
        st.subheader("Anomalies amount")
        fig = pb.anomalies_pie(sarimax.model_df)
        st.plotly_chart(fig, theme=None, use_container_width=True, key='sarimax1')

    with col3:
        st.subheader("Model Quality")
        st.write(sarimax.model_quality_df)







# # #
# Moving Average
