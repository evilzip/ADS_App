import streamlit as st
import pandas as pd
from IncomeDataProcessor import IncomeDataProcessor
import _plot_builder
import _plot_builder

df_for_import = pd.DataFrame()

st.title("Upload dataset here - Only '*.cvs' File (temporarily)")
idp = IncomeDataProcessor()
#
uploaded_file = st.file_uploader("Choose your dataset (csv file only)", key="upload_1")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)
    # Sample data

    col1, col2 = st.columns(2)

    with col1:
        time_column = tuple(df.columns.tolist() + ['Indexes'])
        # Dropdown menu for selecting a category
        selected_time_column = st.selectbox('Select a time column', time_column, key='time_column')
        # Display the selected category
        st.write('You selected time column:', selected_time_column)
    with col2:
        value_column = tuple(df.columns.tolist())
        selected_value_column = st.selectbox("select a value column", value_column, key='value_column')
        st.write('You selected value column:', selected_value_column)
    # тут нужно сделать обраточик ошибки
    income_configured_data_frame = df[[selected_time_column, selected_value_column]]
    # Rename columns to Time and Value
    income_configured_data_frame.columns = ['Time', 'Raw_Data']
    st.write('Yours configured timeseries dataframe', income_configured_data_frame)

    min_value = income_configured_data_frame['Time'].tolist()[0]
    max_value = income_configured_data_frame['Time'].tolist()[-1]
    options = income_configured_data_frame['Time'].tolist()
    #
    start_income_df, end_income_df = st.select_slider(
        "Select a range for analysis",
        options=options,
        value=(min_value, max_value),
        key='slider_time_limits'
    )

    start_index = income_configured_data_frame.index[
        income_configured_data_frame['Time'] == start_income_df][0]

    end_index = income_configured_data_frame.index[income_configured_data_frame['Time'] == end_income_df][0]

    income_configured_data_frame = income_configured_data_frame.iloc[start_index:end_index]
    if st.button("Confirm", key='confirm_btn'):
        st.write("Configured dataframe has been saved")
        # import income_configured_data_frame to plot builder ?????
        if 'df_for_import' not in st.session_state:
            st.session_state['df_for_import'] = income_configured_data_frame
        else:
            st.session_state['df_for_import'] = income_configured_data_frame
        df_for_import = income_configured_data_frame
        income_configured_data_frame.to_csv('Data/configured_csv.csv')
    # Сделать кнопку отправить/сохранить
    # сохранять новый датафрэйм в файл
