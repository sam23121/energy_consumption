import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from datetime import datetime, timedelta
import streamlit as st






def fit_data(df):
    df.columns = ['ds', 'y']
    # print(df.info())
    df['ds'] = pd.to_datetime(df['ds'])# .dt.to_timestamp()
    df['ds'] = df['ds'].dt.strftime('%Y-%m-01')
    df['ds'] = pd.to_datetime(df['ds'])
    # print(df["y"].min())
    # print(df["ds"].min())

    # define the model
    model = Prophet()
    # fit the model
    model.fit(df)

    return model, (df['ds'].tail(1).iloc[0].year, df['ds'].tail(1).iloc[0].month)

def get_next_12_months(date_tuple):
    year, month = date_tuple
    start_date = datetime(year, month, 1)
    next_12_months = []

    for i in range(2, 14):
        next_month = start_date + timedelta(days=30*i)
        formatted_month = next_month.strftime("%Y-%m")
        next_12_months.append([formatted_month])

    # print(next_12_months)

    future = pd.DataFrame(next_12_months)
    future.columns = ['ds']
    future['ds']= pd.to_datetime(future['ds'])
    
    
    return future


def predict(model, future):
    forecast = model.predict(future)
    # summarize the forecast
    # print(forecast[['ds', "yhat", 'yhat_lower', 'yhat_upper']].head())
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
    
    # plot forecast
    fig = model.plot(forecast)
    st.pyplot(fig)


