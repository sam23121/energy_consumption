import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from datetime import datetime, timedelta
from dateutil import parser
import calendar
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

def change_date_format(date_string, new_format):

    parts = date_string.split('/')
    if len(parts) != 3:
        parts = date_string.split('-')
    if len(parts) != 3:
        print("Error: Invalid date format")
        return None

    try:
        for i in parts:
            if int(i) > 40:
                year = int(i)
            
        
        month = int(parts[1])

        

        # Create a new datetime object with the adjusted day value
        days_in_month = calendar.monthrange(year, month)[1]
        date = datetime(year, month, days_in_month)

        # Convert the new datetime object to the desired format
        new_date_string = date.strftime(new_format)
        return new_date_string
    except (ValueError, OverflowError) as e:
        print(f"Error: {e}")
        return None

        
    # Convert input date string to datetime object
    # date = datetime.strptime(date_string, current_format)

    # # Create a new datetime object with the desired day (28)
    # new_date = datetime(date.year, date.month, 28)

    # # Convert the new datetime object to the desired format
    # new_date_string = new_date.strftime(new_format)

    # return new_date_string


def predict(model, future):
    forecast = model.predict(future)
    # summarize the forecast
    # print(forecast[['ds', "yhat", 'yhat_lower', 'yhat_upper']].head())
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
    
    # plot forecast
    fig = model.plot(forecast)
    st.pyplot(fig)


