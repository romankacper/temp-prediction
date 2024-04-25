# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 12:34:37 2024

@author: roman
"""
import docker
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sqlalchemy import create_engine

client = docker.from_env()
# Utwórz silnik SQLAlchemy do połączenia z bazą danych MySQL
engine = create_engine('mysql+pymysql://myuser:mypassword@db/mydatabase')

df_real = pd.read_csv('/app/predictions/temp_real.csv')
df_real['Date'] = pd.to_datetime(df_real['Date'])
df_predictions = pd.read_csv('/app/predictions/temp_predictions.csv')
df_predictions['Date'] = pd.to_datetime(df_predictions['Date'])

st.title("Aplikacja internetowa do przewidywania temperatury powietrza dla Delhi")

# ustawienie zakresu daty za pomocą suwaka streamlit
start_date = df_real['Date'].iloc[13].to_pydatetime()
end_date = df_real['Date'].iloc[-3].to_pydatetime()

selected_end_date = st.slider('Wybierz datę końca okresu:', min_value=start_date, max_value=end_date)

difference = (df_real['Date'].max().to_pydatetime() - selected_end_date).days
# dodanie pola do wprowadzania liczby dni dla prognozy
days_to_predict = st.number_input('Wybierz liczbę dni na ile ma się wyświetlić predykcja:', min_value=2, max_value=difference, value=2, step=1)

# wybór prognozowanych danych
start_index = df_real[df_real['Date'] == selected_end_date].index[0] - 13
df_predicted = df_predictions.iloc[start_index : start_index + days_to_predict]

selected_end_date = selected_end_date + pd.Timedelta(days=days_to_predict)

# wybór danych z ostatnich 14 dni
mask = (df_real['Date'] >= (selected_end_date - pd.Timedelta(days=13))) & (df_real['Date'] <= selected_end_date)
df_selected = df_real.loc[mask]

# rysowanie wykresu za pomocą st.line_chart
chart_data = pd.concat([df_selected.set_index('Date')['Temperature'], df_predicted.set_index('Date')['Temperature']], axis=1)
chart_data.columns = ['Real', 'Predicted']
st.line_chart(chart_data)

flag_1 = 0
flag_2 = 0

uploaded_file_1 = st.file_uploader("Załącz plik csv, na którym mam wykonać predykcję", type=['csv', 'xlsx'])

if uploaded_file_1 is not None:
    # Możesz teraz użyć przesłanego pliku, na przykład wczytując go do ramki danych pandas
    uploaded_test = pd.read_csv(uploaded_file_1)
    # sprawdź, czy ramka danych zawiera wymagane kolumny
    required_columns = ['date', 'meantemp', 'humidity', 'wind_speed', 'meanpressure']
    if not all(column in uploaded_test.columns for column in required_columns):
        st.error('Plik nie zawiera wszystkich wymaganych kolumn. Proszę przesłać inny plik.')
        st.stop()

    # sprawdź, czy ramka danych zawiera co najmniej 16 wierszy
    if len(uploaded_test) < 16:
        st.error('Plik zawiera mniej niż 16 wierszy. Proszę przesłać inny plik.')
        st.stop()
    
    #uploaded_test.to_csv('/app/uploaded/uploaded1.csv', index=False)
    uploaded_test.to_sql('test', con=engine, if_exists='replace', index=False)
    flag_1 = 1

uploaded_file_2 = st.file_uploader("Załącz plik csv, na którym mam trenować model", type=['csv', 'xlsx'])

if uploaded_file_2 is not None:
    # Możesz teraz użyć przesłanego pliku, na przykład wczytując go do ramki danych pandas
    uploaded_train = pd.read_csv(uploaded_file_2)
    # sprawdź, czy ramka danych zawiera wymagane kolumny
    required_columns = ['date', 'meantemp', 'humidity', 'wind_speed', 'meanpressure']
    if not all(column in uploaded_train.columns for column in required_columns):
        st.error('Plik nie zawiera wszystkich wymaganych kolumn. Proszę przesłać inny plik.')
        st.stop()

    # sprawdź, czy ramka danych zawiera co najmniej 16 wierszy
    if len(uploaded_train) < 16:
        st.error('Plik zawiera mniej niż 16 wierszy. Proszę przesłać inny plik.')
        st.stop()
    
    #uploaded_test.to_csv('/app/uploaded/uploaded1.csv', index=False)
    uploaded_train.to_sql('train', con=engine, if_exists='replace', index=False)
    flag_2 = 1

if flag_1 == 1 or flag_2 == 1:
    #znajdź kontener 'train'
    container = client.containers.get('projekt-docker-train-1')
    # zrestartuj kontener
    container.restart()

flag_1 = 0
flag_2 = 0

