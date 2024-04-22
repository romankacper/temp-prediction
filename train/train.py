# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 11:41:35 2024

@author: roman
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import os
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from keras import regularizers
from keras.layers import Dense, LSTM, Dropout
from sqlalchemy import create_engine


def normalize(data, column_names, mean_columns, std_columns):
    def normalize(x, center, scale):
        return (x - center) / scale

    # Inicjalizacja pustej listy do przechowywania znormalizowanych danych
    normalized_data = []

    # Iteracja przez każdą sekwencję w danych
    for sequence in data:
        sequence = pd.DataFrame(sequence, columns=column_names)
        for col in column_names:
            sequence[col] = normalize(sequence[col], mean_columns[col], std_columns[col])
        normalized_data.append(sequence.values)

    # Konwersja znormalizowanych danych z powrotem na tablicę numpy
    normalized_data = np.array(normalized_data)

    return normalized_data


def make_dataset(df, column_names, sequence_length, output_length ):
    data = df[column_names].values
    # print(data)
    # Inicjalizujemy listy dla naszych wejść i wyjść
    inputs = []
    targets = []
    
    # Tworzymy nasze wejścia i wyjścia
    for i in range(len(data) - sequence_length - output_length + 1):
        inputs.append(data[i:i+sequence_length])
        targets.append(data[i+sequence_length:i+sequence_length+output_length, 0])
    
    # Konwertujemy listy do tablic numpy
    inputs = np.array(inputs)
    targets = np.array(targets).round(2)
    
    normalized_inputs = normalize(inputs, column_names, mean_columns, std_columns)
    # Tworzymy nasz zestaw danych
    # Dodajemy dodatkowy wymiar do naszych danych wejściowych i wyjściowych
    normalized_inputs = normalized_inputs[None, :]
    targets = targets[None, :]
    dataset = tf.data.Dataset.from_tensor_slices((normalized_inputs, targets))
    
    return dataset

def replace_outliers_with_mean(df, means):
    for column in df.columns:
        if column in means.index:
            mean_value = means[column]
            std_value = df[column].std()
            df[column] = np.where(np.abs(df[column]-mean_value) > 2*std_value, mean_value, df[column])
    return df

# Utwórz silnik SQLAlchemy do połączenia z bazą danych MySQL
engine = create_engine('mysql+pymysql://myuser:mypassword@db/mydatabase')

# Pobierz dane z tabeli 'train' i 'test'
df = pd.read_sql('SELECT * FROM train', con=engine)
df_test = pd.read_sql('SELECT * FROM test', con=engine)

# Konwertuj kolumnę 'date' na typ datetime
df['date'] = pd.to_datetime(df['date'])
df_test['date'] = pd.to_datetime(df_test['date'])

train_len = int(len(df)*0.66)

df_train = df.iloc[:train_len]
df_val = df.iloc[train_len:train_len + (len(df) - train_len)]

column_names = df.drop(columns=["date"]).columns

# Obliczenie średniej i odchylenia standardowego dla kolumn wejściowych w zbiorze treningowym
mean_columns = df_train[column_names].mean()
std_columns = df_train[column_names].std()

df_test= replace_outliers_with_mean(df_test, mean_columns)
df_val= replace_outliers_with_mean(df_val, mean_columns)
df_train= replace_outliers_with_mean(df_train, mean_columns)

sequence_length = 14
output_length = 1
# Utworzenie zestawów danych treningowych, walidacyjnych i testowych
train_dataset = make_dataset(df_train, column_names, sequence_length, output_length)
val_dataset = make_dataset(df_val, column_names, sequence_length, output_length)
test_dataset = make_dataset(df_test, column_names, sequence_length, output_length)

# Budowanie modelu
model_lstm = Sequential()
model_lstm.add(LSTM(32, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), recurrent_dropout=0.001))
model_lstm.add(Dropout(0.05))
model_lstm.add(LSTM(32, kernel_regularizer=regularizers.l2(0.001), recurrent_dropout=0.001))
model_lstm.add(Dropout(0.05))
model_lstm.add(Dense(1))

# Kompilacja modelu
model_lstm.compile(optimizer=SGD(), loss='mse', metrics=['mae'])

# Trenowanie modelu
history_lstm = model_lstm.fit(train_dataset, epochs=1000, validation_data=val_dataset)

# Zapisanie modelu do pliku
model_lstm.save('/app/models/model_lstm.keras')

# Ewaluacja modelu
test_mae_lstm = model_lstm.evaluate(test_dataset)[1]
print(f"Test MAE: {test_mae_lstm:.2f}")

predictions = model_lstm.predict(test_dataset).flatten()

df_real = pd.DataFrame({
    'Date': df_test['date'],
    'Temperature': df_test['meantemp'].round(2)
})

df_predictions = pd.DataFrame({
    'Date': df_test['date'].iloc[sequence_length:],
    'Temperature': predictions.round(2)
})

df_predictions.to_csv('/app/predictions/temp_predictions.csv', index=False)
df_real.to_csv('/app/predictions/temp_real.csv', index=False)

while True:
    time.sleep(1)

