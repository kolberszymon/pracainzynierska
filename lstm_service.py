import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from configuration_file import configuration_var
from configuration_file import folder_locations
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import amazon_service

#Properties

sample_length = configuration_var['sample_length']
train_range = configuration_var['train_range']
test_range = 1 - train_range



def normalize_data(x):
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    return scaler.fit_transform(x)



def prepare_samples(x, y):
    
    x_samples = []
    y_samples = []
        
    for i in range(sample_length, len(x)):
        x_sample = x[i-sample_length:i]
        y_sample = y[i]
    
        x_samples.append(x_sample)
        y_samples.append(y_sample)
        
    x_samples = np.array(x_samples, dtype='object')
    y_samples = np.array(y_samples, dtype='object')
    
    x_samples = x_samples.reshape(x_samples.shape[0], x_samples.shape[1], 3)

    return x_samples, y_samples

def split_into_training_and_testing_set(x_samples, y_samples):
    
    train_size = int(len(x_samples) * train_range)

    x_samples_train = x_samples[:train_size]
    y_samples_train = y_samples[:train_size]

    x_samples_test = x_samples[train_size:]
    y_samples_test = y_samples[train_size:]
    
    return x_samples_train, y_samples_train, x_samples_test, y_samples_test

def model_compiling_and_saving(model, x_samples_train, y_samples_train, epochs, batch_size):
    
    model.compile(optimizer='adam',loss='mean_squared_error', metrics=['accuracy'])
    history = model.fit(x_samples_train, y_samples_train, epochs=epochs, batch_size=batch_size)
    model.save(folder_locations["trained_models"] + 'lstm_model' + configuration_var["picked_data_period"] + '.h5')
    
    return history
    