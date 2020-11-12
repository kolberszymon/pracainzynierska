import pandas as pd
import numpy as np
import math
from configuration_file import configuration_var
from configuration_file import folder_locations
import amazon_service
import lstm_service

def find_buy_signals(prediction):
    
    markers_buy = []
    
    for n in range(len(prediction)):
        if n > 0:
            if prediction[n] > prediction[n - 1] + configuration_var['broker_spread']:
                markers_buy.append([n, prediction[n - 1].tolist()[0]])
                
    markers_buy_x = np.array([x[0] for x in markers_buy])
    markers_buy_y = np.array([y[1] for y in markers_buy])
    
    return markers_buy_x, markers_buy_y

def find_sell_signals(prediction):
    
    markers_sell = []

    for n in range(len(prediction)):
        if n > 0:
            if prediction[n] < prediction[n - 1] - configuration_var['broker_spread']:
                markers_sell.append([n, prediction[n - 1].tolist()[0]])

    markers_sell_x = np.array([x[0] for x in markers_sell])
    markers_sell_y = np.array([y[1] for y in markers_sell])
    
    return markers_sell_x, markers_sell_y