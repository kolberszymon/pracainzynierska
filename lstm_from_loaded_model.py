import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from configuration_file import configuration_var
from configuration_file import folder_locations
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import amazon_service
import lstm_service
import investment_strategy as invest_s

ticker_to_obtain = 'EURUSD=X30m'

df = amazon_service.get_csv_data_from_amazon(ticker_to_obtain)
df.columns = ['Date', 'Open', 'High', 'Low', 'Close']

#Preparing the data

x = np.array(df.filter(['Open', 'High', 'Low']))
y = np.array(df.filter(['Close']))

#Preparing samples

x_samples, y_samples = lstm_service.prepare_samples(x, y)

#Splitting data into training set and testing set

x_samples_train, y_samples_train, x_samples_test, y_samples_test = lstm_service.split_into_training_and_testing_set(x_samples, y_samples)

#Loading model, and predicting price

model = load_model(folder_locations["trained_models"] + 'lstm_model30m.h5')

X = x_samples
Y = y_samples

prediction = model.predict(X)

Y_plus_spread = Y + configuration_var['broker_spread']
Y_minus_spread = Y - configuration_var['broker_spread']
Y_timestamps = np.array([index for index, value in enumerate(Y_plus_spread)])

Y_plus_spread = Y_plus_spread.tolist()
Y_minus_spread = Y_minus_spread.tolist()


#Buying and selling strategy

markers_buy_x, markers_buy_y = invest_s.find_buy_signals(prediction)
markers_sell_x, markers_sell_y = invest_s.find_sell_signals(prediction)

print(prediction)
print(Y)


plt.title(ticker_to_obtain + ' Buy / Sell Signals')

plt.plot(Y_timestamps - configuration_var['sample_length'], prediction, 'b', label='Currency prices')
plt.plot(markers_buy_x - configuration_var['sample_length'] - 1, markers_buy_y, marker = '+', color = 'g', markersize = 16, linestyle = 'None', label='Buy Signal')
plt.plot(markers_sell_x - configuration_var['sample_length'] - 1, markers_sell_y, marker = 'x', color = 'r', markersize = 16, linestyle = 'None', label='Sell Signal')
plt.xlabel('Timestamps: 1 = ' + configuration_var["picked_data_period"])
plt.ylabel('Price in $')
plt.legend(loc="upper left")
figure = plt.gcf()
figure.set_size_inches(12, 8)
#plt.savefig(ticker_to_obtain + 'Prediction' + '.png', dpi=100)
