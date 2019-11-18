import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error as MSE

import scipy.stats


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def gen_all_data(df):
    High = df["High"]
    Low = df["Low"]
    Open = df["Open"]
    Close = df["Close"]
    return High,Low,Open,Close


def bollinger(data,window):
    mu = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    upper_band = mu + 2*std
    lower_band = mu -2*std
    return mu, upper_band, lower_band


def exp_mov_average(data,window):
    exp = data.ewm(span = window, adjust = False).mean()
    return exp


def get_windows(data,window_size):
    windows = []
    for i in range(len(data)-window_size+1):
        windows.append(data[i: i + window_size])
    return windows


def bollinger_windows(data,window):
    mu,ub,lb = bollinger(data,20)
    ubw = get_windows(ub,5)
    lbw = get_windows(lb,5)
    return ubw,lbw

def mult_mlp(x_train,x_test,y_train,y_test,iter):
    highs = []
    lows = []
    for i in range(iter):
        mlp = MLPRegressor(hidden_layer_sizes=12,
                                        activation='relu', solver='lbfgs',
                                        max_iter = 100, learning_rate= 'constant')

        mlp.fit(x_train,y_train)
        predict = mlp.predict(x_test)

        top_test = [y_test[i][0] for i in range(len(y_test))]
        bottom_test = [y_test[i][1] for i in range(len(y_test))]

        top_pred = [predict[i][0] for i in range(len(predict))]
        bottom_pred = [predict[i][1] for i in range(len(predict))]
        highs.append(top_pred[0])
        lows.append(bottom_pred[0])
        return highs, lows