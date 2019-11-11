import pandas as pd
import numpy as np

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

