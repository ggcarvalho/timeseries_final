import pandas as pd
import numpy as np

def bollinger(df,col,window):
    mu = df[col].rolling(window=window).mean()
    std = df[col].rolling(window=window).std()
    upper_band = mu + 2*std
    lower_band = mu -2*std
    return mu, upper_band, lower_band


def exp_mov_average(df,col,window):
    exp = df[col].ewm(span = window, adjust = False).mean()
    return exp


