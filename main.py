import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from methods import bollinger,exp_mov_average
import seaborn as sns
rcParams['figure.figsize'] =15, 6
sns.set()

petr = pd.read_csv("PETR4.csv").dropna()


# ########################## BOLLINGER BANDS ##################################################
mu,up,low = bollinger(petr,"Close",5)
plt.figure(1)
plt.plot(mu,label="Moving Average")
plt.plot(petr["Close"],label="Close")
x_axis = petr["Close"].index.get_level_values(0)
plt.fill_between(x_axis, up, low, color='silver')
plt.legend()
plt.plot()
plt.show()
plt.close(1)
# ###################### EMA ################################################################
exp = exp_mov_average(petr,"Close",5)
plt.figure(2)
plt.plot(petr["Close"],label="close")
plt.plot(exp, label="EMA 5")
plt.show()
plt.close(2)