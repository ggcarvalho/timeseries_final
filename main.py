import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from methods import *
import seaborn as sns
rcParams['figure.figsize'] =15, 6
sns.set()

petr = pd.read_csv("PETR4.csv")
petr.dropna(inplace=True)
petr.reset_index(drop=True,inplace=True)

hi,lo,opn,close = gen_all_data(petr)


# ########################## BOLLINGER BANDS ##################################################
# mu,up,low = bollinger(close,5)
# plt.figure(1)
# plt.plot(mu,label="Moving Average")
# plt.plot(close,label="Close")
# x_axis = close.index.get_level_values(0)
# plt.fill_between(x_axis, up, low, color='silver')
# plt.legend()
# plt.plot()
# plt.show()
# plt.close(1)
# ####################### EMA ################################################################
# exp = exp_mov_average(close,5)
# plt.figure(2)
# plt.plot(close,label="close")
# plt.plot(exp, label="EMA 5")
# plt.show()
# plt.close(2)

close_windows = get_windows(close,5)
print(close[close.index==100])
print(close_windows[95])