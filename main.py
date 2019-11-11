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
start = petr.index[petr.Date == '2007-05-18'].tolist()[0]
end = petr.index[petr.Date == '2007-11-23'].tolist()[0]
window = 5


windows_hi = get_windows(hi,window)
windows_lo = get_windows(lo,window)
windows_opn = get_windows(opn,window)
windows_close = get_windows(close,window)

exp_lo = exp_mov_average(lo,window)
exp_hi = exp_mov_average(hi,window)

exp_opn = exp_mov_average(opn,window)
exp_close = exp_mov_average(close,window)

bb_low_h , bb_low_l = bollinger_windows(lo,5)
bb_hi_h , bb_hi_l = bollinger_windows(hi,5)

bb_opn_h , bb_opn_l = bollinger_windows(opn,5)
bb_close_h , bb_close_l = bollinger_windows(close,5)


x_train = []
y_train = []
for t in range(start,end):

    x = np.hstack( ( np.array(windows_hi[t-window]), np.array(windows_lo[t-window]), np.array(windows_opn[t-window]),
                    np.array(windows_close[t-window]), exp_lo[t],exp_hi[t], exp_opn[t], exp_close[t],
                    np.array(bb_low_h[t-window]) , np.array(bb_low_l[t-window]),np.array(bb_hi_h[t-window]) , np.array(bb_hi_l[t-window]),
                    np.array(bb_opn_h[t-window]) , np.array(bb_opn_l[t-window]),np.array(bb_close_h[t-window]) , np.array(bb_close_l[t-window]),opn[t] ))

    x_train.append(x)


print(len(x_train))

########################### BOLLINGER BANDS ##################################################
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
######################## EMA ################################################################
# exp = exp_mov_average(close,5)
# plt.figure(2)
# plt.plot(close,label="close")
# plt.plot(exp, label="EMA 5")
# plt.show()
# plt.close(2)

# close_windows = get_windows(close,5)
# print(close[close.index==100])
# print(close_windows[95])