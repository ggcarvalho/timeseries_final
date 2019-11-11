import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error as MSE
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

############################################# DEFINING FEATURES ###################################################################################################
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

############################################# TRAINING SET ###################################################################################################
x_train = []
y_train = []
for t in range(start,end):

    x = np.hstack( ( np.array(windows_hi[t-window]), np.array(windows_lo[t-window]), np.array(windows_opn[t-window]),
                    np.array(windows_close[t-window]), exp_lo[t],exp_hi[t], exp_opn[t], exp_close[t],
                    np.array(bb_low_h[t-window]) , np.array(bb_low_l[t-window]),np.array(bb_hi_h[t-window]) , np.array(bb_hi_l[t-window]),
                    np.array(bb_opn_h[t-window]) , np.array(bb_opn_l[t-window]),np.array(bb_close_h[t-window]) , np.array(bb_close_l[t-window]),opn[t] ))

    x_train.append(x)
    #y = np.hstack( (hi[t],lo[t]) )
    y_train.append([hi[t],lo[t]])


###################################### TEST SET ###############################################################################################################
x_test = []
y_test = []
for t in range(end,end+30):

    x = np.hstack( ( np.array(windows_hi[t-window]), np.array(windows_lo[t-window]), np.array(windows_opn[t-window]),
                    np.array(windows_close[t-window]), exp_lo[t],exp_hi[t], exp_opn[t], exp_close[t],
                    np.array(bb_low_h[t-window]) , np.array(bb_low_l[t-window]),np.array(bb_hi_h[t-window]) , np.array(bb_hi_l[t-window]),
                    np.array(bb_opn_h[t-window]) , np.array(bb_opn_l[t-window]),np.array(bb_close_h[t-window]) , np.array(bb_close_l[t-window]),opn[t] ))

    x_test.append(x)

    y_test.append([hi[t],lo[t]])
###############################################################################################################################################################

mlp = MLPRegressor(hidden_layer_sizes=12,
                                 activation='relu', solver='lbfgs',
                                  max_iter = 10000, learning_rate= 'constant')

mlp.fit(x_train,y_train)
predict = mlp.predict(x_test)

print("MSE = %s" %MSE(y_test, predict))

