import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error as MSE
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from methods import *
from metrics import mape
import seaborn as sns
rcParams['figure.figsize'] =15, 6
sns.set()

########################################################################################################################################################
petr = pd.read_csv("PETR4.csv")
petr.dropna(inplace=True)
petr.reset_index(drop=True,inplace=True)

hi,lo,opn,close = gen_all_data(petr)
start = petr.index[petr.Date == '2007-05-18'].tolist()[0]
end = petr.index[petr.Date == '2007-11-23'].tolist()[0]
window = 5

mu, ub,lb = bollinger(close,5)

plt.figure(0)
x_axis = close.index.get_level_values(0)
plt.plot(close,label="close",linewidth=1)
plt.fill_between(x_axis, ub,lb, color='silver')
plt.title("Closing price with\n5 days Bollinger bands")
plt.legend()
plt.show()
plt.close(0)

############################################# DEFINING FEATURES ###################################################################################################
windows_hi = get_windows(hi,window)
windows_lo = get_windows(lo,window)
windows_opn = get_windows(opn,window)
windows_close = get_windows(close,window)

exp_lo = exp_mov_average(lo,window)
exp_hi = exp_mov_average(hi,window)

exp_opn = exp_mov_average(opn,window)
exp_close = exp_mov_average(close,window)

bb_low_h , bb_low_l = bollinger_windows(lo,window)
bb_hi_h , bb_hi_l = bollinger_windows(hi,window)

bb_opn_h , bb_opn_l = bollinger_windows(opn,window)
bb_close_h , bb_close_l = bollinger_windows(close,window)

############################################# TRAINING SET ###################################################################################################
x_train = []
y_train = []
for t in range(start,end):

    x = np.hstack( ( np.array(windows_hi[t-window]), np.array(windows_lo[t-window]), np.array(windows_opn[t-window]),
                    np.array(windows_close[t-window]), exp_lo[t],exp_hi[t], exp_opn[t], exp_close[t],
                    np.array(bb_low_h[t-window]) , np.array(bb_low_l[t-window]),np.array(bb_hi_h[t-window]) , np.array(bb_hi_l[t-window]),
                    np.array(bb_opn_h[t-window]) , np.array(bb_opn_l[t-window]),np.array(bb_close_h[t-window]) , np.array(bb_close_l[t-window]),opn[t] ))

    x_train.append(x)
    y_train.append([hi[t],lo[t]])


###################################### TEST SET ###############################################################################################################
x_test = []
y_test = []
for t in range(end,end+1):

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

top_test = [y_test[i][0] for i in range(len(y_test))]
bottom_test = [y_test[i][1] for i in range(len(y_test))]

top_pred = [predict[i][0] for i in range(len(predict))]
bottom_pred = [predict[i][1] for i in range(len(predict))]

mse_high = MSE(top_test,top_pred)
mape_high = mape(top_test,top_pred)

mse_low = MSE(bottom_test,bottom_pred)
mape_low = mape(bottom_test,bottom_pred)

############################################ FIGURE ##############################################################################

# plt.figure(1)
# plt.plot(top_test,label="High -- Test",color='green',linewidth = 2)
# plt.plot(top_pred,label='High -- Prediction',linewidth = 1)
# plt.plot(bottom_test,label="Low -- Test",color='red',linewidth = 2)
# plt.plot(bottom_pred,label='Low -- Prediction',linewidth=1)
# plt.legend()
# plt.title("PETR4 High and Low\nPrice predictions")
# plt.ylabel('Price')
# plt.xlabel('Days')
# plt.text(2, 40,'MSE (High) = %f \nMAPE (High) = %f'%(mse_high,mape_high))
# plt.text(4, 40 ,'MSE (Low) = %f \nMAPE (Low) = %f'%(mse_low,mape_low))
# plt.show()
# plt.close(1)

highs = []
lows = []
for i in range(10000):
    mlp = MLPRegressor(hidden_layer_sizes=12,
                                    activation='relu', solver='lbfgs',
                                    max_iter = 200, learning_rate= 'constant')

    mlp.fit(x_train,y_train)
    predict = mlp.predict(x_test)

    top_test = [y_test[i][0] for i in range(len(y_test))]
    bottom_test = [y_test[i][1] for i in range(len(y_test))]

    top_pred = [predict[i][0] for i in range(len(predict))]
    bottom_pred = [predict[i][1] for i in range(len(predict))]
    highs.append(top_pred[0])
    lows.append(bottom_pred[0])

plt.figure(2)
sns.boxplot(highs)
plt.axvline(x=hi[end],label='Real value = %f' %hi[end])
#plt.hist(highs,bins=200)
plt.title("Highs")
plt.legend()
plt.show()
plt.close(2)
print("Mean (Highs) = %f" %np.mean(highs))
print("Std. (Highs) = %f" %np.std(highs))

plt.figure(3)
sns.boxplot(lows)
plt.axvline(x=lo[end],label='Real value = %f' %lo[end])
#plt.hist(lows,bins=200)
plt.title("Lows")
plt.legend()
plt.show()
plt.close(3)
print("Mean (Lows) = %f" %np.mean(lows))
print("Std. (Lows) = %f" %np.std(lows))