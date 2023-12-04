import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from prettytable import PrettyTable

def tabel_pretty(df,title):
    x = PrettyTable()
    for i in range(df.shape[0]):
        x.add_row(df.iloc[i,:])
    x.title = title
    x.field_names = df.columns
    x.float_format = '.2'
    x.hrules = 1
    print(x.get_string())

lag = 10
def cal_Q (y):
    mean = np.mean(y)
    D = 0
    for i in range(len(y)):
        var = (y[i]-mean)**2
        D += var

    R = 0
    for tao in np.arange(1,lag+1):
        S = 0
        for t in range(tao, len(y)):
            N = (y[t]-mean)*(y[t-tao]-mean)
            S += N
        r = S/D
        R += r**2
    Q = len(y)*R
    return round(Q,2)


df = pd.read_csv("clean_hourly_df.csv", parse_dates=["date"])
data = df[["date","Temp_C"]]
data.set_index('date', inplace=True)
train, test = train_test_split(data, test_size=0.2, shuffle=False, random_state=6313)

## Average method
ave = [round(np.mean(train),2)]*len(test)
ave = pd.DataFrame(ave)
ave.index = test.index

err_ave = test - ave
mean_err_ave = np.mean(err_ave)
var_err_ave = np.var(err_ave)

## Naive
naive = [train.iloc[-1, 0]]*len(test)
naive = pd.DataFrame(naive)
naive.index = test.index

err_naive = test - naive
mean_err_naive = np.mean(err_naive)
var_err_naive = np.var(err_naive)


## Drift
b = (train.iloc[-1, 0]-train.iloc[0, 0])/(len(train)-1)
drift = []
for i in range(len(test)):
    d = train.iloc[-1, 0] + b*(i+1)
    drift.append(d)
drift = pd.DataFrame(drift)
drift.index = test.index

err_drift = test - drift
mean_err_drift = np.mean(err_drift)
var_err_drift = np.var(err_drift)

## Simple exponential smoothing
model_ses = SimpleExpSmoothing(train).fit()
y_ses_forecast = model_ses.forecast(len(test))
ses = pd.DataFrame(y_ses_forecast)
ses.index = test.index

err_ses = test - ses
mean_err_ses = np.mean(err_ses)
var_err_ses = np.var(err_ses)

## Plot
plt.figure(figsize=(12, 8))
plt.plot(train, label='Train')#, color='blue')
plt.plot(test, label='Test')#, color='green')
plt.plot(ave, label='ave')#, color='red')
plt.plot(naive, label='naive')#, color='red')
plt.plot(drift, label='drift')#, color='red')
plt.plot(ses, label='ses')#, color='red')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Base model results')
plt.xticks(rotation=45)
plt.legend()
plt.show()
