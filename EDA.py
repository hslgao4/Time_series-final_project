import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#ACF
def cal_ACF(y, lag, sample_plot_name):
    mean = np.mean(y)
    D = 0
    for i in range(len(y)):
        var = (y[i]-mean)**2
        D += var
    R = []
    for tao in range(lag+1):
        S = 0
        for t in range(tao, len(y)):
            N = (y[t]-mean)*(y[t-tao]-mean)
            S += N
        r = S/D
        R.append(r)
    R_inv = R[::-1]
    Magnitute = R_inv + R[1:]
    ax = plt.figure()
    x_values = range(-lag, lag + 1)
    (markers, stemlines, baseline) = plt.stem(x_values, Magnitute, markerfmt='o')
    plt.setp(markers, color = 'red')
    m = 1.96/np.sqrt(len(y))
    plt.axhspan(-m, m, alpha = 0.2, color = 'blue')
    plt.xlabel('Lags')
    plt.ylabel('Magnitute')
    plt.title(f'Autocorrelation Function of {sample_plot_name}' )
    plt.show()
    return ax

hourly_df = pd.read_csv("hourly_df.csv")


plt.figure(figsize=(16, 8))
plt.plot(hourly_df['date'], hourly_df["T (degC)"], linestyle='-', color='b', label='T (degC)')
plt.legend()
plt.tight_layout()
plt.show()

cal_ACF(hourly_df['T (degC)'], 20, "Hourly dataset")