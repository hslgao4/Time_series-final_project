import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import statsmodels.tsa.holtwinters as ets
from sklearn.model_selection import train_test_split
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
    (markers, stemlines, baseline) = plt.stem(
        x_values, Magnitute, markerfmt='o')
    plt.setp(markers, color='red')
    m = 1.96/np.sqrt(len(y))
    plt.axhspan(-m, m, alpha=0.2, color='blue')
    plt.xlabel('Lags')
    plt.ylabel('Magnitute')
    plt.title(f'Autocorrelation Function of {sample_plot_name}')
    plt.show()
    return ax

def cal_rolling_mean_var(df, item_list):
    def rolling_mean_var(df, x):
        rolling_mean = []
        rolling_var = []
        for i in range(1, len(df) + 1):
            new_df = df.iloc[:i, ]
            if i == 1:
                mean = new_df[x]
                var = 0
            else:
                mean = new_df[x].mean()
                var = new_df[x].var()
            rolling_mean.append(mean)
            rolling_var.append(var)
        return rolling_mean, rolling_var
    plt.figure(figsize=(8, 7))
    plt.subplot(2, 1, 1)
    for i in item_list:
        roll_mean, roll_var = rolling_mean_var(df, i)

        plt.plot(roll_mean, label=i, lw=1.5)
    plt.xlabel('Samples')
    plt.ylabel('Magnitude')
    plt.title('Rolling Mean')
    plt.legend()

    plt.subplot(2, 1, 2)
    for i in item_list:
        roll_mean, roll_var = rolling_mean_var(df, i)

        plt.plot(roll_var, label=i, lw=1.5)
    plt.xlabel('Samples')
    plt.ylabel('Magnitude')
    plt.title('Rolling Variance')
    plt.legend()
    plt.tight_layout()
    plt.show()

def ADF_Cal(x):
    result = adfuller(x)
    print("ADF Statistic: %f" % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

# KPSS
def kpss_test(timeseries):
    print('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=[
                            'Test Statistic', 'p-value', 'Lags Used'])
    for key, value in kpsstest[3].items():
        kpss_output['Critical Value (%s)' % key] = value
        print(kpss_output)


df = pd.read_csv("clean_hourly_df.csv", parse_dates=['date'])
plt.figure(figsize=(16, 8))
plt.plot(df['date'], df['Temp_K'])
plt.xticks(rotation=45)
plt.ylabel('Temperature-K')
plt.xlabel('Date')
plt.title('Temperature(K) over time')
plt.show()

cal_ACF(df["Temp_K"], 50, "Temperature-K")
item_list = ["Temp_K"]
cal_rolling_mean_var(df, item_list)
ADF_Cal(df["Temp_K"])
print("--------")
kpss_test(df["Temp_K"])



# Split the dataset into train set (80%) and test set (20%)
df = pd.read_csv("clean_hourly_df.csv", index_col="date", parse_dates=True)
y = df['Temp_K'][:500]
yt, yf = train_test_split(y, shuffle=False, test_size=0.2)

holt_t = ets.ExponentialSmoothing(yt, trend='multiplicative', seasonal='multiplicative',seasonal_periods=12).fit()
holt_f = holt_t.forecast(steps=len(yf))
holt_f = pd.DataFrame(holt_f).set_index(yf.index)
MSE = np.square(np.subtract(yf.values,np.ndarray.flatten(holt_f.values))).mean()
print("Mean square error for holt-winter method is ", MSE)

plt.figure(figsize=(8, 6))
plt.plot(yt,label= "train")
plt.plot(yf,label= "test")
plt.plot(holt_f,label= "Holt-Winter Method")

plt.legend()
plt.title('Holt-Winter Method - mul')
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Temperature-K')
plt.show()

#### additive

y = df['Temp_K']
yt, yf = train_test_split(y, shuffle=False, test_size=0.2)

holt_t = ets.ExponentialSmoothing(yt, trend='add', seasonal='add', seasonal_periods=24).fit()
holt_f = holt_t.forecast(steps=len(yf))
holt_f = pd.DataFrame(holt_f).set_index(yf.index)
MSE = np.square(np.subtract(yf.values,np.ndarray.flatten(holt_f.values))).mean()
print("Mean square error for holt-winter method is ", MSE)

plt.figure(figsize=(8, 6))
plt.plot(yt,label= "train")
plt.plot(yf,label= "test")
plt.plot(holt_f,label= "Holt-Winter Method")

plt.legend()
plt.title('Holt-Winter Method - add')
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Temperature-K')
plt.show()