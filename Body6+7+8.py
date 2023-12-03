import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import STL

##### Section 6 ######

# Rolling-mean/var
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

# ACF
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

# ADF
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

# Plot of the dependent variable versus time.
plt.figure(figsize=(16, 8))
plt.plot(df['date'], df['Temp_K'])
plt.xticks(rotation=45)
plt.ylabel('Temperature-K')
plt.xlabel('Date')
plt.title('Temperature(K) over time')
plt.show()

# ACF
ACF_Temp_K = cal_ACF(df["Temp_K"], 100, "Temperature-K")

# Correlation matrix
corr_matrix = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Split the dataset into train set (80%) and test set (20%)
train_df, test_df = train_test_split(
    df, test_size=0.2, shuffle=False, random_state=6313)
print(f"Training set size: {len(train_df)} samples")
print(f"Test set size: {len(test_df)} samples")


##### Section 7. Stationarity ######
# Rolling mean/var
item_list = ["Temp_K"]
cal_rolling_mean_var(df, item_list)

# AD / KPSS
ADF_Cal(df["Temp_C"])
print("--------")
kpss_test(df["Temp_C"])

#1st order non-seasonal differencing for Temp-C (for later ARMA)
# diff1 = [0]
# for i in range(1, len(df)):
#     diff = df.loc[i, 'Temp_C'] - df.loc[i-1, 'Temp_C']
#     diff1.append(diff)
# df['diff_1st'] = diff1
# item_list = ['diff_1st']
# cal_rolling_mean_var(df, item_list)


##### Section 8. Time series decomposition ######
df = pd.read_csv("clean_hourly_df.csv", parse_dates=['date'])
df.set_index("date", inplace=True)
stl = STL(df['Temp_K'], period=24)
res = stl.fit()
T = res.trend
S = res.seasonal
R = res.resid
fig = res.plot()
plt.show()


F = np.maximum(0, 1 - np.var(np.array(R))/np.var(np.array(T+R)))
print(f'The strength of trend for this data set is {100*F:.2f}%')

FS = np.maximum(0, 1 - np.var(np.array(R)) / np.var(np.array(S + R)))
print(f'The strength of seasonality for this data set is  {100*FS:.2f}%')
