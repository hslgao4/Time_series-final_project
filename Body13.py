import pandas as pd
import numpy as np
import statsmodels.api as sm
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
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
def cal_fi(j, k, ACF):
    if k == 1:
        up = ACF[j+1]
        bottom = ACF[j]
        if bottom == 0:
            fi = "inf"
        else:
            fi = up / bottom
    else:
        den = []
        for a in range(j, j + k):
            row = []
            for b in range(a - (k - 1), a + 1):
                b = abs(b)
                R = ACF[b]
                row.append(R)
            row = row[::-1]
            den.append(row)

        num = copy.deepcopy(den)
        for i in range(k):
            num[i][-1] = ACF[j + 1 + i]
        up = np.linalg.det(num)
        bottom = np.linalg.det(den)

        if bottom == 0:
            fi = "inf"
        else:
            fi = up / bottom
            if abs(fi) < 0.0000001:
                fi = 0
    return fi

def GPAC_table(ACF, J=7, K=7):
    temp = np.zeros((J, K - 1))
    for k in range(1, K):
        for j in range(J):
            value = cal_fi(j, k, ACF)
            temp[j][k-1] = value
    table = pd.DataFrame(temp)
    table = table.round(2)
    table.columns = range(1, K)
    plt.figure()
    sns.heatmap(table, annot=True)
    plt.title("Generalized Partial Autocorrelation(GPAC) Table")
    plt.show()
    return table

def ACF_PACF_Plot(y,lags):
    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)
    fig = plt.figure()
    plt.subplot(2,1,1)
    plt.title('ACF/PACF of the raw data')
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.subplot(2,1,2)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    fig.tight_layout(pad=3)
    plt.show()


df = pd.read_csv("clean_hourly_df.csv")
data = df["Temp_C"]

ACF = sm.tsa.acf(data, nlags=80)
table = GPAC_table(ACF, J=30, K=30)

ACF_PACF_Plot(data, lags=60)

train, test = train_test_split(data, test_size=0.2, shuffle=False, random_state=6313)
model = sm.tsa.SARIMAX(train, order = (0, 0, 0), seasonal_order= (1,0,0,24))
model_fit = model.fit()
result = model_fit.predict(start=1, end=(len(train)-1))
train_arima = train.tolist()
result_sarima = result.tolist()
err_sarima = []
for i in range(len(result_sarima)):
    e = train_arima[i+1] - result_sarima[i]
    err_sarima.append(e)

plot = cal_ACF(err_sarima, 40, "Residual")
ACF = sm.tsa.acf(err_sarima, nlags=80)
table_error = GPAC_table(ACF, J=12, K=12)

#### Differencing
def diff(y, n):
    y_diff = []
    for i in range(n, len(y)):
        diff = y[i] - y[i-n]
        y_diff.append(diff)
    y_diff = np.array(y_diff)
    return y_diff

df = pd.read_csv("clean_hourly_df.csv")
data = df["Temp_C"]
diff_data = diff(data, 24)
diff_data_1 = diff(diff_data, 1)
diff_data_2 = diff(diff_data_1, 1)

ACF_PACF_Plot(diff_data_2, lags=60)

ACF = sm.tsa.acf(diff_data_2, nlags=80)
table_diff_2 = GPAC_table(ACF, J=12, K=12)


train, test = train_test_split(diff_data_2, test_size=0.2, shuffle=False, random_state=6313)
model = sm.tsa.SARIMAX(train, order = (0, 2, 0), seasonal_order= (0,0,1,24))
model_fit = model.fit()
result = model_fit.predict(start=1, end=(len(train)-1))
train_arima = train.tolist()
result_sarima = result.tolist()
err_sarima = []
for i in range(len(result_sarima)):
    e = train_arima[i+1] - result_sarima[i]
    err_sarima.append(e)

plot = cal_ACF(err_sarima, 40, "Residual")
ACF = sm.tsa.acf(err_sarima, nlags=80)
table_error = GPAC_table(ACF, J=12, K=12)

plt.figure(figsize=(8, 7))
plt.plot(train, label="train", lw=1.5)
plt.plot(result, label="1-step Prediction", lw=1.5)
plt.xlabel('Samples')
plt.ylabel('Magnitude')
plt.legend()
plt.title('sample')
plt.tight_layout()
plt.show()

sarima_pred_test = model_fit.forecast(steps=len(test))

plt.figure(figsize=(8, 7))
plt.plot(test, label="test", lw=1.5)
plt.plot(sarima_pred_test, label="forecast", lw=1.5)
plt.xlabel('Samples')
plt.ylabel('Magnitude')
plt.legend()
plt.title('sample')
plt.tight_layout()
plt.show()
