from sklearn.model_selection import train_test_split
from allfunction import *

def diff(y, n):
    y_diff = []
    for i in range(n, len(y)):
        diff = y[i] - y[i-n]
        y_diff.append(diff)
    y_diff = np.array(y_diff)
    return y_diff

df = pd.read_csv("clean_hourly_df.csv")
data = df["Temp_C"]
diff_data = diff(data, 744)
diff_data_24 = diff(data, 24)
# diff_data_1 = diff(diff_data, 1)
# diff_data_2 = diff(diff_data_1, 1)

ACF1 = sm.tsa.acf(diff_data_24, nlags=200)
table1 = GPAC_table(ACF1, J=12, K=12)   # na=1, nb=0
ACF_PACF_Plot(diff_data_24, lags=200)

### GPAC - na=1, nb=0
train, test = train_test_split(diff_data, test_size=0.2, shuffle=False, random_state=6313)
model1 = sm.tsa.SARIMAX(train, order = (0, 0, 0), seasonal_order= (1,0,0,24))
model_fit_1 = model1.fit()
result_1 = model_fit_1.predict(start=0, end=len(train)-1)

# plot train vs. 1-step
plt.figure(figsize=(8, 7))
plt.plot(train.tolist()[:1000-24], label="train", lw=1.5)
plt.plot(result_1.tolist()[24:1000], label="1-step Prediction", lw=1.5)
plt.xlabel('Samples')
plt.ylabel('Magnitude')
plt.legend()
plt.title('sample')
plt.tight_layout()
plt.show()

# calculate error
train_arima = pd.DataFrame(train)
result_sarima = pd.DataFrame(result_1)
err_sarima_1 = []
for i in range(len(result_sarima)-24):
    e = train_arima.iloc[i, 0] - result_sarima.iloc[i+24, 0]
    err_sarima_1.append(e)

plot = cal_ACF(err_sarima_1, 40, "Residual")
ACF_PACF_Plot(err_sarima_1, lags=100)
ACF = sm.tsa.acf(err_sarima_1, nlags=80)
table_error = GPAC_table(ACF, J=12, K=12)

#####  GPAC of error & PACF, na = 1, nb = 0
train, test = train_test_split(diff_data, test_size=0.2, shuffle=False, random_state=6313)
model1 = sm.tsa.SARIMAX(train, order = (1, 0, 0), seasonal_order= (1,0,0,24))
model_fit_1 = model1.fit()
result_1 = model_fit_1.predict(start=0, end=len(train)-1)

# plot train vs. 1-step
plt.figure(figsize=(8, 7))
plt.plot(train.tolist()[:1000], label="train", lw=1.5)
plt.plot(result_1.tolist()[:1000], label="1-step Prediction", lw=1.5)
plt.xlabel('Samples')
plt.ylabel('Magnitude')
plt.legend()
plt.title('sample')
plt.tight_layout()
plt.show()

# calculate error
train_arima = pd.DataFrame(train)
result_sarima = pd.DataFrame(result_1)
err_sarima_1 = []
for i in range(len(result_sarima)):
    e = train_arima.iloc[i, 0] - result_sarima.iloc[i, 0]
    err_sarima_1.append(e)

plot = cal_ACF(err_sarima_1, 40, "Residual")
ACF_PACF_Plot(err_sarima_1, lags=100)
ACF = sm.tsa.acf(err_sarima_1, nlags=80)
table_error = GPAC_table(ACF, J=12, K=12)


### add na/nb 1
train, test = train_test_split(diff_data, test_size=0.2, shuffle=False, random_state=6313)
model1 = sm.tsa.SARIMAX(train, order = (2, 0, 1), seasonal_order= (1,0,0,24))
model_fit_1 = model1.fit()
result_1 = model_fit_1.predict(start=0, end=len(train)-1)

# plot train vs. 1-step
plt.figure(figsize=(8, 7))
plt.plot(train.tolist()[:1000], label="train", lw=1.5)
plt.plot(result_1.tolist()[:1000], label="1-step Prediction", lw=1.5)
plt.xlabel('Samples')
plt.ylabel('Magnitude')
plt.legend()
plt.title('sample')
plt.tight_layout()
plt.show()

# calculate error
train_arima = pd.DataFrame(train)
result_sarima = pd.DataFrame(result_1)
err_sarima_1 = []
for i in range(len(result_sarima)):
    e = train_arima.iloc[i, 0] - result_sarima.iloc[i, 0]
    err_sarima_1.append(e)

plot = cal_ACF(err_sarima_1, 40, "Residual")
ACF_PACF_Plot(err_sarima_1, lags=100)
ACF = sm.tsa.acf(err_sarima_1, nlags=80)
table_error = GPAC_table(ACF, J=12, K=12)


sarima_pred_test = model_fit_1.forecast(steps=len(test))

plt.figure(figsize=(8, 7))
#plt.plot(train, label="test", lw=1.5)
plt.plot(test, label="test", lw=1.5)
plt.plot(sarima_pred_test, label="forecast", lw=1.5)
plt.xlabel('Samples')
plt.ylabel('Magnitude')
plt.legend()
plt.title('sample')
plt.tight_layout()
plt.show()
