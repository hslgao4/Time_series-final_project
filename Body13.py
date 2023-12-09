import pandas as pd
from sklearn.model_selection import train_test_split
from allfunction import *

df = pd.read_csv("clean_hourly_df.csv")
data = df["Temp_C"]

#########################  Raw data  #########################
ACF1 = sm.tsa.acf(data, nlags=200)
table1 = GPAC_table(ACF1, J=12, K=12)   # na=1, nb=0

ACF_PACF_Plot(data, lags=200)

train, test = train_test_split(data, test_size=0.2, shuffle=False, random_state=6313)
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

train, test = train_test_split(data, test_size=0.2, shuffle=False, random_state=6313)
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

########
train, test = train_test_split(data, test_size=0.2, shuffle=False, random_state=6313)
model1 = sm.tsa.SARIMAX(train, order = (1, 0, 0), seasonal_order= (1,1,0,24))
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



########
train, test = train_test_split(data, test_size=0.2, shuffle=False, random_state=6313)
model1 = sm.tsa.SARIMAX(train, order = (1, 0, 0), seasonal_order= (1,1,2,24))
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

error, error_mean, error_var, error_mse = cal_err(result_sarima, train_arima)

########
train, test = train_test_split(data, test_size=0.2, shuffle=False, random_state=6313)
model1 = sm.tsa.SARIMAX(train, order = (2, 0, 1), seasonal_order= (1,1,2,24))
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

error, error_mean, error_var, error_mse = cal_err(result_sarima, train_arima)


#####
train, test = train_test_split(data, test_size=0.2, shuffle=False, random_state=6313)
model1 = sm.tsa.SARIMAX(train, order = (2, 0, 1), seasonal_order= (1,1,2,720))
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

error, error_mean, error_var, error_mse = cal_err(result_sarima, train_arima)


# forecast
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







