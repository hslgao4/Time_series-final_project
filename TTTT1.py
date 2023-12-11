from sklearn.model_selection import train_test_split
from allfunction import *

df = pd.read_csv("clean_hourly_df.csv")
data = df["Temp_C"]

ACF1 = sm.tsa.acf(data, nlags=200)
table1 = GPAC_table(ACF1, J=12, K=12)
ACF_PACF_Plot(data, lags=200)

train, test = train_test_split(data, test_size=0.2, shuffle=False, random_state=6313)
model1 = sm.tsa.SARIMAX(train, order = (1, 0, 1), seasonal_order= (1,0,0,24))
model_fit_1 = model1.fit()
result_1 = model_fit_1.predict(start=0, end=len(train)-1)

# plot train vs. 1-step
plt.figure(figsize=(8, 7))
plt.plot(train.tolist()[:1000], label="train", lw=1.5)
plt.plot(result_1.tolist()[:1000], label="1-step Prediction", lw=1.5)
plt.xlabel('Index')
plt.ylabel('Temperature')
plt.legend()
plt.title('train vs. 1-step prediction [:1000]')
plt.tight_layout()
plt.show()

train_arima = pd.DataFrame(train)
result_sarima = pd.DataFrame(result_1)
err_sarima_1 = []
for i in range(len(result_sarima)):
    e = train_arima.iloc[i, 0] - result_sarima.iloc[i, 0]
    err_sarima_1.append(e)

plot = cal_ACF(err_sarima_1, 100, "Residual")
ACF_PACF_Plot(err_sarima_1, lags=100)
ACF = sm.tsa.acf(err_sarima_1, nlags=80)
table_error = GPAC_table(ACF, J=12, K=12)

error_final, error_mean_final, error_var_final, error_mse_final = cal_err(result_sarima, train_arima)


# forecast
sarima_pred_test = model_fit_1.forecast(steps=len(test))

##plot
df = pd.read_csv("clean_hourly_df.csv", parse_dates=['date'])
data = pd.DataFrame(df[["date","Temp_C"]])
data.set_index('date', inplace=True)

train, test = train_test_split(data, test_size=0.2, shuffle=False, random_state=6313)
sarima_pred_test = pd.DataFrame(sarima_pred_test)
sarima_pred_test.index = test.index

plt.figure(figsize=(8, 7))
plt.plot(train, label="test", lw=1.5)
plt.plot(test, label="test", color = 'green', lw=1.5)
plt.plot(sarima_pred_test, label="forecast", color = 'orange', lw=1.5)
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.xticks(rotation=45)
plt.title('Final SARIMA model ')
plt.tight_layout()
plt.show()

error_f = []
for i in range(len(test)):
    err = test.iloc[i, 0] - sarima_pred_test.iloc[i, 0]
    error_f.append(err)
error_f = pd.DataFrame(error_f)


error_f, error_mean_f, error_var_f, error_mse_f = cal_err(sarima_pred_test, test)

############################################################################
df = pd.read_csv("clean_hourly_df.csv")
data = df["Temp_C"]
train, test = train_test_split(data, test_size=0.2, shuffle=False, random_state=6313)
model1 = sm.tsa.SARIMAX(train, order = (1, 0, 3), seasonal_order= (1,0,1,24))
model_fit_1 = model1.fit()
result_1 = model_fit_1.predict(start=0, end=len(train)-1)

# plot train vs. 1-step
plt.figure(figsize=(8, 7))
plt.plot(train.tolist()[:1000], label="train", lw=1.5)
plt.plot(result_1.tolist()[:1000], label="1-step Prediction", lw=1.5)
plt.xlabel('Index')
plt.ylabel('Temperature')
plt.legend()
plt.title('train vs. 1-step prediction [:1000]')
plt.tight_layout()
plt.show()

train_arima = pd.DataFrame(train)
result_sarima = pd.DataFrame(result_1)
err_sarima_1 = []
for i in range(len(result_sarima)):
    e = train_arima.iloc[i, 0] - result_sarima.iloc[i, 0]
    err_sarima_1.append(e)

plot = cal_ACF(err_sarima_1, 100, "Residual")
ACF_PACF_Plot(err_sarima_1, lags=100)
ACF = sm.tsa.acf(err_sarima_1, nlags=80)
table_error = GPAC_table(ACF, J=12, K=12)

error_final, error_mean_final, error_var_final, error_mse_final = cal_err(result_sarima, train_arima)


# forecast
sarima_pred_test = model_fit_1.forecast(steps=len(test))

##plot
df = pd.read_csv("clean_hourly_df.csv", parse_dates=['date'])
data = pd.DataFrame(df[["date","Temp_C"]])
data.set_index('date', inplace=True)

train, test = train_test_split(data, test_size=0.2, shuffle=False, random_state=6313)
sarima_pred_test = pd.DataFrame(sarima_pred_test)
sarima_pred_test.index = test.index

plt.figure(figsize=(8, 7))
plt.plot(train, label="test", lw=1.5)
plt.plot(test, label="test", color = 'green', lw=1.5)
plt.plot(sarima_pred_test, label="forecast", color = 'orange', lw=1.5)
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.xticks(rotation=45)
plt.title('Final SARIMA model ')
plt.tight_layout()
plt.show()

error_f = []
for i in range(len(test)):
    err = test.iloc[i, 0] - sarima_pred_test.iloc[i, 0]
    error_f.append(err)
error_f = pd.DataFrame(error_f)


error_f, error_mean_f, error_var_f, error_mse_f = cal_err(sarima_pred_test, test)

############################################################
df = pd.read_csv("clean_hourly_df.csv")
data = df["Temp_C"]
train, test = train_test_split(data, test_size=0.2, shuffle=False, random_state=6313)
model1 = sm.tsa.SARIMAX(train, order = (1, 0, 3), seasonal_order= (2,0,2,24))
model_fit_1 = model1.fit()
result_1 = model_fit_1.predict(start=0, end=len(train)-1)

# plot train vs. 1-step
plt.figure(figsize=(8, 7))
plt.plot(train.tolist()[:1000], label="train", lw=1.5)
plt.plot(result_1.tolist()[:1000], label="1-step Prediction", lw=1.5)
plt.xlabel('Index')
plt.ylabel('Temperature')
plt.legend()
plt.title('train vs. 1-step prediction [:1000]')
plt.tight_layout()
plt.show()

train_arima = pd.DataFrame(train)
result_sarima = pd.DataFrame(result_1)
err_sarima_1 = []
for i in range(len(result_sarima)):
    e = train_arima.iloc[i, 0] - result_sarima.iloc[i, 0]
    err_sarima_1.append(e)

plot = cal_ACF(err_sarima_1, 100, "Residual")
ACF_PACF_Plot(err_sarima_1, lags=100)
ACF = sm.tsa.acf(err_sarima_1, nlags=80)
table_error = GPAC_table(ACF, J=12, K=12)

error_final, error_mean_final, error_var_final, error_mse_final = cal_err(result_sarima, train_arima)


# forecast
sarima_pred_test = model_fit_1.forecast(steps=len(test))

##plot
df = pd.read_csv("clean_hourly_df.csv", parse_dates=['date'])
data = pd.DataFrame(df[["date","Temp_C"]])
data.set_index('date', inplace=True)

train, test = train_test_split(data, test_size=0.2, shuffle=False, random_state=6313)
sarima_pred_test = pd.DataFrame(sarima_pred_test)
sarima_pred_test.index = test.index

plt.figure(figsize=(8, 7))
plt.plot(train, label="test", lw=1.5)
plt.plot(test, label="test", color = 'green', lw=1.5)
plt.plot(sarima_pred_test, label="forecast", color = 'orange', lw=1.5)
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.xticks(rotation=45)
plt.title('Final SARIMA model ')
plt.tight_layout()
plt.show()

error_f = []
for i in range(len(test)):
    err = test.iloc[i, 0] - sarima_pred_test.iloc[i, 0]
    error_f.append(err)
error_f = pd.DataFrame(error_f)


error_f, error_mean_f, error_var_f, error_mse_f = cal_err(sarima_pred_test, test)