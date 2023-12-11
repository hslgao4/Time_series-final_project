import pandas as pd
from sklearn.model_selection import train_test_split
from allfunction import *

df = pd.read_csv("clean_hourly_df.csv")
data = df["Temp_C"]

#########################  Raw data  #########################
ACF1 = sm.tsa.acf(data, nlags=200)
table1 = GPAC_table(ACF1, J=12, K=12)   # na=1, nb=0
ACF_PACF_Plot(data, lags=100)

train, test = train_test_split(data, test_size=0.2, shuffle=False, random_state=6313)
model1 = sm.tsa.SARIMAX(train, order = (0, 0, 0), seasonal_order= (1,0,0,24))
model_fit_1 = model1.fit()
result_1 = model_fit_1.predict(start=0, end=len(train)-1)

# plot train vs. 1-step
plt.figure(figsize=(8, 7))
plt.plot(train.tolist()[:1000-24], label="train", lw=1.5)
plt.plot(result_1.tolist()[24:1000], label="1-step Prediction", lw=1.5)
plt.xlabel('Index')
plt.ylabel('Temperature')
plt.legend()
plt.title('train vs. 1-step prediction [:1000]')
plt.tight_layout()
plt.show()

# calculate error
train_arima = pd.DataFrame(train)
result_sarima = pd.DataFrame(result_1)
err_sarima_1 = []
for i in range(len(result_sarima)-24):
    e = train_arima.iloc[i, 0] - result_sarima.iloc[i+24, 0]
    err_sarima_1.append(e)

plot = cal_ACF(err_sarima_1, 100, "Residual")
ACF_PACF_Plot(err_sarima_1, lags=100)
ACF = sm.tsa.acf(err_sarima_1, nlags=80)
table_error = GPAC_table(ACF, J=12, K=12)

######################################################
train, test = train_test_split(data, test_size=0.2, shuffle=False, random_state=6313)
model1 = sm.tsa.SARIMAX(train, order = (1, 0, 0), seasonal_order= (1,0,0,24))
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

########
train, test = train_test_split(data, test_size=0.2, shuffle=False, random_state=6313)
model1 = sm.tsa.SARIMAX(train, order = (1, 0, 0), seasonal_order= (2,0,2,24))
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

error_1, error_mean_1, error_var_1, error_mse_1 = cal_err(result_sarima, train_arima)


plot = cal_ACF(err_sarima_1, 100, "Residual")
ACF_PACF_Plot(err_sarima_1, lags=100)
ACF = sm.tsa.acf(err_sarima_1, nlags=80)
table_error = GPAC_table(ACF, J=12, K=12)

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

sarima_pred_test = pd.DataFrame(sarima_pred_test)
test = pd.DataFrame(test)
error_f, error_mean_f, error_var_f, error_mse_f = cal_err(sarima_pred_test, test)

################################################
train, test = train_test_split(data, test_size=0.2, shuffle=False, random_state=6313)
model1 = sm.tsa.SARIMAX(train, order = (1, 0, 0), seasonal_order= (1,1,2,24))
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

error_2, error_mean_2, error_var_2, error_mse_2 = cal_err(result_sarima, train_arima)

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

sarima_pred_test = pd.DataFrame(sarima_pred_test)
test = pd.DataFrame(test)
error_f, error_mean_f, error_var_f, error_mse_f = cal_err(sarima_pred_test, test)
################################################################
train, test = train_test_split(data, test_size=0.2, shuffle=False, random_state=6313)
model1 = sm.tsa.SARIMAX(train, order = (2, 0, 1), seasonal_order= (1,1,2,24))
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

error_3, error_mean_3, error_var_3, error_mse_3 = cal_err(result_sarima, train_arima)

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

##################################################Final Model##########################
df = pd.read_csv("clean_hourly_df.csv")
data = df["Temp_C"]

train, test = train_test_split(data, test_size=0.2, shuffle=False, random_state=6313)
model1 = sm.tsa.SARIMAX(train, order = (3, 0, 2), seasonal_order= (1,1,2,24))
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



from scipy.stats import chi2
ACF = sm.tsa.acf(error_final, nlags=80)
Q = len(error_final)*sum(np.square(ACF[1:]))
DOF = 80 - 24 - 48
alfa = 0.01
chi_critical = chi2.ppf(1-alfa, DOF)
if Q < chi_critical:
    print("Q:", round(Q,2))
    print("Q*:", round(chi_critical,2))
    print("The residual is white")
else:
    print("The residual is not white")

print("Residual error variance", round(error_var_final, 2))
print("Forecast error variance", round(np.var(error_f), 2))







