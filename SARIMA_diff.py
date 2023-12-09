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
diff_data = diff(data, 365)
diff_data = diff(data, 24)
# diff_data_1 = diff(diff_data, 1)
# diff_data_2 = diff(diff_data_1, 1)

ACF_PACF_Plot(diff_data, lags=400)

ACF = sm.tsa.acf(diff_data, nlags=80)
table_diff_2 = GPAC_table(ACF, J=12, K=12)


train, test = train_test_split(diff_data, test_size=0.2, shuffle=False, random_state=6313)
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