from allfunction import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def cal_err(y_pred, Y_test):
    error = []
    error_se = []
    for i in range(len(y_pred)):
        e = Y_test.iloc[i,0] - y_pred.iloc[i,0]
        error.append(e)
        error_se.append(e**2)
    error_mean = np.mean(error)
    error_var = np.var(error)
    error_mse = np.mean(error_se)
    return error, error_mean, error_var, error_mse

df = pd.read_csv("clean_hourly_df.csv")
df = df.drop("Temp_K", axis=1)
df = df.drop(["date"], axis = 1)
temp = df.pop("Temp_C")
df.insert(0, "Temp_C", temp)

scaler = StandardScaler()
stand_df = scaler.fit_transform(df)

Y = stand_df[:, 0].reshape(-1, 1)
Y = pd.DataFrame(Y)
Y.index = df.index
column_y = ["Temp_C"]
Y.columns = column_y
X = stand_df[:, 1:]
X = np.column_stack((np.ones(X.shape[0]), X))
X = pd.DataFrame(X)
X.index = df.index
column_x = ['constant', 'atmos_p', 'Temp_C_humi', 'rel_humi%', 'Vapor_p_max',
       'Vapor_p', 'Vapor_p_deficit', 'spe_humi', 'H2O_conc', 'air_density',
       'wind_sp', 'wind_sp_max', 'wind_direction', 'rain_depth', 'rain_time',
       'SWDR', 'PAR', 'max_PAR', 'Tlog', 'CO2']
X.columns = column_x

varibale = ['constant', 'rel_humi%', 'Vapor_p_deficit', 'spe_humi']

X = X[varibale]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False, random_state=6313)
LR_model = sm.OLS(Y_train, X_train).fit()
print(LR_model.summary())

prediction = LR_model.predict(X_test)
prediction = pd.DataFrame(prediction)
prediction.index = Y_test.index
err_vif, err_vif_mean, err_vif_var, err_vif_mse = cal_err(prediction, Y_test)
ACF_PACF_Plot(err_vif,40)

plt.figure(figsize=(10, 6))
plt.plot( Y_train, label='Train Data')
plt.plot( Y_test, label='Test Data')
plt.plot(prediction, label='Multiple linear regression')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.xticks(rotation=45)
plt.legend()
plt.title('Multiple linear regression')
plt.show()

# T-test
t_test_results = LR_model.t_test(np.identity(X_train.shape[1]))
print("T-Test Results:")
print(t_test_results)

# F-test
f_test_results = LR_model.f_test(np.identity(X_train.shape[1]))
print("\nF-Test Results:")
print(f_test_results)
