import pandas as pd
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
varibale = ["Temp_C", 'rel_humi%', 'Vapor_p_deficit', 'spe_humi']
df_new = df[varibale]

scaler = StandardScaler()
stand_df = scaler.fit_transform(df_new)

Y = stand_df[:, 0].reshape(-1, 1)
Y = pd.DataFrame(Y)
Y.index = df.index
column_y = ["Temp_C"]
Y.columns = column_y
X = stand_df[:, 1:]
X = np.column_stack((np.ones(X.shape[0]), X))
X = pd.DataFrame(X)
X.index = df.index
X.columns = ['constant', 'rel_humi%', 'Vapor_p_deficit', 'spe_humi']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False, random_state=6313)

LR_model = sm.OLS(Y_train, X_train).fit()
print(LR_model.summary())

prediction = LR_model.predict(X_test)
prediction = pd.DataFrame(prediction)
prediction.index = Y_test.index
err_vif, err_vif_mean, err_vif_var, err_vif_mse = cal_err(prediction, Y_test)
ACF_PACF_Plot(err_vif,40)

cal_ACF(err_vif, 40, "Residual")

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


from sklearn.model_selection import TimeSeriesSplit
cv = TimeSeriesSplit(n_splits=5)
rmse = []
mse = []
R_squared = []
R_squared_adj = []
for train_index, valid_index in cv.split(X_train):
    X_train_cv, X_valid_cv = X_train.iloc[train_index], X_train.iloc[valid_index]
    y_train_cv, y_valid_cv = Y_train.iloc[train_index], Y_train.iloc[valid_index]
    model_cv = sm.OLS(y_train_cv,X_train_cv).fit()
    prediction = model_cv.predict(X_valid_cv)
    prediction = pd.DataFrame(prediction)
    error, error_mean, error_var, error_mse = cal_err(prediction, y_valid_cv)
    R_squared.append(model_cv.rsquared)
    R_squared_adj.append(model_cv.rsquared_adj)
    rmse.append(np.sqrt(error_mse))
    mse.append(error_mse)


cv_data = {"MSE": mse,
           "RMSE": rmse,
           "R square": R_squared,
           "Adj R square": R_squared_adj
}
cv_data = pd.DataFrame(cv_data)
def tabel_pretty(df,title):
    x = PrettyTable()
    for i in range(df.shape[0]):
        x.add_row(df.iloc[i,:])
    x.title = title
    x.field_names = df.columns
    x.float_format = '.2'
    x.hrules = 1
    print(x.get_string())

cv_table = tabel_pretty(cv_data,"Cross validation")

from statsmodels.stats.diagnostic import acorr_ljungbox
print('Ljung-Box test')
print(acorr_ljungbox(err_vif,lags=20))

