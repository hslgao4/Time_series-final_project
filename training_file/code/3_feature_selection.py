from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from allfunction import *
from prettytable import PrettyTable

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

def tabel_pretty(df,title):
    x = PrettyTable()
    for i in range(df.shape[0]):
        x.add_row(df.iloc[i,:])
    x.title = title
    x.field_names = df.columns
    x.float_format = '.2'
    x.hrules = 1
    print(x.get_string())

df = pd.read_csv("clean_hourly_df.csv", parse_dates=['date'])
df = df.drop("Temp_K", axis=1)
df.set_index('date', inplace=True)
temp = df.pop("Temp_C")
df.insert(0, "Temp_C", temp)

# SVD analysis
df_svd = df.drop(['Temp_C'], axis = 1)
U, S, V = np.linalg.svd(df_svd)
print('Singular values\n', S)

# Condition number
print(f'Condition number is {np.linalg.cond(df_svd):.2f}')

# Normalize data
scaler = StandardScaler()
stand_df = scaler.fit_transform(df)

# PCA
Y = stand_df[:, 0].reshape(-1, 1)
X = stand_df[:, 1:]
X = np.column_stack((np.ones(X.shape[0]), X))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False, random_state=6313)

pca = PCA(n_components=19)
pca.fit(X_train)
var_ratio = np.cumsum(pca.explained_variance_ratio_)
print(var_ratio)

pca_reduced = PCA(n_components=7)
pca_reduced.fit(X_train)
X_train_pca = pca_reduced.transform(X_train)
model_pca = sm.OLS(Y_train,X_train_pca).fit()
print(model_pca.summary())

X_test_pca = pca_reduced.transform(X_test)
y_pred = model_pca.predict(X_test_pca)

Y_temp = df.iloc[:, 0:1]
train, test = train_test_split( Y_temp, test_size=0.2, shuffle=False, random_state=6313)

Y_test = pd.DataFrame(Y_test)
Y_test.index = test.index
y_pred_pca = pd.DataFrame(y_pred)
y_pred_pca.index = test.index
Y_train = pd.DataFrame(Y_train)
Y_train.index = train.index

### error
err_pca, err_pca_mean, err_pca_var, err_pca_mse = cal_err(y_pred_pca, Y_test)
ACF_PACF_Plot(err_pca,40)


plt.figure(figsize=(10, 6))
plt.plot( Y_train, label='Train Data')
plt.plot( Y_test, label='Test Data')
plt.plot(y_pred_pca, label='Prediction-PCA')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.xticks(rotation=45)
plt.legend()
plt.title('PCA residual')
plt.show()


# Backward stepwise regression
Y = stand_df[:, 0].reshape(-1, 1)
Y = pd.DataFrame(Y)
Y.index = df.index
column_y = ["Temp_C"]
Y.columns = column_y
X = stand_df[:, 1:]
X = np.column_stack((np.ones(X.shape[0]), X))
X = pd.DataFrame(X)
X.index = df.index
column_x = ["constant", 'atmos_p', 'Temp_C_humi', 'rel_humi%', 'Vapor_p_max',
       'Vapor_p', 'Vapor_p_deficit', 'spe_humi', 'H2O_conc', 'air_density',
       'wind_sp', 'wind_sp_max', 'wind_direction', 'rain_depth', 'rain_time',
       'SWDR', 'PAR', 'max_PAR', 'Tlog', 'CO2']
X.columns = column_x

varis = copy.deepcopy(column_x)

while True:
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False, random_state=6313)
    model = sm.OLS(Y_train, X_train).fit()
    #print(model.summary())
    print(f"\nAIC is {model.aic:.1f}")
    print(f"\nBIC is {model.bic:.1f}")
    print(f"\nR_squared_adj is {model.rsquared_adj:.3f}")
    print("---------")
    p_values = model.pvalues
    max_p_value_index = np.argmax(p_values)
    max_p_value = p_values[max_p_value_index]
    name = varis[max_p_value_index]
    print(f"Max p-vlue is: {max_p_value:.2f}")

    if max_p_value > 0.05:
        X.drop(X.columns[max_p_value_index], axis=1, inplace=True)
        varis.pop(max_p_value_index)
        print(f"\nremove variable is: {name}")

    if max_p_value <= 0.05:
        break

print(varis)

#Final model from Backward stepwise regression
back_X = X[varis]
X_train, X_test, Y_train, Y_test = train_test_split(back_X, Y, test_size=0.2, shuffle=False, random_state=6313)
model_backward = sm.OLS(Y_train, X_train).fit()
print(model_backward.summary())

varis_2 = varis[:9]
back_X = X[varis_2]
X_train, X_test, Y_train, Y_test = train_test_split(back_X, Y, test_size=0.2, shuffle=False, random_state=6313)
model_backward = sm.OLS(Y_train, X_train).fit()
print(model_backward.summary())

back_predic = model_backward.predict(X_test)
back_predic = pd.DataFrame(back_predic)
back_predic.index = Y_test.index
### error
err_back, err_back_mean, err_back_var, err_back_mse = cal_err(back_predic, Y_test)
ACF_PACF_Plot(err_back,40)

plt.figure(figsize=(10, 6))
plt.plot( Y_train, label='Train Data')
plt.plot( Y_test, label='Test Data')
plt.plot(back_predic, label='Backward')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.xticks(rotation=45)
plt.legend()
plt.title('Backward Stepwide Regression')
plt.show()

# VIF
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

varibale = copy.deepcopy(column_x)
while True:
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False, random_state=6313)
    model = sm.OLS(Y_train, X_train).fit()
    print(f"AIC is {model.aic:.1f}")
    print(f"\nBIC is {model.bic:.1f}")
    print(f"\nR_squared_adj is {model.rsquared_adj:.3f}")
    vif = []
    for i in range(X_train.shape[1]):
        vif.append(variance_inflation_factor(X_train.values, i))
    print("vif:", vif)
    m_value = np.max(vif)
    m_index = vif.index(m_value)
    name = varibale[m_index]

    if m_value > 10:
        X.drop(X.columns[m_index], axis=1, inplace=True)
        varibale.pop(m_index)
        print("max vif: ", m_value)
        print(f"\nremove variable is: {name}")

    if m_value <= 10:
        break

VIF_X = X[varibale]
X_train, X_test, Y_train, Y_test = train_test_split(VIF_X, Y, test_size=0.2, shuffle=False, random_state=6313)
model_vif = sm.OLS(Y_train, X_train).fit()
print(model_vif.summary())

# delete the p-value over 0.05
varibale.pop(6)
VIF_X = X[varibale]
X_train, X_test, Y_train, Y_test = train_test_split(VIF_X, Y, test_size=0.2, shuffle=False, random_state=6313)
model_vif = sm.OLS(Y_train, X_train).fit()
print(model_vif.summary())
#
varibale.pop(6)
#
varibale.pop(-1)
#
varibale.pop(5)
#
varibale.pop(5)
#
varibale.pop(5)
#
varibale.pop(1)
#
VIF_X = X[varibale]
X_train, X_test, Y_train, Y_test = train_test_split(VIF_X, Y, test_size=0.2, shuffle=False, random_state=6313)
model_vif = sm.OLS(Y_train, X_train).fit()
print(model_vif.summary())


vif_predic = model_vif.predict(X_test)
vif_predic = pd.DataFrame(vif_predic)
vif_predic.index = Y_test.index
err_vif, err_vif_mean, err_vif_var, err_vif_mse = cal_err(vif_predic, Y_test)
ACF_PACF_Plot(err_vif,40)

plt.figure(figsize=(10, 6))
plt.plot( Y_train, label='Train Data')
plt.plot( Y_test, label='Test Data')
plt.plot(vif_predic, label='VIF model')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.xticks(rotation=45)
plt.legend()
plt.title('VIF')
plt.show()


data = {"Method": ["PCA", "Backwards", "VIF"],
        "Mean": [err_pca_mean, err_back_mean, err_vif_mean],
        "Variance": [err_pca_var, err_back_var, err_vif_var],
        "MSE": [err_pca_mse, err_back_mse, err_vif_mse]
}

data = pd.DataFrame(data)
tabel_pretty(data,"Residual")

