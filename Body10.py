import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

df = pd.read_csv("clean_hourly_df.csv")
df = df.drop("Temp_K", axis=1)

# SVD analysis
df_svd = df.drop(['Temp_C', "date"], axis = 1)
U, S, V = np.linalg.svd(df_svd)
print('Singular values\n', S)

# Condition number
print(f'Condition number is {np.linalg.cond(df_svd):.2f}')

# Normalize data
df = df.drop(["date"], axis = 1)
temp = df.pop("Temp_C")
df.insert(0, "Temp_C", temp)
scaler = StandardScaler()
stand_df = scaler.fit_transform(df)


# Backward stepwise regression
variable = ['Temp_C', 'atmos_p', 'Temp_C_humi', 'rel_humi%', 'Vapor_p_max',
       'Vapor_p', 'Vapor_p_deficit', 'spe_humi', 'H2O_conc', 'air_density',
       'wind_sp', 'wind_sp_max', 'wind_direction', 'rain_depth', 'rain_time',
       'SWDR', 'PAR', 'max_PAR', 'Tlog', 'CO2']

Y = stand_df[:, 0].reshape(-1, 1)
X = stand_df[:, 1:]

while True:
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False, random_state=6313)
    model = sm.OLS(Y_train, X_train).fit()
    # print(model.summary())
    print(f"\nAIC is {model.aic:.1f}")
    print(f"\nBIC is {model.bic:.1f}")
    print(f"\nR_squared_adj is {model.rsquared_adj:.3f}")
    print("---------")
    p_values = model.pvalues[1:]
    max_p_value_index = np.argmax(p_values)
    max_p_value = p_values[max_p_value_index]
    # print(f"Max p-vlue is: {max_p_value:.2f}")

    if max_p_value > 0.05:
        X = np.delete(X, max_p_value_index + 1, axis=1)
        rem_var = variable.pop(max_p_value_index + 1)
        print(f"\nremove variable is: {rem_var}")

    if max_p_value <= 0.05:
        break

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False, random_state=6313)
model = sm.OLS(Y_train, X_train).fit()
model.summary()


# PCA
Y = stand_df[:, 0].reshape(-1, 1)
X = stand_df[:, 1:]
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

train_indices = np.arange(len(X_train))
test_indices = np.arange(len(X_train), len(X_train) + len(X_test))
pred_indices = np.arange(len(X_train) + len(X_test), len(X_train) + len(X_test) + len(y_pred))

plt.figure(figsize=(10, 6))
plt.plot(train_indices, Y_train, label='Train Data', color='blue')
plt.plot(test_indices, Y_test, label='Test Data', color='red')
plt.plot(test_indices, y_pred, label='Predicted Values', color='green')
plt.xlabel('Index')
plt.ylabel('Price')
plt.legend()
plt.title('auto_dataset - Train, Test, and Predicted Values')
plt.show()



# VIF
df = pd.read_csv("clean_hourly_df.csv")
df = df.drop("Temp_K", axis=1)
df = df.drop(["date"], axis = 1)
temp = df.pop("Temp_C")
df.insert(0, "Temp_C", temp)

variable = ['Temp_C', 'atmos_p', 'Temp_C_humi', 'rel_humi%', 'Vapor_p_max',
       'Vapor_p', 'Vapor_p_deficit', 'spe_humi', 'H2O_conc', 'air_density',
       'wind_sp', 'wind_sp_max', 'wind_direction', 'rain_depth', 'rain_time',
       'SWDR', 'PAR', 'max_PAR', 'Tlog', 'CO2']
scaler = StandardScaler()
stand_df = scaler.fit_transform(df)
Y = stand_df[:, 0].reshape(-1, 1)
X = stand_df[:, 1:]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False, random_state=6313)

model1 = sm.OLS(Y_train, X_train).fit()
# print(f"AIC is {model1.aic:.1f}")
# print(f"\nBIC is {model1.bic:.1f}")
# print(f"\nR_squared_adj is {model1.rsquared_adj:.3f}")
X_train_df = pd.DataFrame(X_train)
vif = pd.DataFrame()
vif["Features"] = X_train_df.columns
vif["VIF"] = [variance_inflation_factor(X_train_df.values, i) for i in range(X_train_df.shape[1])]
print(f"\nVIF is: \n{vif}")