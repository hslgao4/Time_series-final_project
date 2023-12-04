import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import copy

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
X = pd.DataFrame(X)
X.index = df.index
column_x = ['atmos_p', 'Temp_C_humi', 'rel_humi%', 'Vapor_p_max',
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
model_backward = sm.OLS(Y_train, X_train).fit()
print(model_backward.summary())

# delete the p-value over 0.05
varibale.pop(5)
VIF_X = X[varibale]
X_train, X_test, Y_train, Y_test = train_test_split(VIF_X, Y, test_size=0.2, shuffle=False, random_state=6313)
model_backward = sm.OLS(Y_train, X_train).fit()
print(model_backward.summary())