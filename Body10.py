import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

df = pd.read_csv("clean_hourly_df.csv")

# SVD analysis
df_svd = df.drop(['Temp_K', "date"], axis = 1)
U, S, V = np.linalg.svd(df_svd)
print('Singular values\n', S)

# Condition number
print(f'Condition number is {np.linalg.cond(df_svd):.2f}')

# Normalize data
scaler = StandardScaler()
stand_df = scaler.fit_transform(df)