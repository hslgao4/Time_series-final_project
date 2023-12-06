from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import STL
from allfunction import *


##### Section 6 ######
df = pd.read_csv("clean_hourly_df.csv", parse_dates=['date'])

# Plot of the dependent variable versus time.
plt.figure(figsize=(16, 8))
plt.plot(df['date'], df['Temp_C'])
plt.xticks(rotation=45)
plt.ylabel('Temperature-C')
plt.xlabel('Date')
plt.title('Temperature(C) over time')
plt.show()

# ACF/PACF
ACF_PACF_Plot(df["Temp_C"],60)

# ACF
ACF_Temp_K = cal_ACF(df["Temp_C"], 80, "Temperature-C")

# Correlation matrix
corr_matrix = df.corr()
#mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Split the dataset into train set (80%) and test set (20%)
train_df, test_df = train_test_split(
    df, test_size=0.2, shuffle=False, random_state=6313)
print(f"Training set size: {len(train_df)} samples")
print(f"Test set size: {len(test_df)} samples")


##### Section 7. Stationarity ######
# Rolling mean/var
item_list = ["Temp_C"]
cal_rolling_mean_var(df, item_list)

# AD / KPSS
ADF_Cal(df["Temp_C"])
print("--------")
kpss_test(df["Temp_C"])

#1st order non-seasonal differencing for Temp-C (for later ARMA)
# diff1 = [0]
# for i in range(1, len(df)):
#     diff = df.loc[i, 'Temp_C'] - df.loc[i-1, 'Temp_C']
#     diff1.append(diff)
# df['diff_1st'] = diff1
# item_list = ['diff_1st']
# cal_rolling_mean_var(df, item_list)


##### Section 8. Time series decomposition ######
df = pd.read_csv("clean_hourly_df.csv", parse_dates=['date'])
df.set_index("date", inplace=True)
stl = STL(df['Temp_C'], period=24)
res = stl.fit()
T = res.trend
S = res.seasonal
R = res.resid
fig = res.plot()
plt.show()

df["season"] = S.tolist()
df["trend"] = T.tolist()

adjutsed_df = df['Temp_C'] - df["season"] - df["trend"]
adjutsed_df.index = df.index

plt.figure(figsize=(12, 8))
plt.plot(df['Temp_C'], label="Original", lw=1.5)
plt.plot(adjutsed_df, label="Detrend & Season_adj", lw=1.5)
plt.title("Original vs. Detrend & Season adjusted")
plt.ylabel("Temerature")
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.tight_layout()
plt.legend()
plt.show()

F = np.maximum(0, 1 - np.var(np.array(R))/np.var(np.array(T+R)))
print(f'The strength of trend for this data set is {100*F:.2f}%')

FS = np.maximum(0, 1 - np.var(np.array(R)) / np.var(np.array(S + R)))
print(f'The strength of seasonality for this data set is  {100*FS:.2f}%')
