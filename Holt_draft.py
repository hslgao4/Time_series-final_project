import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import statsmodels.tsa.holtwinters as ets
from sklearn.model_selection import train_test_split

# Split the dataset into train set (80%) and test set (20%)
df = pd.read_csv("clean_hourly_df.csv", index_col="date", parse_dates=True)

#### Raw data
y = df["Temp_K"]
yt, yf = train_test_split(y, shuffle=False, test_size=0.2)

holt_t = ets.ExponentialSmoothing(yt, trend='mul', seasonal='mul', seasonal_periods=24).fit()
holt_f = holt_t.forecast(steps=len(yf))
holt_f = pd.DataFrame(holt_f).set_index(yf.index)
MSE = np.square(np.subtract(yf.values,np.ndarray.flatten(holt_f.values))).mean()
print("Mean square error for holt-winter method is ", MSE)

plt.figure(figsize=(8, 6))
plt.plot(yt,label= "train")
plt.plot(yf,label= "test")
plt.plot(holt_f,label= "Holt-Winter Method")

plt.legend()
plt.title('Holt-Winter Method - Raw-mul')
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Temperature-K')
plt.show()

# Raw - add
holt_t = ets.ExponentialSmoothing(yt, trend='add', seasonal='add', seasonal_periods=24).fit()
holt_f = holt_t.forecast(steps=len(yf))
holt_f = pd.DataFrame(holt_f).set_index(yf.index)
MSE = np.square(np.subtract(yf.values,np.ndarray.flatten(holt_f.values))).mean()
print("Mean square error for holt-winter method is ", MSE)

plt.figure(figsize=(8, 6))
plt.plot(yt,label= "train")
plt.plot(yf,label= "test")
plt.plot(holt_f,label= "Holt-Winter Method")

plt.legend()
plt.title('Holt-Winter Method - Raw-add')
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Temperature-K')
plt.show()

### Log-Mul
y = np.log(df['Temp_K'])
yt, yf = train_test_split(y, shuffle=False, test_size=0.2)

holt_t = ets.ExponentialSmoothing(yt, trend='mul', seasonal='mul', seasonal_periods=24).fit()
holt_f = holt_t.forecast(steps=len(yf))
holt_f = pd.DataFrame(holt_f).set_index(yf.index)
MSE = np.square(np.subtract(yf.values,np.ndarray.flatten(holt_f.values))).mean()
print("Mean square error for holt-winter method is ", MSE)

plt.figure(figsize=(8, 6))
plt.plot(yt,label= "train")
plt.plot(yf,label= "test")
plt.plot(holt_f,label= "Holt-Winter Method")

plt.legend()
plt.title('Holt-Winter Method - Log-mul')
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Temperature-K')
plt.show()

### Log-add
y = np.log(df['Temp_K'])
yt, yf = train_test_split(y, shuffle=False, test_size=0.2)

holt_t = ets.ExponentialSmoothing(yt, trend='add', seasonal='add', seasonal_periods=24).fit()
holt_f = holt_t.forecast(steps=len(yf))
holt_f = pd.DataFrame(holt_f).set_index(yf.index)
MSE = np.square(np.subtract(yf.values,np.ndarray.flatten(holt_f.values))).mean()
print("Mean square error for holt-winter method is ", MSE)

plt.figure(figsize=(8, 6))
plt.plot(yt,label= "train")
plt.plot(yf,label= "test")
plt.plot(holt_f,label= "Holt-Winter Method")

plt.legend()
plt.title('Holt-Winter Method - Log-add')
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Temperature-K')
plt.show()


### 1st-diff-add
diff1 = [0]
for i in range(1, len(df)):
    diff = df.iloc[i]['Temp_K'] - df.iloc[i-1]['Temp_K']
    diff1.append(diff)
df['diff_1st_K'] = diff1

# y = df['diff_1st']
# yt, yf = train_test_split(y, shuffle=False, test_size=0.2)
#
# holt_t = ets.ExponentialSmoothing(yt, trend='mul', seasonal='mul', seasonal_periods=24).fit()
# holt_f = holt_t.forecast(steps=len(yf))
# holt_f = pd.DataFrame(holt_f).set_index(yf.index)
# MSE = np.square(np.subtract(yf.values,np.ndarray.flatten(holt_f.values))).mean()
# print("Mean square error for holt-winter method is ", MSE)
#
# plt.figure(figsize=(8, 6))
# plt.plot(yt,label= "train")
# plt.plot(yf,label= "test")
# plt.plot(holt_f,label= "Holt-Winter Method")
#
# plt.legend()
# plt.title('Holt-Winter Method - mul')
# plt.xticks(rotation=45)
# plt.xlabel('Date')
# plt.ylabel('Temperature-C')
# plt.show()

#### additive

y = df['diff_1st_K']
yt, yf = train_test_split(y, shuffle=False, test_size=0.2)

holt_t = ets.ExponentialSmoothing(yt, trend='add', seasonal='add', seasonal_periods=24).fit()
holt_f = holt_t.forecast(steps=len(yf))
holt_f = pd.DataFrame(holt_f).set_index(yf.index)

y_rev = []
for i in range(len(holt_f)):
    if i == 0:
        y_r = df.iloc[-1]['Temp_K'] + holt_f[i]
    else:
        y_r = df.iloc[-1]['Temp_K'] + holt_f[:i-1]
    y_rev.append(y_r)

y_rev = pd.DataFrame(y_rev).set_index(yf.index)
plt.figure(figsize=(8, 6))
plt.plot(yt,label= "train")
plt.plot(yf,label= "test")
plt.plot(y_rev,label= "Holt-Winter Method")

plt.legend()
plt.title('Holt-Winter Method - 1st_diff-add')
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Temperature-K')
plt.show()
