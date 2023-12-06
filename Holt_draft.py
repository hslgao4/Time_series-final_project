import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import statsmodels.tsa.holtwinters as ets
from sklearn.model_selection import train_test_split
import copy

# Split the dataset into train set (80%) and test set (20%)
df = pd.read_csv("clean_hourly_df.csv", index_col="date", parse_dates=True)

#### Raw data
y = df["Temp_C"] + 12
yt, yf = train_test_split(y, shuffle=False, test_size=0.2)

holt_t = ets.ExponentialSmoothing(yt, trend='mul', seasonal='mul', seasonal_periods=24).fit()
holt_f = holt_t.forecast(steps=len(yf))
holt_f = pd.DataFrame(holt_f).set_index(yf.index)
# MSE = np.square(np.subtract(yf.values,np.ndarray.flatten(holt_f.values))).mean()
# print("Mean square error for holt-winter method is ", MSE)

plt.figure(figsize=(8, 6))
plt.plot(yt - 12,label= "train")
plt.plot(yf - 12,label= "test")
plt.plot(holt_f - 12,label= "Holt-Winter Method")
plt.legend()
plt.title('Holt-Winter Method - Raw(mul)')
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Temperature-C')
plt.show()

# Raw - add
y = df["Temp_C"]
yt, yf = train_test_split(y, shuffle=False, test_size=0.2)

holt_t = ets.ExponentialSmoothing(yt, trend='add', seasonal='add', seasonal_periods=24).fit()
holt_f = holt_t.forecast(steps=len(yf))
holt_f = pd.DataFrame(holt_f).set_index(yf.index)
# MSE = np.square(np.subtract(yf.values,np.ndarray.flatten(holt_f.values))).mean()
# print("Mean square error for holt-winter method is ", MSE)

plt.figure(figsize=(8, 6))
plt.plot(yt,label= "train")
plt.plot(yf,label= "test")
plt.plot(holt_f,label= "Holt-Winter Method")
plt.legend()
plt.title('Holt-Winter Method - Raw(add)')
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Temperature-C')
plt.show()

### Log-Mul
y = df["Temp_C"] + 12
y = np.log(y)
yt, yf = train_test_split(y, shuffle=False, test_size=0.2)

holt_t = ets.ExponentialSmoothing(yt, trend='mul', seasonal='mul', seasonal_periods=24).fit()
holt_f = holt_t.forecast(steps=len(yf))
holt_f = pd.DataFrame(holt_f).set_index(yf.index)
# MSE = np.square(np.subtract(yf.values,np.ndarray.flatten(holt_f.values))).mean()
# print("Mean square error for holt-winter method is ", MSE)
holt_f = np.exp(holt_f) - 12

plt.figure(figsize=(8, 6))
plt.plot(yt,label= "train")
plt.plot(yf,label= "test")
plt.plot(holt_f,label= "Holt-Winter Method")
plt.legend()
plt.title('Holt-Winter Method - Log(mul)')
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Temperature-C')
plt.show()

### Log-add
y = df["Temp_C"] + 12
y = np.log(y)
yt, yf = train_test_split(y, shuffle=False, test_size=0.2)

holt_t = ets.ExponentialSmoothing(yt, trend='add', seasonal='add', seasonal_periods=24).fit()
holt_f = holt_t.forecast(steps=len(yf))
holt_f = pd.DataFrame(holt_f).set_index(yf.index)
MSE = np.square(np.subtract(yf.values,np.ndarray.flatten(holt_f.values))).mean()
print("Mean square error for holt-winter method is ", MSE)

holt_f = np.exp(holt_f) - 12

plt.figure(figsize=(8, 6))
plt.plot(yt,label= "train")
plt.plot(yf,label= "test")
plt.plot(holt_f,label= "Holt-Winter Method")
plt.legend()
plt.title('Holt-Winter Method - Log(add)')
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Temperature-C')
plt.show()


### 1st-diff-add
diff1 = [0]
for i in range(1, len(df)):
    diff = df.iloc[i]['Temp_C'] - df.iloc[i-1]['Temp_C']
    diff1.append(diff)
df['diff_1st_C'] = diff1

y = df['diff_1st_C']
yt, yf = train_test_split(y, shuffle=False, test_size=0.2)

holt_t = ets.ExponentialSmoothing(yt, trend='add', seasonal='add', seasonal_periods=24).fit()
holt_f = holt_t.forecast(steps=len(yf))

y_rev = copy.deepcopy(holt_f)
for i in range(len(holt_f)):
    if i == 0:
        y_rev[i] = df.iloc[-1]['Temp_K'] + holt_f[i]
    else:
        y_rev[i] = df.iloc[-1]['Temp_K'] + sum(holt_f[:i-1])
y_rev = pd.DataFrame(y_rev).set_index(yf.index)

y_rev = pd.DataFrame(y_rev)
y_rev.index = yf.index
plt.figure(figsize=(8, 6))
plt.plot(yt,label= "train")
plt.plot(yf,label= "test")
plt.plot(y_rev,label= "Holt-Winter Method")

plt.legend()
plt.title('Holt-Winter Method - 1st_diff(add)')
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Temperature-C')
plt.show()
