import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.holtwinters as ets
from sklearn.model_selection import train_test_split
import copy

df = pd.read_csv("clean_hourly_df.csv", index_col="date", parse_dates=True)

diff1 = [0]
for i in range(1, len(df)):
    diff = df.iloc[i]['Temp_K'] - df.iloc[i-1]['Temp_K']
    diff1.append(diff)
df['diff_1st_K'] = diff1

y = df['diff_1st_K']
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


data = df["Temp_K"]
data_t, data_f = train_test_split(data, shuffle=False, test_size=0.2)

plt.figure(figsize=(8, 6))
#plt.plot(data_t,label= "train")
plt.plot(data_f,label= "test")
plt.plot(y_rev,label= "Holt-Winter Method")

plt.legend()
plt.title('Holt-Winter Method - 1st_diff-add')
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Temperature-K')
plt.show()


### log
y = np.log(df['Temp_K'])
yt, yf = train_test_split(y, shuffle=False, test_size=0.2)

holt_t = ets.ExponentialSmoothing(yt, trend='mul', seasonal='mul', seasonal_periods=24).fit()
holt_f = holt_t.forecast(steps=len(yf))
holt_f = pd.DataFrame(holt_f).set_index(yf.index)

holt_f_rev = np.exp(holt_f)

plt.figure(figsize=(8, 6))
#plt.plot(data_t,label= "train")
plt.plot(data_f,label= "test")
plt.plot(holt_f_rev,label= "Holt-Winter Method")

plt.legend()
plt.title('Holt-Winter Method - Log-mul')
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Temperature-K')
plt.show()

###
y = np.log(df['Temp_K'])
yt, yf = train_test_split(y, shuffle=False, test_size=0.2)

holt_t = ets.ExponentialSmoothing(yt, trend='add', seasonal='add', seasonal_periods=24).fit()
holt_f = holt_t.forecast(steps=len(yf))

holt_f_rev = np.exp(holt_f)

plt.figure(figsize=(8, 6))
#plt.plot(data_t,label= "train")
plt.plot(data_f,label= "test")
plt.plot(holt_f_rev,label= "Holt-Winter Method")

plt.legend()
plt.title('Holt-Winter Method - Log-add')
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Temperature-K')
plt.show()
