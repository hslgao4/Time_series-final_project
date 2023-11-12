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
y = df['Temp_C'] + 12
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
plt.title('Holt-Winter Method - mul')
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Temperature-C')
plt.show()

#### additive

y = df['Temp_C']
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
plt.title('Holt-Winter Method - add')
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Temperature-C')
plt.show()
