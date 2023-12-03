import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

df = pd.read_csv("clean_hourly_df.csv", parse_dates=["date"])
data = df[["date","Temp_C"]]
data.set_index('date', inplace=True)
train, test = train_test_split(data, test_size=0.2, shuffle=False, random_state=6313)

## Average method
ave = [round(np.mean(train),2)]*len(test)
ave = pd.DataFrame(ave)
ave.index = test.index

plt.figure(figsize=(12, 8))
plt.plot(train, label='Train', color='blue')
plt.plot(test, label='Test', color='green')
plt.plot(ave, label='Forecast', color='red')
plt.xlabel('Index')
plt.ylabel('Y')
plt.title('Average method')
plt.xticks(rotation=45)
plt.legend()
plt.show()


## Naive
naive = [train.iloc[-1, 0]]*len(test)
naive = pd.DataFrame(naive)
naive.index = test.index

plt.figure(figsize=(12, 8))
plt.plot(train, label='Train', color='blue')
plt.plot(test, label='Test', color='green')
plt.plot(naive, label='Forecast', color='red')
plt.xlabel('Index')
plt.ylabel('Y')
plt.title('Naive method')
plt.xticks(rotation=45)
plt.legend()
plt.show()

## Drift
b = (train.iloc[-1, 0]-train.iloc[0, 0])/(len(train)-1)
drift = []
for i in range(len(test)):
    d = train.iloc[-1, 0] + b*(i+1)
    drift.append(d)
drift = pd.DataFrame(drift)
drift.index = test.index
plt.figure(figsize=(12, 8))
plt.plot(train, label='Train', color='blue')
plt.plot(test, label='Test', color='green')
plt.plot(drift, label='Forecast', color='red')
plt.xlabel('Index')
plt.ylabel('Y')
plt.title('Drift method')
plt.xticks(rotation=45)
plt.legend()
plt.show()

## Simple exponential smoothing
model_ses = SimpleExpSmoothing(train).fit()
y_ses_forecast = model_ses.forecast(len(test))
ses = pd.DataFrame(y_ses_forecast)
ses.index = test.index
plt.figure(figsize=(12, 8))
plt.plot(train, label='Train', color='blue')
plt.plot(test, label='Test', color='green')
plt.plot(ses, label='Forecast', color='red')
plt.xlabel('Index')
plt.ylabel('Y')
plt.title('SES method')
plt.xticks(rotation=45)
plt.legend()
plt.show()