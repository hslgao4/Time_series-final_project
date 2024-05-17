from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import STL
from toolbox import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def null_check(df):
    null_value = df.isnull()
    row_null = null_value.any(axis=1)
    rows = df[row_null]
    return rows

# fill missing with Drift method
def fill_data(df):
    filldf = df.groupby(pd.Grouper(freq='10T')).mean()
    df_null = null_check(filldf)
    print(f"{len(df_null)} rows have been filled")
    # Drift method
    filldf = filldf.interpolate().round(2)
    return filldf

# load data
def loaddata(start, end):
    path = 'https://www.bgc-jena.mpg.de/wetter/'
    list = []
    df = pd.DataFrame()
    for year in np.arange(start, end, 1):
        list.append(path+"mpi_roof_"+str(year)+"a.zip")
        list.append(path+"mpi_roof_"+str(year)+"b.zip")
    for url in list:
        df = df.append(pd.read_csv(url, encoding='unicode_escape',
                       parse_dates=True, index_col="Date Time"))
    df.index.name = 'datetime'
    return df

# Outliers check
def statistics_and_plt(df):
    for i in range(1, df.shape[1]):
        print(f"{df.columns[i]} statistics: \n{df.iloc[:,i].describe()}")
        plt.figure(figsize=(8, 6))
        plt.plot(df['date'], df.iloc[:, i])
        plt.ylabel('Magnitude')
        plt.xlabel('Date')
        plt.title(df.columns[i])
        plt.grid()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


#raw_data = loaddata(start=2019, end=2021)

raw_data = pd.read_csv("raw_dataset.csv", parse_dates=['datetime'])

# raw dataset
print(f"Shape of raw dataset: {raw_data.shape}")
print(f"NA in the raw dataset: {null_check(raw_data)}")
# save the raw dataset
df = fill_data(raw_data)
df.to_csv("filled_raw_dataset.csv")
# Outliers check
def statistics_and_plt(df):
    for i in range(1, df.shape[1]):
        print(f"{df.columns[i]} statistics: \n{df.iloc[:,i].describe()}")
        plt.figure(figsize=(8, 6))
        plt.plot(df['date'], df.iloc[:, i])
        plt.ylabel('Magnitude')
        plt.xlabel('Date')
        plt.title(df.columns[i])
        plt.grid()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
# Missing observations
df = pd.read_csv("filled_raw_dataset.csv")

# change date format, remove index, change to hourly-df
date_range = pd.date_range(start="2019-01-01 00:10:00",
                           end="2021-01-01 00:00:00", freq="10T")
df.insert(1, "date", date_range)
df = df.iloc[:, 1:]
df = df.set_index('date')
hourly_df = df.resample("60T").mean()
print(f"Hourly_df shape {hourly_df.shape}")
hourly_df = hourly_df.reset_index()

# Change column names
names = ['date', 'atmos_p', 'Temp_C', "Temp_K", 'Temp_C_humi', "rel_humi%", "Vapor_p_max", "Vapor_p",
         "Vapor_p_deficit", "spe_humi", "H2O_conc", "air_density", "wind_sp", "wind_sp_max", "wind_direction",
         "rain_depth", "rain_time", "SWDR", "PAR", "max_PAR", "Tlog", "CO2"]
hourly_df.columns = names
# round to 2 decimals
#hourly_df = hourly_df.round(2)
hourly_df.to_csv("hourly_df.csv", index=False)

# Check outliers and plot
df = pd.read_csv("hourly_df.csv", parse_dates=['date'])
statistics_and_plt(df)

# Fix wind_speed - change minimum to mean (Average method)
mean_wind = df[df["wind_sp"] >= 0]["wind_sp"].mean()
min_wind = df["wind_sp"].min()
df["wind_sp"] = df["wind_sp"].replace(min_wind, mean_wind)

print(f"new wind_sp statistics: \n{df['wind_sp'].describe()}")
plt.figure(figsize=(8, 6))
plt.plot(df['date'], df["wind_sp"])
plt.tight_layout()
plt.ylabel('Magnitude')
plt.xlabel('Date')
plt.title('new wind_sp')
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# change negative CO2 to mean(after dropping negative) - Average method
mean_CO2 = df[df["CO2"] >= 0]["CO2"].mean()
df.loc[df["CO2"] < 0, "CO2"] = mean_CO2

print(f"new CO2 statistics: \n{df['CO2'].describe()}")
plt.figure(figsize=(8, 6))
plt.plot(df['date'], df["CO2"])
plt.tight_layout()
plt.ylabel('Magnitude')
plt.xlabel('Date')
plt.title('new CO2')
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# change negative max_PAR to the last observation - Naive method
mean_max_PAR = df[df["max_PAR"] >= 0]["max_PAR"].mean()
df.loc[df["max_PAR"] < 0, "max_PAR"] = mean_max_PAR

for i in range(1, len(df)):
    if df.at[i, "max_PAR"] < 0:
        df.at[i, "max_PAR"] = df.at[i-1, "max_PAR"]

print(f"new max_PAR statistics: {df['max_PAR'].describe()}")

plt.figure(figsize=(8, 6))
plt.plot(df['date'], df["max_PAR"])
plt.tight_layout()
plt.ylabel('Magnitude')
plt.xlabel('Date')
plt.title('new max_PAR ')
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

df.to_csv('clean_hourly_df.csv', index=False)

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
