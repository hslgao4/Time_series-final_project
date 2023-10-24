import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Defined functions
# NA check
def findnanrows(df):
    is_NaN = df.isnull() 
    row_has_NaN = is_NaN.any(axis=1) 
    rows_with_NaN = df[row_has_NaN] 
    return rows_with_NaN

# missing 
def interpolatedata(df):
    filldf = df.groupby(pd.Grouper(freq='10T')).mean()
    dfnan = findnanrows(filldf)
    print("==> %s rows have been filled <==" %len(dfnan))
    filldf = filldf.interpolate().round(2)
    return filldf

# load data
def loaddata(sartyear, endyear):
    urlpath = 'https://www.bgc-jena.mpg.de/wetter/'
    urllist = []
    df = pd.DataFrame()
    for year in np.arange(sartyear, endyear, 1):
        urllist.append(urlpath+"mpi_roof_"+str(year)+"a.zip")
        urllist.append(urlpath+"mpi_roof_"+str(year)+"b.zip")
    for url in urllist:
        df = df.append(pd.read_csv(url, encoding='unicode_escape', parse_dates=True, index_col="Date Time"))
    df.index.name = 'datetime'
    return df

#ACF
def cal_ACF(y, lag, sample_plot_name):
    mean = np.mean(y)
    D = 0
    for i in range(len(y)):
        var = (y[i]-mean)**2
        D += var
    R = []
    for tao in range(lag+1):
        S = 0
        for t in range(tao, len(y)):
            N = (y[t]-mean)*(y[t-tao]-mean)
            S += N
        r = S/D
        R.append(r)
    R_inv = R[::-1]
    Magnitute = R_inv + R[1:]
    ax = plt.figure()
    x_values = range(-lag, lag + 1)
    (markers, stemlines, baseline) = plt.stem(x_values, Magnitute, markerfmt='o')
    plt.setp(markers, color = 'red')
    m = 1.96/np.sqrt(len(y))
    plt.axhspan(-m, m, alpha = 0.2, color = 'blue')
    plt.xlabel('Lags')
    plt.ylabel('Magnitute')
    plt.title(f'Autocorrelation Function of {sample_plot_name}' )
    plt.show()
    return ax

raw_data = loaddata(sartyear=2019, endyear=2021)
# raw dataset 
print(f"Shape of raw dataset: {raw_data.shape}")
print(f"NA in the raw dataset: {findnanrows(raw_data)}")
# save the raw dataset 
raw_data.to_csv("raw_dataset.csv")
# Missing observations
df = interpolatedata(raw_data)
#print(df.shape)
#change date format, remove index
date_range = pd.date_range(start="2019-01-01 00:10:00", end="2021-01-01 00:00:00", freq="10T")
df.insert(0, "date", date_range)
df = df.reset_index()
df = df.iloc[:, 1:]
df = df.set_index('date')
hourly_df = df.resample("60T").mean()
print(f"Hourly_df shape {hourly_df.shape}")
# Target varibale statistics
hourly_df["wv (m/s)"].describe()
# Outlier - change the minimum to mean
mean_wind = hourly_df["wv (m/s)"].mean()
min_wind = hourly_df["wv (m/s)"].min()
hourly_df["wv (m/s)"] = hourly_df["wv (m/s)"].replace(min_wind, mean_wind)
hourly_df = hourly_df.reset_index()
#hourly_df.to_csv("hourly_df.csv", index = False)
#df = pd.read_csv("hourly_df.csv")
plt.figure(figsize=(10, 6))
plt.plot(hourly_df['date'], hourly_df["wv (m/s)"], marker='o', linestyle='-', color='b', label='wind speed')
plt.legend()
plt.tight_layout()
plt.show()
cal_ACF(hourly_df['wv (m/s)'], 20, "Hourly dataset")