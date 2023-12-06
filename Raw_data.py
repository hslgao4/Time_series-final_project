import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define functions
# NA check
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