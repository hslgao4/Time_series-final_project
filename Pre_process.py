import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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