from scipy import signal
import pandas as pd
import numpy as np
import statsmodels.api as sm
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from prettytable import PrettyTable
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.seasonal import STL
import statsmodels.tsa.holtwinters as ets



def e_dlsim(num, den, y):
    system = (den, num, 1)
    _, e_dlsim = signal.dlsim(system, y)
    return e_dlsim

def step_1(theta, delta, na, nb, y, N):
    n = na+nb
    if nb < na:
        corr_term = na - nb
    else:
        corr_term = 0
    den = np.r_[1, theta[:na, 0]] # AR
    num = np.r_[1, theta[na:, 0], [0] * corr_term] # MA
    e_0 = e_dlsim(num, den, y)
    X = np.zeros((N, 1))

    for i in range(n):
        if i < na:
            den_new = copy.deepcopy(den)
            den_new[i + 1] += delta
            e = e_dlsim(num, den_new, y)
            x = (e_0 - e) / delta
            X = np.concatenate((X, x), axis=1)
        else:
            num_new = copy.deepcopy(num)
            num_new[i - na + 1] += delta
            e = e_dlsim(num_new, den, y)
            x = (e_0 - e) / delta
            X = np.concatenate((X, x), axis=1)
    X = X[:, 1:]
    A = X.T @ X
    g = X.T @ e_0
    SSE_0 = e_0.T @ e_0
    return A, g, SSE_0

def step_2(A, g, miu, na, nb, theta, y):
    l = np.eye(na+nb)
    d_theta = np.linalg.inv(A + miu*l) @ g
    theta_new = theta + d_theta

    if nb < na:
        corr_term = na - nb
    else:
        corr_term = 0
    den = np.r_[1, theta_new[:na, 0]]  # MA
    num = np.r_[1, theta_new[na:, 0], [0] * corr_term]  # AR

    e_theta = e_dlsim(num, den, y)
    SSE_new = e_theta.T @ e_theta
    return SSE_new, d_theta, theta_new

def LM_algorithm(y, na, nb):
    #y, na, nb, coe = gen_arma_df()
    max_iteration = 50
    miu_max = 10 ** 10
    iteration = 1
    theta = np.zeros((na+nb, 1))
    delta = 10**-6
    N = len(y)
    SSE = [y.T @ y]
    miu = 0.01

    for i in range(1, max_iteration):

        A, g, SSE_0 = step_1(theta, delta, na, nb, y, N)
        SSE_new, d_theta, theta_new = step_2(A, g, miu, na, nb, theta, y)
        SSE.append(SSE_new.item())
        if SSE_new < SSE_0:
            if np.linalg.norm(d_theta) < 10 ** -3:
                theta_hat = theta_new
                var = SSE_new / (N-na-nb)
                co_var = var * np.linalg.inv(A)
                ar = theta_hat[:na,0].flatten()
                ma = theta_hat[na:, 0].flatten()
                SSE = [round(sse, 3) for sse in SSE]
                print(f"The estimated parameters are: {np.around(theta_hat, decimals =3)}")
                #print(f"The true parameters are: {coe}")

                for i in range(len(theta)):
                    if i < na:
                        cl_low = theta_hat[i] - 2 * np.sqrt(co_var[i, i])
                        cl_high = theta_hat[i] + 2 * np.sqrt(co_var[i, i])
                        print(f"The confidence interval for a{i+1} is: {np.around(cl_low, decimals =3)} to {np.around(cl_high, decimals =3)}")
                    else:
                        cl_low = theta_hat[i] - 2 * np.sqrt(co_var[i, i])
                        cl_high = theta_hat[i] + 2 * np.sqrt(co_var[i, i])
                        print(f"The confidence interval for b{i-na+1} is: {np.around(cl_low, decimals =3)} to {np.around(cl_high, decimals =3)}")

                print(f"The estimated covariance matrix is: {np.around(co_var, decimals =3)}")
                print(f"The estimated estimated variance of error is: {np.around(var, decimals =3)}")
                #print(f"The roots of AR coe are: {np.around(np.roots(np.r_[1,ar]), decimals=3)}")
                #print(f"The roots of MA coe are: {np.around(np.roots(np.r_[1,ma]), decimals=3)}")

                index = np.arange(iteration+1)
                plt.figure()
                plt.plot(index, SSE, label = "SSE")
                plt.xlabel("Number of Iteration")
                plt.ylabel("Sum of Square Error (SSE)")
                plt.title("SSE vs. Iteration")
                plt.xticks(index)
                plt.show()
                break

            else:
                theta = theta_new
                miu = miu/10

        while SSE_new > SSE_0:
            miu = miu * 10
            if miu > miu_max:
                theta_hat = theta_new
                var = SSE_new / (N - na - nb)
                co_var = var * np.linalg.inv(A)
                print("miu is too big! Stop")
                break

            else:
                SSE_new, d_theta, theta_new = step_2(A, g, miu, na, nb, theta, y)

        if miu > miu_max:
            print("Too many iterations. Stop!")
            break

        iteration += 1

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
    (markers, stemlines, baseline) = plt.stem(
        x_values, Magnitute, markerfmt='o')
    plt.setp(markers, color='red')
    m = 1.96/np.sqrt(len(y))
    plt.axhspan(-m, m, alpha=0.2, color='blue')
    plt.xlabel('Lags')
    plt.ylabel('Magnitute')
    plt.title(f'Autocorrelation Function of {sample_plot_name}')
    plt.show()
    return ax

def cal_fi(j, k, ACF):
    if k == 1:
        up = ACF[j+1]
        bottom = ACF[j]
        if bottom == 0:
            fi = "inf"
        else:
            fi = up / bottom
    else:
        den = []
        for a in range(j, j + k):
            row = []
            for b in range(a - (k - 1), a + 1):
                b = abs(b)
                R = ACF[b]
                row.append(R)
            row = row[::-1]
            den.append(row)

        num = copy.deepcopy(den)
        for i in range(k):
            num[i][-1] = ACF[j + 1 + i]
        up = np.linalg.det(num)
        bottom = np.linalg.det(den)

        if bottom == 0:
            fi = "inf"
        else:
            fi = up / bottom
            if abs(fi) < 0.0000001:
                fi = 0
    return fi

def GPAC_table(ACF, J=7, K=7):
    temp = np.zeros((J, K - 1))
    for k in range(1, K):
        for j in range(J):
            value = cal_fi(j, k, ACF)
            temp[j][k-1] = value
    table = pd.DataFrame(temp)
    table = table.round(2)
    table.columns = range(1, K)
    plt.figure()
    sns.heatmap(table, annot=True)
    plt.title("Generalized Partial Autocorrelation(GPAC) Table")
    plt.show()
    return table

def ACF_PACF_Plot(y,lags):
    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)
    fig = plt.figure()
    plt.subplot(2,1,1)
    plt.title('ACF/PACF of the raw data')
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.subplot(2,1,2)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    fig.tight_layout(pad=3)
    plt.show()

def cal_rolling_mean_var(df, item_list):
    def rolling_mean_var(df, x):
        rolling_mean = []
        rolling_var = []
        for i in range(1, len(df) + 1):
            new_df = df.iloc[:i, ]
            if i == 1:
                mean = new_df[x]
                var = 0
            else:
                mean = new_df[x].mean()
                var = new_df[x].var()
            rolling_mean.append(mean)
            rolling_var.append(var)
        return rolling_mean, rolling_var
    plt.figure(figsize=(8, 7))
    plt.subplot(2, 1, 1)
    for i in item_list:
        roll_mean, roll_var = rolling_mean_var(df, i)

        plt.plot(roll_mean, label=i, lw=1.5)
    plt.xlabel('Samples')
    plt.ylabel('Magnitude')
    plt.title('Rolling Mean')
    plt.legend()

    plt.subplot(2, 1, 2)
    for i in item_list:
        roll_mean, roll_var = rolling_mean_var(df, i)

        plt.plot(roll_var, label=i, lw=1.5)
    plt.xlabel('Samples')
    plt.ylabel('Magnitude')
    plt.title('Rolling Variance')
    plt.legend()
    plt.tight_layout()
    plt.show()

# ACF

# ADF
def ADF_Cal(x):
    result = adfuller(x)
    print("ADF Statistic: %f" % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
# KPSS
def kpss_test(timeseries):
    print('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=[
                            'Test Statistic', 'p-value', 'Lags Used'])
    for key, value in kpsstest[3].items():
        kpss_output['Critical Value (%s)' % key] = value
        print(kpss_output)

def Q_test(error, lag, na, nb):
    ACF = sm.tsa.acf(error, nlags=lag)
    Q = len(error)*sum(np.square(ACF))
    DOF = lag - na - nb
    alfa = 0.01
    chi_critical = chi2.ppf(1-alfa, DOF)
    if Q < chi_critical:
        print("Q:", round(Q,2))
        print("Q*:", round(chi_critical,2))
        print("The residual is white")
    else:
        print("The residual is not white")


def tabel_pretty(df,title):
    x = PrettyTable()
    for i in range(df.shape[0]):
        x.add_row(df.iloc[i,:])
    x.title = title
    x.field_names = df.columns
    x.float_format = '.2'
    x.hrules = 1
    print(x.get_string())

def cal_err(y_pred, Y_test):
    error = []
    error_se = []
    for i in range(len(y_pred)):
        e = Y_test.iloc[i,0] - y_pred.iloc[i,0]
        error.append(e)
        error_se.append(e**2)
    error_mean = np.mean(error)
    error_var = np.var(error)
    error_mse = np.mean(error_se)
    return error, error_mean, error_var, error_mse
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


raw_data = loaddata(start=2019, end=2021)

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
train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False, random_state=6313)
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

df = pd.read_csv("clean_hourly_df.csv", index_col="date", parse_dates=True)
y = df["Temp_C"]
yt, yf = train_test_split(y, shuffle=False, test_size=0.2)

holt_t = ets.ExponentialSmoothing(yt, trend='add', seasonal='add', seasonal_periods=744).fit()
holt_f = holt_t.forecast(steps=len(yf))
holt_f = pd.DataFrame(holt_f).set_index(yf.index)
# MSE = np.square(np.subtract(yf.values,np.ndarray.flatten(holt_f.values))).mean()
# print("Mean square error for holt-winter method is ", MSE)

plt.figure(figsize=(8, 6))
plt.plot(yt,label= "train")
plt.plot(yf,label= "test")
plt.plot(holt_f,label= "Holt-Winter Method")
plt.legend()
plt.title('Holt-Winter Method - Raw(add)-744')
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Temperature-C')
plt.show()

def cal_err(y_pred, Y_test):
    error = []
    error_se = []
    for i in range(len(y_pred)):
        e = Y_test.iloc[i,0] - y_pred.iloc[i,0]
        error.append(e)
        error_se.append(e**2)
    error_mean = np.mean(error)
    error_var = np.var(error)
    error_mse = np.mean(error_se)
    return error, error_mean, error_var, error_mse

def tabel_pretty(df,title):
    x = PrettyTable()
    for i in range(df.shape[0]):
        x.add_row(df.iloc[i,:])
    x.title = title
    x.field_names = df.columns
    x.float_format = '.2'
    x.hrules = 1
    print(x.get_string())

df = pd.read_csv("clean_hourly_df.csv", parse_dates=['date'])
df = df.drop("Temp_K", axis=1)
df.set_index('date', inplace=True)
temp = df.pop("Temp_C")
df.insert(0, "Temp_C", temp)

# SVD analysis
df_svd = df.drop(['Temp_C'], axis = 1)
U, S, V = np.linalg.svd(df_svd)
print('Singular values\n', S)

# Condition number
print(f'Condition number is {np.linalg.cond(df_svd):.2f}')

# Normalize data
scaler = StandardScaler()
stand_df = scaler.fit_transform(df)

# PCA
Y = stand_df[:, 0].reshape(-1, 1)
X = stand_df[:, 1:]
X = np.column_stack((np.ones(X.shape[0]), X))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False, random_state=6313)

pca = PCA(n_components=19)
pca.fit(X_train)
var_ratio = np.cumsum(pca.explained_variance_ratio_)
print(var_ratio)

pca_reduced = PCA(n_components=7)
pca_reduced.fit(X_train)
X_train_pca = pca_reduced.transform(X_train)
model_pca = sm.OLS(Y_train,X_train_pca).fit()
print(model_pca.summary())

X_test_pca = pca_reduced.transform(X_test)
y_pred = model_pca.predict(X_test_pca)

Y_temp = df.iloc[:, 0:1]
train, test = train_test_split( Y_temp, test_size=0.2, shuffle=False, random_state=6313)

Y_test = pd.DataFrame(Y_test)
Y_test.index = test.index
y_pred_pca = pd.DataFrame(y_pred)
y_pred_pca.index = test.index
Y_train = pd.DataFrame(Y_train)
Y_train.index = train.index

### error
err_pca, err_pca_mean, err_pca_var, err_pca_mse = cal_err(y_pred_pca, Y_test)
ACF_PACF_Plot(err_pca,40)


plt.figure(figsize=(10, 6))
plt.plot( Y_train, label='Train Data')
plt.plot( Y_test, label='Test Data')
plt.plot(y_pred_pca, label='Prediction-PCA')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.xticks(rotation=45)
plt.legend()
plt.title('PCA residual')
plt.show()


# Backward stepwise regression
Y = stand_df[:, 0].reshape(-1, 1)
Y = pd.DataFrame(Y)
Y.index = df.index
column_y = ["Temp_C"]
Y.columns = column_y
X = stand_df[:, 1:]
X = np.column_stack((np.ones(X.shape[0]), X))
X = pd.DataFrame(X)
X.index = df.index
column_x = ["constant", 'atmos_p', 'Temp_C_humi', 'rel_humi%', 'Vapor_p_max',
       'Vapor_p', 'Vapor_p_deficit', 'spe_humi', 'H2O_conc', 'air_density',
       'wind_sp', 'wind_sp_max', 'wind_direction', 'rain_depth', 'rain_time',
       'SWDR', 'PAR', 'max_PAR', 'Tlog', 'CO2']
X.columns = column_x

varis = copy.deepcopy(column_x)

while True:
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False, random_state=6313)
    model = sm.OLS(Y_train, X_train).fit()
    #print(model.summary())
    print(f"\nAIC is {model.aic:.1f}")
    print(f"\nBIC is {model.bic:.1f}")
    print(f"\nR_squared_adj is {model.rsquared_adj:.3f}")
    print("---------")
    p_values = model.pvalues
    max_p_value_index = np.argmax(p_values)
    max_p_value = p_values[max_p_value_index]
    name = varis[max_p_value_index]
    print(f"Max p-vlue is: {max_p_value:.2f}")

    if max_p_value > 0.05:
        X.drop(X.columns[max_p_value_index], axis=1, inplace=True)
        varis.pop(max_p_value_index)
        print(f"\nremove variable is: {name}")

    if max_p_value <= 0.05:
        break

print(varis)

#Final model from Backward stepwise regression
back_X = X[varis]
X_train, X_test, Y_train, Y_test = train_test_split(back_X, Y, test_size=0.2, shuffle=False, random_state=6313)
model_backward = sm.OLS(Y_train, X_train).fit()
print(model_backward.summary())

varis_2 = varis[:9]
back_X = X[varis_2]
X_train, X_test, Y_train, Y_test = train_test_split(back_X, Y, test_size=0.2, shuffle=False, random_state=6313)
model_backward = sm.OLS(Y_train, X_train).fit()
print(model_backward.summary())

back_predic = model_backward.predict(X_test)
back_predic = pd.DataFrame(back_predic)
back_predic.index = Y_test.index
### error
err_back, err_back_mean, err_back_var, err_back_mse = cal_err(back_predic, Y_test)
ACF_PACF_Plot(err_back,40)

plt.figure(figsize=(10, 6))
plt.plot( Y_train, label='Train Data')
plt.plot( Y_test, label='Test Data')
plt.plot(back_predic, label='Backward')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.xticks(rotation=45)
plt.legend()
plt.title('Backward Stepwide Regression')
plt.show()

# VIF
df = pd.read_csv("clean_hourly_df.csv")
df = df.drop("Temp_K", axis=1)
df = df.drop(["date"], axis = 1)
temp = df.pop("Temp_C")
df.insert(0, "Temp_C", temp)

scaler = StandardScaler()
stand_df = scaler.fit_transform(df)

Y = stand_df[:, 0].reshape(-1, 1)
Y = pd.DataFrame(Y)
Y.index = df.index
column_y = ["Temp_C"]
Y.columns = column_y
X = stand_df[:, 1:]
X = np.column_stack((np.ones(X.shape[0]), X))
X = pd.DataFrame(X)
X.index = df.index
column_x = ['constant', 'atmos_p', 'Temp_C_humi', 'rel_humi%', 'Vapor_p_max',
       'Vapor_p', 'Vapor_p_deficit', 'spe_humi', 'H2O_conc', 'air_density',
       'wind_sp', 'wind_sp_max', 'wind_direction', 'rain_depth', 'rain_time',
       'SWDR', 'PAR', 'max_PAR', 'Tlog', 'CO2']
X.columns = column_x

varibale = copy.deepcopy(column_x)
while True:
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False, random_state=6313)
    model = sm.OLS(Y_train, X_train).fit()
    print(f"AIC is {model.aic:.1f}")
    print(f"\nBIC is {model.bic:.1f}")
    print(f"\nR_squared_adj is {model.rsquared_adj:.3f}")
    vif = []
    for i in range(X_train.shape[1]):
        vif.append(variance_inflation_factor(X_train.values, i))
    print("vif:", vif)
    m_value = np.max(vif)
    m_index = vif.index(m_value)
    name = varibale[m_index]

    if m_value > 10:
        X.drop(X.columns[m_index], axis=1, inplace=True)
        varibale.pop(m_index)
        print("max vif: ", m_value)
        print(f"\nremove variable is: {name}")

    if m_value <= 10:
        break

VIF_X = X[varibale]
X_train, X_test, Y_train, Y_test = train_test_split(VIF_X, Y, test_size=0.2, shuffle=False, random_state=6313)
model_vif = sm.OLS(Y_train, X_train).fit()
print(model_vif.summary())

# delete the p-value over 0.05
varibale.pop(6)
VIF_X = X[varibale]
X_train, X_test, Y_train, Y_test = train_test_split(VIF_X, Y, test_size=0.2, shuffle=False, random_state=6313)
model_vif = sm.OLS(Y_train, X_train).fit()
print(model_vif.summary())
#
varibale.pop(6)
#
varibale.pop(-1)
#
varibale.pop(5)
#
varibale.pop(5)
#
varibale.pop(5)
#
varibale.pop(1)
#
VIF_X = X[varibale]
X_train, X_test, Y_train, Y_test = train_test_split(VIF_X, Y, test_size=0.2, shuffle=False, random_state=6313)
model_vif = sm.OLS(Y_train, X_train).fit()
print(model_vif.summary())


vif_predic = model_vif.predict(X_test)
vif_predic = pd.DataFrame(vif_predic)
vif_predic.index = Y_test.index
err_vif, err_vif_mean, err_vif_var, err_vif_mse = cal_err(vif_predic, Y_test)
ACF_PACF_Plot(err_vif,40)

plt.figure(figsize=(10, 6))
plt.plot( Y_train, label='Train Data')
plt.plot( Y_test, label='Test Data')
plt.plot(vif_predic, label='VIF model')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.xticks(rotation=45)
plt.legend()
plt.title('VIF')
plt.show()


data = {"Method": ["PCA", "Backwards", "VIF"],
        "Mean": [err_pca_mean, err_back_mean, err_vif_mean],
        "Variance": [err_pca_var, err_back_var, err_vif_var],
        "MSE": [err_pca_mse, err_back_mse, err_vif_mse]
}

data = pd.DataFrame(data)
tabel_pretty(data,"Residual")

def tabel_pretty(df,title):
    x = PrettyTable()
    for i in range(df.shape[0]):
        x.add_row(df.iloc[i,:])
    x.title = title
    x.field_names = df.columns
    x.float_format = '.2'
    x.hrules = 1
    print(x.get_string())

lag = 10
def cal_Q (y):
    mean = np.mean(y)
    D = 0
    for i in range(len(y)):
        var = (y[i]-mean)**2
        D += var

    R = 0
    for tao in np.arange(1,lag+1):
        S = 0
        for t in range(tao, len(y)):
            N = (y[t]-mean)*(y[t-tao]-mean)
            S += N
        r = S/D
        R += r**2
    Q = len(y)*R
    return round(Q,2)

def cal_mse(y):
    temp = []
    for i in range(len(y)):
        err_sqr = y.iloc[i] ** 2
        temp.append(err_sqr)
    mse = np.mean(temp)
    return mse

df = pd.read_csv("clean_hourly_df.csv", parse_dates=["date"])
data = df[["date","Temp_C"]]
data.set_index('date', inplace=True)
train, test = train_test_split(data, test_size=0.2, shuffle=False, random_state=6313)

## Average method
ave = [round(np.mean(train),2)]*len(test)
ave = pd.DataFrame(ave)
ave.index = test.index

err_ave = test.iloc[:,0]- ave.iloc[:,0]
mean_err_ave = np.mean(err_ave)
var_err_ave = np.var(err_ave)
mse_ave = cal_mse(err_ave)

## Naive
naive = [train.iloc[-1, 0]]*len(test)
naive = pd.DataFrame(naive)
naive.index = test.index

err_naive = test.iloc[:,0]- naive.iloc[:,0]
mean_err_naive = np.mean(err_naive)
var_err_naive = np.var(err_naive)
mse_naive = cal_mse(err_naive)

## Drift
b = (train.iloc[-1, 0]-train.iloc[0, 0])/(len(train)-1)
drift = []
for i in range(len(test)):
    d = train.iloc[-1, 0] + b*(i+1)
    drift.append(d)
drift = pd.DataFrame(drift)
drift.index = test.index

err_drift = test.iloc[:,0] - drift.iloc[:,0]
mean_err_drift = np.mean(err_drift)
var_err_drift = np.var(err_drift)
mse_drift = cal_mse(err_drift)

## Simple exponential smoothing
model_ses = SimpleExpSmoothing(train).fit()
y_ses_forecast = model_ses.forecast(len(test))
ses = pd.DataFrame(y_ses_forecast)
ses.index = test.index

err_ses = test.iloc[:,0] - ses.iloc[:,0]
mean_err_ses = np.mean(err_ses)
var_err_ses = np.var(err_ses)
mse_ses = cal_mse(err_ses)
## Plot
plt.figure(figsize=(12, 8))
plt.plot(train, label='Train')#, color='blue')
plt.plot(test, label='Test')#, color='green')
plt.plot(ave, label='ave')#, color='red')
plt.plot(naive, label='naive')#, color='red')
plt.plot(drift, label='drift')#, color='red')
plt.plot(ses, label='ses')#, color='red')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Base model results')
plt.xticks(rotation=45)
plt.legend()
plt.show()

data = {"Method": ["Average", "Naive", "Dirft", "SES"],
        "Mean": [mean_err_ave, mean_err_naive, mean_err_drift, mean_err_ses],
        "Variance": [var_err_ave, var_err_naive, var_err_drift, var_err_ses],
        "MSE": [mse_ave, mse_naive, mse_drift, mse_ses]
}

data = pd.DataFrame(data)
tabel_pretty(data,"Base Model - Residual")



def cal_err(y_pred, Y_test):
    error = []
    error_se = []
    for i in range(len(y_pred)):
        e = Y_test.iloc[i,0] - y_pred.iloc[i,0]
        error.append(e)
        error_se.append(e**2)
    error_mean = np.mean(error)
    error_var = np.var(error)
    error_mse = np.mean(error_se)
    return error, error_mean, error_var, error_mse

df = pd.read_csv("clean_hourly_df.csv")
df = df.drop("Temp_K", axis=1)
df = df.drop(["date"], axis = 1)
temp = df.pop("Temp_C")
df.insert(0, "Temp_C", temp)
varibale = ["Temp_C", 'rel_humi%', 'Vapor_p_deficit', 'spe_humi']
df_new = df[varibale]

scaler = StandardScaler()
stand_df = scaler.fit_transform(df_new)

Y = stand_df[:, 0].reshape(-1, 1)
Y = pd.DataFrame(Y)
Y.index = df.index
column_y = ["Temp_C"]
Y.columns = column_y
X = stand_df[:, 1:]
X = np.column_stack((np.ones(X.shape[0]), X))
X = pd.DataFrame(X)
X.index = df.index
X.columns = ['constant', 'rel_humi%', 'Vapor_p_deficit', 'spe_humi']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False, random_state=6313)

LR_model = sm.OLS(Y_train, X_train).fit()
print(LR_model.summary())

prediction = LR_model.predict(X_test)
prediction = pd.DataFrame(prediction)
prediction.index = Y_test.index
err_vif, err_vif_mean, err_vif_var, err_vif_mse = cal_err(prediction, Y_test)
ACF_PACF_Plot(err_vif,40)

cal_ACF(err_vif, 40, "Residual")

plt.figure(figsize=(10, 6))
plt.plot( Y_train, label='Train Data')
plt.plot( Y_test, label='Test Data')
plt.plot(prediction, label='Multiple linear regression')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.xticks(rotation=45)
plt.legend()
plt.title('Multiple linear regression')
plt.show()

# T-test
t_test_results = LR_model.t_test(np.identity(X_train.shape[1]))
print("T-Test Results:")
print(t_test_results)

# F-test
f_test_results = LR_model.f_test(np.identity(X_train.shape[1]))
print("\nF-Test Results:")
print(f_test_results)


from sklearn.model_selection import TimeSeriesSplit
cv = TimeSeriesSplit(n_splits=5)
rmse = []
mse = []
R_squared = []
R_squared_adj = []
for train_index, valid_index in cv.split(X_train):
    X_train_cv, X_valid_cv = X_train.iloc[train_index], X_train.iloc[valid_index]
    y_train_cv, y_valid_cv = Y_train.iloc[train_index], Y_train.iloc[valid_index]
    model_cv = sm.OLS(y_train_cv,X_train_cv).fit()
    prediction = model_cv.predict(X_valid_cv)
    prediction = pd.DataFrame(prediction)
    error, error_mean, error_var, error_mse = cal_err(prediction, y_valid_cv)
    R_squared.append(model_cv.rsquared)
    R_squared_adj.append(model_cv.rsquared_adj)
    rmse.append(np.sqrt(error_mse))
    mse.append(error_mse)


cv_data = {"MSE": mse,
           "RMSE": rmse,
           "R square": R_squared,
           "Adj R square": R_squared_adj
}
cv_data = pd.DataFrame(cv_data)
def tabel_pretty(df,title):
    x = PrettyTable()
    for i in range(df.shape[0]):
        x.add_row(df.iloc[i,:])
    x.title = title
    x.field_names = df.columns
    x.float_format = '.2'
    x.hrules = 1
    print(x.get_string())

cv_table = tabel_pretty(cv_data,"Cross validation")

from statsmodels.stats.diagnostic import acorr_ljungbox
print('Ljung-Box test')
print(acorr_ljungbox(err_vif,lags=20))

df = pd.read_csv("clean_hourly_df.csv")
data = df["Temp_C"]

ACF1 = sm.tsa.acf(data, nlags=200)
table1 = GPAC_table(ACF1, J=12, K=12)
ACF_PACF_Plot(data, lags=200)

train, test = train_test_split(data, test_size=0.2, shuffle=False, random_state=6313)
model1 = sm.tsa.SARIMAX(train, order = (1, 0, 0), seasonal_order= (1,0,0,24))
model_fit_1 = model1.fit()
result_1 = model_fit_1.predict(start=0, end=len(train)-1)

# plot train vs. 1-step
plt.figure(figsize=(8, 7))
plt.plot(train.tolist()[:1000], label="train", lw=1.5)
plt.plot(result_1.tolist()[:1000], label="1-step Prediction", lw=1.5)
plt.xlabel('Index')
plt.ylabel('Temperature')
plt.legend()
plt.title('train vs. 1-step prediction [:1000]')
plt.tight_layout()
plt.show()

train_arima = pd.DataFrame(train)
result_sarima = pd.DataFrame(result_1)
err_sarima_1 = []
for i in range(len(result_sarima)):
    e = train_arima.iloc[i, 0] - result_sarima.iloc[i, 0]
    err_sarima_1.append(e)

plot = cal_ACF(err_sarima_1, 100, "Residual")
ACF_PACF_Plot(err_sarima_1, lags=100)
ACF = sm.tsa.acf(err_sarima_1, nlags=80)
table_error = GPAC_table(ACF, J=12, K=12)

error_final, error_mean_final, error_var_final, error_mse_final = cal_err(result_sarima, train_arima)


# forecast
sarima_pred_test = model_fit_1.forecast(steps=len(test))

##plot
df = pd.read_csv("clean_hourly_df.csv", parse_dates=['date'])
data = pd.DataFrame(df[["date","Temp_C"]])
data.set_index('date', inplace=True)

train, test = train_test_split(data, test_size=0.2, shuffle=False, random_state=6313)
sarima_pred_test = pd.DataFrame(sarima_pred_test)
sarima_pred_test.index = test.index

plt.figure(figsize=(8, 7))
plt.plot(train, label="test", lw=1.5)
plt.plot(test, label="test", color = 'green', lw=1.5)
plt.plot(sarima_pred_test, label="forecast", color = 'orange', lw=1.5)
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.xticks(rotation=45)
plt.title('Final SARIMA model ')
plt.tight_layout()
plt.show()

error_f = []
for i in range(len(test)):
    err = test.iloc[i, 0] - sarima_pred_test.iloc[i, 0]
    error_f.append(err)
error_f = pd.DataFrame(error_f)


error_f, error_mean_f, error_var_f, error_mse_f = cal_err(sarima_pred_test, test)

############################################################################
df = pd.read_csv("clean_hourly_df.csv")
data = df["Temp_C"]
train, test = train_test_split(data, test_size=0.2, shuffle=False, random_state=6313)
model1 = sm.tsa.SARIMAX(train, order = (1, 0, 3), seasonal_order= (1,0,0,24))
model_fit_1 = model1.fit()
result_1 = model_fit_1.predict(start=0, end=len(train)-1)

# plot train vs. 1-step
plt.figure(figsize=(8, 7))
plt.plot(train.tolist()[:1000], label="train", lw=1.5)
plt.plot(result_1.tolist()[:1000], label="1-step Prediction", lw=1.5)
plt.xlabel('Index')
plt.ylabel('Temperature')
plt.legend()
plt.title('train vs. 1-step prediction [:1000]')
plt.tight_layout()
plt.show()

train_arima = pd.DataFrame(train)
result_sarima = pd.DataFrame(result_1)
err_sarima_1 = []
for i in range(len(result_sarima)):
    e = train_arima.iloc[i, 0] - result_sarima.iloc[i, 0]
    err_sarima_1.append(e)

plot = cal_ACF(err_sarima_1, 100, "Residual")
ACF_PACF_Plot(err_sarima_1, lags=100)
ACF = sm.tsa.acf(err_sarima_1, nlags=80)
table_error = GPAC_table(ACF, J=12, K=12)

error_final, error_mean_final, error_var_final, error_mse_final = cal_err(result_sarima, train_arima)


# forecast
sarima_pred_test = model_fit_1.forecast(steps=len(test))

##plot
df = pd.read_csv("clean_hourly_df.csv", parse_dates=['date'])
data = pd.DataFrame(df[["date","Temp_C"]])
data.set_index('date', inplace=True)

train, test = train_test_split(data, test_size=0.2, shuffle=False, random_state=6313)
sarima_pred_test = pd.DataFrame(sarima_pred_test)
sarima_pred_test.index = test.index

plt.figure(figsize=(8, 7))
plt.plot(train, label="test", lw=1.5)
plt.plot(test, label="test", color = 'green', lw=1.5)
plt.plot(sarima_pred_test, label="forecast", color = 'orange', lw=1.5)
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.xticks(rotation=45)
plt.title('Final SARIMA model ')
plt.tight_layout()
plt.show()

error_f = []
for i in range(len(test)):
    err = test.iloc[i, 0] - sarima_pred_test.iloc[i, 0]
    error_f.append(err)
error_f = pd.DataFrame(error_f)


error_f, error_mean_f, error_var_f, error_mse_f = cal_err(sarima_pred_test, test)

############################################################
df = pd.read_csv("clean_hourly_df.csv")
data = df["Temp_C"]
train, test = train_test_split(data, test_size=0.2, shuffle=False, random_state=6313)
model1 = sm.tsa.SARIMAX(train, order = (1, 0, 3), seasonal_order= (1,0,1,24))
model_fit_1 = model1.fit()
result_1 = model_fit_1.predict(start=0, end=len(train)-1)

# plot train vs. 1-step
plt.figure(figsize=(8, 7))
plt.plot(train.tolist()[:1000], label="train", lw=1.5)
plt.plot(result_1.tolist()[:1000], label="1-step Prediction", lw=1.5)
plt.xlabel('Index')
plt.ylabel('Temperature')
plt.legend()
plt.title('train vs. 1-step prediction [:1000]')
plt.tight_layout()
plt.show()

train_arima = pd.DataFrame(train)
result_sarima = pd.DataFrame(result_1)
err_sarima_1 = []
for i in range(len(result_sarima)):
    e = train_arima.iloc[i, 0] - result_sarima.iloc[i, 0]
    err_sarima_1.append(e)

plot = cal_ACF(err_sarima_1, 100, "Residual")
ACF_PACF_Plot(err_sarima_1, lags=100)
ACF = sm.tsa.acf(err_sarima_1, nlags=80)
table_error = GPAC_table(ACF, J=12, K=12)

error_final, error_mean_final, error_var_final, error_mse_final = cal_err(result_sarima, train_arima)


# forecast
sarima_pred_test = model_fit_1.forecast(steps=len(test))

##plot
df = pd.read_csv("clean_hourly_df.csv", parse_dates=['date'])
data = pd.DataFrame(df[["date","Temp_C"]])
data.set_index('date', inplace=True)

train, test = train_test_split(data, test_size=0.2, shuffle=False, random_state=6313)
sarima_pred_test = pd.DataFrame(sarima_pred_test)
sarima_pred_test.index = test.index

plt.figure(figsize=(8, 7))
plt.plot(train, label="test", lw=1.5)
plt.plot(test, label="test", color = 'green', lw=1.5)
plt.plot(sarima_pred_test, label="forecast", color = 'orange', lw=1.5)
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.xticks(rotation=45)
plt.title('Final SARIMA model ')
plt.tight_layout()
plt.show()

error_f = []
for i in range(len(test)):
    err = test.iloc[i, 0] - sarima_pred_test.iloc[i, 0]
    error_f.append(err)
error_f = pd.DataFrame(error_f)


error_f, error_mean_f, error_var_f, error_mse_f = cal_err(sarima_pred_test, test)

cov_matrix = model_fit_1.cov_params()

print(f"The roots of AR coe are: {np.around(np.roots(np.r_[1, -0.97, -0.07, 0.93]), decimals=3)}")
print(f"The roots of MA coe are: {np.around(np.roots(np.r_[1, 0.21, 0.11, 0.06,-0.04, -0.92]), decimals=3)}")
