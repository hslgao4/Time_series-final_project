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
from prettytable import PrettyTable




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