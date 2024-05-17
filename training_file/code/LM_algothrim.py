import copy
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.model_selection import train_test_split


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

# df = pd.read_csv("clean_hourly_df.csv")
# data = df["Temp_C"]

data = pd.read_csv("y_train.csv")
data = data['Energy delta[Wh]']
train, test = train_test_split(data, test_size=0.2, shuffle=False, random_state=6313)


LM_algorithm(train, 24, 24)
