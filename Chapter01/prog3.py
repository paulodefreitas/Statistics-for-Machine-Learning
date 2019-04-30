# Hypothesis testing
from scipy import stats
import numpy as np


def funcTestStatistic(xbar, mu0, s, n):
    t_smple = (xbar-mu0)/(s/np.sqrt(float(n)))
    print("Test Statistic:", round(t_smple, 2))


def funcCriticalValueTtable(alpha, n):
    t_alpha = stats.t.ppf(alpha, n-1)
    print("Critical value from t-table:", round(t_alpha, 3))


def funcLowerTailPValueTtable(xbar, mu0, s, n):
    t_smple = (xbar-mu0)/(s/np.sqrt(float(n)))
    p_val = stats.t.sf(np.abs(t_smple), n-1)
    print("Lower tail p-value from t-table", p_val)


if __name__ == "__main__":
    xbar = 990
    mu0 = 1000
    s = 12.5
    n = 30
    alpha = 0.05
    funcTestStatistic(xbar, mu0, s, n)
    funcCriticalValueTtable(alpha, n)
    funcLowerTailPValueTtable(xbar, mu0, s, n)

