# Normal Distribution
from scipy import stats


def funcCalculateZscore(xbar, mu0, s):
    z = (xbar-mu0)/s
    print("Calculate z-score : ", z)


def funcCalculateProbabilityUnderCurve(xbar, mu0, s):
    z = (xbar-mu0)/s
    p_val = 1 - stats.norm.cdf(z)
    print("Prob. to score more than 67 is ", round(p_val*100, 2), "%")


if __name__ == "__main__":
    xbar = 67
    mu0 = 52
    s = 16.3
    funcCalculateZscore(xbar, mu0, s)
    funcCalculateProbabilityUnderCurve(xbar, mu0, s)