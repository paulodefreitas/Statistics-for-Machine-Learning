import numpy as np
from scipy import stats


def funcCalculateMean(data):
    # Calculate Mean
    dt_mean = np.mean(data)
    print("Mean :", round(dt_mean, 2))


def funcCalculateMedian(data):
    # Calculate Median
    dt_median = np.median(data)
    print("Median :", dt_median)


def funcCalculateMode(data):
    # Calculate Mode
    dt_mode = stats.mode(data)
    print("Mode :", dt_mode[0][0])


if __name__ == "__main__":
    data = np.array([4, 5, 1, 2, 7, 2, 6, 9, 3])
    funcCalculateMean(data)
    funcCalculateMedian(data)
    funcCalculateMode(data)
