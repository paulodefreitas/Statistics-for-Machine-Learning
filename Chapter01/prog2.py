import numpy as np
from statistics import variance, stdev


def funcCalculateVariance(game_points):
    dt_var = variance(game_points)
    print("Sample variance:", round(dt_var, 2))


def funcCalculateStandardDeviation(game_points):
    dt_std = stdev(game_points)
    print("Sample std.dev:", round(dt_std, 2))


def funcCalculateRange(game_points):
    dt_rng = np.max(game_points, axis=0) - np.min(game_points, axis=0)
    print("Range:", dt_rng)


def funcCalculatePercentiles(game_points):
    print("Quantiles:")
    for val in [20, 80, 100]:
        dt_qntls = np.percentile(game_points, val)
        print(str(val)+"%", dt_qntls)


def funcCalculateIQR(game_points):
    q75, q25 = np.percentile(game_points, [75, 25])
    print("Inter quartile range:", q75-q25)


if __name__ == "__main__":
    game_points = np.array(
        [35, 56, 43, 59, 63, 79, 35, 41, 64, 43, 93, 60, 77, 24, 82])
    funcCalculateVariance(game_points)
    funcCalculateStandardDeviation(game_points)
    funcCalculateRange(game_points)
    funcCalculatePercentiles(game_points)
    funcCalculateIQR(game_points)
