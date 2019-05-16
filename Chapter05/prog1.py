import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# KNN Curse of Dimensionality
import random
import math


def random_point_gen(dimension):
    return [random.random() for _ in range(dimension)]


def distance(v, w):
    vec_sub = [v_i-w_i for v_i, w_i in zip(v, w)]
    sum_of_sqrs = sum(v_i*v_i for v_i in vec_sub)
    return math.sqrt(sum_of_sqrs)


def random_distances_comparison(dimension, number_pairs):
    return [distance(random_point_gen(dimension), random_point_gen(dimension))
            for _ in range(number_pairs)]


def mean(x):
    return sum(x) / len(x)


if __name__ == "__main__":
    dimensions = range(1, 201, 5)
    #print(dimensions)

    avg_distances = []
    min_distances = []

    dummyarray = np.empty((20, 4))
    #print(dummyarray)
    dist_vals = pd.DataFrame(dummyarray)
    #print(dist_vals)
    dist_vals.columns = ["Dimension", "Min_Distance",
                         "Avg_Distance", "Min/Avg_Distance"]
    #print(dist_vals.columns)

    random.seed(34)
    i = 0
    for dims in dimensions:
        distances = random_distances_comparison(dims, 1000)
        avg_distances.append(mean(distances))
        min_distances.append(min(distances))

        dist_vals.loc[i, "Dimension"] = dims
        dist_vals.loc[i, "Min_Distance"] = min(distances)
        dist_vals.loc[i, "Avg_Distance"] = mean(distances)
        dist_vals.loc[i, "Min/Avg_Distance"] = min(distances)/mean(distances)

        print(dims, min(distances), mean(distances),
              min(distances)*1.0 / mean(distances))
        i = i+1

    # Ploting Average distances for Various Dimensions
    plt.figure()
    #plt.title('Avg. Distance Change with Number of Dimensions for 1K Obs')
    plt.xlabel('Dimensions')
    plt.ylabel('Avg. Distance')
    plt.plot(dist_vals["Dimension"], dist_vals["Avg_Distance"])
    plt.legend(loc='best')
    plt.show()
