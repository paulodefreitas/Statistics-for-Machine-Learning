# 1-Dimension Plot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def func1DimensionPlot(one_d_data_df):
    plt.figure()
    plt.scatter(one_d_data_df['1D_Data'], one_d_data_df["height"])
    plt.yticks([])
    plt.xlabel("1-D points")
    plt.show()


if __name__ == "__main__":
    # 1-Dimension Plot
    one_d_data = np.random.rand(60, 1)
    one_d_data_df = pd.DataFrame(one_d_data)
    one_d_data_df.columns = ["1D_Data"]
    one_d_data_df["height"] = 1

    func1DimensionPlot(one_d_data_df)
