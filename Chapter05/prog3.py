import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def func2DimensionsPlot(two_d_data_df):
    plt.figure()
    plt.scatter(two_d_data_df['x_axis'], two_d_data_df["y_axis"])
    plt.xlabel("x_axis")
    plt.ylabel("y_axis")
    plt.show()


if __name__ == "__main__":
    # 2- Dimensions Plot
    two_d_data = np.random.rand(60, 2)
    two_d_data_df = pd.DataFrame(two_d_data)
    two_d_data_df.columns = ["x_axis", "y_axis"]

    func2DimensionsPlot(two_d_data_df)
