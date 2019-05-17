import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def func3DimensionsPlot(three_d_data_df):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(three_d_data_df['x_axis'],
               three_d_data_df["y_axis"], three_d_data_df["z_axis"])
    ax.set_xlabel('x_axis')
    ax.set_ylabel('y_axis')
    ax.set_zlabel('z_axis')
    plt.show()


if __name__ == "__main__":
    # 3- Dimensions Plot
    three_d_data = np.random.rand(60, 3)
    three_d_data_df = pd.DataFrame(three_d_data)
    three_d_data_df.columns = ["x_axis", "y_axis", "z_axis"]

    func3DimensionsPlot(three_d_data_df)
