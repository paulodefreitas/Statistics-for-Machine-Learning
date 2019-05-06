import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    wine_quality = pd.read_csv("winequality-red.csv", sep=';')
    wine_quality.rename(columns=lambda x: x.replace(" ", "_"), inplace=True)
    # Plots - pair plots
    eda_colnms = ['volatile_acidity',  'chlorides',
                  'sulphates', 'alcohol', 'quality']
    # Correlation coefficients
    corr_mat = np.corrcoef(wine_quality[eda_colnms].values.T)
    sns.set(font_scale=1)
    full_mat = sns.heatmap(corr_mat, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={
                           'size': 15}, yticklabels=eda_colnms, xticklabels=eda_colnms)

    plt.show()
