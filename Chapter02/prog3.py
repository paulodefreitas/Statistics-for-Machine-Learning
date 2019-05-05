import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    wine_quality = pd.read_csv("winequality-red.csv", sep=';')
    wine_quality.rename(columns=lambda x: x.replace(" ", "_"), inplace=True)
    # Plots - pair plots
    eda_colnms = ['volatile_acidity',  'chlorides',
                  'sulphates', 'alcohol', 'quality']
    sns.set(style='whitegrid', context='notebook')
    sns.pairplot(wine_quality[eda_colnms], size=2.5,
                 x_vars=eda_colnms, y_vars=eda_colnms)
    plt.show()
