# Simple Linear Regression - Model fit
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def mean(values):
    return round(sum(values)/float(len(values)), 2)


def variance(values):
    return round(sum((values - mean(values))**2), 2)


def funcCovariance(Xtrain, Ytrain, meanX, meanY):
    return round(sum((Xtrain - meanX) * (Ytrain - meanY)), 2)


if __name__ == "__main__":
    wine_quality = pd.read_csv("winequality-red.csv", sep=';')

    wine_quality.rename(columns=lambda x: x.replace(" ", "_"), inplace=True)

    x_train, x_test, y_train, y_test = train_test_split(
        wine_quality['alcohol'], wine_quality["quality"], train_size=0.7, random_state=42)

    x_train = pd.DataFrame(x_train)
    x_test = pd.DataFrame(x_test)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)

    print("Alcohol mean", mean(x_train['alcohol']))
    print("Quality mean", mean(y_train['quality']))

    print("Alcohol variance: ", variance(x_train['alcohol']))
    print("Quality variance: ", variance(y_train['quality']))

    print("Covariance: ", funcCovariance(x_train['alcohol'], y_train['quality'], mean(
        x_train['alcohol']), mean(y_train['quality'])))

    b1 = funcCovariance(x_train['alcohol'], y_train['quality'], mean(
        x_train['alcohol']), mean(y_train['quality']))/variance(x_train['alcohol'])
    b0 = mean(y_train['quality']) - b1*mean(x_train['alcohol'])

    print("\n\nIntercept (B0):", round(b0, 4),
          "Co-efficient (B1):", round(b1, 4))

    y_test["y_pred"] = pd.DataFrame(b0+b1*x_test['alcohol'])
    R_sqrd = 1 - (sum((y_test['quality']-y_test['y_pred'])**2) /
                  sum((y_test['quality'] - mean(y_test['quality']))**2))
    print("Test R-squared value:", round(R_sqrd, 4))
