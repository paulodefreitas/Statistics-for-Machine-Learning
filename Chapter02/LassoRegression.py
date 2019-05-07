# Lasso Regression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso


def funcDisplayBestParameters(alphas, x_train, y_train, x_test, y_test):
    initrsq = 0
    print("\nLasso Regression: Best Parameters\n")

    for alph in alphas:
        lasso_reg = Lasso(alpha=alph)
        lasso_reg.fit(x_train, y_train)
        tr_rsqrd = lasso_reg.score(x_train, y_train)
        ts_rsqrd = lasso_reg.score(x_test, y_test)

        if ts_rsqrd > initrsq:
            print("Lambda: ", alph, "Train R-Squared value:",
                  round(tr_rsqrd, 5), "Test R-squared value:", round(ts_rsqrd, 5))
            initrsq = ts_rsqrd


if __name__ == "__main__":
    wine_quality = pd.read_csv("winequality-red.csv", sep=';')
    wine_quality.rename(columns=lambda x: x.replace(" ", "_"), inplace=True)
    all_colnms = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
                  'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
                  'pH', 'sulphates', 'alcohol']
    pdx = wine_quality[all_colnms]
    pdy = wine_quality["quality"]
    x_train, x_test, y_train, y_test = train_test_split(
        pdx, pdy, train_size=0.7, random_state=42)

    alphas = [1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 5.0, 10.0]

    funcDisplayBestParameters(alphas, x_train, y_train, x_test, y_test)

    # Coeffients of Lasso regression of best alpha value
    lasso_reg = Lasso(alpha=0.001)
    lasso_reg.fit(x_train, y_train)

    print("\nLasso Regression coefficient values of Alpha = 0.001\n")
    for i in range(11):
        print(all_colnms[i], ": ", lasso_reg.coef_[i])
