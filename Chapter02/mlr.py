import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import statsmodels.api as sm

if __name__ == "__main__":
    wine_quality = pd.read_csv("winequality-red.csv", sep=';')
    # Step for converting white space in columns to _ value for better handling
    wine_quality.rename(columns=lambda x: x.replace(" ", "_"), inplace=True)
    # Multi linear regression model
    colnms = ['volatile_acidity',  'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide',
              'pH', 'sulphates', 'alcohol']

    pdx = wine_quality[colnms]
    pdy = wine_quality["quality"]

    x_train, x_test, y_train, y_test = train_test_split(
        pdx, pdy, train_size=0.7, random_state=42)
    x_train_new = sm.add_constant(x_train)
    x_test_new = sm.add_constant(x_test)

    # random.seed(434)
    full_mod = sm.OLS(y_train, x_train_new)
    full_res = full_mod.fit()
    print("\n \n", full_res.summary())

    print("\nVariance Inflation Factor")
    cnames = x_train.columns
    for i in np.arange(0, len(cnames)):
        xvars = list(cnames)
        yvar = xvars.pop(i)
        mod = sm.OLS(x_train[yvar], sm.add_constant(x_train_new[xvars]))
        res = mod.fit()
        vif = 1/(1-res.rsquared)
        print(yvar, round(vif, 3))

    # Predition of data
    y_pred = full_res.predict(x_test_new)
    y_pred_df = pd.DataFrame(y_pred)
    y_pred_df.columns = ['y_pred']
    pred_data = pd.DataFrame(y_pred_df['y_pred'])
    y_test_new = pd.DataFrame(y_test)
    # y_test_new.reset_index(inplace=True)

    pred_data['y_test'] = pd.DataFrame(y_test_new['quality'])

    # R-square calculation
    rsqd = r2_score(y_test_new['quality'].tolist(),
                    y_pred_df['y_pred'].tolist())
    print("\nTest R-squared value:", round(rsqd, 4))
