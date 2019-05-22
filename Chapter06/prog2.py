import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def funcGridSearchRBF(x_train, x_test, y_train, y_test):
    grid_search_rbf = GridSearchCV(
        pipeline, parameters, n_jobs=-1, cv=5, verbose=1, scoring='accuracy')
    grid_search_rbf.fit(x_train, y_train)

    print('RBF Kernel Grid Search Best Training score: %0.3f' %
          grid_search_rbf.best_score_)
    print('RBF Kernel Grid Search Best parameters set:')
    best_parameters = grid_search_rbf.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print('\t%s: %r' % (param_name, best_parameters[param_name]))

    predictions = grid_search_rbf.predict(x_test)

    print("\nRBF Kernel Grid Search - Testing accuracy:",
          round(accuracy_score(y_test, predictions), 4))
    print("\nRBF Kernel Grid Search - Test Classification Report\n",
          classification_report(y_test, predictions))
    print("\n\nRBF Kernel Grid Search- Test Confusion Matrix\n\n",
          pd.crosstab(y_test, predictions, rownames=["Actuall"], colnames=["Predicted"]))


if __name__ == "__main__":
    letterdata = pd.read_csv("letterdata.csv")
    # print(letterdata.head())
    x_vars = letterdata.drop(['letter'], axis=1)
    # print(x_vars)
    y_var = letterdata["letter"]
    # print(y_var)
    y_var = y_var.replace({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 'J': 10,
                           'K': 11, 'L': 12, 'M': 13, 'N': 14, 'O': 15, 'P': 16, 'Q': 17, 'R': 18, 'S': 19, 'T': 20,
                           'U': 21, 'V': 22, 'W': 23, 'X': 24, 'Y': 25, 'Z': 26})
    # print(y_var)

    x_train, x_test, y_train, y_test = train_test_split(
        x_vars, y_var, train_size=0.7, random_state=42)

    # print(x_train)
    # print(x_test)
    # print(y_train)
    # print(y_test)

    pipeline = Pipeline([('clf', SVC(kernel='rbf', C=1, gamma=0.1))])

    parameters = {'clf__C': (0.1, 0.3, 1, 3, 10, 30),
                  'clf__gamma': (0.001, 0.01, 0.1, 0.3, 1)}

    funcGridSearchRBF(x_train, x_test, y_train, y_test)
