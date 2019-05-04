# Grid search on Decision Trees
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

if __name__ == "__main__":
    input_data = pd.read_csv("ad.csv", header=None)

    X_columns = set(input_data.columns.values)
    y = input_data[len(input_data.columns.values)-1]
    X_columns.remove(len(input_data.columns.values)-1)
    X = input_data[list(X_columns)]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, random_state=33)

    pipeline = Pipeline([('clf', DecisionTreeClassifier(criterion='entropy'))])
    parameters = {'clf__max_depth': (50, 100, 150), 'clf__min_samples_split': (
        2, 3), 'clf__min_samples_leaf': (1, 2, 3)}

    grid_search = GridSearchCV(
        pipeline, parameters, n_jobs=-1, verbose=1, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    y_pred = grid_search.predict(X_test)

    print('\n Best score: \n', grid_search.best_score_)
    print('\n Best parameters set: \n')
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print('\t%s: %r' % (param_name, best_parameters[param_name]))
    print("\n Confusion Matrix on Test data \n",
          confusion_matrix(y_test, y_pred))
    print("\n Test Accuracy \n", accuracy_score(y_test, y_pred))
    print("\nPrecision Recall f1 table \n",
          classification_report(y_test, y_pred))
