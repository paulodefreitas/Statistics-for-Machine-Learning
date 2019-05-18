# KNN CLassifier - Breast Cancer
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

if __name__ == "__main__":
    breast_cancer = pd.read_csv("Breast_Cancer_Wisconsin.csv")
    #print (breast_cancer.head())
    '''
    for col in breast_cancer.columns:
        print(col)
    '''
    breast_cancer['bare_nucleoli'] = breast_cancer['bare_nucleoli'].replace(
        '?', np.NAN)
    # print(breast_cancer['bare_nucleoli'])
    breast_cancer['bare_nucleoli'] = breast_cancer['bare_nucleoli'].fillna(
        breast_cancer['bare_nucleoli'].value_counts().index[0])
    # print(breast_cancer['bare_nucleoli'])

    breast_cancer['Cancer_Ind'] = 0
    # print(breast_cancer['Cancer_Ind'])
    breast_cancer.loc[breast_cancer['class'] == 4, 'Cancer_Ind'] = 1
    # print(breast_cancer.loc[breast_cancer['class']==4,'Cancer_Ind'])

    x_vars = breast_cancer.drop(['id', 'class', 'Cancer_Ind'], axis=1)
    # print(x_vars)
    y_var = breast_cancer['Cancer_Ind']
    # print(y_var)

    x_vars_stdscle = StandardScaler().fit_transform(x_vars.values)
    # print(x_vars_stdscle)

    x_vars_stdscle_df = pd.DataFrame(
        x_vars_stdscle, index=x_vars.index, columns=x_vars.columns)
    # print(x_vars_stdscle_df)
    x_train, x_test, y_train, y_test = train_test_split(
        x_vars_stdscle_df, y_var, train_size=0.7, random_state=42)

    # print(x_train)
    # print(x_test)
    # print(y_train)
    # print(y_test)

    knn_fit = KNeighborsClassifier(n_neighbors=3, p=2, metric='minkowski')
    knn_fit.fit(x_train, y_train)

    print("\nK-Nearest Neighbors - Train Confusion Matrix\n\n", pd.crosstab(y_train,
                                                                            knn_fit.predict(x_train), rownames=["Actuall"], colnames=["Predicted"]))
    print("\nK-Nearest Neighbors - Train accuracy:",
          round(accuracy_score(y_train, knn_fit.predict(x_train)), 3))
    print("\nK-Nearest Neighbors - Train Classification Report\n",
          classification_report(y_train, knn_fit.predict(x_train)))

    print("\n\nK-Nearest Neighbors - Test Confusion Matrix\n\n", pd.crosstab(y_test,
                                                                             knn_fit.predict(x_test), rownames=["Actuall"], colnames=["Predicted"]))
    print("\nK-Nearest Neighbors - Test accuracy:",
          round(accuracy_score(y_test, knn_fit.predict(x_test)), 3))
    print("\nK-Nearest Neighbors - Test Classification Report\n",
          classification_report(y_test, knn_fit.predict(x_test)))
