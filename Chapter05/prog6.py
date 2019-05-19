import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

    # Tuning of K- value for Train & Test data
    dummyarray = np.empty((5, 3))
    # print(dummyarray)
    k_valchart = pd.DataFrame(dummyarray)
    # print(k_valchart)
    k_valchart.columns = ["K_value", "Train_acc", "Test_acc"]
    # print(k_valchart.columns)
    k_vals = [1, 2, 3, 4, 5]
    for i in range(len(k_vals)):
        knn_fit = KNeighborsClassifier(
            n_neighbors=k_vals[i], p=2, metric='minkowski')
        knn_fit.fit(x_train, y_train)

        print("\nK-value", k_vals[i])

        tr_accscore = round(accuracy_score(
            y_train, knn_fit.predict(x_train)), 3)
        print("\nK-Nearest Neighbors - Train Confusion Matrix\n\n", pd.crosstab(y_train,
                                                                                knn_fit.predict(x_train), rownames=["Actuall"], colnames=["Predicted"]))
        print("\nK-Nearest Neighbors - Train accuracy:", tr_accscore)
        print("\nK-Nearest Neighbors - Train Classification Report\n",
              classification_report(y_train, knn_fit.predict(x_train)))

        ts_accscore = round(accuracy_score(y_test, knn_fit.predict(x_test)), 3)
        print("\n\nK-Nearest Neighbors - Test Confusion Matrix\n\n", pd.crosstab(y_test,
                                                                                 knn_fit.predict(x_test), rownames=["Actuall"], colnames=["Predicted"]))
        print("\nK-Nearest Neighbors - Test accuracy:", ts_accscore)
        print("\nK-Nearest Neighbors - Test Classification Report\n",
              classification_report(y_test, knn_fit.predict(x_test)))

        k_valchart.loc[i, 'K_value'] = k_vals[i]
        k_valchart.loc[i, 'Train_acc'] = tr_accscore
        k_valchart.loc[i, 'Test_acc'] = ts_accscore

    plt.figure()
    #plt.title('KNN Train & Test Accuracy change with K-value')

    plt.xlabel('K-value')
    plt.ylabel('Accuracy')
    plt.plot(k_valchart["K_value"], k_valchart["Train_acc"])
    plt.plot(k_valchart["K_value"], k_valchart["Test_acc"])

    plt.axis([0.9, 5, 0.92, 1.005])
    plt.xticks([1, 2, 3, 4, 5])

    for a, b in zip(k_valchart["K_value"], k_valchart["Train_acc"]):
        plt.text(a, b, str(b), fontsize=10)

    for a, b in zip(k_valchart["K_value"], k_valchart["Test_acc"]):
        plt.text(a, b, str(b), fontsize=10)

    plt.legend(loc='upper right')

    plt.show()
