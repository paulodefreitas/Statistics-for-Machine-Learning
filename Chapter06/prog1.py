import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
# Linear Classifier
from sklearn.svm import SVC


def funcSVMLinearClassifier(x_train, x_test, y_train, y_test):
    svm_fit = SVC(kernel='linear', C=1.0, random_state=43)
    svm_fit.fit(x_train, y_train)

    print("\nSVM Linear Classifier - Train Confusion Matrix\n\n", pd.crosstab(y_train,
                                                                              svm_fit.predict(x_train), rownames=["Actuall"], colnames=["Predicted"]))
    print("\nSVM Linear Classifier - Train accuracy:",
          round(accuracy_score(y_train, svm_fit.predict(x_train)), 3))
    print("\nSVM Linear Classifier - Train Classification Report\n",
          classification_report(y_train, svm_fit.predict(x_train)))

    print("\n\nSVM Linear Classifier - Test Confusion Matrix\n\n", pd.crosstab(y_test,
                                                                               svm_fit.predict(x_test), rownames=["Actuall"], colnames=["Predicted"]))
    print("\nSVM Linear Classifier - Test accuracy:",
          round(accuracy_score(y_test, svm_fit.predict(x_test)), 3))
    print("\nSVM Linear Classifier - Test Classification Report\n",
          classification_report(y_test, svm_fit.predict(x_test)))


def funcSVMPolynomialKernelClassifier(x_train, x_test, y_train, y_test):
    # Polynomial Kernel
    svm_poly_fit = SVC(kernel='poly', C=1.0, degree=2)
    svm_poly_fit.fit(x_train, y_train)

    print("\nSVM Polynomial Kernel Classifier - Train Confusion Matrix\n\n", pd.crosstab(y_train,
                                                                                         svm_poly_fit.predict(x_train), rownames=["Actuall"], colnames=["Predicted"]))
    print("\nSVM Polynomial Kernel Classifier - Train accuracy:",
          round(accuracy_score(y_train, svm_poly_fit.predict(x_train)), 3))
    print("\nSVM Polynomial Kernel Classifier - Train Classification Report\n",
          classification_report(y_train, svm_poly_fit.predict(x_train)))

    print("\n\nSVM Polynomial Kernel Classifier - Test Confusion Matrix\n\n", pd.crosstab(y_test,
                                                                                          svm_poly_fit.predict(x_test), rownames=["Actuall"], colnames=["Predicted"]))
    print("\nSVM Polynomial Kernel Classifier - Test accuracy:",
          round(accuracy_score(y_test, svm_poly_fit.predict(x_test)), 3))
    print("\nSVM Polynomial Kernel Classifier - Test Classification Report\n",
          classification_report(y_test, svm_poly_fit.predict(x_test)))


def funcSVMRBFKernelClassifier(x_train, x_test, y_train, y_test):
    # RBF Kernel
    svm_rbf_fit = SVC(kernel='rbf', C=1.0, gamma=0.1)
    svm_rbf_fit.fit(x_train, y_train)

    print("\nSVM RBF Kernel Classifier - Train Confusion Matrix\n\n", pd.crosstab(y_train,
                                                                                  svm_rbf_fit.predict(x_train), rownames=["Actuall"], colnames=["Predicted"]))
    print("\nSVM RBF Kernel Classifier - Train accuracy:",
          round(accuracy_score(y_train, svm_rbf_fit.predict(x_train)), 3))
    print("\nSVM RBF Kernel Classifier - Train Classification Report\n",
          classification_report(y_train, svm_rbf_fit.predict(x_train)))

    print("\n\nSVM RBF Kernel Classifier - Test Confusion Matrix\n\n", pd.crosstab(y_test,
                                                                                   svm_rbf_fit.predict(x_test), rownames=["Actuall"], colnames=["Predicted"]))
    print("\nSVM RBF Kernel Classifier - Test accuracy:",
          round(accuracy_score(y_test, svm_rbf_fit.predict(x_test)), 3))
    print("\nSVM RBF Kernel Classifier - Test Classification Report\n",
          classification_report(y_test, svm_rbf_fit.predict(x_test)))


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

    funcSVMLinearClassifier(x_train, x_test, y_train, y_test)
    funcSVMRBFKernelClassifier(x_train, x_test, y_train, y_test)
    funcSVMRBFKernelClassifier(x_train, x_test, y_train, y_test)
