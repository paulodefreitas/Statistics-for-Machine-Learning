import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

if __name__ == "__main__":
    hrattr_data = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
    # iterating the columns
    '''
    for col in hrattr_data.columns:
        print(col)
    '''
    # print (hrattr_data.head())

    hrattr_data['Attrition_ind'] = 0
    hrattr_data.loc[hrattr_data['Attrition'] == 'Yes', 'Attrition_ind'] = 1

    dummy_busnstrvl = pd.get_dummies(
        hrattr_data['BusinessTravel'], prefix='busns_trvl')
    # print("dummy_busnstrvl: ", dummy_busnstrvl)
    dummy_dept = pd.get_dummies(hrattr_data['Department'], prefix='dept')
    # print("dummy_dept: ", dummy_dept)
    dummy_edufield = pd.get_dummies(
        hrattr_data['EducationField'], prefix='edufield')
    # print("dummy_edufield: ", dummy_edufield)
    dummy_gender = pd.get_dummies(hrattr_data['Gender'], prefix='gend')
    # print("dummy_gender: ", dummy_gender)
    dummy_jobrole = pd.get_dummies(hrattr_data['JobRole'], prefix='jobrole')
    # print("dummy_jobrole: ", dummy_jobrole)
    dummy_maritstat = pd.get_dummies(
        hrattr_data['MaritalStatus'], prefix='maritalstat')
    # print("dummy_maritstat: ", dummy_maritstat)
    dummy_overtime = pd.get_dummies(hrattr_data['OverTime'], prefix='overtime')
    # print("dummy_overtime: ", dummy_overtime)

    continuous_columns = ['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
                          'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
                          'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
                          'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
                          'YearsWithCurrManager']
    # print(continuous_columns)

    hrattr_continuous = hrattr_data[continuous_columns]
    # print(hrattr_continuous)

    hrattr_continuous['Age'].describe()
    # print("Describe age\n" ,hrattr_continuous['Age'].describe())

    hrattr_data['BusinessTravel'].value_counts()
    # print("BusinessTravel value_counts\n", hrattr_data['BusinessTravel'].value_counts())

    hrattr_data_new = pd.concat([dummy_busnstrvl, dummy_dept, dummy_edufield, dummy_gender, dummy_jobrole,
                                 dummy_maritstat, dummy_overtime, hrattr_continuous, hrattr_data['Attrition_ind']], axis=1)
    # print("hrattr_data_new: \n", hrattr_data_new)

    # Train & Test split
    x_train, x_test, y_train, y_test = train_test_split(hrattr_data_new.drop(['Attrition_ind'], axis=1),
                                                        hrattr_data_new['Attrition_ind'], train_size=0.7, random_state=42)

    '''
    print("x train: ", x_train)
    print("y train: ", y_train)
    print("x test: ", x_test)
    print("y test: ", y_test)
    '''

    # Tuning class weights to analyze accuracy, precision & recall
    dummyarray = np.empty((6, 10))
    dt_wttune = pd.DataFrame(dummyarray)

    dt_wttune.columns = ["zero_wght", "one_wght", "tr_accuracy", "tst_accuracy", "prec_zero", "prec_one",
                         "prec_ovll", "recl_zero", "recl_one", "recl_ovll"]

    zero_clwghts = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]

    for i in range(len(zero_clwghts)):
        clwght = {0: zero_clwghts[i], 1: 1.0-zero_clwghts[i]}
        dt_fit = DecisionTreeClassifier(criterion="gini", max_depth=5, min_samples_split=2,
                                        min_samples_leaf=1, random_state=42, class_weight=clwght)
        dt_fit.fit(x_train, y_train)
        dt_wttune.loc[i, 'zero_wght'] = clwght[0]
        dt_wttune.loc[i, 'one_wght'] = clwght[1]
        dt_wttune.loc[i, 'tr_accuracy'] = round(
            accuracy_score(y_train, dt_fit.predict(x_train)), 3)
        dt_wttune.loc[i, 'tst_accuracy'] = round(
            accuracy_score(y_test, dt_fit.predict(x_test)), 3)

        clf_sp = classification_report(y_test, dt_fit.predict(x_test)).split()
        dt_wttune.loc[i, 'prec_zero'] = float(clf_sp[5])
        dt_wttune.loc[i, 'prec_one'] = float(clf_sp[10])
        dt_wttune.loc[i, 'prec_ovll'] = float(clf_sp[17])

        dt_wttune.loc[i, 'recl_zero'] = float(clf_sp[6])
        dt_wttune.loc[i, 'recl_one'] = float(clf_sp[11])
        dt_wttune.loc[i, 'recl_ovll'] = float(clf_sp[18])
        print("\nClass Weights", clwght, "Train accuracy:", round(accuracy_score(y_train, dt_fit.predict(
            x_train)), 3), "Test accuracy:", round(accuracy_score(y_test, dt_fit.predict(x_test)), 3))
        print("Test Confusion Matrix\n\n", pd.crosstab(y_test, dt_fit.predict(
            x_test), rownames=["Actuall"], colnames=["Predicted"]))
