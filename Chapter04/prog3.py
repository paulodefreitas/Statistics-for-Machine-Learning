import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
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

    dt_fit = DecisionTreeClassifier(criterion="gini", max_depth=5, min_samples_split=2, min_samples_leaf=1, random_state=42,
                                    class_weight={0: 0.3, 1: 0.7})

    bag_fit = BaggingClassifier(base_estimator=dt_fit, n_estimators=5000, max_samples=0.67, max_features=1.0,
                                bootstrap=True, bootstrap_features=True, n_jobs=-1, random_state=42)

    bag_fit.fit(x_train, y_train)

    print("\nBagging - Train Confusion Matrix\n\n", pd.crosstab(y_train,
                                                                bag_fit.predict(x_train), rownames=["Actuall"], colnames=["Predicted"]))
    print("\nBagging- Train accuracy",
          round(accuracy_score(y_train, bag_fit.predict(x_train)), 3))
    print("\nBagging  - Train Classification Report\n",
          classification_report(y_train, bag_fit.predict(x_train)))

    print("\n\nBagging - Test Confusion Matrix\n\n", pd.crosstab(y_test,
                                                                 bag_fit.predict(x_test), rownames=["Actuall"], colnames=["Predicted"]))
    print("\nBagging - Test accuracy",
          round(accuracy_score(y_test, bag_fit.predict(x_test)), 3))
    print("\nBagging - Test Classification Report\n",
          classification_report(y_test, bag_fit.predict(x_test)))
