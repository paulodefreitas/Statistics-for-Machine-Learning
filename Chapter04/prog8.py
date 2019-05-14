import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
# Classifier 1
from sklearn.linear_model import LogisticRegression

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

    # Ensemble of Ensembles - by fitting various classifiers
    clwght = {0: 0.3, 1: 0.7}

    clf1_logreg_fit = LogisticRegression(
        fit_intercept=True, class_weight=clwght)
    clf1_logreg_fit.fit(x_train, y_train)

    print("\nLogistic Regression for Ensemble - Train Confusion Matrix\n\n", pd.crosstab(y_train,
                                                                                         clf1_logreg_fit.predict(x_train), rownames=["Actuall"], colnames=["Predicted"]))
    print("\nLogistic Regression for Ensemble - Train accuracy",
          round(accuracy_score(y_train, clf1_logreg_fit.predict(x_train)), 3))
    print("\nLogistic Regression for Ensemble - Train Classification Report\n",
          classification_report(y_train, clf1_logreg_fit.predict(x_train)))

    print("\n\nLogistic Regression for Ensemble - Test Confusion Matrix\n\n", pd.crosstab(y_test,
                                                                                          clf1_logreg_fit.predict(x_test), rownames=["Actuall"], colnames=["Predicted"]))
    print("\nLogistic Regression for Ensemble - Test accuracy",
          round(accuracy_score(y_test, clf1_logreg_fit.predict(x_test)), 3))
    print("\nLogistic Regression for Ensemble - Test Classification Report\n",
          classification_report(y_test, clf1_logreg_fit.predict(x_test)))
