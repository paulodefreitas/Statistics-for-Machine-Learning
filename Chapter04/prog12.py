import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
# Classifier 1
from sklearn.linear_model import LogisticRegression
# Classifier 2
from sklearn.tree import DecisionTreeClassifier
# Classifier 3
from sklearn.ensemble import RandomForestClassifier
# Classifier 4
from sklearn.ensemble import AdaBoostClassifier

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

    clf2_dt_fit = DecisionTreeClassifier(criterion="gini", max_depth=5, min_samples_split=2,
                                         min_samples_leaf=1, random_state=42, class_weight=clwght)
    clf2_dt_fit.fit(x_train, y_train)

    clf3_rf_fit = RandomForestClassifier(n_estimators=10000, criterion="gini", max_depth=6,
                                         min_samples_split=2, min_samples_leaf=1, class_weight=clwght)
    clf3_rf_fit.fit(x_train, y_train)

    clf4_dtree = DecisionTreeClassifier(
        criterion='gini', max_depth=1, class_weight=clwght)
    clf4_adabst_fit = AdaBoostClassifier(base_estimator=clf4_dtree,
                                         n_estimators=5000, learning_rate=0.05, random_state=42)

    clf4_adabst_fit.fit(x_train, y_train)

    ensemble = pd.DataFrame()

    ensemble["log_output_one"] = pd.DataFrame(
        clf1_logreg_fit.predict_proba(x_train))[1]
    ensemble["dtr_output_one"] = pd.DataFrame(
        clf2_dt_fit.predict_proba(x_train))[1]
    ensemble["rf_output_one"] = pd.DataFrame(
        clf3_rf_fit.predict_proba(x_train))[1]
    ensemble["adb_output_one"] = pd.DataFrame(
        clf4_adabst_fit.predict_proba(x_train))[1]

    ensemble = pd.concat([ensemble, pd.DataFrame(
        y_train).reset_index(drop=True)], axis=1)

    # Fitting meta-classifier
    meta_logit_fit = LogisticRegression(fit_intercept=False)
    meta_logit_fit.fit(ensemble[['log_output_one', 'dtr_output_one',
                                 'rf_output_one', 'adb_output_one']], ensemble['Attrition_ind'])

    coefs = meta_logit_fit.coef_
    print("Co-efficients for LR, DT, RF & AB are:", coefs)

    ensemble_test = pd.DataFrame()
    ensemble_test["log_output_one"] = pd.DataFrame(
        clf1_logreg_fit.predict_proba(x_test))[1]
    ensemble_test["dtr_output_one"] = pd.DataFrame(
        clf2_dt_fit.predict_proba(x_test))[1]
    ensemble_test["rf_output_one"] = pd.DataFrame(
        clf3_rf_fit.predict_proba(x_test))[1]
    ensemble_test["adb_output_one"] = pd.DataFrame(
        clf4_adabst_fit.predict_proba(x_test))[1]

    ensemble_test["all_one"] = meta_logit_fit.predict(
        ensemble_test[['log_output_one', 'dtr_output_one', 'rf_output_one', 'adb_output_one']])

    ensemble_test = pd.concat(
        [ensemble_test, pd.DataFrame(y_test).reset_index(drop=True)], axis=1)

    print("\n\nEnsemble of Models - Test Confusion Matrix\n\n", pd.crosstab(
        ensemble_test['Attrition_ind'], ensemble_test['all_one'], rownames=["Actuall"], colnames=["Predicted"]))
    print("\nEnsemble of Models - Test accuracy",
          round(accuracy_score(ensemble_test['Attrition_ind'], ensemble_test['all_one']), 3))
    print("\nEnsemble of Models - Test Classification Report\n",
          classification_report(ensemble_test['Attrition_ind'], ensemble_test['all_one']))

    