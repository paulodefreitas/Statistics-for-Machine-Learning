# Chi-square independence test
import pandas as pd
from scipy import stats


def funcChiSquareContingency(data):
    contg = stats.chi2_contingency(observed=data)
    print("chi2: ", contg[0])
    print("P-value is:", contg[1])
    print("Degrees of freedom is: ", contg[2])
    print("Matrix:\n", data)
    # The expected frequencies, based on the marginal sums of the table.
    print("Expected frequency distribution: \n", contg[3])


if __name__ == "__main__":
    fileName = "survey.csv"
    survey = pd.read_csv(fileName)
    '''
    for i in survey.columns:
        print("Columns[",i,"]")
    '''
    # Tabulating 2 variables with row & column variables respectively
    survey_tab = pd.crosstab(survey.Smoke, survey.Exer, margins=True)
    # Creating observed table for analysis
    observed = survey_tab.ix[0:4, 0:3]
    funcChiSquareContingency(observed)
