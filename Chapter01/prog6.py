#ANOVA
import pandas as pd
from scipy import stats
'''
Parameters:	
    sample1, sample2, … : array_like
    The sample measurements for each group.
'''
def funcANOVA(sample1, sample2, sample3):
    one_way_anova = stats.f_oneway(sample1, sample2, sample3)
    print("F-value is: ", one_way_anova[0])
    print("p-value is: ", one_way_anova[1])

if __name__ == "__main__":
    fileName = "fetilizers.csv"
    fetilizers = pd.read_csv("fetilizers.csv")
    '''
    for i in fetilizers.columns:
        print("Columns[",i,"]")
    '''
    #one_way_anova = stats.f_oneway(fetilizers["fertilizer1"], fetilizers["fertilizer2"], fetilizers["fertilizer3"])
    #print ("Statistic :", round(one_way_anova[0],2),", p-value :",round(one_way_anova[1],3))
    funcANOVA(fetilizers["fertilizer1"], fetilizers["fertilizer2"], fetilizers["fertilizer3"])