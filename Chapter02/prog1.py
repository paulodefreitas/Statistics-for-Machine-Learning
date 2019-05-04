import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.model_selection import train_test_split    
#from sklearn.metrics import r2_score

if __name__ == "__main__":
    wine_quality = pd.read_csv("winequality-red.csv",sep=';')  
    # Step for converting white space in columns to _ value for better handling 
    wine_quality.rename(columns=lambda x: x.replace(" ", "_"), inplace=True)

    # Simple Linear Regression - chart
    model = sm.OLS(wine_quality['quality'],sm.add_constant(wine_quality['alcohol'])).fit()
    
    print (model.summary())

    plt.scatter(wine_quality['alcohol'],wine_quality['quality'],label = 'Actual Data')
    plt.plot(wine_quality['alcohol'],model.params[0]+model.params[1]*wine_quality['alcohol'],
            c ='r',label="Regression fit")
    plt.title('Wine Quality regressed on Alchohol')
    plt.xlabel('Alcohol')
    plt.ylabel('Quality')
    plt.show()