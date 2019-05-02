# Train & Test split
# Linear Regressio
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    fileName = "mtcars.csv"
    original_data = pd.read_csv(fileName)
    print("Orignal data: \n", original_data)
    train_data, test_data = train_test_split(original_data, train_size=0.7, random_state=42)
    print("Train data: \n", train_data)
    print("Test data: \n", test_data)
    X = np.array(train_data["hp"])
    y = np.array(train_data["mpg"]) 
    X = X.reshape(22,1)
    y = y.reshape(22,1)
    #Linear Regression
    model = LinearRegression(fit_intercept = True)
    model.fit(X,y)
    print ("Linear Regression Results")
    print ("Intercept: ",model.intercept_[0])
    print("Coefficient: ",model.coef_[0])
    print("Predict using the linear model: \n", model.predict(X))
    print("Coefficient of determination R^2: ", model.score(X, y))