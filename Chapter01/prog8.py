import numpy as np
import pandas as pd

'''
Gradient descent

code:

https://github.com/PacktPublishing/Statistics-for-Machine-Learning/tree/master/Chapter01
'''

def gradient_descent(x, y, learn_rate, conv_threshold, batch_size, max_iter):
    converged = False
    iter = 0
    m = batch_size

    t0 = np.random.random(x.shape[1])
    t1 = np.random.random(x.shape[1])

    MSE = (sum([(t0 + t1*x[i] - y[i])**2 for i in range(m)]) / m)

    while not converged:
        grad0 = 1.0/m * sum([(t0 + t1*x[i] - y[i]) for i in range(m)])
        grad1 = 1.0/m * sum([(t0 + t1*x[i] - y[i])*x[i] for i in range(m)])

        temp0 = t0 - learn_rate * grad0
        temp1 = t1 - learn_rate * grad1

        t0 = temp0
        t1 = temp1

        MSE_New = (sum([(t0 + t1*x[i] - y[i])**2 for i in range(m)]) / m)

        if abs(MSE - MSE_New) <= conv_threshold:
            print('Converged, iterations: ', iter)
            converged = True

        MSE = MSE_New
        iter += 1

        if iter == max_iter:
            print('Max interactions reached')
            converged = True

    return t0, t1


if __name__ == "__main__":
    train_data = pd.read_csv("mtcars.csv")
    X = np.array(train_data["hp"])
    y = np.array(train_data["mpg"])
    X = X.reshape(32, 1)
    y = y.reshape(32, 1)
    Inter, Coeff = gradient_descent(
        x=X, y=y, learn_rate=0.00003, conv_threshold=1e-8, batch_size=32, max_iter=1500)
    print("Gradient Descent Results")
    print(('Intercept = %s Coefficient = %s') % (Inter, Coeff))
