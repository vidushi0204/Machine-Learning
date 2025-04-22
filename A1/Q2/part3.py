import numpy as np
from sampling_sgd import generate, StochasticLinearRegressor

X = np.loadtxt("X_train.csv", delimiter=",")  
y = np.loadtxt("y_train.csv", delimiter=",")

n_samples, n_features = X.shape
X = np.c_[(np.ones(n_samples), X)]

theta_closed = np.dot(np.linalg.inv(np.dot(X.T, X)), (np.dot(X.T,y)))

print(theta_closed)