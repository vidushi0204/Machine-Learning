# run.py
import numpy as np
from sampling_sgd import generate, StochasticLinearRegressor


N = 1_000_000
theta_true = np.array([3, 1, 2])
input_mean = np.array([3, -1])
input_sigma = np.array([2, 2]) 
noise_sigma = np.sqrt(2)     

X, y = generate(N, theta_true, input_mean, input_sigma, noise_sigma)

indices = np.arange(N)
np.random.shuffle(indices)
train_size = int(0.8 * N)
train_indices = indices[:train_size]
test_indices = indices[train_size:]

X_train, y_train = X[train_indices], y[train_indices]
X_test, y_test = X[test_indices], y[test_indices]

np.savetxt('X_train.csv', X_train, delimiter=',')
np.savetxt('y_train.csv', y_train, delimiter=',')
np.savetxt('X_test.csv', X_test, delimiter=',')
np.savetxt('y_test.csv', y_test, delimiter=',')
