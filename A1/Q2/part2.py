import numpy as np
from sampling_sgd import StochasticLinearRegressor

X_train = np.loadtxt("X_train.csv", delimiter=",")  
y_train = np.loadtxt("y_train.csv", delimiter=",")
batch_sizes = [1, 80, 8000, 800000]
lr=0.001
n_epochs=[100,100,200,2000]
eps=[1e-3,1e-4,1e-5,1e-5]


for i in range(4):
    print(f"Batch size: {batch_sizes[i]}")
    model = StochasticLinearRegressor()
    _ = model.fit(X_train, y_train, learning_rate=lr, batch_size=batch_sizes[i], n_epochs=n_epochs[i],epsilon=eps[i])
    print(f"Theta: {model.theta}\n")
