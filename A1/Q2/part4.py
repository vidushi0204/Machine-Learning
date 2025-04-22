import numpy as np
from sampling_sgd import generate, StochasticLinearRegressor

X_train = np.loadtxt("X_train.csv", delimiter=",")  
y_train = np.loadtxt("y_train.csv", delimiter=",")
X_test = np.loadtxt("X_test.csv", delimiter=",")  
y_test = np.loadtxt("y_test.csv", delimiter=",")

n_train = X_train.shape[0]
n_test = X_test.shape[0]
X_cl_train = np.column_stack((np.ones(n_train), X_train))
X_cl_test = np.column_stack((np.ones(n_test), X_test))

theta_closed = np.dot(np.linalg.inv(np.dot(X_cl_train.T, X_cl_train)), (np.dot(X_cl_train.T,y_train)))
print(f"Closed Form Theta", theta_closed)
y_train_pred = np.dot(X_cl_train, theta_closed)
y_test_pred = np.dot(X_cl_test, theta_closed)

train_error = np.mean((y_train - y_train_pred) ** 2)
test_error = np.mean((y_test - y_test_pred) ** 2)

print("Training error (MSE):", train_error)
print("Test error (MSE):", test_error)
print("\n")
batch_sizes = [1, 80, 8000, 800000]
lr=0.001
n_epochs=[100,100,200,2000]
eps=[1e-3,1e-4,1e-5,1e-5]


for i in range(4):
    print(f"Batch size: {batch_sizes[i]}")
    model = StochasticLinearRegressor()
    _ = model.fit(X_train, y_train, learning_rate=lr, batch_size=batch_sizes[i], n_epochs=n_epochs[i],epsilon=eps[i])
    print(f"Theta: {model.theta}\n")

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_error = np.mean((y_train - y_train_pred) ** 2)
    test_error = np.mean((y_test - y_test_pred) ** 2)
    print("Training error (MSE):", train_error)
    print("Test error (MSE):", test_error)
    print("\n")


