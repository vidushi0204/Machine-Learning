import numpy as np
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegressor

def plot_decision_boundary(X, y, model):
    label0 = y == 0
    label1 = y == 1
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X[label0, 0], X[label0, 1], marker='o', color='red', label='Class 0')
    plt.scatter(X[label1, 0], X[label1, 1], marker='x', color='blue', label='Class 1')
    
    theta = model.theta      
    mu = model.mean          
    sigma = model.std    
    
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x1_range = np.linspace(x1_min, x1_max, 100)
    
    if abs(theta[2]) > 1e-6: 
        x2_range = mu[1] - (sigma[1]/theta[2]) * (theta[0] + theta[1]*((x1_range - mu[0]) / sigma[0]))
        plt.plot(x1_range, x2_range, 'g-', linewidth=2, label='Decision Boundary')
    else:
        x1_boundary = mu[0] - (theta[0]*sigma[0])/theta[1]
        plt.axvline(x=x1_boundary, color='g', linewidth=2, label='Decision Boundary')
    
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Training Data and Logistic Regression Decision Boundary')
    plt.legend()
    plt.show()

X = np.loadtxt("./data/Q3/logisticX.csv", delimiter=",")
y = np.loadtxt("./data/Q3/logisticY.csv", delimiter=",")

model = LogisticRegressor()
model.fit(X, y, learning_rate=0.01)

plot_decision_boundary(X, y, model)