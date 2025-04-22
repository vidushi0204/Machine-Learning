import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegressor

lr=0.025
X = np.loadtxt('./data/Q1/linearX.csv', delimiter=',')
y = np.loadtxt('./data/Q1/linearY.csv', delimiter=',')

model = LinearRegressor()
params_list = model.fit(X, y, learning_rate=lr)

theta0_vals = np.linspace(-20, 32, 100)
theta1_vals = np.linspace(-15, 70, 100)
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

n = X.shape[0]
X_bias = np.c_[np.ones(n), X]

for i, theta0 in enumerate(theta0_vals):
    for j, theta1 in enumerate(theta1_vals):
        theta = np.array([theta0, theta1])
        J_vals[i, j] = model.loss(np.dot(X_bias, theta), y, n)

levels = np.concatenate((np.arange(0, 100, 10), np.arange(100, np.max(J_vals), 100)))

plt.figure()
T0, T1 = np.meshgrid(theta0_vals, theta1_vals)
plt.contourf(T0, T1, J_vals.T, levels=levels, cmap='coolwarm', alpha=0.8)
cp = plt.contour(T0, T1, J_vals.T, levels=levels, colors='black', linewidths=0.5)

label = [0, 10, 20, 40, 60, 80, 100, 200, 300, 400]
plt.clabel(cp, levels=label, inline=True, fontsize=8, fmt='%1.1f')
plt.xlabel('Theta0 (Intercept)')
plt.ylabel('Theta1')
plt.title('Contour Plot of Cost Function')

for param in params_list:
    theta0, theta1 = param
    cost_val = model.loss(np.dot(X_bias, param), y, n)
    marker = plt.plot(theta0, theta1, 'ro', markersize=5)[0]
    label = plt.text(theta0, theta1, f'Error = {cost_val:.4f}', color='black', fontsize=8, ha='left')
    
    plt.plot(theta0, theta1, 'ro', markersize=1)[0]
    plt.draw()
    plt.pause(0.2)
    marker.remove()
    label.remove()

theta0, theta1 = params_list[-1]
plt.plot(theta0, theta1, 'ro', markersize=1)[0]
plt.text(theta0, theta1, f'Final Error = {model.loss(model.predict(X), y, n):.4f}', color='black', fontsize=8, ha='left')   

plt.show()
