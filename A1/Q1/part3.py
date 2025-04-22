import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegressor 

X = np.loadtxt('./data/Q1/linearX.csv', delimiter=',')
y = np.loadtxt('./data/Q1/linearY.csv', delimiter=',')


model = LinearRegressor()
params_list = model.fit(X, y, learning_rate=0.025)

theta0_vals = np.linspace(-20, 32, 100)
theta1_vals = np.linspace(-15,70, 100)
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

n = X.shape[0]
X = np.c_[np.ones(n), X]

for i, theta0 in enumerate(theta0_vals):
    for j, theta1 in enumerate(theta1_vals):
        theta = np.array([theta0, theta1])
        y_pred = np.dot(X,theta)
        J_vals[i, j] = model.loss(y_pred,y,n)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

T0, T1 = np.meshgrid(theta0_vals, theta1_vals)
surf = ax.plot_surface(T0, T1, J_vals.T, cmap='coolwarm', edgecolor='none', alpha=0.5)
ax.set_xlabel('Theta0 (Intercept)')
ax.set_ylabel('Theta1')
ax.set_zlabel('Cost J(θ)')
ax.set_title('3D Mesh of Cost Function')

# plt.pause(10)
for param in params_list:
    theta0, theta1 = param
    pred=np.dot(X,param)
    cost_val = model.loss(pred,y,n)
    
    point = ax.scatter(theta0, theta1, cost_val, color='red', s=15)
    label = ax.text(theta0, theta1, cost_val, f'Error = {cost_val:.4f}', color='black', fontsize=8, ha='left')
    ax.scatter(theta0, theta1, cost_val, color='red', s=2)
    plt.pause(0.2)
    point.remove()
    label.remove()

theta0, theta1 = params_list[-1]
pred=np.dot(X,params_list[-1])
cost_val = model.loss(pred,y,n)

point = ax.scatter(theta0, theta1, cost_val, color='red', s=15)
label = ax.text(theta0, theta1, cost_val, f'Final Error = {cost_val:.4f}', color='black', fontsize=8, ha='left')
plt.show()
