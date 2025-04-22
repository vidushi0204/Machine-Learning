import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegressor  

X = np.loadtxt('./data/Q1/linearX.csv', delimiter=',')
y = np.loadtxt('./data/Q1/linearY.csv', delimiter=',')

model = LinearRegressor()
model.fit(X, y, learning_rate=0.025)

plt.scatter(X, y, color='blue', label='Data Points',s=10) 
y_pred = model.predict(X)
plt.plot(X, y_pred, color='red', label='Hypothesis')
# plt.xlim(-0.1, 0.1)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Fit')
plt.legend()
plt.show()
