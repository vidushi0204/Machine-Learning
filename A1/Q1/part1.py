import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegressor  

X = np.loadtxt("./data/Q1/linearX.csv", delimiter=",")  
y = np.loadtxt("./data/Q1/linearY.csv", delimiter=",")  

model = LinearRegressor()
params = model.fit(X, y, 0.025)
n=X.shape[0]

y_pred = model.predict(X)

print("Final Weights:", params[-1])
print("Number of Iterations: ", len(params))
print("Final Loss: ", model.loss(y_pred, y, n))
losses=[]
X_new = np.c_[np.ones(X.shape[0]), X]
for w in params:
    loss = model.loss(np.dot(X_new,w),y,n)
    losses.append(loss)

plt.plot(range(len(losses)), losses)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Loss Curve (Training)")
plt.show()
