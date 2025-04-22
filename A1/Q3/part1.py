import numpy as np
from logistic_regression import LogisticRegressor

X = np.loadtxt("./data/Q3/logisticX.csv", delimiter=",")
y = np.loadtxt("./data/Q3/logisticY.csv", delimiter=",")

model = LogisticRegressor()

history = model.fit(X, y, learning_rate=1,max_iter=10000)

print("Learned Theta:")
print(model.theta)