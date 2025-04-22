import numpy as np
import matplotlib.pyplot as plt

X = np.loadtxt('./data/Q4/q4x.dat')
y = np.loadtxt('./data/Q4/q4y.dat', dtype=str)
y = np.where(y == 'Alaska', 0, 1)

plt.figure(figsize=(8, 6))
plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='o', label='Alaska')
plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='x', label='Canada')
plt.xlabel('Feature 1 (Freshwater Growth)')
plt.ylabel('Feature 2 (Marine Growth)')
plt.title('Salmon Data from Alaska and Canada (Original)')
plt.legend()
plt.grid(True)
plt.show()


mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X_norm = (X - mean) / std

plt.figure(figsize=(8, 6))
plt.scatter(X_norm[y == 0, 0], X_norm[y == 0, 1], marker='o', label='Alaska')
plt.scatter(X_norm[y == 1, 0], X_norm[y == 1, 1], marker='x', label='Canada')
plt.xlabel('Feature 1 (Freshwater Growth)')
plt.ylabel('Feature 2 (Marine Growth)')
plt.title('Salmon Data from Alaska and Canada (Normalized)')
plt.legend()
plt.grid(True)
plt.show()
