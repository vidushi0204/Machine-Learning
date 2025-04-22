import numpy as np
import matplotlib.pyplot as plt
from gda import GaussianDiscriminantAnalysis

X = np.loadtxt('./data/Q4/q4x.dat')
y = np.loadtxt('./data/Q4/q4y.dat', dtype=str)
y = np.where(y == 'Alaska', 0, 1)

gda = GaussianDiscriminantAnalysis()
mu_0, mu_1, sigma_0, sigma_1 = gda.fit(X, y)

mu_0_orig = mu_0 * gda.std + gda.mean
mu_1_orig = mu_1 * gda.std + gda.mean
sigma_0_orig = (gda.std[:, None] * sigma_0 * gda.std)
sigma_1_orig = (gda.std[:, None] * sigma_1 * gda.std)

plt.figure(figsize=(8, 6))
plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='o', label='Alaska')
plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='x', label='Canada')

inv_cov = np.linalg.inv((sigma_0_orig + sigma_1_orig) / 2)
w = np.dot(inv_cov, (mu_1_orig - mu_0_orig))
intercept = -0.5 * np.dot(mu_1_orig.T, np.dot(inv_cov, mu_1_orig)) + 0.5 * np.dot(mu_0_orig.T, np.dot(inv_cov, mu_0_orig))
x_vals = np.linspace(X[:, 0].min()-2, X[:, 0].max()+2, 200)
y_vals = -(w[0] * x_vals + intercept) / w[1]
plt.plot(x_vals, y_vals, 'b--', label='Linear Decision Boundary')

x, y_grid = np.meshgrid(np.linspace(X[:, 0].min()-2, X[:, 0].max()+2, 300), np.linspace(X[:, 1].min()-2, X[:, 1].max()+2, 300))
z = np.zeros(x.shape)
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        diff_0 = np.array([x[i, j], y_grid[i, j]]) - mu_0_orig
        diff_1 = np.array([x[i, j], y_grid[i, j]]) - mu_1_orig
        z[i, j] = diff_0.T @ np.linalg.inv(sigma_0_orig) @ diff_0 - diff_1.T @ np.linalg.inv(sigma_1_orig) @ diff_1

contour = plt.contour(x, y_grid, z, levels=[0], colors='red', linewidths=2, linestyles='-')
# contour.collections[0].set_label('Quadratic Decision Boundary')
plt.xlabel('Freshwater Growth')
plt.ylabel('Marine Growth')
plt.title('GDA Decision Boundaries (Linear and Quadratic)')
plt.legend()
plt.grid(True)
plt.show()
