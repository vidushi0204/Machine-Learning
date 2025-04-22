import numpy as np
import matplotlib.pyplot as plt
from gda import GaussianDiscriminantAnalysis

X = np.loadtxt('./data/Q4/q4x.dat')
y = np.loadtxt('./data/Q4/q4y.dat', dtype=str)
y = np.where(y == 'Alaska', 0, 1)

gda = GaussianDiscriminantAnalysis()
gda.fit(X, y, assume_same_covariance=True)
mu_0 = gda.mu_0
mu_1 = gda.mu_1
cov_matrix = gda.sigma

mu_0_orig = gda.mean + gda.std * mu_0
mu_1_orig = gda.mean + gda.std * mu_1

scaling = np.diag(gda.std)
cov_matrix_orig = scaling @ cov_matrix @ scaling

inv_cov = np.linalg.inv(cov_matrix_orig)
w = np.dot(inv_cov, (mu_1_orig - mu_0_orig))
phi = np.mean(y)
intercept = -0.5 * np.dot(mu_1_orig.T, np.dot(inv_cov, mu_1_orig)) + 0.5 * np.dot(mu_0_orig.T, np.dot(inv_cov, mu_0_orig)) - np.log((1 - phi) / phi)

plt.figure(figsize=(8, 6))
plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='o', color='blue', label='Alaska')
plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='x', color='green', label='Canada')

x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
y_vals = -(w[0] * x_vals + intercept) / w[1]
plt.plot(x_vals, y_vals, 'r-', label='Decision Boundary')

plt.xlabel('Freshwater Growth')
plt.ylabel('Marine Growth')
plt.title('GDA Decision Boundary')
plt.legend()
plt.grid(True)
plt.show()
