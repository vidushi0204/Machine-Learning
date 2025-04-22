import numpy as np
from gda import GaussianDiscriminantAnalysis

X = np.loadtxt('./data/Q4/q4x.dat')
with open('./data/Q4/q4y.dat', 'r') as f:
    y = np.array([1 if line.strip() == 'Canada' else 0 for line in f])

gda = GaussianDiscriminantAnalysis()
mu_0, mu_1, cov_matrix = gda.fit(X, y, assume_same_covariance=True)

print("Normalized Scale:")
print("mu_0 (Alaska):", mu_0)
print("mu_1 (Canada):", mu_1)
print("Shared Covariance Matrix Σ:")    
print(cov_matrix)
mu_0_orig = mu_0 * gda.std + gda.mean
mu_1_orig = mu_1 * gda.std + gda.mean
cov_matrix_orig = cov_matrix * np.outer(gda.std, gda.std)

print("Original Scale:")
print("mu_0 (Alaska):", mu_0_orig)
print("mu_1 (Canada):", mu_1_orig)
print("Shared Covariance Matrix Σ:")
print(cov_matrix_orig)
