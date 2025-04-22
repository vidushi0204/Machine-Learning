import numpy as np
from gda import GaussianDiscriminantAnalysis

X = np.loadtxt('./data/Q4/q4x.dat')
y = np.loadtxt('./data/Q4/q4y.dat', dtype=str)
y = np.where(y == 'Alaska', 0, 1)

gda = GaussianDiscriminantAnalysis()
mu_0, mu_1, sigma_0, sigma_1 = gda.fit(X, y)

print("Normalized Scale:")
print("mu_0 (Alaska):", mu_0)
print("mu_1 (Canada):", mu_1)
print("Sigma_0:\n", sigma_0)
print("Sigma_1:\n", sigma_1)

# Convert back to original scale
mu_0_orig = mu_0 * gda.std + gda.mean
mu_1_orig = mu_1 * gda.std + gda.mean

print("\nOriginal Scale:")
print("mu_0 (Alaska):", mu_0_orig)
print("mu_1 (Canada):", mu_1_orig)
print("Sigma_0:\n", sigma_0 * np.outer(gda.std, gda.std))
print("Sigma_1:\n", sigma_1 * np.outer(gda.std, gda.std))
