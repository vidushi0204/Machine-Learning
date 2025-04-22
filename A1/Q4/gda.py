# Imports - you can add any other permitted libraries
import numpy as np

# You may add any other functions to make your code more modular. However,
# do not change the function signatures (name and arguments) of the given functions,
# as these functions will be called by the autograder.


class GaussianDiscriminantAnalysis:
    # Assume Binary Classification
    def __init__(self):
        self.mean = None
        self.std = None
        self.mu_0 = None
        self.mu_1 = None
        self.sigma = None
        self.sigma_0 = None
        self.sigma_1 = None
    
    def fit(self, X, y, assume_same_covariance=False):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        X_norm = (X - self.mean) / self.std

        self.mu_0 = np.mean(X_norm[y == 0], axis=0)
        self.mu_1 = np.mean(X_norm[y == 1], axis=0)

        if assume_same_covariance:
            m = X_norm.shape[0]
            self.sigma = np.zeros((X_norm.shape[1], X_norm.shape[1]))
            for i in range(m):
                x_i = X_norm[i].reshape(-1, 1)
                mu = self.mu_1.reshape(-1, 1) if y[i] == 1 else self.mu_0.reshape(-1, 1)
                self.sigma += (x_i - mu) @ (x_i - mu).T
            self.sigma /= m
            return (self.mu_0, self.mu_1, self.sigma)
        
        else:
            X_0 = X_norm[y == 0]
            X_1 = X_norm[y == 1]
            self.sigma_0 = np.cov(X_0, rowvar=False,bias=True)
            self.sigma_1 = np.cov(X_1, rowvar=False,bias=True)
            
            return (self.mu_0, self.mu_1, self.sigma_0, self.sigma_1)

        
    
    def predict(self, X):
        X_norm = (X - self.mean) / self.std
        
        if self.sigma is not None:  
            inv_sigma = np.linalg.inv(self.sigma)
            discriminant_score_0 = -0.5 * np.sum((X_norm - self.mu_0) @ inv_sigma * (X_norm - self.mu_0), axis=1)
            discriminant_score_1 = -0.5 * np.sum((X_norm - self.mu_1) @ inv_sigma * (X_norm - self.mu_1), axis=1)
        
        else: 
            discriminant_score_0 = -0.5 * np.sum((X_norm - self.mu_0) @ np.linalg.inv(self.sigma_0) * (X_norm - self.mu_0), axis=1)
            discriminant_score_1 = -0.5 * np.sum((X_norm - self.mu_1) @ np.linalg.inv(self.sigma_1) * (X_norm - self.mu_1), axis=1)

        y_pred = np.where(discriminant_score_0 > discriminant_score_1, 0, 1)
        
        return y_pred