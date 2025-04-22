import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class LogisticRegressor:
    def __init__(self):
        self.theta = None
        self.mean = None
        self.std = None
    
    def fit(self, X, y, learning_rate=1,max_iter=100):
        """
        Fit the logistic regression model to the data using Newton's Method.
        Remember to normalize the input data X before fitting the model.
        
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input data.
            
        y : numpy array of shape (n_samples,)
            The target labels - 0 or 1.
        
        learning_rate : float
            The learning rate to use in the update rule.
        
        Returns
        -------
        params_history : numpy array of shape (n_iter, n_features+1)
            The list of parameters obtained after each iteration of Newton's Method.
        """
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1.0
        X = (X - self.mean) / self.std
        
        n_samples, n_features = X.shape

        X = np.c_[np.ones((n_samples, 1)), X]
        theta = np.zeros(n_features + 1)
        params_history = []

        eps = 1e-7
        
        for i in range(max_iter):
            z = X.dot(theta)
            p = sigmoid(z)
            
            grad = X.T.dot(p - y)
            
            W = np.diag(p * (1 - p))
            H = X.T.dot(W).dot(X)
            
            try:
                delta = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                delta = np.linalg.pinv(H).dot(grad)
            
            theta_new = theta - learning_rate * delta
            params_history.append(theta_new.copy())
            
            if np.linalg.norm(theta_new - theta, ord=2) < eps:
                theta = theta_new
                # print(i)
                break
            
            theta = theta_new
        
        self.theta = theta
        return np.array(params_history)
    
    def predict(self, X):
        """
        Predict the target values for the input data.
        
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input data.
            
        Returns
        -------
        y_pred : numpy array of shape (n_samples,)
            The predicted target label.
        """
        
        X = (X - self.mean) / self.std
        n_samples = X.shape[0]
        X = np.hstack([np.ones((n_samples, 1)), X])
        
        probs = sigmoid(X.dot(self.theta))
        return (probs >= 0.5).astype(int)
