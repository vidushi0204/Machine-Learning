import numpy as np

def generate(N, theta, input_mean, input_sigma, noise_sigma):
    np.random.seed(42)
    x1 = np.random.normal(loc=input_mean[0], scale=input_sigma[0], size=N)
    x2 = np.random.normal(loc=input_mean[1], scale=input_sigma[1], size=N)
    
    epsilon = np.random.normal(loc=0, scale=noise_sigma, size=N)
    
    y = theta[0] + theta[1] * x1 + theta[2] * x2 + epsilon
    X = np.column_stack((x1, x2))
    
    return X, y

class StochasticLinearRegressor:
    def __init__(self):
        self.theta = None
    
    def loss(self, y_pred, y, n_samples):
        loss=(1/(2*n_samples))*np.sum((y_pred-y)**2)
        return loss
    
    def fit(self, X, y, learning_rate=0.01, n_epochs=10, batch_size=32,epsilon = 1e-5):
        n_samples, n_features = X.shape
        X=np.c_[np.ones(n_samples),X]
        self.theta = np.zeros(n_features + 1)

        num_batches = int(np.ceil(n_samples / batch_size))
        params_history = []

        
        prev_loss = float('inf')

        for epoch in range(n_epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            loss = self.loss(np.dot(X,self.theta), y, n_samples)   

            if abs(loss-prev_loss)<epsilon:
                # print(epoch)
                break
            prev_loss=loss
            
            
            for batch in range(num_batches):
                start = batch * batch_size
                end = min(start + batch_size, n_samples)

                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                y_pred = np.dot(X_batch,self.theta)
                
                m_batch = X_batch.shape[0]
                gradient = - (1 / (m_batch)) * (X_batch.T).dot(y_batch - y_pred)
                
                self.theta = self.theta - learning_rate * gradient
                
                params_history.append(self.theta.copy())
                

        return np.array(params_history)

    
    def predict(self, X):

        n_samples = X.shape[0]
        X = np.c_[(np.ones(n_samples), X)]
        return np.dot(X,self.theta)
