# Imports - you can add any other permitted libraries
import numpy as np
# You may add any other functions to make your code more modular. However,
# do not change the function signatures (name and arguments) of the given functions,
# as these functions will be called by the autograder.

class LinearRegressor:
    def __init__(self):
        self.weights=None
    
    def loss(self, y_pred, y, n_samples):
        loss=(1/(2*n_samples))*np.sum((y_pred-y)**2)
        return loss

    def train(self, X, y, lrate, n_iter, eps):
        n_samples, n_features = X.shape if X.ndim == 2 else (X.shape[0], 1)
        X=np.c_[np.ones(n_samples),X] # Concatenate column of ones to X
        Xt=X.T
        # Shape of X: (n_samples, n_features+1)

        self.W=np.zeros(n_features+1) #Initializing weights to 0
        param_list=[]
        prev_loss = float('inf')

        # loss J(θ)=(1/2n) * Σ(hθ(x(i)) - y(i))^2 
        for i in range(n_iter):
            y_pred=np.dot(X,self.W)
            loss=self.loss(y_pred,y,n_samples)

            if abs(loss-prev_loss)<eps:
                # print("iterations: ", i)
                break

            prev_loss=loss
            gradient=np.dot(Xt,(y_pred-y))*(1/n_samples)
            
            self.W -=lrate*gradient
            param_list.append(self.W.copy())


        return np.array(param_list)        



    def fit(self, X, y, learning_rate=0.025):
        n_iter=1500
        eps=1e-5
        return self.train(X,y,learning_rate,n_iter,eps)
        
    
    def predict(self, X):
       
        n_samples =X.shape[0]
        X=np.c_[np.ones(n_samples),X]
        return np.dot(X,self.W)