import cvxopt
import numpy as np

class SupportVectorMachine:
    '''
    Binary Classifier using Support Vector Machine
    '''
    def __init__(self):
        self.alphas = None
        self.w = None
        self.b = None
        self.support_vectors = None
        self.kernel = None
        self.X_train = None
        self.y_train = None
        self.C = None
        self.gamma = None

    
    def linear_fit(self, X, y, C=1.0):
        y = y * 2 - 1  # Convert {0,1} labels to {-1,1}
        N, D = X.shape
        K = np.dot(X, X.T)
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-np.ones(N))
        G = cvxopt.matrix(np.vstack((-np.eye(N), np.eye(N))))
        h = cvxopt.matrix(np.hstack((np.zeros(N), np.ones(N) * self.C)))
        A = cvxopt.matrix(y.astype(float), (1, N))
        b = cvxopt.matrix(0.0)
        
        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        
        self.alphas = np.ravel(solution['x'])
        support_indices = np.where(self.alphas > 1e-5)[0]
        self.support_vectors = X[support_indices]
        
        # Fix broadcasting issue
        self.w = np.sum((self.alphas[support_indices] * y[support_indices])[:, None] * X[support_indices], axis=0)
        
        # Compute bias term
        self.b = np.mean(y[support_indices] - np.dot(X[support_indices], self.w))
    
    def _gaussian_kernel(self, X1, X2):
        """Compute the Gaussian (RBF) kernel matrix."""
        gamma = self.gamma if self.gamma else 1.0 / X1.shape[1]
        X1_norm = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        X2_norm = np.sum(X2 ** 2, axis=1).reshape(1, -1)
        K = np.exp(-gamma * (X1_norm + X2_norm - 2 * np.dot(X1, X2.T)))
        return K

    def gaussian_fit(self, X, y, C=1.0, gamma=0.001):
        """Fit the SVM model using a Gaussian (RBF) kernel."""
        y = y * 2 - 1  # Convert {0,1} labels to {-1,1}
        N, D = X.shape
        self.C = C
        self.gamma = gamma

        K = self._gaussian_kernel(X, X)

        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-np.ones(N))
        G = cvxopt.matrix(np.vstack((-np.eye(N), np.eye(N))))
        h = cvxopt.matrix(np.hstack((np.zeros(N), np.ones(N) * C)))
        A = cvxopt.matrix(y.astype(float), (1, N))
        b = cvxopt.matrix(0.0)

        cvxopt.solvers.options['show_progress'] = False
        cvxopt.solvers.options['maxiters'] = 100

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        
        self.alphas = np.ravel(solution['x'])
        support_indices = self.alphas > 1e-5  # Only keep nonzero alphas
        self.support_vectors = X[support_indices]
        self.support_y = y[support_indices]
        self.support_alphas = self.alphas[support_indices]

        # Compute bias term
        K_sv = self._gaussian_kernel(self.support_vectors, self.support_vectors)
        self.b = np.mean(self.support_y - np.sum(self.support_alphas[:, None] * self.support_y[:, None] * K_sv, axis=1))


    def fit(self, X, y, kernel='linear', C=1.0, gamma=0.001):
        '''
        Learn the parameters from the given training data
        Classes are 0 or 1
        
        Args:
            X: np.array of shape (N, D) 
                where N is the number of samples and D is the flattened dimension of each image
            
            y: np.array of shape (N,)
                where N is the number of samples and y[i] is the class of the ith sample
            
            kernel: str
                The kernel to be used. Can be 'linear' or 'gaussian'
            
            C: float
                The regularization parameter
            
            gamma: float
                The gamma parameter for Gaussian kernel, ignored for linear kernel
        '''
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        

        if self.kernel == 'linear':
            self.linear_fit(X, y, C)
        elif self.kernel == 'gaussian':
            self.gaussian_fit(X, y, C, gamma)

    
    def predict(self, X):
        '''
        Predict the class of the input data
        
        Args:
            X: np.array of shape (N, D) 
                where N is the number of samples and D is the flattened dimension of each image
        
        Returns:
            np.array of shape (N,)
                where N is the number of samples and y[i] is the class of the ith sample (0 or 1)
        '''
        if self.kernel == 'linear':
            return (np.sign(np.dot(X, self.w) + self.b) > 0).astype(int)
        else:
            K_test = self._gaussian_kernel(X, self.support_vectors)
            decision = np.sum(self.support_alphas * self.support_y * K_test, axis=1) + self.b
            return (decision > 0).astype(int)
            
