import numpy as np
from svm import SupportVectorMachine

class Multi_SVM:
    def __init__(self, C=1.0, gamma=0.001):
        self.C = C
        self.gamma = gamma
        self.models = None
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.models = {}
        
        n = len(self.classes)
        for i in range(n):
            for j in range(i + 1, n):
                i_indices = np.where(y == self.classes[i])[0]
                j_indices = np.where(y == self.classes[j])[0]
                
                ij = np.concatenate((i_indices, j_indices))
                y_ij = np.zeros(len(ij))
                y_ij[len(i_indices):] = 1
                X_ij = X[ij]
                
                svm = SupportVectorMachine()
                svm.fit(X_ij, y_ij, kernel='gaussian', C=self.C, gamma=self.gamma)
                
                model_key = f"{self.classes[i]}_{self.classes[j]}"
                self.models[model_key] = svm

    def predict(self, X):
        votes = np.zeros((len(X), len(self.classes)))
        models = self.models
        for model_key, svm in models.items():
            class_i, class_j = map(int, model_key.split('_'))
            predictions = svm.predict(X)
            
            for i, pred in enumerate(predictions):
                if pred == 0:
                    votes[i, class_i] += 1
                else:
                    votes[i, class_j] += 1
        
        predicted_classes = np.argmax(votes, axis=1)
        
        tied_indices = np.where(np.sum(votes == np.max(votes, axis=1)[:, None], axis=1) > 1)[0]
        for ind in tied_indices:
            max_score_class = np.argmax(votes[ind])
            predicted_classes[ind] = max_score_class
        
        return self.classes[predicted_classes]
