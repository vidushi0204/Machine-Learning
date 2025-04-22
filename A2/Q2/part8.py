import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from methods import load_images

X_train, y_train = load_images("data/Q2/train")
X_test, y_test = load_images("data/Q2/test")

C_values = [1e-5, 1e-3, 1, 5, 10]
gamma = 0.001  

cv_accuracies = []
test_accuracies = []

for C in C_values:
    svm_rbf = SVC(kernel='rbf', C=C, gamma=gamma)
    cv_accuracy = np.mean(cross_val_score(svm_rbf, X_train, y_train, cv=5))
    cv_accuracies.append(cv_accuracy)
    
    svm_rbf.fit(X_train, y_train)
    test_accuracy = accuracy_score(y_test, svm_rbf.predict(X_test))
    test_accuracies.append(test_accuracy)

    print(f"C={C}, Cross-validation Accuracy: {cv_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(C_values, cv_accuracies, label="5-Fold CV Accuracy", marker='o')
plt.plot(C_values, test_accuracies, label="Test Accuracy", marker='s', linestyle='dashed')
plt.xscale('log') 
plt.xlabel("C (log scale)")
plt.ylabel("Accuracy")
plt.title("5-Fold Cross-Validation vs Test Accuracy")
plt.legend()
plt.grid(True)
plt.show()
