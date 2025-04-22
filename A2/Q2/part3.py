import numpy as np
import pandas as pd
import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from methods import load_images
from matplotlib import pyplot as plt

sv_linear_cv = np.loadtxt("sv_linear.csv", delimiter=",")
sv_gaussian_cv = np.loadtxt("sv_gaussian.csv", delimiter=",")
wb_cv = np.loadtxt("weights_bias_linear.csv", delimiter=",")

w_cvxopt = wb_cv[:-1]
b_cvxopt = wb_cv[-1]

X_train, y_train = load_images("data/Q2/train")
X_test, y_test = load_images("data/Q2/test")

selected_classes = [83 % 11, (83 + 1) % 11]
train_filter = np.isin(y_train, selected_classes)
test_filter = np.isin(y_test, selected_classes)

X_train, y_train = X_train[train_filter], y_train[train_filter]
X_test, y_test = X_test[test_filter], y_test[test_filter]

y_train = (y_train == selected_classes[1]).astype(int)
y_test = (y_test == selected_classes[1]).astype(int)


start_time = time.time()
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train, y_train)
time_linear = time.time() - start_time
sv_linear = X_train[svm_linear.support_]

start_time = time.time()
svm_gaussian = SVC(kernel='rbf', C=1.0, gamma=0.001)
svm_gaussian.fit(X_train, y_train)
time_gaussian = time.time() - start_time
sv_gaussian = X_train[svm_gaussian.support_]

y_pred_linear = svm_linear.predict(X_test)
y_pred_gaussian = svm_gaussian.predict(X_test)
accuracy_linear = accuracy_score(y_test, y_pred_linear) * 100
accuracy_gaussian = accuracy_score(y_test, y_pred_gaussian) * 100

w = svm_linear.coef_.flatten()
b = svm_linear.intercept_[0]

w_diff = np.linalg.norm(w_cvxopt - w)
b_diff = abs(b_cvxopt - b)

def count_matching(sv1, sv2, tol=1e-6):
    count = 0
    for v1 in sv1:
        if np.any(np.all(np.abs(sv2 - v1) < tol, axis=1)):
            count += 1
    return count

models = ["Linear", "Gaussian", "SL Linear", "SL Gaussian"]
support_vectors = [sv_linear_cv, sv_gaussian_cv, sv_linear, sv_gaussian]

matching_matrix = np.zeros((4, 4), dtype=int)
for i in range(4):
    for j in range(4):
        matching_matrix[i, j] = count_matching(support_vectors[i], support_vectors[j])

df = pd.DataFrame(matching_matrix, index=models, columns=models)

print("\nNumber of Matching Support Vectors:")
print(df)

print("\nSupport Vector Counts:")
print(f"Linear: {len(svm_linear.support_)}")
print(f"Gaussian: {len(svm_gaussian.support_)}")

print("\nTest Accuracy:")
print(f"Linear: {accuracy_linear:.2f}%")
print(f"Gaussian: {accuracy_gaussian:.2f}%")

print("\nWeight and Bias Comparison (Linear SVM):")
print(f"||w_CVXOPT - w|| (L2 Norm): {w_diff:.6f}")
print(f"|b_CVXOPT - b| (Absolute Difference): {b_diff:.6f}")

print("\nComputational Cost (Training Time):")
print(f"Linear: {time_linear:.4f} sec")
print(f"Gaussian: {time_gaussian:.4f} sec")


w = svm_linear.coef_[0]
w_min, w_max = w.min(), w.max()
w_image = (w - w_min) / (w_max - w_min) 
w_image = w_image.reshape(100, 100, 3)
plt.imshow(w_image)
plt.axis("off")
plt.show()
