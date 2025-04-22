import numpy as np
import cv2
import os
import time
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from methods import load_images

X_train, y_train = load_images("data/Q2/train")
X_test, y_test = load_images("data/Q2/test")

selected_classes = [83 % 11, (83 + 1) % 11]
train_filter = np.isin(y_train, selected_classes)
test_filter = np.isin(y_test, selected_classes)
X_train, y_train = X_train[train_filter], y_train[train_filter]
X_test, y_test = X_test[test_filter], y_test[test_filter]


y_train = np.where(y_train == selected_classes[1], 1, -1)
y_test = np.where(y_test == selected_classes[1], 1, -1)

start_time = time.time()
svm = SVC(kernel='linear', C=1.0)
svm.fit(X_train, y_train)
linear_time = time.time() - start_time

start_time = time.time()
sgd_svm = SGDClassifier(loss='hinge', alpha=0.01, max_iter=10000, tol=1e-6)
sgd_svm.fit(X_train, y_train)
sgd_time = time.time() - start_time

accuracy_liblinear = np.mean(svm.predict(X_test) == y_test) * 100
accuracy_sgd = np.mean(sgd_svm.predict(X_test) == y_test) * 100

print(f"Training Time (LIBLINEAR): {linear_time:.4f} sec")
print(f"Training Time (SGD): {sgd_time:.4f} sec")
print(f"Test Accuracy (LIBLINEAR): {accuracy_liblinear:.2f}%")
print(f"Test Accuracy (SGD): {accuracy_sgd:.2f}%")
