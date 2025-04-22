import numpy as np
import time
from svm import SupportVectorMachine
from methods import load_images

X_train, y_train = load_images("data/Q2/train")
X_test, y_test = load_images("data/Q2/test")

selected_classes = [83 % 11, (83 + 1) % 11]

train_filter = np.isin(y_train, selected_classes)
test_filter = np.isin(y_test, selected_classes)
X_train, y_train = X_train[train_filter], y_train[train_filter]
X_test, y_test = X_test[test_filter], y_test[test_filter]

y_train = (y_train == selected_classes[1]).astype(int)
y_test = (y_test == selected_classes[1]).astype(int)


svm_linear = SupportVectorMachine()
svm_linear.fit(X_train, y_train, kernel='linear', C=1.0)
linear_support_vectors = set(map(tuple, svm_linear.support_vectors))

svm = SupportVectorMachine()
start_time = time.time()
svm.fit(X_train, y_train, kernel='gaussian', C=1.0, gamma=0.001)
training_time = time.time() - start_time
print(f"Training Time: {training_time:.2f} sec")

y_pred_gaussian = svm.predict(X_test)
accuracy_gaussian = np.mean(y_pred_gaussian == y_test) * 100
print(f"Test Accuracy: {accuracy_gaussian:.2f}%")

num_support_vectors_gaussian = len(svm.support_vectors)
percentage_support_vectors_gaussian = (num_support_vectors_gaussian / len(y_train)) * 100
print(f"Number of Support Vectors: {num_support_vectors_gaussian}")
print(f"Percentage of Support Vectors: {percentage_support_vectors_gaussian:.2f}%")


gaussian_support_vectors = set(map(tuple, svm.support_vectors))
matching_support_vectors = len(linear_support_vectors & gaussian_support_vectors)
print(f"Number of matching support vectors: {matching_support_vectors}")


np.savetxt("sv_gaussian.csv", svm.support_vectors, delimiter=",")