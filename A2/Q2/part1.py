import numpy as np
import matplotlib.pyplot as plt
from svm import SupportVectorMachine
from methods import load_images
import time

X_train, y_train = load_images("data/Q2/train")
X_test, y_test = load_images("data/Q2/test")

selected_classes = [83 % 11, (83 + 1) % 11]

train_filter = np.isin(y_train, selected_classes)
test_filter = np.isin(y_test, selected_classes)
X_train, y_train = X_train[train_filter], y_train[train_filter]
X_test, y_test = X_test[test_filter], y_test[test_filter]


y_train = (y_train == selected_classes[1]).astype(int)
y_test = (y_test == selected_classes[1]).astype(int)


svm = SupportVectorMachine()
start_time = time.time()
svm.fit(X_train, y_train, kernel='linear', C=1.0)
training_time = time.time() - start_time
print(f"Training Time: {training_time:.2f} sec")
# print(svm.b)

y_pred = svm.predict(X_test)
accuracy = np.mean(y_pred == y_test) * 100
print(f"Test Accuracy: {accuracy:.2f}%")


num_support_vectors = len(svm.support_vectors)
percentage_support_vectors = (num_support_vectors / len(y_train)) * 100
print(f"Number of Support Vectors: {num_support_vectors}")
print(f"Percentage of Support Vectors: {percentage_support_vectors:.2f}%")


support_alphas = svm.alphas[svm.alphas > 1e-5]  
top_5_indices = np.argsort(support_alphas)[-5:]  

fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i, idx in enumerate(top_5_indices):
    if idx < len(svm.support_vectors):  
        img = svm.support_vectors[idx].reshape(100, 100, 3)
        axes[i].imshow(img)
        axes[i].axis("off")
plt.show()


w_min, w_max = svm.w.min(), svm.w.max()
w_image = (svm.w - w_min) / (w_max - w_min) 


w_image = w_image.reshape(100, 100, 3)
plt.imshow(w_image)
plt.axis("off")
plt.show()


wb = np.append(svm.w, svm.b) 
np.savetxt("weights_bias_linear.csv", wb.reshape(1, -1), delimiter=",")
np.savetxt("sv_linear.csv", svm.support_vectors, delimiter=",")