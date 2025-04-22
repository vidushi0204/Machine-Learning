import numpy as np
import pandas as pd
import cv2
import os
import time
from multi_svm import Multi_SVM
from methods import load_images_2, load_images

X_train, y_train = load_images("data/Q2/train")
X_test, y_test, filenames_test = load_images_2("data/Q2/test")

svm = Multi_SVM(C=1.0, gamma=0.001)
start_time = time.time()
svm.fit(X_train, y_train)
training_time = time.time() - start_time

y_pred = svm.predict(X_test)
accuracy = np.mean(y_pred == y_test) * 100

print(f"Test Accuracy: {accuracy:.2f}%")
print(f"Training Time: {training_time:.4f} sec")

pd.DataFrame(y_test, columns=["y_test"]).to_csv("y_test5.csv", index=False)
pd.DataFrame(y_pred, columns=["y_pred"]).to_csv("y_pred5.csv", index=False)

misclassified_indices = np.where(y_pred != y_test)[0]
if len(misclassified_indices) > 10:
    misclassified_indices = np.random.choice(misclassified_indices, 10, replace=False)

df_misclassified = pd.DataFrame({
    "image_name": [filenames_test[i] for i in misclassified_indices],
    "y_test": [y_test[i] for i in misclassified_indices],
    "y_pred": [y_pred[i] for i in misclassified_indices]
})
df_misclassified.to_csv("misclassified_images_5.csv", index=False)
