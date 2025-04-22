import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_confusion_matrix(y_test, y_pred, title, labels, normalize=False):
    cm = confusion_matrix(y_test, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] 
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title + (" (Relative)" if normalize else " (Absolute) "))
    plt.show()

class_labels = ["dew", "fogsmog", "frost", "glaze", "hail", "lightning", "rain", "rainbow", "rime", "sandstorm", "snow"]

y_test5 = pd.read_csv("y_test5.csv").values.flatten()
y_pred5 = pd.read_csv("y_pred5.csv").values.flatten()
y_test6 = pd.read_csv("y_test6.csv").values.flatten()
y_pred6 = pd.read_csv("y_pred6.csv").values.flatten()

plot_confusion_matrix(y_test5, y_pred5, "Confusion Matrix - CVXOPT SVM", class_labels)
plot_confusion_matrix(y_test6, y_pred6, "Confusion Matrix - LIBSVM SVM", class_labels)

plot_confusion_matrix(y_test5, y_pred5, "Confusion Matrix - CVXOPT SVM", class_labels, normalize=True)
plot_confusion_matrix(y_test6, y_pred6, "Confusion Matrix - LIBSVM SVM", class_labels, normalize=True)
