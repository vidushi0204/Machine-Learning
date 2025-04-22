import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_pred = pd.read_csv("y_pred6.csv").values.flatten()
y_true = pd.read_csv("y_test6.csv").values.flatten()

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["World", "Sports", "Business", "Science/Technology"], yticklabels=["World", "Sports", "Business", "Science/Technology"])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
