import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

y_test1 = pd.read_csv("y_test1.csv").values.flatten()  
y_pred1 = pd.read_csv("y_pred1.csv").values.flatten()  
y_test2 = pd.read_csv("y_test2.csv").values.flatten()
y_pred2 = pd.read_csv("y_pred2.csv").values.flatten()
y_test3 = pd.read_csv("y_test3.csv").values.flatten()
y_pred3 = pd.read_csv("y_pred3.csv").values.flatten()

def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    return accuracy, precision, recall, f1

results = []
for i, (y_test, y_pred) in enumerate([(y_test1, y_pred1), (y_test2, y_pred2), (y_test3, y_pred3)], start=1):
    accuracy, precision, recall, f1 = evaluate_model(y_test, y_pred)
    results.append([f"Model {i}", accuracy, precision, recall, f1])

df_results = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-score"])
print(df_results.to_string(index=False))
