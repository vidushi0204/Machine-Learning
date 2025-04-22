import numpy as np
import pandas as pd

test_df = pd.read_csv("./data/Q1/test.csv")

class_col = "Class Index"

num_classes = len(test_df[class_col].unique())
classes = test_df[class_col].unique()
random_predictions = np.random.choice(classes, size=len(test_df))
random_accuracy = (random_predictions == test_df[class_col]).mean() * 100

most_frequent_class = test_df[class_col].mode()[0]
positive_predictions = np.full(len(test_df), most_frequent_class)
positive_accuracy = (positive_predictions == test_df[class_col]).mean() * 100

print(f"Random Guessing Accuracy: {random_accuracy:.2f}%")
print(f"Positive Accuracy: {positive_accuracy:.2f}%")
