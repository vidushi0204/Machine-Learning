import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('prune_accuracy_depth_25.csv')

print(df)
plt.figure(figsize=(10, 6))
plt.plot(df['NumNodes'], df['TrainAccuracy'], label='Train Accuracy', marker='o')
plt.plot(df['NumNodes'], df['ValidAccuracy'], label='Validation Accuracy', marker='o')
plt.plot(df['NumNodes'], df['TestAccuracy'], label='Test Accuracy', marker='o')

plt.xlabel('Number of Nodes')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
