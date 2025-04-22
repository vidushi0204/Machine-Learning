import numpy as np
import matplotlib.pyplot as plt

C_values = [1e-5, 1e-3, 1, 5, 10]
validation_accuracies = [0.1693, 0.1693, 0.6506, 0.6655, 0.6615]
test_accuracies = [0.1685, 0.1685, 0.6630, 0.6834, 0.6870]

plt.figure(figsize=(8, 5))
plt.plot(C_values, validation_accuracies, marker='o', linestyle='-', label='5-fold CV Accuracy')
plt.plot(C_values, test_accuracies, marker='s', linestyle='--', label='Test Accuracy')

plt.xscale('log')

plt.xlabel('C (Log Scale)')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

plt.show()
