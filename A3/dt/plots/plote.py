import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

results = [
    {'n_estimators': 50, 'max_features': 0.1, 'min_samples_split': 2, 'OOB Accuracy': 0.8506},
    {'n_estimators': 150, 'max_features': 0.1, 'min_samples_split': 2, 'OOB Accuracy': 0.8533},
    {'n_estimators': 250, 'max_features': 0.1, 'min_samples_split': 2, 'OOB Accuracy': 0.8540},
    {'n_estimators': 350, 'max_features': 0.1, 'min_samples_split': 2, 'OOB Accuracy': 0.8544},
    {'n_estimators': 50, 'max_features': 0.1, 'min_samples_split': 4, 'OOB Accuracy': 0.8520},
    {'n_estimators': 150, 'max_features': 0.1, 'min_samples_split': 4, 'OOB Accuracy': 0.8546},
    {'n_estimators': 250, 'max_features': 0.1, 'min_samples_split': 4, 'OOB Accuracy': 0.8545},
    {'n_estimators': 350, 'max_features': 0.1, 'min_samples_split': 4, 'OOB Accuracy': 0.8553},
    {'n_estimators': 50, 'max_features': 0.1, 'min_samples_split': 6, 'OOB Accuracy': 0.8526},
    {'n_estimators': 150, 'max_features': 0.1, 'min_samples_split': 6, 'OOB Accuracy': 0.8552},
    {'n_estimators': 250, 'max_features': 0.1, 'min_samples_split': 6, 'OOB Accuracy': 0.8544},
    {'n_estimators': 350, 'max_features': 0.1, 'min_samples_split': 6, 'OOB Accuracy': 0.8547},
    {'n_estimators': 50, 'max_features': 0.1, 'min_samples_split': 8, 'OOB Accuracy': 0.8532},
    {'n_estimators': 150, 'max_features': 0.1, 'min_samples_split': 8, 'OOB Accuracy': 0.8547},
    {'n_estimators': 250, 'max_features': 0.1, 'min_samples_split': 8, 'OOB Accuracy': 0.8544},
    {'n_estimators': 350, 'max_features': 0.1, 'min_samples_split': 8, 'OOB Accuracy': 0.8544},
    {'n_estimators': 50, 'max_features': 0.1, 'min_samples_split': 10, 'OOB Accuracy': 0.8542},
    {'n_estimators': 150, 'max_features': 0.1, 'min_samples_split': 10, 'OOB Accuracy': 0.8553},
    {'n_estimators': 250, 'max_features': 0.1, 'min_samples_split': 10, 'OOB Accuracy': 0.8557},
    {'n_estimators': 350, 'max_features': 0.1, 'min_samples_split': 10, 'OOB Accuracy': 0.8553},
    {'n_estimators': 50, 'max_features': 0.3, 'min_samples_split': 2, 'OOB Accuracy': 0.8542},
    {'n_estimators': 150, 'max_features': 0.3, 'min_samples_split': 2, 'OOB Accuracy': 0.8569},
    {'n_estimators': 250, 'max_features': 0.3, 'min_samples_split': 2, 'OOB Accuracy': 0.8581},
    {'n_estimators': 350, 'max_features': 0.3, 'min_samples_split': 2, 'OOB Accuracy': 0.8581},
    {'n_estimators': 50, 'max_features': 0.3, 'min_samples_split': 4, 'OOB Accuracy': 0.8551},
    {'n_estimators': 150, 'max_features': 0.3, 'min_samples_split': 4, 'OOB Accuracy': 0.8575},
    {'n_estimators': 250, 'max_features': 0.3, 'min_samples_split': 4, 'OOB Accuracy': 0.8575},
    {'n_estimators': 350, 'max_features': 0.3, 'min_samples_split': 4, 'OOB Accuracy': 0.8569},
    {'n_estimators': 50, 'max_features': 0.3, 'min_samples_split': 6, 'OOB Accuracy': 0.8559},
    {'n_estimators': 150, 'max_features': 0.3, 'min_samples_split': 6, 'OOB Accuracy': 0.8579},
    {'n_estimators': 250, 'max_features': 0.3, 'min_samples_split': 6, 'OOB Accuracy': 0.8585},
    {'n_estimators': 350, 'max_features': 0.3, 'min_samples_split': 6, 'OOB Accuracy': 0.8583},
    {'n_estimators': 50, 'max_features': 0.3, 'min_samples_split': 8, 'OOB Accuracy': 0.8549},
    {'n_estimators': 150, 'max_features': 0.3, 'min_samples_split': 8, 'OOB Accuracy': 0.8573},
    {'n_estimators': 250, 'max_features': 0.3, 'min_samples_split': 8, 'OOB Accuracy': 0.8579},
    {'n_estimators': 350, 'max_features': 0.3, 'min_samples_split': 8, 'OOB Accuracy': 0.8582},
    {'n_estimators': 50, 'max_features': 0.3, 'min_samples_split': 10, 'OOB Accuracy': 0.8561},
    {'n_estimators': 150, 'max_features': 0.3, 'min_samples_split': 10, 'OOB Accuracy': 0.8578},
    {'n_estimators': 250, 'max_features': 0.3, 'min_samples_split': 10, 'OOB Accuracy': 0.8586},
    {'n_estimators': 350, 'max_features': 0.3, 'min_samples_split': 10, 'OOB Accuracy': 0.8581},
    {'n_estimators': 50, 'max_features': 0.5, 'min_samples_split': 2, 'OOB Accuracy': 0.8557},
    {'n_estimators': 150, 'max_features': 0.5, 'min_samples_split': 2, 'OOB Accuracy': 0.8581},
    {'n_estimators': 250, 'max_features': 0.5, 'min_samples_split': 2, 'OOB Accuracy': 0.8581},
    {'n_estimators': 350, 'max_features': 0.5, 'min_samples_split': 2, 'OOB Accuracy': 0.8586},
    {'n_estimators': 50, 'max_features': 0.5, 'min_samples_split': 4, 'OOB Accuracy': 0.8579},
    {'n_estimators': 150, 'max_features': 0.5, 'min_samples_split': 4, 'OOB Accuracy': 0.8595},
    {'n_estimators': 250, 'max_features': 0.5, 'min_samples_split': 4, 'OOB Accuracy': 0.8588},
    {'n_estimators': 350, 'max_features': 0.5, 'min_samples_split': 4, 'OOB Accuracy': 0.8587},
    {'n_estimators': 50, 'max_features': 0.5, 'min_samples_split': 6, 'OOB Accuracy': 0.8568},
    {'n_estimators': 150, 'max_features': 0.5, 'min_samples_split': 6, 'OOB Accuracy': 0.8588},
    {'n_estimators': 250, 'max_features': 0.5, 'min_samples_split': 6, 'OOB Accuracy': 0.8586},
    {'n_estimators': 350, 'max_features': 0.5, 'min_samples_split': 6, 'OOB Accuracy': 0.8585},
    {'n_estimators': 50, 'max_features': 0.5, 'min_samples_split': 8, 'OOB Accuracy': 0.8569},
    {'n_estimators': 150, 'max_features': 0.5, 'min_samples_split': 8, 'OOB Accuracy': 0.8601},
    {'n_estimators': 250, 'max_features': 0.5, 'min_samples_split': 8, 'OOB Accuracy': 0.8594},
    {'n_estimators': 350, 'max_features': 0.5, 'min_samples_split': 8, 'OOB Accuracy': 0.8594},
    {'n_estimators': 50, 'max_features': 0.5, 'min_samples_split': 10, 'OOB Accuracy': 0.8574},
    {'n_estimators': 150, 'max_features': 0.5, 'min_samples_split': 10, 'OOB Accuracy': 0.8603},
    {'n_estimators': 250, 'max_features': 0.5, 'min_samples_split': 10, 'OOB Accuracy': 0.8601},
    {'n_estimators': 350, 'max_features': 0.5, 'min_samples_split': 10, 'OOB Accuracy': 0.8600}
]


# Convert the results to a DataFrame
df = pd.DataFrame(results)

# Set the style
sns.set(style="whitegrid")

# Create the FacetGrid
g = sns.FacetGrid(df, col="max_features", col_wrap=3, height=4)

# Draw heatmaps for each max_features value
def heatmap(data, **kwargs):
    pivot = data.pivot_table(index="min_samples_split", columns="n_estimators", values="OOB Accuracy")
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="viridis", **kwargs)

g.map_dataframe(heatmap)

# Adjust the plot
g.set_titles(col_template="max_features = {col_name}")
g.set_axis_labels("n_estimators", "min_samples_split")
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Grid Search: OOB Accuracy Across All Hyperparameters", fontsize=16)

plt.show()