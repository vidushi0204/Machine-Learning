import numpy as np
import math
from collections import Counter, defaultdict
import csv
import matplotlib.pyplot as plt
np.random.seed(42)

class Node:
    def __init__(self, is_leaf, prediction=None, feature=None, threshold=None, children=None):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.feature = feature
        self.threshold = threshold
        self.children = children or {}

def read_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    header = lines[0].strip().split(',')
    data = [line.strip().split(',') for line in lines[1:] if line.strip()]
    return header, data

def entropy(y):
    """Vectorized entropy using NumPy."""
    y = np.array(y)
    if len(y) == 0:
        return 0
    counts = np.bincount(y)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs + 1e-9)) 

def information_gain_categorical(X, y, feature_idx):
    H = entropy(y)
    subsets = defaultdict(list)
    for i in range(len(X)):
        subsets[X[i][feature_idx]].append(y[i])
    weighted_entropy = sum((len(subset) / len(y)) * entropy(subset) for subset in subsets.values())
    return H - weighted_entropy

def information_gain_numeric(X, y, feature_idx):
    values = np.array([float(row[feature_idx]) for row in X])
    median = np.median(values)
    
    left_mask = values <= median
    right_mask = values > median
    
    left_y = np.array(y)[left_mask]
    right_y = np.array(y)[right_mask]

    if len(left_y) == 0 or len(right_y) == 0:
        return 0, median

    H = entropy(y)
    w_left, w_right = len(left_y) / len(y), len(right_y) / len(y)
    info_gain = H - (w_left * entropy(left_y) + w_right * entropy(right_y))
    return info_gain, median

def best_split(X, y, categorical_indices, numeric_indices):
    best_feature = None
    best_info_gain = -1
    best_threshold = None
    is_categorical = True

    for idx in categorical_indices:
        ig = information_gain_categorical(X, y, idx)
        if ig > best_info_gain:
            best_info_gain = ig
            best_feature = idx
            best_threshold = None
            is_categorical = True

    for idx in numeric_indices:
        ig, median = information_gain_numeric(X, y, idx)
        if ig > best_info_gain:
            best_info_gain = ig
            best_feature = idx
            best_threshold = median
            is_categorical = False

    return best_feature, best_threshold, is_categorical

def build_tree(X, y, depth, max_depth, categorical_indices, numeric_indices):
    y_array = np.array(y)
    if depth == max_depth or len(set(y_array)) == 1:
        prediction = Counter(y_array).most_common(1)[0][0]
        return Node(is_leaf=True, prediction=prediction)

    feature, threshold, is_cat = best_split(X, y_array, categorical_indices, numeric_indices)
    if feature is None:
        prediction = Counter(y_array).most_common(1)[0][0]
        return Node(is_leaf=True, prediction=prediction)

    if is_cat:
        children = {}
        subsets = defaultdict(lambda: ([], []))
        for i in range(len(X)):
            val = X[i][feature]
            subsets[val][0].append(X[i])
            subsets[val][1].append(y[i])
        for val, (sub_X, sub_y) in subsets.items():
            children[val] = build_tree(sub_X, sub_y, depth + 1, max_depth, categorical_indices, numeric_indices)
        return Node(is_leaf=False, feature=feature, children=children)
    else:
        left_X, left_y, right_X, right_y = [], [], [], []
        for i in range(len(X)):
            val = float(X[i][feature])
            if val <= threshold:
                left_X.append(X[i])
                left_y.append(y[i])
            else:
                right_X.append(X[i])
                right_y.append(y[i])
        left_child = build_tree(left_X, left_y, depth + 1, max_depth, categorical_indices, numeric_indices)
        right_child = build_tree(right_X, right_y, depth + 1, max_depth, categorical_indices, numeric_indices)
        return Node(is_leaf=False, feature=feature, threshold=threshold, children={'<=': left_child, '>': right_child})

def predict_one(x, node):
    while not node.is_leaf:
        val = x[node.feature]
        if node.threshold is not None:
            try:
                val = float(val)
                node = node.children['<='] if val <= node.threshold else node.children['>']
            except ValueError:
                break
        else:
            if val in node.children:
                node = node.children[val]
            else:
                break
    return node.prediction

def predict(X, tree):
    return [predict_one(x, tree) for x in X]

def accuracy(y_true, y_pred):
    return np.mean(np.array(y_true) == np.array(y_pred))

def process_labels(data):
    return [1 if row[-1].strip() == '>50K' else 0 for row in data]

def get_feature_indices(header):
    categorical_features = {'workclass', 'education', 'marital.status', 'occupation',
                            'relationship', 'race', 'sex', 'native.country'}
    categorical_indices = [i for i, name in enumerate(header[:-1]) if name.strip() in categorical_features]
    numeric_indices = [i for i, name in enumerate(header[:-1]) if name.strip() not in categorical_features]
    return categorical_indices, numeric_indices

def save_predictions(predictions, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Prediction'])
        for pred in predictions:
            writer.writerow([pred])

def plot_acc(depths, train_accs, test_accs):
    plt.plot(depths, train_accs, label='Train Accuracy', marker='o')
    plt.plot(depths, test_accs, label='Test Accuracy', marker='o')
    plt.xlabel('Tree Depth')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.show()
def parta(train_data_path, validation_data_path, test_data_path, output_path):
    output_path = output_path + "/prediction_a.csv"
    header, train_data = read_data(train_data_path)
    _, valid_data = read_data(validation_data_path)
    _, test_data = read_data(test_data_path)

    categorical_indices, numeric_indices = get_feature_indices(header)

    train_X, train_y = [row[:-1] for row in train_data], process_labels(train_data)
    valid_X, valid_y = [row[:-1] for row in valid_data], process_labels(valid_data)
    test_X, test_y = [row[:-1] for row in test_data], process_labels(test_data)
    depths = [1, 5, 10, 15, 20]
    # test_accs = []
    # train_accs = []
    for depth in depths:
        tree = build_tree(train_X, train_y, 0, depth, categorical_indices, numeric_indices)
        train_acc = accuracy(train_y, predict(train_X, tree))
        test_preds = predict(test_X, tree)
        test_acc = accuracy(test_y, test_preds)
        
        print(f"{depth} {train_acc:.4f} {test_acc:.4f}")
        # test_accs.append(test_acc)
        # train_accs.append(train_acc)
        # plot_acc(depths, train_accs, test_accs)
        save_predictions(test_preds, output_path)

# if __name__ == '__main__':
#     main()
