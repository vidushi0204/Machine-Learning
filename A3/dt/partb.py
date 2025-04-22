import numpy as np
import math
from collections import Counter
import csv
from sklearn.preprocessing import OneHotEncoder
from queue import Queue
import time
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
    counts = np.bincount(y)  
    probs = counts / len(y)  
    return -np.sum(probs * np.log2(probs + 1e-9))  

def information_gain(X, y, feature_idx):
    values = X[:, feature_idx]
    
    median = np.median(values)
    
    left_mask = values <= median
    right_mask = values > median

    left_y = y[left_mask]
    right_y = y[right_mask]

    if len(left_y) == 0 or len(right_y) == 0:
        return 0, median

    H = entropy(y)
    w_left = len(left_y) / len(y)
    w_right = len(right_y) / len(y)
    info_gain = H - (w_left * entropy(left_y) + w_right * entropy(right_y))
    return info_gain, median


def best_split(X, y):
    """Find the best feature and threshold to split on."""
    best_feature = None
    best_info_gain = -1
    best_threshold = None
    
    H = entropy(y) 
    
    for idx in range(X.shape[1]):
        values = X[:, idx]
        
        median = np.median(values)
        
        left_mask = values <= median
        right_mask = values > median
        
        left_y = y[left_mask]
        right_y = y[right_mask]

        if len(left_y) == 0 or len(right_y) == 0:
            continue

        w_left = len(left_y) / len(y)
        w_right = len(right_y) / len(y)
        
        info_gain = H - (w_left * entropy(left_y) + w_right * entropy(right_y))
        
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = idx
            best_threshold = median
    
    return best_feature, best_threshold

def build_tree(X, y, max_depth):
    """Build the decision tree using an iterative breadth-first approach."""
    n_samples = X.shape[0]
    root = Node(is_leaf=False)
    
    queue = Queue()
    queue.put((root, np.arange(n_samples), 0))

    while not queue.empty():
        node, indices, depth = queue.get()
        y_sub = y[indices]

        if depth == max_depth or len(set(y_sub)) == 1:
            node.is_leaf = True
            node.prediction = Counter(y_sub).most_common(1)[0][0]
            continue

        best_feature, best_threshold = best_split(X[indices], y_sub)

        if best_feature is None:
            node.is_leaf = True
            node.prediction = Counter(y_sub).most_common(1)[0][0]
            continue

        
        values = X[indices, best_feature]
        
        left_indices = indices[values <= best_threshold]
        right_indices = indices[values > best_threshold]

        node.feature = best_feature
        node.threshold = best_threshold
        
        left_child = Node(is_leaf=False)
        right_child = Node(is_leaf=False)
        
        node.children['<='] = left_child
        node.children['>'] = right_child
        
        queue.put((left_child, left_indices, depth + 1))
        queue.put((right_child, right_indices, depth + 1))

    return root

def predict_one(x, node):
    """Predict the label for a single sample by traversing the tree."""
    while not node.is_leaf:
        val = x[node.feature]
        node = node.children['<='] if val <= node.threshold else node.children['>']
    return node.prediction

def predict(X, tree):
    """Predict labels for all samples in X."""
    return np.array([predict_one(x, tree) for x in X])

def accuracy(y_true, y_pred):
    """Calculate accuracy of predictions."""
    return np.mean(y_true == y_pred)

def save_predictions(predictions, filename):
    """Save predictions to a CSV file."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Prediction'])
        for pred in predictions:
            writer.writerow([pred])

def partb(train_data_path, validation_data_path, test_data_path, output_path):
    output_path = output_path + "/prediction_b.csv"
    header, train_data = read_data(train_data_path)
    _, valid_data = read_data(validation_data_path)
    _, test_data = read_data(test_data_path)

    X_train_raw = np.array([row[:-1] for row in train_data])
    y_train = np.array([1 if row[-1].strip() == '>50K' else 0 for row in train_data])

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train = encoder.fit_transform(X_train_raw)

    X_valid_raw = np.array([row[:-1] for row in valid_data])
    y_valid = np.array([1 if row[-1].strip() == '>50K' else 0 for row in valid_data])
    X_valid = encoder.transform(X_valid_raw)

    X_test_raw = np.array([row[:-1] for row in test_data])
    y_test = np.array([1 if row[-1].strip() == '>50K' else 0 for row in test_data])
    X_test = encoder.transform(X_test_raw)

    for depth in [25,35,45,55]:
    
        tree = build_tree(X_train, y_train, depth)

        train_preds = predict(X_train, tree)
        # valid_preds = predict(X_valid, tree)
        test_preds = predict(X_test, tree)

        train_acc = accuracy(y_train, train_preds)
        # valid_acc = accuracy(y_valid, valid_preds)
        test_acc = accuracy(y_test, test_preds)
        print(f"{depth} {train_acc:.4f} {test_acc:.4f}")
        
        # print(f"{depth} {train_acc:.4f} {valid_acc:.4f} {test_acc:.4f}")
        
        save_predictions(test_preds, output_path)

# if __name__ == '__main__':
#     start_time = time.time()
#     main()
#     end_time = time.time()
#     print(f"Total runtime: {end_time - start_time:.2f} seconds")