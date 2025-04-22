import numpy as np
import math
from collections import Counter
import csv
from sklearn.preprocessing import OneHotEncoder
from queue import Queue

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

def best_split(X, y):
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
    while not node.is_leaf:
        val = x[node.feature]
        node = node.children['<='] if val <= node.threshold else node.children['>']
    return node.prediction

def predict(X, tree):
    return np.array([predict_one(x, tree) for x in X])

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def save_predictions(predictions, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Prediction'])
        for pred in predictions:
            writer.writerow([pred])

def collect_prunable_nodes(node, path=()):
    prunable = []
    if not node.is_leaf and node.children:
        prunable.append((node, path))
        for direction, child in node.children.items():
            prunable.extend(collect_prunable_nodes(child, path + ((node, direction),)))
    return prunable

def prune_node(node):
    leaf_labels = []
    def collect_labels(n):
        if n.is_leaf:
            leaf_labels.append(n.prediction)
        else:
            for child in n.children.values():
                collect_labels(child)
    collect_labels(node)
    if leaf_labels:
        node.is_leaf = True
        node.prediction = Counter(leaf_labels).most_common(1)[0][0]
        node.children = {}

def post_prune(tree, X_valid, y_valid):
    best_acc = accuracy(y_valid, predict(X_valid, tree))

    while True:
        prunable_nodes = collect_prunable_nodes(tree)
        best_improvement = 0
        best_node_to_prune = None

        for node, _ in prunable_nodes:
            backup_children = node.children
            backup_leaf = node.is_leaf
            backup_prediction = node.prediction

            prune_node(node)
            new_acc = accuracy(y_valid, predict(X_valid, tree))

            improvement = new_acc - best_acc

            if improvement > best_improvement:
                best_improvement = improvement
                best_node_to_prune = node
            else:
                node.is_leaf = backup_leaf
                node.children = backup_children
                node.prediction = backup_prediction

        if best_node_to_prune:
            prune_node(best_node_to_prune)
            best_acc += best_improvement
        else:
            break

    return tree

def partc(train_data_path, validation_data_path, test_data_path, output_path):
    output_path = output_path + "/prediction_c.csv"
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

    for depth in [25, 35, 45, 55]:
        tree = build_tree(X_train, y_train, depth)
        tree = post_prune(tree, X_valid, y_valid)

        train_preds = predict(X_train, tree)
        valid_preds = predict(X_valid, tree)
        test_preds = predict(X_test, tree)

        train_acc = accuracy(y_train, train_preds)
        valid_acc = accuracy(y_valid, valid_preds)
        test_acc = accuracy(y_test, test_preds)

        print(f"{depth} {train_acc:.4f} {valid_acc:.4f} {test_acc:.4f}")
        save_predictions(test_preds, output_path)

# if __name__ == '__main__':
#     main()
