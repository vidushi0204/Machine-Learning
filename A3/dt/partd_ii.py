from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import csv

def read_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    header = lines[0].strip().split(',')
    data = [line.strip().split(',') for line in lines[1:] if line.strip()]
    return header, data

def process_labels(data):
    return np.array([1 if row[-1].strip() == '>50K' else 0 for row in data])


def save_predictions(predictions, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Prediction'])
        for pred in predictions:
            writer.writerow([pred])
def partdii(train_data_path, validation_data_path, test_data_path, output_path):
    output_path = output_path + "/prediction_d.csv"
    header, train_data = read_data(train_data_path)
    _, valid_data = read_data(validation_data_path)
    _, test_data = read_data(test_data_path)
 
    X_train_raw = np.array([row[:-1] for row in train_data])
    y_train = process_labels(train_data)

    X_valid_raw = np.array([row[:-1] for row in valid_data])
    y_valid = process_labels(valid_data)

    X_test_raw = np.array([row[:-1] for row in test_data])
    y_test = process_labels(test_data)

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train = encoder.fit_transform(X_train_raw)
    X_valid = encoder.transform(X_valid_raw)
    X_test = encoder.transform(X_test_raw)

    ccp_alphas = [0.001, 0.01, 0.1, 0.2]

    for alpha in ccp_alphas:
        clf = DecisionTreeClassifier(criterion='entropy', ccp_alpha=alpha, random_state=42)
        clf.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, clf.predict(X_train))
        valid_acc = accuracy_score(y_valid, clf.predict(X_valid))
        test_acc = accuracy_score(y_test, clf.predict(X_test))

        print(f"{alpha:.3f}\t\t{train_acc:.4f}\t\t{valid_acc:.4f}\t\t{test_acc:.4f}")
        save_predictions(clf.predict(X_test), output_path)

