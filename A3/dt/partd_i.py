import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
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

def partdi(train_data_path, validation_data_path, test_data_path, output_path):
    output_path = output_path + "/prediction_d.csv"
    header, train_data = read_data(train_data_path)
    _, valid_data = read_data(validation_data_path)
    _, test_data = read_data(test_data_path)

    train_X_raw = np.array([row[:-1] for row in train_data])
    valid_X_raw = np.array([row[:-1] for row in valid_data])
    test_X_raw = np.array([row[:-1] for row in test_data])

    train_y = process_labels(train_data)
    valid_y = process_labels(valid_data)
    test_y = process_labels(test_data)

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train = encoder.fit_transform(train_X_raw)
    X_valid = encoder.transform(valid_X_raw)
    X_test = encoder.transform(test_X_raw)

    depths = [25, 35, 45, 55]
    
    for depth in depths:
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=depth, random_state=42)
        clf.fit(X_train, train_y)

        train_acc = accuracy_score(train_y, clf.predict(X_train))
        valid_acc = accuracy_score(valid_y, clf.predict(X_valid))
        test_acc = accuracy_score(test_y, clf.predict(X_test))

        print(f"{depth}\t{train_acc:.4f}\t\t{valid_acc:.4f}\t\t{test_acc:.4f}")
        save_predictions(clf.predict(X_test), output_path)


# if __name__ == '__main__':
#     main()
