import numpy as np
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

def read_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    header = lines[0].strip().split(',')
    data = [line.strip().split(',') for line in lines[1:] if line.strip()]
    return header, data

def prepare_data(data, encoder=None, fit_encoder=False):
    X_raw = np.array([row[:-1] for row in data])
    y = np.array([1 if row[-1].strip() == '>50K' else 0 for row in data])
    if fit_encoder:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X = encoder.fit_transform(X_raw)
    else:
        X = encoder.transform(X_raw)
    return X, y, encoder

def save_predictions(predictions, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Prediction'])
        for pred in predictions:
            writer.writerow([pred])

def parte(train_data_path, validation_data_path, test_data_path, output_path):
    output_path = output_path + "/prediction_e.csv"
    header, train_data = read_data(train_data_path)
    _, valid_data = read_data(validation_data_path)
    _, test_data = read_data(test_data_path)

    X_train, y_train, encoder = prepare_data(train_data, fit_encoder=True)
    X_valid, y_valid, _ = prepare_data(valid_data, encoder)
    X_test, y_test, _ = prepare_data(test_data, encoder)

    param_grid = {
        'n_estimators': [50, 150, 250, 350],
        'max_features': [0.1, 0.3, 0.5, 0.7, 1.0],
        'min_samples_split': [2, 4, 6, 8, 10]
    }


    best_oob_accuracy = 0
    best_params = None

    for params in ParameterGrid(param_grid):
        print(f"Params: {params}")
        model = RandomForestClassifier( criterion='entropy', n_estimators=params['n_estimators'],
            max_features=params['max_features'], min_samples_split=params['min_samples_split'],
            oob_score=True, bootstrap=True,
            n_jobs=-1, random_state=42
        )

        model.fit(X_train, y_train)

        if hasattr(model, 'oob_score_'):
            oob_acc = model.oob_score_
            print(f"OOB Accuracy: {oob_acc:.4f}")

            if oob_acc > best_oob_accuracy:
                best_oob_accuracy = oob_acc
                best_params = params

    print(f"\nBest Parameters: {best_params}")
    print(f"Best OOB Accuracy: {best_oob_accuracy:.4f}")

    final_model = RandomForestClassifier(
        criterion='entropy', n_estimators=best_params['n_estimators'], max_features=best_params['max_features'],
        min_samples_split=best_params['min_samples_split'], oob_score=False,
        bootstrap=True, n_jobs=-1, random_state=42
    )
    final_model.fit(X_train, y_train)

    train_preds = final_model.predict(X_train)
    valid_preds = final_model.predict(X_valid)
    test_preds = final_model.predict(X_test)

    train_acc = accuracy_score(y_train, train_preds)
    valid_acc = accuracy_score(y_valid, valid_preds)
    test_acc = accuracy_score(y_test, test_preds)

    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {valid_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    save_predictions(test_preds, output_path)

# if __name__ == '__main__':
#     main()
