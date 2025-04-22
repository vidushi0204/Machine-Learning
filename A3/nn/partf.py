import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import os
import cv2

def load_preprocessed_gtsrb(train_data_path, test_data_path):
    X_train, y_train = [], []

    train_path = os.path.join(train_data_path, 'train')
    class_folders = sorted([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])


    for label, class_folder in enumerate(class_folders):
        folder_path = os.path.join(train_path, class_folder)
        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)
            if img is not None:
                X_train.append(img)
                y_train.append(label)

    X_train = np.array(X_train).astype('float32') / 255.0
    y_train = np.array(y_train)
    X_train = X_train.reshape(X_train.shape[0], -1)

    test_img_dir = os.path.join(test_data_path, 'test')
    labels_df = pd.read_csv(os.path.join(test_data_path, 'test_labels.csv'))

    X_test, y_test = [], []
    for _, row in labels_df.iterrows():
        img_path = os.path.join(test_img_dir, row['image'])
        img = cv2.imread(img_path)
        if img is not None:
            X_test.append(img)
            y_test.append(row['label'])

    X_test = np.array(X_test).astype('float32') / 255.0
    y_test = np.array(y_test)
    X_test = X_test.reshape(X_test.shape[0], -1)

    return X_train, y_train, X_test, y_test

import csv

def save_predictions(predictions, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Prediction'])
        for pred in predictions:
            writer.writerow([pred])

def save_classification_report(output_size, report_train, report_test, h):
    train_metrics = {
        'Class': list(range(output_size)),
        'Precision': [round(report_train[str(i)]['precision'], 4) for i in range(output_size)],
        'Recall': [round(report_train[str(i)]['recall'], 4) for i in range(output_size)],
        'F1 Score': [round(report_train[str(i)]['f1-score'], 4) for i in range(output_size)]
    }
    train_df = pd.DataFrame(train_metrics)
    train_df.to_csv(f"train_{h}.csv", index=False)

    test_metrics = {
        'Class': list(range(output_size)),
        'Precision': [round(report_test[str(i)]['precision'], 4) for i in range(output_size)],
        'Recall': [round(report_test[str(i)]['recall'], 4) for i in range(output_size)],
        'F1 Score': [round(report_test[str(i)]['f1-score'], 4) for i in range(output_size)]
    }
    test_df = pd.DataFrame(test_metrics)
    test_df.to_csv(f"test_{h}.csv", index=False)


def score(report_train, report_test, output_size):
    train_f1_scores = [report_train[str(i)]['f1-score'] for i in range(output_size)]
    avg_f1_train = np.mean(train_f1_scores)
    
    train_precision_scores = [report_train[str(i)]['precision'] for i in range(output_size)]
    avg_precision_train = np.mean(train_precision_scores)
    
    train_recall_scores = [report_train[str(i)]['recall'] for i in range(output_size)]
    avg_recall_train = np.mean(train_recall_scores)

    test_f1_scores = [report_test[str(i)]['f1-score'] for i in range(output_size)]
    avg_f1_test = np.mean(test_f1_scores)
    
    test_precision_scores = [report_test[str(i)]['precision'] for i in range(output_size)]
    avg_precision_test = np.mean(test_precision_scores)
    
    test_recall_scores = [report_test[str(i)]['recall'] for i in range(output_size)]
    avg_recall_test = np.mean(test_recall_scores)

    print(f"{avg_f1_train:.4f} | {avg_precision_train:.4f} | {avg_recall_train:.4f} | {avg_f1_test:.4f} | {avg_precision_test:.4f} | {avg_recall_test:.4f}")


def partf(train_data_path, test_data_path, output_folder_path):
    np.random.seed(42)
    X_train, y_train, X_test, y_test = load_preprocessed_gtsrb(train_data_path, test_data_path)


    hidden_layer = [[512],[512,256],[512,256,128],[512,256,128,64]]
    output_size = 43
    for h in hidden_layer:
        clf = MLPClassifier(hidden_layer_sizes=h, activation='relu', solver='sgd', alpha = 0.0, batch_size=32, learning_rate='invscaling', learning_rate_init=0.01, max_iter = 100, shuffle = True,
                            early_stopping=True, n_iter_no_change=3, random_state=42, verbose=False)
        clf.fit(X_train, y_train)
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        report_train = classification_report(y_train, y_train_pred, output_dict=True, zero_division=0)
        report_test = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)

        save_classification_report(output_size, report_train, report_test, h)
        score(report_train, report_test, output_size)

        save_predictions(y_test_pred, output_folder_path)
