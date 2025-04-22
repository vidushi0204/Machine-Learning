import numpy as np
import os
import cv2
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(a):
    return a * (1 - a)


def softmax(z):
    e_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # numerical stability
    return e_z / np.sum(e_z, axis=1, keepdims=True)


def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / m


def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]


class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size):
        self.layer_sizes = [input_size] + hidden_layers + [output_size]
        self.num_layers = len(self.layer_sizes)

        # Initialize weights and biases
        self.weights = []
        self.biases = []
        for i in range(self.num_layers - 1):
            w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * np.sqrt(1. / self.layer_sizes[i])
            b = np.zeros((1, self.layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, X):
        activations = [X]
        zs = []
        for i in range(self.num_layers - 2):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = sigmoid(z)
            zs.append(z)
            activations.append(a)

        z_final = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        a_final = softmax(z_final)
        zs.append(z_final)
        activations.append(a_final)
        return activations, zs

    def backward(self, X, y_true, activations, zs):
        grads_w = [None] * (self.num_layers - 1)
        grads_b = [None] * (self.num_layers - 1)

        m = X.shape[0]
        delta = activations[-1] - y_true  # final layer softmax + cross-entropy

        grads_w[-1] = np.dot(activations[-2].T, delta) / m
        grads_b[-1] = np.sum(delta, axis=0, keepdims=True) / m

        for l in range(self.num_layers - 3, -1, -1):
            delta = np.dot(delta, self.weights[l+1].T) * sigmoid_derivative(activations[l+1])
            grads_w[l] = np.dot(activations[l].T, delta) / m
            grads_b[l] = np.sum(delta, axis=0, keepdims=True) / m

        return grads_w, grads_b

    def update_parameters(self, grads_w, grads_b, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * grads_w[i]
            self.biases[i] -= learning_rate * grads_b[i]

    def predict(self, X):
        a, _ = self.forward(X)
        return np.argmax(a[-1], axis=1)

    def fit(self, X, y, batch_size, epochs, learning_rate, early_stopping=True, patience=2):
        num_samples = X.shape[0]
        best_train_loss = float('inf')
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            # print(epoch)
            perm = np.random.permutation(num_samples)
            X_shuffled = X[perm]
            y_shuffled = y[perm]

            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                y_encoded = one_hot_encode(y_batch, self.layer_sizes[-1])
                activations, zs = self.forward(X_batch)
                grads_w, grads_b = self.backward(X_batch, y_encoded, activations, zs)
                self.update_parameters(grads_w, grads_b, learning_rate)

            # Compute training loss at end of each epoch
            y_encoded_full = one_hot_encode(y, self.layer_sizes[-1])
            train_loss = cross_entropy_loss(y_encoded_full, self.forward(X)[0][-1])
            
            if early_stopping:
                if train_loss < best_train_loss - 1e-5:  # small threshold to avoid floating point wiggle
                    best_train_loss = train_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        # print(f"Early stopping at epoch {epoch} (no improvement in training loss for {patience} epochs).")
                        break

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


def score(report_train, report_test,output_size):
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

def partc(train_data_path, test_data_path, output_folder_path):
    np.random.seed(42)

    X_train, y_train, X_test, y_test = load_preprocessed_gtsrb(train_data_path, test_data_path)

    hidden = [[512],[512,256],[512,256,128],[512,256,128,64]]
    input_size = 28 * 28 * 3 
    output_size = 43         
    batch_size = 32
    epochs = 100
    learning_rate = 0.01

    for h in hidden:
        nn = NeuralNetwork(input_size, h, output_size)
        print(f"Training with hidden layers: {h}")
        nn.fit(X_train, y_train, batch_size, epochs, learning_rate)
        
        y_train_pred = nn.predict(X_train)
        y_test_pred = nn.predict(X_test)

        report_train = classification_report(y_train, y_train_pred, output_dict=True, zero_division=0)
        report_test = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)

        save_classification_report(output_size, report_train, report_test, h)
        score(report_train, report_test,output_size)

        save_predictions(y_test_pred, output_folder_path)