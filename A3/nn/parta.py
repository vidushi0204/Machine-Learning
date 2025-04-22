import numpy as np
import os
import cv2
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score

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

    def fit(self, X, y, batch_size, epochs, learning_rate, verbose=True):
        num_samples = X.shape[0]
        for epoch in range(1, epochs + 1):
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

            if verbose and epoch % 1 == 0:
                y_encoded = one_hot_encode(y, self.layer_sizes[-1])
                train_loss = cross_entropy_loss(y_encoded, self.forward(X)[0][-1])
                print(f"Epoch {epoch}, Loss: {train_loss:.4f}")

# --- Data loader assuming preprocessing is done ---

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

def parta(train_data_path, test_data_path, output_folder_path):
    np.random.seed(42)

    X_train, y_train, X_test, y_test = load_preprocessed_gtsrb(train_data_path, test_data_path)

    input_size = 28 * 28 * 3 
    hidden_layers = [50]
    output_size = 43         
    batch_size = 32
    epochs = 200
    learning_rate = 0.01

    nn = NeuralNetwork(input_size, hidden_layers, output_size)
    nn.fit(X_train, y_train, batch_size, epochs, learning_rate)
    
    y_pred = nn.predict(X_test)

    save_predictions(y_pred, output_folder_path)
    
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_accuracy:.4f}")

