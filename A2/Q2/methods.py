import os
import cv2
import numpy as np
def load_images(folder):
    images = []
    labels = []
    class_mapping = {"dew": 0, "fogsmog": 1, "frost": 2, "glaze": 3, "hail": 4,
                     "lightning": 5, "rain": 6, "rainbow": 7, "rime": 8, "sandstorm": 9, "snow": 10}
    for label in class_mapping:
        class_path = f"{folder}/{label}"
        for filename in os.listdir(class_path):
            img = cv2.imread(os.path.join(class_path, filename))
            if img is not None:
                img = cv2.resize(img, (100, 100)) / 255.0  
                images.append(img.flatten())
                labels.append(class_mapping[label])
    return np.array(images), np.array(labels)


def load_images_2(folder):
    images = []
    labels = []
    filenames = []
    
    class_mapping = {"dew": 0, "fogsmog": 1, "frost": 2, "glaze": 3, "hail": 4,
                     "lightning": 5, "rain": 6, "rainbow": 7, "rime": 8, "sandstorm": 9, "snow": 10}
    
    for label in class_mapping:
        class_path = f"{folder}/{label}"
        for filename in os.listdir(class_path):
            img_path = os.path.join(class_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (100, 100)) / 255.0  
                images.append(img.flatten())
                labels.append(class_mapping[label])
                filenames.append(filename)

    return np.array(images), np.array(labels), np.array(filenames)
