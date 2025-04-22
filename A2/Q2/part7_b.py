import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class_mapping = {0: "dew", 1: "fogsmog", 2: "frost", 3: "glaze", 4: "hail",
                 5: "lightning", 6: "rain", 7: "rainbow", 8: "rime", 9: "sandstorm", 10: "snow"}
test_dir = "./data/Q2/test"
def show_misclassified_images(file_name):
    misclassified_df = pd.read_csv(file_name)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for i, ax in enumerate(axes.flat):
        img_name = misclassified_df.iloc[i]['image_name']
        true_label_idx = misclassified_df.iloc[i]['y_test']
        pred_label_idx = misclassified_df.iloc[i]['y_pred']
        
        true_label = class_mapping[true_label_idx]
        pred_label = class_mapping[pred_label_idx]
        
        img_path = os.path.join(test_dir, true_label, img_name)
        img = cv2.imread(img_path)
        
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
            ax.imshow(img)
        
        ax.set_title(f"True: {true_label}, Pred: {pred_label}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

show_misclassified_images("misclassified_images_5.csv")
show_misclassified_images("misclassified_images_6.csv")