import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
import cv2
from tqdm import tqdm

# Folder paths
BASE_PATH = "Datasets"
DATA_SETS = {
    "clean": ("Cat_Normalized", "Dog_Normalized"),
    "noisy": ("Cat_GausNois", "Dog_GausNois"),
    "gauss_filtered": ("Result_Denoising/Cat_GausDenois", "Result_Denoising/Dog_GausDenois"),
    "cnn_filtered": ("Result_Denoising/Cat_CNN_GausDenois", "Result_Denoising/Dog_CNN_GausDenois")
}

# Loading a trained model
print("Loading a trained ResNet50 model...")
model = load_model("ResNet50.h5")
print("The model has been successfully uploaded!")

# Image upload function
def load_images(folder):
    images, labels = [], []
    label = 0 if "Cat" in folder else 1  

    filenames = os.listdir(folder)
    
    for filename in tqdm(filenames, desc=f"Loading: {folder}"):
        img_path = os.path.join(BASE_PATH, folder, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (128, 128))
        img = img / 255.0  # Normalization
        images.append(img)
        labels.append(label)
    
    return np.array(images), np.array(labels)

# Assessing accuracy on different datasets
results = {}

for key, (cat_folder, dog_folder) in DATA_SETS.items():
    print(f"\n🔎 Testing on {key} data...")
    
    X_cats, y_cats = load_images(cat_folder)
    X_dogs, y_dogs = load_images(dog_folder)
    
    X = np.concatenate((X_cats, X_dogs), axis=0)
    y = np.concatenate((y_cats, y_dogs), axis=0)

    # Prediction
    y_pred = np.argmax(model.predict(X), axis=1)
    accuracy = accuracy_score(y, y_pred)
    results[key] = accuracy

    print(f"📊 Accuracy for {key}: {accuracy:.4f}")

# === Итоговые результаты ===
print("Final Test Results:")
for key, acc in results.items():
    print(f"{key}: {acc:.4f}")
