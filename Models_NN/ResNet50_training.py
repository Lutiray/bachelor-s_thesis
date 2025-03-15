import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import cv2
from tqdm import tqdm

# Folder paths
BASE_PATH = "Datasets"
CLEAN_CATS = os.path.join(BASE_PATH, "Cat_Normalized")
CLEAN_DOGS = os.path.join(BASE_PATH, "Dog_Normalized")

# Image upload function
def load_images(folder, max_images=1000):
    images, labels = [], []
    label = 0 if "Cat" in folder else 1  # 0 - cats, 1 - dogs

    filenames = os.listdir(folder)[:max_images]  # Only take the first max_images of the files
    
    for filename in tqdm(filenames, desc=f"Loading... {folder}"):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (128, 128))
        img = img / 255.0  # Normalization
        images.append(img)
        labels.append(label)
    
    return np.array(images), np.array(labels)

# Loading data
X_cats, y_cats = load_images(CLEAN_CATS, max_images=1000)
X_dogs, y_dogs = load_images(CLEAN_DOGS, max_images=1000)

X = np.concatenate((X_cats, X_dogs), axis=0)
y = np.concatenate((y_cats, y_dogs), axis=0)

# Split data
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

#  Model creation and compilation 
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
x = Flatten()(base_model.output)
x = Dense(256, activation="relu")(x)
out = Dense(2, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=out)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Training the model
print("\n Training the model ResNet50...")
model.fit(X_train, to_categorical(y_train, 2), epochs=5, batch_size=16, verbose=1)

# Svaing the model
model.save("ResNet50.h5")
print("Training is complete. The model has been saved as 'ResNet50.h5'")
