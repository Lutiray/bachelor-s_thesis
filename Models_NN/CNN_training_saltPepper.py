import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
from sklearn.model_selection import train_test_split

# Folder paths
BASE_PATH = "Datasets"
CLEAN_CATS = os.path.join(BASE_PATH, "Cat_Normalized")
CLEAN_DOGS = os.path.join(BASE_PATH, "Dog_Normalized")
NOISY_CATS = os.path.join(BASE_PATH, "Cat_SaltPepper")
NOISY_DOGS = os.path.join(BASE_PATH, "Dog_SaltPepper")

# Image upload function
def load_images(clean_folder, noisy_folder):
    clean_images, noisy_images = [], []

    for filename in os.listdir(clean_folder):
        clean_img_path = os.path.join(clean_folder, filename)
        noisy_img_path = os.path.join(noisy_folder, filename)

        clean_img = cv2.imread(clean_img_path)
        noisy_img = cv2.imread(noisy_img_path)

        if clean_img is not None and noisy_img is not None:
            clean_img = cv2.resize(clean_img, (128, 128)) / 255.0  # Normalization
            noisy_img = cv2.resize(noisy_img, (128, 128)) / 255.0

            clean_images.append(clean_img)
            noisy_images.append(noisy_img)

    return np.array(noisy_images), np.array(clean_images)

# Upload datasets
noisy_cats, clean_cats = load_images(CLEAN_CATS, NOISY_CATS)
noisy_dogs, clean_dogs = load_images(CLEAN_DOGS, NOISY_DOGS)

X = np.concatenate((noisy_cats, noisy_dogs), axis=0)
Y = np.concatenate((clean_cats, clean_dogs), axis=0)

# Split into training and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

# Creating a CNN architecture to remove the noise 
def build_denoising_cnn(input_shape=(128, 128, 3)):
    model = keras.Sequential([
        layers.Conv2D(64, (3, 3), activation="relu", padding="same", input_shape=input_shape),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.UpSampling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),

        layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same")  # 3 channels (RGB)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

# Creating and training the model
model = build_denoising_cnn()
model.summary()

model.fit(X_train, Y_train, epochs=20, batch_size=16, validation_data=(X_test, Y_test))

# Saving the trained model
model.save(os.path.join(BASE_PATH, "denoising_cnn_model_saltpepper.h5"))

print("Training completed, model was.")
