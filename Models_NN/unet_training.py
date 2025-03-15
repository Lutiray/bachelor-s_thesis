import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from glob import glob

# Paths to dataset
IMAGE_FOLDER = "Datasets/images"  # Folder with X-ray images
MASK_FOLDER = "Datasets/mask"    # Folder with segmentation masks

# --- 1. Data Loading ---
def load_data(image_folder, mask_folder, img_size=(256, 256)):
    images, masks = [], []

    for mask_path in glob(os.path.join(mask_folder, "*.png")):
        name = os.path.basename(mask_path).replace("_mask", "").split(".")[0]
        image_path = os.path.join(image_folder, name + ".dcm")

        if not os.path.exists(image_path):
            continue

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, img_size) / 255.0

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, img_size) / 255.0
        mask = np.expand_dims(mask, axis=-1)

        images.append(img)
        masks.append(mask)

    return np.array(images).reshape(-1, *img_size, 1), np.array(masks)

# Load dataset
X, Y = load_data(IMAGE_FOLDER, MASK_FOLDER)

# --- 2. U-Net Model ---
def build_unet(input_shape=(256, 256, 1)):
    inputs = keras.Input(input_shape)

    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)

    # Decoder
    u1 = layers.UpSampling2D((2, 2))(c3)
    u1 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u1)
    u1 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u1)

    u2 = layers.UpSampling2D((2, 2))(u1)
    u2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    u2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u2)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(u2)

    model = keras.Model(inputs, outputs)
    return model

# Compile and train U-Net
model = build_unet()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
model.fit(X, Y, epochs=20, batch_size=8, validation_split=0.2)

# Save the trained model
model.save("unet_model.h5")
print("Training complete. Model saved as 'unet_model.h5'.")
