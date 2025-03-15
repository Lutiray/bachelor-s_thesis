import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from glob import glob

# Paths
MODEL_PATH = "unet_model.h5"  # Pre-trained model
TEST_IMAGE_FOLDER = "Datasets/images"  # Folder with new X-ray images
OUTPUT_MASKS_FOLDER = "Datasets/Result_Segmentation/unet"  # Output folder for segmented masks

# Ensure the output directory exists
os.makedirs(OUTPUT_MASKS_FOLDER, exist_ok=True)

# Load trained U-Net model
model = load_model(MODEL_PATH)
print("Loaded pre-trained U-Net model.")

# Function to preprocess images
def preprocess_image(image_path, img_size=(256, 256)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, img_size) / 255.0
    img = img.reshape(1, *img_size, 1)
    return img

# Function to apply U-Net model to images
def apply_unet_to_images(image_folder):
    image_paths = glob(os.path.join(image_folder, "*.dcm"))

    for img_path in image_paths:
        filename = os.path.basename(img_path).replace(".dcm", ".png")

        # Preprocess image
        img = preprocess_image(img_path)

        # Predict mask
        predicted_mask = model.predict(img)[0]

        # Convert mask to binary format
        mask = (predicted_mask > 0.5).astype(np.uint8) * 255

        # Save the mask
        output_path = os.path.join(OUTPUT_MASKS_FOLDER, filename)
        cv2.imwrite(output_path, mask)
        print(f"Saved segmented mask: {output_path}")

# Apply segmentation to test images
apply_unet_to_images(TEST_IMAGE_FOLDER)
print("Segmentation completed.")
