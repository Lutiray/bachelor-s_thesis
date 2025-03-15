import os
import cv2
import numpy as np
from skimage.util import random_noise
from Utils.image_loader import load_images_from_folder
from Utils.image_saver import save_image

# Paths to input image folders
INPUT_CAT_FOLDER = "Cat_Normalized"
INPUT_DOG_FOLDER = "Dog_Normalized"

# Paths for noisy images
NOISY_CAT_FOLDER = "Cat_SaltPepper"
NOISY_DOG_FOLDER = "Dog_SaltPepper"

# Paths for filtered images
FILTERED_CAT_FOLDER = "Result_Denoising/Cat_MedianDenois"
FILTERED_DOG_FOLDER = "Result_Denoising/Dog_MedianDenois"

# Ensure output directories exist
for folder in [NOISY_CAT_FOLDER, NOISY_DOG_FOLDER, FILTERED_CAT_FOLDER, FILTERED_DOG_FOLDER]:
    os.makedirs(folder, exist_ok=True)  # Create folder if it doesn't exist

def add_salt_and_pepper_noise(image, amount=0.02):
    noisy = random_noise(image, mode='s&p', amount=amount)
    noisy = (noisy * 255).astype(np.uint8)
    return noisy

def apply_median_filter(image, kernel_size=5):
    """Applies a median filter to smooth the image."""
    return cv2.medianBlur(image, kernel_size)

def process_images(input_folder, noisy_folder, filtered_folder):
    """Loads images, applies Gaussian noise, then filters them and saves results."""
    images, filenames = load_images_from_folder(input_folder)

    for img, filename in zip(images, filenames):
        # Add salt and pepper noise
        noisy_img = add_salt_and_pepper_noise(img)
        save_image(noisy_img, noisy_folder, filename)

        # Apply median filter to the noisy image
        filtered_img = apply_median_filter(noisy_img)
        save_image(filtered_img, filtered_folder, filename)

if __name__ == "__main__":
    # Process cats and dogs images together
    process_images(INPUT_CAT_FOLDER, NOISY_CAT_FOLDER, FILTERED_CAT_FOLDER)
    process_images(INPUT_DOG_FOLDER, NOISY_DOG_FOLDER, FILTERED_DOG_FOLDER)
