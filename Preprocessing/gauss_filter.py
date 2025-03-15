import os
import cv2
import numpy as np
from Utils.image_loader import load_images_from_folder
from Utils.image_saver import save_image

# Paths to input image folders
INPUT_CAT_FOLDER = "Cat_Normalized"
INPUT_DOG_FOLDER = "Dog_Normalized"

# Paths for noisy images
NOISY_CAT_FOLDER = "Cat_GausNois"
NOISY_DOG_FOLDER = "Dog_GausNois"

# Paths for filtered images
FILTERED_CAT_FOLDER = "Result_Denoising/Cat_GausDenois"
FILTERED_DOG_FOLDER = "Result_Denoising/Dog_GausDenois"

# Ensure output directories exist
for folder in [NOISY_CAT_FOLDER, NOISY_DOG_FOLDER, FILTERED_CAT_FOLDER, FILTERED_DOG_FOLDER]:
    os.makedirs(folder, exist_ok=True)  # Create folder if it doesn't exist

def add_gaussian_noise(image, mean=0, sigma=25):
    """Adds Gaussian noise to an image."""
    noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

def apply_gaussian_filter(image, kernel_size=5):
    """Applies a Gaussian filter to smooth the image."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def process_images(input_folder, noisy_folder, filtered_folder):
    """Loads images, applies Gaussian noise, then filters them and saves results."""
    images, filenames = load_images_from_folder(input_folder)

    for img, filename in zip(images, filenames):
        # Add Gaussian noise
        noisy_img = add_gaussian_noise(img)
        save_image(noisy_img, noisy_folder, filename)

        # Apply Gaussian filter to the noisy image
        filtered_img = apply_gaussian_filter(noisy_img)
        save_image(filtered_img, filtered_folder, filename)

if __name__ == "__main__":
    # Process cats and dogs images together
    process_images(INPUT_CAT_FOLDER, NOISY_CAT_FOLDER, FILTERED_CAT_FOLDER)
    process_images(INPUT_DOG_FOLDER, NOISY_DOG_FOLDER, FILTERED_DOG_FOLDER)
