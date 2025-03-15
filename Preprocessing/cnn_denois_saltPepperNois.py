import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm 

BASE_PATH = "Datasets"
NOISY_CATS = os.path.join(BASE_PATH, "Cat_SaltPepper")
NOISY_DOGS = os.path.join(BASE_PATH, "Dog_SaltPepper")
OUTPUT_CATS = os.path.join(BASE_PATH, "Result_Denoising/Cat_CNN_SaltPepperDenois")
OUTPUT_DOGS = os.path.join(BASE_PATH, "Result_Denoising/Dog_CNN_SaltPepperDenois")

os.makedirs(OUTPUT_CATS, exist_ok=True)
os.makedirs(OUTPUT_DOGS, exist_ok=True)

# Load the trained model
model = keras.models.load_model("denoising_cnn_model_saltpepper.h5")
model.compile(optimizer="adam", loss="mean_squared_error")  # Define MSE explicitly

# Function for image processing
def denoise_images(input_folder, output_folder):
    filenames = os.listdir(input_folder)

    print(f"Start processing of {len(filenames)} images in the {input_folder}...")

    for filename in tqdm(filenames, desc=f"Processing... {input_folder}"):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        if img is not None:
            img = cv2.resize(img, (128, 128)) / 255.0  # Normalization
            img = np.expand_dims(img, axis=0)  # Adding batch-dimension

            denoised_img = model.predict(img)[0]  # Apply the neural net
            denoised_img = (denoised_img * 255).astype(np.uint8)  # Back to 0-255.

            cv2.imwrite(os.path.join(output_folder, filename), denoised_img)

    print(f"Processing {input_folder} is complete. Results are saved in the {output_folder}.\n")

# Apply the neural network to noisy images
denoise_images(NOISY_CATS, OUTPUT_CATS)
denoise_images(NOISY_DOGS, OUTPUT_DOGS)

print("Noise removal is complete and the results have been saved.")
