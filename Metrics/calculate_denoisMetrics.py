import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from tqdm import tqdm

BASE_PATH = "Datasets"

ORIGINAL_CATS = os.path.join(BASE_PATH, "Cat_Normalized")
ORIGINAL_DOGS = os.path.join(BASE_PATH, "Dog_Normalized")

GAUS_CATS = os.path.join(BASE_PATH, "Result_Denoising/Cat_GausDenois")
GAUS_DOGS = os.path.join(BASE_PATH, "Result_Denoising/Cat_GausDenois")

MEDIAN_CATS = os.path.join(BASE_PATH, "Result_Denoising/Cat_SaltPepperDenois")
MEDIAN_DOGS = os.path.join(BASE_PATH, "Result_Denoising/Dog_SaltPepperDenois")

CNN_GAUS_CATS = os.path.join(BASE_PATH, "Result_Denoising/Cat_CNN_GausDenois")
CNN_GAUS_DOGS = os.path.join(BASE_PATH, "Result_Denoising/Dog_CNN_GausDenois")

CNN_MEDIAN_CATS = os.path.join(BASE_PATH, "Result_Denoising/Cat_CNN_SaltPepperDenois")
CNN_MEDIAN_DOGS = os.path.join(BASE_PATH, "Result_Denoising/Dog_CNN_SaltPepperDenois")

# Function for calculating metrics
def calculate_avg_metrics_combined(original_folder_cats, original_folder_dogs, 
                                   processed_folder_cats, processed_folder_dogs):
    psnr_list, ssim_list, mse_list = [], [], []
    filenames_1 = os.listdir(original_folder_cats)
    filenames_2 = os.listdir(original_folder_dogs)
    
    filenames = filenames_1 + filenames_2

    for filename in tqdm(filenames, desc=f"Processing..."):
        original_path_1 = os.path.join(original_folder_cats, filename)
        original_path_2 = os.path.join(original_folder_dogs, filename)
        
        processed_path_1 = os.path.join(processed_folder_cats, filename)
        processed_path_2 = os.path.join(processed_folder_dogs, filename)
        
        # Open images from both folders
        original = None
        processed = None
        
        if os.path.exists(original_path_1):
            original = cv2.imread(original_path_1)
            processed = cv2.imread(processed_path_1)
        
        if os.path.exists(original_path_2) and original is None:
            original = cv2.imread(original_path_2)
            processed = cv2.imread(processed_path_2)

        if original is not None and processed is not None:
            original = cv2.resize(original, (128, 128))
            processed = cv2.resize(processed, (128, 128))

            psnr_list.append(psnr(original, processed, data_range=255))
            ssim_list.append(ssim(original, processed, data_range=255, channel_axis=-1, win_size=7))
            mse_list.append(mse(original, processed))

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_mse = np.mean(mse_list)

    return avg_psnr, avg_ssim, avg_mse

# Calculate the averages of the metrics for salt and pepper noise
gaus_psnr, gaus_cats_ssim, gaus_mse = calculate_avg_metrics_combined(ORIGINAL_CATS, ORIGINAL_DOGS, GAUS_CATS, GAUS_DOGS)
cnn_gaus_psnr, cnn_gaus_ssim, cnn_gaus_mse = calculate_avg_metrics_combined(ORIGINAL_CATS, ORIGINAL_DOGS, CNN_GAUS_CATS, CNN_GAUS_DOGS)

# Calculate the averages of the metrics for salt and pepper noise
median_psnr, median_cats_ssim, median_mse = calculate_avg_metrics_combined(ORIGINAL_CATS, ORIGINAL_DOGS, MEDIAN_CATS, MEDIAN_DOGS)
cnn_median_psnr, cnn_median_ssim, cnn_median_mse = calculate_avg_metrics_combined(ORIGINAL_CATS, ORIGINAL_DOGS, CNN_MEDIAN_CATS, CNN_MEDIAN_DOGS)

# Create a DataFrame for easy analysis
df = pd.DataFrame({
    "Method": ["Gaussian filter",  "CNN"],
    "PSNR": [gaus_psnr, cnn_gaus_psnr],
    "SSIM": [gaus_cats_ssim, cnn_gaus_ssim],
    "MSE": [gaus_mse, cnn_gaus_mse],
    
    "Method": ["Median filter",  "CNN"],
    "PSNR": [median_psnr, cnn_median_psnr],
    "SSIM": [median_cats_ssim, cnn_median_ssim],
    "MSE": [median_mse, cnn_median_mse]
})

# Save the results to CSV
df.to_csv(os.path.join("metricsDenoising.csv"), index=False)

print("Metrics calculated! The results are saved in metricsDenoising.csv.")