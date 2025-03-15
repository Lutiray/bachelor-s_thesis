import os
import numpy as np
import cv2
from tqdm import tqdm

# Folder paths
BASE_PATH = "Datasets"
images_folder = os.path.join(BASE_PATH, "images")
masks_folder = os.path.join(BASE_PATH,"mask")
unet_folder = os.path.join(BASE_PATH,"Result_Segmentation/unet")
region_based_folder = os.path.join(BASE_PATH,"Result_Segmentation/region_based")

# Function to search for a similar file in a folder
def find_matching_file(folder, filename_base):
    for file in os.listdir(folder):
        if filename_base in file:  # If the image name is in the file
            return os.path.join(folder, file)
    return None  # If nothing is found

# Function for uploading images
def load_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(image_path) else None

# Function for calculating metrics (IoU, Dice, Precision, Recall, F1-score)
def calculate_metrics(mask, prediction):
    if mask.shape != prediction.shape:
        prediction = cv2.resize(prediction, (mask.shape[1], mask.shape[0]))  # Adjusting the size

    mask = (mask > 0).astype(np.uint8)
    prediction = (prediction > 0).astype(np.uint8)

    intersection = np.sum(mask * prediction)
    union = np.sum(mask) + np.sum(prediction) - intersection
    iou = intersection / (union + 1e-6) 
    dice = 2 * intersection / (np.sum(mask) + np.sum(prediction) + 1e-6)
    precision = intersection / (np.sum(prediction) + 1e-6)
    recall = intersection / (np.sum(mask) + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    return iou, dice, precision, recall, f1

iou_unet, dice_unet, precision_unet, recall_unet, f1_unet = [], [], [], [], []
iou_region, dice_region, precision_region, recall_region, f1_region = [], [], [], [], []

for filename in tqdm(os.listdir(images_folder), desc="Processing images"):
    if filename.endswith(".png"):  
        filename_base = filename.replace(".png", "")

        # Looking for files
        mask_path = find_matching_file(masks_folder, filename_base)
        unet_path = find_matching_file(unet_folder, filename_base)
        region_path = find_matching_file(region_based_folder, filename_base)

        if mask_path and unet_path and region_path:
            mask = load_image(mask_path)
            unet_pred = load_image(unet_path)
            region_pred = load_image(region_path)

            if mask is not None and unet_pred is not None and region_pred is not None:
                iou_r, dice_r, prec_r, rec_r, f1_r = calculate_metrics(mask, region_pred)
                iou_u, dice_u, prec_u, rec_u, f1_u = calculate_metrics(mask, unet_pred)

                iou_region.append(iou_r)
                dice_region.append(dice_r)
                precision_region.append(prec_r)
                recall_region.append(rec_r)
                f1_region.append(f1_r)

                iou_unet.append(iou_u)
                dice_unet.append(dice_u)
                precision_unet.append(prec_u)
                recall_unet.append(rec_u)
                f1_unet.append(f1_u)

# Average metrics output
print("Metrics for U-Net:")
print(f"IoU: {np.mean(iou_unet):.4f}, Dice: {np.mean(dice_unet):.4f}, Precision: {np.mean(precision_unet):.4f}, Recall: {np.mean(recall_unet):.4f}, F1-score: {np.mean(f1_unet):.4f}")

print("Metrics for region based method:")
print(f"IoU: {np.mean(iou_region):.4f}, Dice: {np.mean(dice_region):.4f}, Precision: {np.mean(precision_region):.4f}, Recall: {np.mean(recall_region):.4f}, F1-score: {np.mean(f1_region):.4f}")

