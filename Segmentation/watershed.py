import os
import cv2
import numpy as np
from tqdm import tqdm

# Folder paths
BASE_PATH = "Datasets"
INPUT_FOLDER = os.path.join(BASE_PATH, "images")
OUTPUT_FOLDER = os.path.join(BASE_PATH, "Result_Segmentation/region_based")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Function for improved mask generation
def create_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Conversion to grayscale
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)  # Increase blur to remove noise
    
    # Add adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Remove noise with morphological processing
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)  

    # Boundary selection (Canny)
    edges = cv2.Canny(closing, 50, 150)  # Boundaries for better mask definition

    # Morphologic operations: dilation and constriction
    sure_bg = cv2.dilate(edges, kernel, iterations=3)  # Expand the background
    sure_fg = cv2.erode(sure_bg, kernel, iterations=3)  # Minimize the mask, removing the excess background
    sure_fg = np.uint8(sure_fg)

    # Create the final mask
    mask = np.zeros_like(gray)
    mask[sure_fg > 0] = 255  # Everything that's found turns white (255)

    return mask

# Generate masks for all images
for filename in tqdm(os.listdir(INPUT_FOLDER), desc="Generation of improved masks"):
    img_path = os.path.join(INPUT_FOLDER, filename)
    img = cv2.imread(img_path)

    if img is not None:
        mask = create_mask(img)
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, filename), mask)

print("Improved mask generation is complete. The masks have been saved.")
