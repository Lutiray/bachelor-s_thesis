import os
import cv2
import numpy as np

BASE_PATH = "Datasets"

def load_images_from_folder(folder_path, target_size=(128, 128)):
    """Loads images from a specified folder and resizes them."""
    images = []
    filenames = []
    full_path = os.path.join(BASE_PATH, folder_path)
    
    for filename in os.listdir(full_path):
        img_path = os.path.join(full_path, filename)
        img = cv2.imread(img_path)
        
        if img is not None:
            img = cv2.resize(img, target_size)
            images.append(img)
            filenames.append(filename)
    
    return np.array(images), filenames
