# prepare_dataset.py
import os
import cv2
import pandas as pd
import shutil
import numpy as np

# === SET THIS to your BUS-UCLM base folder ===
BASE = r"./BUS_UCLM/BUS-UCLM Breast ultrasound lesion segmentation dataset/BUS-UCLM Breast ultrasound lesion segmentation dataset/BUS-UCLM"
IMG_DIR = os.path.join(BASE, "images")   # ultrasound images
MASK_DIR = os.path.join(BASE, "masks")    # ground truth RGB masks

# Destination folder
DST = "./BUS_RCNN2"
os.makedirs(os.path.join(DST, "images"), exist_ok=True)
os.makedirs(os.path.join(DST, "masks"), exist_ok=True)

print("Using BASE:", BASE)
print("Images:", IMG_DIR)
print("Masks:", MASK_DIR)

def process_mask(mask_path):
    """
    Convert RGB mask into class mask:
    - 0: background
    - 1: benign (green)
    - 2: malignant (red)
    """
    mask = cv2.imread(mask_path)
    if mask is None:
        return None

    # Convert to RGB (cv2 loads as BGR)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    # Initialize class mask
    h, w = mask.shape[:2]
    class_mask = np.zeros((h, w), dtype=np.uint8)

    # Define benign (green) and malignant (red) ranges
    benign_lower = np.array([0, 200, 0], dtype=np.uint8)
    benign_upper = np.array([100, 255, 100], dtype=np.uint8)

    malignant_lower = np.array([200, 0, 0], dtype=np.uint8)
    malignant_upper = np.array([255, 100, 100], dtype=np.uint8)

    # Create masks
    benign_mask = cv2.inRange(mask, benign_lower, benign_upper)
    malignant_mask = cv2.inRange(mask, malignant_lower, malignant_upper)

    # Assign labels
    class_mask[benign_mask > 0] = 1
    class_mask[malignant_mask > 0] = 2

    return class_mask

rows = []
missing = 0

for img_name in os.listdir(IMG_DIR):
    img_path = os.path.join(IMG_DIR, img_name)
    mask_path = os.path.join(MASK_DIR, img_name)

    if not os.path.exists(img_path):
        print(f"Warning: image missing {img_path}")
        missing += 1
        continue

    # Copy image
    dst_img = os.path.join(DST, "images", img_name)
    shutil.copy(img_path, dst_img)

    # Process mask
    if os.path.exists(mask_path):
        class_mask = process_mask(mask_path)
        if class_mask is None:
            im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            h, w = im.shape[:2]
            class_mask = np.zeros((h, w), dtype=np.uint8)
    else:
        im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        h, w = im.shape[:2]
        class_mask = np.zeros((h, w), dtype=np.uint8)
        print(f"Note: mask missing for {img_name}; created empty mask.")

    # Save class mask
    dst_mask = os.path.join(DST, "masks", img_name)
    cv2.imwrite(dst_mask, class_mask)

    rows.append([img_name, img_name])

print(f"Done. Copied {len(rows)} items. Missing images: {missing}")

# Save CSV mapping
out_df = pd.DataFrame(rows, columns=["image", "mask"])
out_df.to_csv(os.path.join(DST, "labels.csv"), index=False)
print("Saved:", os.path.join(DST, "labels.csv"))
print("Prepared dataset at:", DST)
