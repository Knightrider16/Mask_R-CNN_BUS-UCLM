# prepare_dataset.py
import os
import cv2
import pandas as pd
import shutil
import numpy as np

# === SET THIS to your nested BUS-UCLM base folder (update as needed) ===
# Example from your screenshot:
BASE = r"./BUS_UCLM/BUS-UCLM Breast ultrasound lesion segmentation dataset/BUS-UCLM Breast ultrasound lesion segmentation dataset/BUS-UCLM"
IMG_DIR = os.path.join(BASE, "images")
MASK_DIR = os.path.join(BASE, "masks")
CSV_FILE = os.path.join(BASE, "INFO.csv")

DST = "./BUS_RCNN"
os.makedirs(os.path.join(DST, "images"), exist_ok=True)
os.makedirs(os.path.join(DST, "masks"), exist_ok=True)

print("Using BASE:", BASE)
print("Images:", IMG_DIR)
print("Masks:", MASK_DIR)
print("CSV:", CSV_FILE)

# read semicolon-separated CSV
df = pd.read_csv(CSV_FILE, sep=';')
print("INFO.csv columns:", df.columns.tolist())
print("First rows:\n", df.head())

label_map = {"normal": 0, "benign": 1, "malignant": 2}

def process_mask(mask_path, label_id, fallback_size=None):
    """
    Read color mask (BGR via cv2) and produce a binary mask (255 everywhere lesion).
    - If masked color is not found (e.g., variation in colors), fall back to non-black pixels.
    """
    mask = cv2.imread(mask_path)
    if mask is None:
        # if mask missing, return None so caller can decide
        return None

    # If already single-channel binary or grayscale
    if len(mask.shape) == 2 or mask.shape[2] == 1:
        gray = mask if len(mask.shape) == 2 else cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        return binary

    # mask is color (BGR)
    # try to detect green and red specifically (BGR order)
    if label_id == 1:  # benign => green (BGR approx [0,255,0])
        lower = np.array([0, 200, 0], dtype=np.uint8)
        upper = np.array([100, 255, 100], dtype=np.uint8)
        binary = cv2.inRange(mask, lower, upper)
        if np.count_nonzero(binary) == 0:
            # fallback: any non-black pixel
            gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        return binary

    elif label_id == 2:  # malignant => red (BGR approx [0,0,255])
        lower = np.array([0, 0, 200], dtype=np.uint8)
        upper = np.array([100, 100, 255], dtype=np.uint8)
        binary = cv2.inRange(mask, lower, upper)
        if np.count_nonzero(binary) == 0:
            gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        return binary

    else:  # Normal: produce empty mask (all zeros)
        h, w = mask.shape[:2]
        return np.zeros((h, w), dtype=np.uint8)

rows = []
missing_imgs = 0
for idx, row in df.iterrows():
    img_name = str(row["Image"]).strip()
    label_text = str(row["Label"]).strip().lower()
    label_id = label_map.get(label_text, 0)

    src_img = os.path.join(IMG_DIR, img_name)
    src_mask = os.path.join(MASK_DIR, img_name)  # assume mask has same filename

    dst_img = os.path.join(DST, "images", img_name)
    dst_mask = os.path.join(DST, "masks", img_name)

    if not os.path.exists(src_img):
        print(f"Warning: image not found: {src_img}")
        missing_imgs += 1
        continue

    # copy image
    shutil.copy(src_img, dst_img)

    # process mask: if mask file missing, create a zero mask with image size
    if os.path.exists(src_mask):
        binary = process_mask(src_mask, label_id)
        if binary is None:
            # fallback: create zero mask using image size
            im = cv2.imread(src_img, cv2.IMREAD_GRAYSCALE)
            h, w = im.shape[:2]
            binary = np.zeros((h, w), dtype=np.uint8)
    else:
        # no mask file: create zero mask
        im = cv2.imread(src_img, cv2.IMREAD_GRAYSCALE)
        if im is None:
            h, w = 0, 0
        else:
            h, w = im.shape[:2]
        binary = np.zeros((h, w), dtype=np.uint8)
        print(f"Note: mask missing for {img_name}; created empty mask.")

    # save binary mask
    if binary is not None and binary.size > 0:
        cv2.imwrite(dst_mask, binary)
    rows.append([img_name, img_name, label_id])

print(f"Done. copied {len(rows)} items. missing images: {missing_imgs}")

out_df = pd.DataFrame(rows, columns=["image", "mask", "label"])
out_df.to_csv(os.path.join(DST, "labels.csv"), index=False)
print("Saved", os.path.join(DST, "labels.csv"))
print("Prepared dataset at:", DST)
