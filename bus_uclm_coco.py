import os
import json
import cv2
from PIL import Image
import numpy as np

# Paths
dataset_dir = "E:/S3/OT/project/BUS_RCNN/images"
masks_dir = "E:/S3/OT/project/BUS_RCNN/masks"  # mask images
output_json = "E:/S3/OT/project/bus_uclm_coco.json"

categories = [{"id": 1, "name": "bus"}]

images = []
annotations = []
ann_id = 0

for img_id, img_name in enumerate(os.listdir(dataset_dir)):
    if not img_name.endswith((".jpg", ".png")):
        continue

    img_path = os.path.join(dataset_dir, img_name)
    width, height = Image.open(img_path).size

    images.append({
        "id": img_id,
        "file_name": img_name,
        "width": width,
        "height": height
    })

    # Corresponding mask
    mask_name = os.path.splitext(img_name)[0] + ".png"  # adjust if masks have different extension
    mask_path = os.path.join(masks_dir, mask_name)
    if not os.path.exists(mask_path):
        continue

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # Binarize just in case
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        contour = contour.flatten().tolist()
        if len(contour) < 6:  # skip too small/invalid polygons
            continue

        x, y, w, h = cv2.boundingRect(np.array(contour).reshape(-1,1,2))
        area = cv2.contourArea(np.array(contour).reshape(-1,1,2))

        annotations.append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": 1,
            "segmentation": [contour],
            "bbox": [x, y, w, h],
            "area": float(area),
            "iscrowd": 0
        })
        ann_id += 1

# Create COCO JSON
coco_dict = {
    "images": images,
    "annotations": annotations,
    "categories": categories
}

with open(output_json, "w") as f:
    json.dump(coco_dict, f, indent=4)

print("COCO JSON with masks created at:", output_json)
