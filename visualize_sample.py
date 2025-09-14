import cv2
import matplotlib.pyplot as plt
import os

# Define paths
img_dir = "BUS_RCNN\images"   # change if needed
mask_dir = "BUS_RCNN\masks"   # change if needed

# Pick a sample filename
sample_name = "ALWI_000.png"

img_path = os.path.join(img_dir, sample_name)
mask_path = os.path.join(mask_dir, sample_name.replace(".png", ".png"))

print("Image path:", img_path)
print("Mask path:", mask_path)

# Load image and mask
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# Safety check
if img is None:
    raise FileNotFoundError(f"Image not found at {img_path}")
if mask is None:
    raise FileNotFoundError(f"Mask not found at {mask_path}")

# Create overlay
overlay = img.copy()
_, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
overlay[mask_bin > 0] = (0, 0, 255)

# Blend images
alpha = 0.5
blended = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)

# Show
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original")

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap="gray")
plt.title("Mask")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
plt.title("Overlay")

plt.show()
