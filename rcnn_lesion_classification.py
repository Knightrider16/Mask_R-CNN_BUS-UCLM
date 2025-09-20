import os
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import KFold
import torch.nn.functional as nnF
import matplotlib.pyplot as plt

from engine import train_one_epoch
import utils

# ---------------------------
# Dataset Class for BUS + Masks
# ---------------------------
class BUSDataset(Dataset):
    def __init__(self, dataframe, img_dir, mask_dir, transforms=None):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transforms = transforms

    def __getitem__(self, idx):
        # Get image and mask filenames
        img_name = self.df.iloc[idx]["image"]
        mask_name = self.df.iloc[idx]["mask"]

        # Load image and mask
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        img = Image.open(img_path).convert("RGB")
        mask = np.array(Image.open(mask_path))

        # Get object IDs (1=Benign, 2=Malignant)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids != 0]

        # If no objects, return empty tensors
        if len(obj_ids) == 0:
            masks = torch.zeros((0, mask.shape[0], mask.shape[1]), dtype=torch.uint8)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            # Create masks and labels for each object
            masks = []
            labels = []
            for obj_id in obj_ids:
                m = (mask == obj_id).astype(np.uint8)
                masks.append(m)
                labels.append(obj_id)

            masks = torch.as_tensor(np.stack(masks, axis=0), dtype=torch.uint8)
            labels = torch.as_tensor(labels, dtype=torch.int64)

            # Compute bounding boxes from masks
            boxes = []
            for m in masks:
                pos = np.where(m)
                xmin, xmax = np.min(pos[1]), np.max(pos[1])
                ymin, ymax = np.min(pos[0]), np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])
            boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Create target dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks
        }

        # Apply any transforms
        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target


    def __len__(self):
        return len(self.df)

# ---------------------------
# Transforms
# ---------------------------
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor:
    def __call__(self, image, target):
        image = torchvision.transforms.functional.to_tensor(image)
        return image, target

class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if torch.rand(1) < self.prob:
            image = torchvision.transforms.functional.hflip(image)
            width = image.shape[2]
            boxes = target["boxes"]
            if boxes.numel() > 0:
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                target["boxes"] = boxes
            if "masks" in target and target["masks"].numel() > 0:
                target["masks"] = target["masks"].flip(-1)
        return image, target

def get_transform(train):
    transforms = [ToTensor()]
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)

# ---------------------------
# Model: Mask R-CNN
# ---------------------------
def get_instance_segmentation_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    return model

# ---------------------------
# Custom Evaluation (per class)
# ---------------------------
def evaluate_custom(model, data_loader, device, visualize=False, num_visualize=3):
    model.eval()
    metrics_per_class = {1: {"IoU": [], "Dice": [], "Precision": [], "Recall": []},
                         2: {"IoU": [], "Dice": [], "Precision": [], "Recall": []}}

    vis_count = 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)

            for output, target in zip(outputs, targets):
                if len(output["masks"]) == 0 or target["masks"].numel() == 0:
                    continue

                for cls_id in [1, 2]:
                    pred_masks_cls = [(m[0] > 0.5).cpu().int()
                                      for m, lbl in zip(output["masks"], output["labels"]) if lbl == cls_id]
                    true_masks_cls = [m.cpu().int()
                                      for m, lbl in zip(target["masks"], target["labels"]) if lbl == cls_id]

                    if len(pred_masks_cls) == 0 or len(true_masks_cls) == 0:
                        continue

                    pred_mask = pred_masks_cls[0]
                    true_mask = true_masks_cls[0]

                    if pred_mask.shape != true_mask.shape:
                        pred_mask = nnF.interpolate(
                            pred_mask.unsqueeze(0).unsqueeze(0).float(),
                            size=true_mask.shape,
                            mode="nearest"
                        ).squeeze().int()

                    intersection = (pred_mask & true_mask).sum().item()
                    union = (pred_mask | true_mask).sum().item()
                    tp = intersection
                    fp = pred_mask.sum().item() - tp
                    fn = true_mask.sum().item() - tp

                    iou = intersection / union if union > 0 else 0
                    dice = 2 * intersection / (pred_mask.sum().item() + true_mask.sum().item()) if (pred_mask.sum().item() + true_mask.sum().item()) > 0 else 0
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

                    metrics_per_class[cls_id]["IoU"].append(iou)
                    metrics_per_class[cls_id]["Dice"].append(dice)
                    metrics_per_class[cls_id]["Precision"].append(precision)
                    metrics_per_class[cls_id]["Recall"].append(recall)

                if visualize and vis_count < num_visualize:
                    img = images[0].cpu().permute(1,2,0).numpy()
                    pred_mask_vis = (output["masks"][0,0] > 0.5).cpu().numpy()
                    true_mask_vis = target["masks"][0].cpu().numpy()

                    fig, axes = plt.subplots(1,3, figsize=(12,4))
                    axes[0].imshow(img); axes[0].set_title("Image"); axes[0].axis("off")
                    axes[1].imshow(true_mask_vis); axes[1].set_title("Ground Truth"); axes[1].axis("off")
                    axes[2].imshow(pred_mask_vis); axes[2].set_title("Prediction"); axes[2].axis("off")

                    plt.show(block=False)
                    plt.pause(2)
                    plt.close(fig)
                    vis_count += 1
    class_map = {1: "Benign", 2: "Malignant"}
    final_results = {class_map.get(cls, f"class_{cls}"): {metric: float(np.mean(vals)) if vals else 0
                           for metric, vals in metrics.items()}
                     for cls, metrics in metrics_per_class.items()}
    return final_results

# ---------------------------
# 5-Fold Cross Validation
# ---------------------------
def run_cross_validation(csv_path, img_dir, mask_dir, num_classes, num_epochs):
    df = pd.read_csv(csv_path)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(kfold.split(df)):
        print(f"\n========== Fold {fold+1} ==========")
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

        dataset = BUSDataset(train_df, img_dir, mask_dir, transforms=get_transform(train=True))
        dataset_test = BUSDataset(test_df, img_dir, mask_dir, transforms=get_transform(train=False))

        data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=utils.collate_fn)
        data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=utils.collate_fn)

        model = get_instance_segmentation_model(num_classes)
        model.to(device)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        for epoch in range(num_epochs):
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
            lr_scheduler.step()

        metrics = evaluate_custom(model, data_loader_test, device, visualize=True, num_visualize=3)
        print(f"Fold {fold+1} Results: {metrics}")
        fold_results.append(metrics)

    avg_results = {}
    for cls_name in ["Benign", "Malignant"]:
        avg_results[cls_name] = {
            metric: float(np.mean([fold[cls_name][metric] for fold in fold_results]))
            for metric in fold_results[0][cls_name]
        }

    print("\n========== Final Cross-Validation Results ==========")
    print(avg_results)
    return avg_results

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    csv_path = "BUS_RCNN2/labels.csv"
    img_dir = "BUS_RCNN2/images"
    mask_dir = "BUS_RCNN2/masks"
    run_cross_validation(csv_path, img_dir, mask_dir, num_classes=3, num_epochs=5)
