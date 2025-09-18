import os
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import KFold
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import torch.nn.functional as nnF


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
        img_name = self.df.iloc[idx]["image"]
        mask_name = self.df.iloc[idx]["mask"]

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        img = Image.open(img_path).convert("RGB")
        mask = np.array(Image.open(mask_path))

        # Object ids (ignore background=0)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids != 0]

        if len(obj_ids) == 0:
            masks = np.zeros((0, mask.shape[0], mask.shape[1]), dtype=np.uint8)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            masks = mask == obj_ids[:, None, None]
            if masks.ndim == 2:
                masks = masks[None, :, :]

            boxes = []
            for m in masks:
                pos = np.where(m)
                xmin, xmax = np.min(pos[1]), np.max(pos[1])
                ymin, ymax = np.min(pos[0]), np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.ones((len(boxes),), dtype=torch.int64)

        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = int(idx)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.tensor([])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }

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
        image = F.to_tensor(image)
        return image, target

class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if torch.rand(1) < self.prob:
            image = F.hflip(image)
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
# Visualization
# ---------------------------
def visualize_predictions(model, data_loader, device, num_images=5):
    model.eval()
    shown = 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)

            for img, output, target in zip(images, outputs, targets):
                if len(output["masks"]) == 0 or target["masks"].numel() == 0:
                    continue

                # Take first predicted mask
                pred_mask = (output["masks"][0, 0] > 0.5).float()

                # Resize to match GT
                true_mask = target["masks"][0].cpu().float()
                pred_mask = nnF.interpolate(
                    pred_mask.unsqueeze(0).unsqueeze(0),
                    size=true_mask.shape[-2:], mode="nearest"
                ).squeeze().cpu()

                # Plot
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(img.permute(1, 2, 0).cpu())
                axs[0].set_title("Image")
                axs[1].imshow(true_mask, cmap="gray")
                axs[1].set_title("Ground Truth Mask")
                axs[2].imshow(pred_mask, cmap="gray")
                axs[2].set_title("Predicted Mask")
                plt.show()

                shown += 1
                if shown >= num_images:
                    return


# ---------------------------
# Custom Evaluation (IoU, Dice, Precision, Recall)
# ---------------------------
def evaluate_custom(model, data_loader, device):
    model.eval()
    ious, dices, precisions, recalls = [], [], [], []

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)

            for output, target in zip(outputs, targets):
                if len(output["masks"]) == 0 or target["masks"].numel() == 0:
                    continue

                pred_mask = (output["masks"][0, 0] > 0.5).float()

                true_mask = target["masks"][0].cpu().float()
                pred_mask = nnF.interpolate(
                    pred_mask.unsqueeze(0).unsqueeze(0),
                    size=true_mask.shape[-2:], mode="nearest"
                ).squeeze().cpu().int()

                true_mask = true_mask.int()

                intersection = (pred_mask & true_mask).sum().item()
                union = (pred_mask | true_mask).sum().item()
                tp = intersection
                fp = (pred_mask.sum().item() - tp)
                fn = (true_mask.sum().item() - tp)

                iou = intersection / union if union > 0 else 0
                dice = (2 * intersection) / (pred_mask.sum().item() + true_mask.sum().item()) if (pred_mask.sum().item() + true_mask.sum().item()) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0

                ious.append(iou)
                dices.append(dice)
                precisions.append(precision)
                recalls.append(recall)

    return {
        "IoU": np.mean(ious) if ious else 0,
        "Dice": np.mean(dices) if dices else 0,
        "Precision": np.mean(precisions) if precisions else 0,
        "Recall": np.mean(recalls) if recalls else 0
    }


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

        metrics = evaluate_custom(model, data_loader_test, device)
        print(f"Fold {fold+1} Results: {metrics}")
        fold_results.append(metrics)

        # Show some predictions for this fold
        visualize_predictions(model, data_loader_test, device, num_images=5)

    # Average across folds
    final_results = {metric: np.mean([fold[metric] for fold in fold_results]) for metric in fold_results[0]}
    print("\n========== Final Cross-Validation Results ==========")
    print(final_results)
    return final_results


# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    csv_path = "BUS_RCNN/labels.csv"
    img_dir = "BUS_RCNN/images"
    mask_dir = "BUS_RCNN/masks"

    run_cross_validation(csv_path, img_dir, mask_dir, num_classes=2, num_epochs=5)
