import os
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from PIL import Image
from engine import train_one_epoch, evaluate
import utils
import torchvision.transforms.functional as F

# ---------------------------
# Dataset Class for BUS + Masks
# ---------------------------
class BUSDataset(Dataset):
    def __init__(self, dataframe, img_dir, mask_dir, transforms=None):
        self.df = dataframe
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

        # Get object ids (ignore background = 0)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids != 0]  # drop background

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
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.ones((len(boxes),), dtype=torch.int64)

        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = int(idx)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
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
# Load Data
# ---------------------------
csv_path = "BUS_RCNN/labels.csv"
img_dir = "BUS_RCNN/images"
mask_dir = "BUS_RCNN/masks"

df = pd.read_csv(csv_path)
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

dataset = BUSDataset(train_df, img_dir, mask_dir, transforms=get_transform(train=True))
dataset_test = BUSDataset(test_df, img_dir, mask_dir, transforms=get_transform(train=False))

data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=utils.collate_fn)
data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=utils.collate_fn)


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

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
num_classes = 2  # background + lesion
model = get_instance_segmentation_model(num_classes)
model.to(device)


# ---------------------------
# Optimizer
# ---------------------------
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


# ---------------------------
# Training
# ---------------------------
num_epochs = 10
for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    lr_scheduler.step()
    evaluate(model, data_loader_test, device=device)

# Save model
torch.save(model.state_dict(), "mask_rcnn_bus.pth")
