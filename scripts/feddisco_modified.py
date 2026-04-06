import os
import glob
import time
import random
from datetime import datetime
import logging
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (precision_recall_fscore_support, roc_auc_score, accuracy_score, balanced_accuracy_score, confusion_matrix, auc, roc_curve, precision_recall_curve, matthews_corrcoef )
from sklearn.utils.class_weight import compute_class_weight
from monai import transforms
from monai.data import Dataset, CacheDataset, SmartCacheDataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, CropForegroundd, SpatialPadd, ScaleIntensityRanged, ToTensor, RandZoomd, SpatialCropd, RandGaussianNoised,
    RandScaleIntensityd, RandShiftIntensityd, RandFlipd, RandRotate90d, OneOf, ScaleIntensityd, Lambda, Lambdad, ToTensord, RandSpatialCropd)
from monai.utils import set_determinism
from lighter_zoo import SegResNet
from skimage import exposure
from tqdm import tqdm
import wandb
from timm.scheduler import CosineLRScheduler
from monai.transforms import MapTransform
import copy 

#---------------------------------------
# Setup and Reproducibility
seed = 42
set_determinism(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current CUDA Device: {torch.cuda.current_device()}")

roi_size = (256, 256, 128)

# Logging and Directories
# Override any of these with environment variables before running.
DATA_DIR      = os.environ.get("DATA_DIR",      os.path.join(os.path.dirname(__file__), "..", "data", "all_combine_retrain_files"))
save_dir      = os.environ.get("SAVE_DIR",      os.path.join(os.path.dirname(__file__), "..", "results", "feddisco"))
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "pdac-ctfm")
WANDB_ENTITY  = os.environ.get("WANDB_ENTITY",  None)
os.makedirs(save_dir, exist_ok=True)

#Creating per-run directory
current_date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join(save_dir, f"run_{current_date_time}")
os.makedirs(run_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join(run_dir, "fl_adapt_training.log"),
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FL_Training")
writer = SummaryWriter(log_dir=run_dir)

#---------------------------------------
# Helper Functions
def to_plain_tensor(x, device):
    """Converts a MONAI MetaTensor to a plain tensor and moves it to the specified device."""
    if hasattr(x, "as_tensor"):
        return x.as_tensor().to(device)
    else:
        return x.clone().to(device)

def move_batch_to_device(batch, device):
    """Recursively moves all tensor elements in a batch to the target device."""
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    return batch

def check_and_create_dir(directory):
    """Creates a directory if it does not exist or is not writable."""
    if not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        except Exception as e:
            raise OSError(f"Unable to create directory {directory}: {e}")
    elif not os.access(directory, os.W_OK):
        raise PermissionError(f"Directory {directory} is not writable.")

def save_model(model, path):
    """Saves the model's state_dict with error handling."""
    try:
        torch.save(model.state_dict(), path)
        logger.info(f"Model saved successfully to {path}")
    except Exception as e:
        logger.error(f"Error saving model to {path}: {e}")
        temp_path = os.path.join(os.path.dirname(path), "temp_best_model.pth")
        try:
            torch.save(model.state_dict(), temp_path)
            logger.info(f"Model saved successfully to alternative path {temp_path}")
        except Exception as alt_e:
            logger.error(f"Alternative model save also failed: {alt_e}")

class LabelCrop(transforms.Transform):
    def __init__(self, keys, roi_size):
        super().__init__()
        self.keys = keys
        self.roi_size = roi_size
    def __call__(self, data):
        label = data[self.keys[1]]
        center = np.array(np.where(label == 1)).mean(axis=1).astype(float)
        center = center[1:]  # Exclude the first dimension
        if center.size == 0:
            logger.warning("Empty center computed, skipping cropping.")
            return data
        center = tuple(center.astype(np.int16).tolist())
        data['center'] = center
        cropper = transforms.SpatialCropd(keys=["image", "seg"], roi_center=data['center'], roi_size=self.roi_size)
        return cropper(data)

class ApplyCLAHE(MapTransform):
    def __init__(self, keys, clip_limit=0.03):
        super().__init__(keys)
        self.clip_limit = clip_limit
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            # Convert to numpy if it's a torch tensor
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            # Apply CLAHE per slice if needed
            if img.ndim == 3:  # [C, H, W] or [H, W, D]
                # If it's a single channel, apply CLAHE to each slice along the last axis
                img = np.stack([
                    exposure.equalize_adapthist(img[..., i], clip_limit=self.clip_limit)
                    for i in range(img.shape[-1])
                ], axis=-1)
            elif img.ndim == 4:  # [C, H, W, D]
                img = np.stack([
                    np.stack([
                        exposure.equalize_adapthist(img[c, ..., i], clip_limit=self.clip_limit)
                        for i in range(img.shape[-1])
                    ], axis=-1)
                    for c in range(img.shape[0])
                ], axis=0)
            else:
                img = exposure.equalize_adapthist(img, clip_limit=self.clip_limit)
            d[key] = torch.from_numpy(img.astype(np.float32))
        return d

def parse_data_combination_name(name):
    label_crop = "labelcrop" in name
    windowing = None
    for w in ["none", "micad", "narrow", "clahe"]:
        if f"{w}window" in name:
            windowing = w
            break
    augmentation = None
    for aug in ["micad", "common"]:
        if f"{aug}aug" in name:
            augmentation = aug
            break
    return label_crop, windowing, augmentation

#---------------------------------------
# Modular Transform Functions
clip_and_norm = ScaleIntensityRanged(
    keys=["image"],
    a_min=-1000,
    a_max=2000,
    b_min=0.0,
    b_max=1.0,
    clip=True,
)

# Label cropping or not
def get_crop_transform(label_crop, roi_size):
    if label_crop:
        return transforms.Compose([
            LabelCrop(keys=["image", "seg"], roi_size=roi_size),
        ])
    else:
        return transforms.Compose([transforms.CenterSpatialCropd(keys=["image", "seg"], roi_size=roi_size),])


def get_windowing_transform(windowing_type="none"):
    if windowing_type == "micad":
        return transforms.Compose([transforms.ScaleIntensityRanged(
            keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
        ),])
    elif windowing_type == "narrow":
        return transforms.Compose([transforms.ScaleIntensityRanged(
            keys=["image"], a_min=-90, a_max=210, b_min=0.0, b_max=1.0, clip=True
        ),])
    elif windowing_type == "clahe":
        return transforms.Compose([
            clip_and_norm,
            ApplyCLAHE(keys=["image"], clip_limit=0.03),
        ])
    else:
        return transforms.Compose([transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),])


def get_augmentation(aug_type="none"):
    if aug_type == "micad":
        return transforms.Compose([
            transforms.RandSpatialCropd(keys=["image", "seg"], roi_size=roi_size, max_roi_size=roi_size, random_size=False, random_center=False),
            transforms.RandZoomd(keys=["image", "seg"], prob=0.3, min_zoom=1.3, max_zoom=1.5, mode=['area', 'nearest']),
            transforms.RandRotate90d(keys=["image", "seg"], prob=0.10, max_k=3),
            transforms.RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.20),
        ])
    elif aug_type == "common":
        return transforms.Compose([
            transforms.RandZoomd(keys=["image", "seg"], min_zoom=0.9, max_zoom=1.2, mode=("trilinear", "nearest"),
                            align_corners=(True, None), prob=0.20),
            transforms.RandGaussianNoised(keys=["image"], std=0.01, prob=0.15),
            transforms.RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.20),
            transforms.RandFlipd(keys=["image", "seg"], spatial_axis=[0], prob=0.5),
            transforms.RandFlipd(keys=["image", "seg"], spatial_axis=[1], prob=0.5),
            transforms.RandFlipd(keys=["image", "seg"], spatial_axis=[2], prob=0.5),
            transforms.RandRotate90d(keys=["image", "seg"], prob=0.10, max_k=3),
        ])
    else:
        return transforms.Compose([])


#---------------------------------------
# Data Transforms & Dataset
def create_train_transform(label_crop, windowing, augment, roi_size):
    return transforms.Compose([
        transforms.LoadImaged(keys=["image", "seg"], image_only=False),
        transforms.EnsureChannelFirstd(keys=["image", "seg"]),
        transforms.Orientationd(keys=["image", "seg"], axcodes="RAS"),
        transforms.Spacingd(keys=["image", "seg"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        transforms.CropForegroundd(keys=["image", "seg"], source_key="image", allow_smaller=True),
        get_crop_transform(label_crop=label_crop, roi_size=roi_size),
        transforms.SpatialPadd(keys=["image", "seg"], spatial_size=roi_size),
        get_windowing_transform(windowing),
        get_augmentation(augment),
        transforms.ToTensord(keys=["image", "seg"])
    ])

def create_val_transform(label_crop, windowing, roi_size):
    return transforms.Compose([
        transforms.LoadImaged(keys=["image", "seg"], image_only=False),
        transforms.EnsureChannelFirstd(keys=["image", "seg"]),
        transforms.Orientationd(keys=["image", "seg"], axcodes="RAS"),
        transforms.Spacingd(keys=["image", "seg"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        transforms.CropForegroundd(keys=["image", "seg"], source_key="image", allow_smaller=True),
        get_crop_transform(label_crop=label_crop, roi_size=roi_size),
        transforms.SpatialPadd(keys=["image", "seg"], spatial_size=roi_size),
        get_windowing_transform(windowing),
        transforms.ToTensord(keys=["image", "seg"])
    ])

#---------------------------------------
# Metrics and plots
def calculate_metrics(y_true, y_pred):
    """Calculates various metrics for binary classification."""
    try:
        # y_pred is (N, 2) softmax probabilities
        y_pred_binary = (y_pred[:, 1] >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred_binary)
        tn, fp, fn, tp = cm.ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

        bal_acc = balanced_accuracy_score(y_true, y_pred_binary)
        auc_score = roc_auc_score(y_true, y_pred[:, 1])
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred[:, 1])
        auprc_score = auc(recall_vals, precision_vals)
        mcc = matthews_corrcoef(y_true, y_pred_binary)

        dor = (tp / fn) / (fp / tn) if (tp > 0 and fn > 0 and fp > 0 and tn > 0) else 0
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        precision = recall = specificity = f1_score = accuracy = bal_acc = auc_score = auprc_score = mcc = dor = 0
    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1_score,
        "accuracy": accuracy,
        "balanced_accuracy": bal_acc,
        "auc": auc_score,
        "auprc": auprc_score,
        "dor": dor,
        "mcc": mcc
    }

def plot_confusion_matrix(y_true, y_pred, save_path):
    y_pred_binary = (y_pred[:, 1] >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred_binary)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["N0", "N1"], yticklabels=["N0", "N1"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    plt.close()

def plot_roc_pr_curves(y_true, y_pred, save_dir):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_score(y_true, y_pred):.3f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "roc_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(recall_vals, precision_vals, label=f"Precision-Recall Curve (AUPRC = {auc(recall_vals, precision_vals):.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "pr_curve.png"))
    plt.close()

def plot_loss_curves(train_losses, val_losses, round_plots_dir, rounds):
    rounds = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(rounds, train_losses, label="Train Loss", marker='o')
    plt.plot(rounds, val_losses, label="Val Loss", marker='o')
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.title(f"Loss Curves up to Round {rounds[-1]}") # {rounds[-1]} or len(rounds) as rounds is an array
    plt.legend()
    plt.grid(True)
    loss_curve_path = os.path.join(round_plots_dir, f"loss_curve_epoch_{rounds[-1]}.png")
    plt.savefig(loss_curve_path)
    plt.close()

#---------------------------------------
# Model Definition
class CTClassificationModel(nn.Module):
    def __init__(self, encoder, feature_dim=512, dropout_rate=None):
        super(CTClassificationModel, self).__init__()
        self.encoder = encoder
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(feature_dim, feature_dim // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate else nn.Identity()
        self.fc2 = nn.Linear(feature_dim // 2, 2)

    def forward(self, x):
        features = self.encoder(x)
        x = features[-1] if isinstance(features, (list, tuple)) else features
        device = next(self.parameters()).device
        x = to_plain_tensor(x, device)
        x = self.pool(x)
        x = self.flatten(x)
        x = to_plain_tensor(x, device)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

#---------------------------------------
# Helper Functions for FL Aggregation
def dict_weight(state_dict, weight):
    return {k: v * weight for k, v in state_dict.items()}

def dict_add(dict1, dict2):
    return {k: dict1[k] + dict2[k] for k in dict1}

def FedUpdate(global_model, local_models):
    global_dict = global_model.state_dict()
    for local_model in local_models:
        local_model.load_state_dict(global_dict)

def FedAvg(global_model, local_models, client_weights):
    new_model_dict = None
    for i in range(len(local_models)):
        local_dict = local_models[i].state_dict()
        weighted = dict_weight(local_dict, client_weights[i])
        if new_model_dict is None:
            new_model_dict = weighted
        else:
            new_model_dict = dict_add(new_model_dict, weighted)
    global_model.load_state_dict(new_model_dict)

def compute_discrepancy(labels, num_classes=2, eps=1e-12):
    if len(labels) == 0:
        return 0.0
    # Labels: tensor or list of client labels
    dist = np.bincount(labels, minlength=num_classes).astype(float) / len(labels)
    uniform = np.full(num_classes, 1.0 / num_classes)
    dist = torch.from_numpy(dist) + eps  # add epsilon to avoid inf/NaN
    uniform = torch.from_numpy(uniform)
    return F.kl_div(uniform.log(), dist, reduction='batchmean').item()

def compute_weight_drift(local_model, global_model):
    # Compares the last classification layer (fc2) between local and global models
    local_weights = local_model.fc2.weight.data.flatten()
    print(local_model.fc2.weight.data.flatten().shape)
    print(local_model.fc2.weight.shape)
    global_weights = global_model.fc2.weight.data.flatten()
    if local_weights.norm() == 0 or global_weights.norm() == 0:
        return 0.0
    cos_sim = F.cosine_similarity(local_weights.unsqueeze(0), global_weights.unsqueeze(0)).item()
    delta_k = 1 - cos_sim  # Cosine distance
    return delta_k

def FedDiscoAvg(global_model, local_models, client_weights, discrepancies, deltas=None, a=1.0, b=0.0, gamma = 0.0):
    if deltas is None:
        deltas = [0.0] * len(local_models) #Fallback if not using drift
    adjusted_weights = [max(client_weights[i] - a * discrepancies[i] -gamma * deltas[i] + b, 0) for i in range(len(local_models))]
    total = sum(adjusted_weights)
    if total == 0:  # Avoid division by zero
        logger.warning("Adjusted weights sum to zero — falling back to FedAvg using original client weights.")
        adjusted_weights = client_weights  # Fallback to FedAvg
    else:
        adjusted_weights = [w / total for w in adjusted_weights]
    new_model_dict = None
    for i in range(len(local_models)):
        logger.info(f"Client {i}: disc={discrepancies[i]:.3f}, orig_w={client_weights[i]:.3f}, adj_w={adjusted_weights[i]:.3f}")
        local_dict = local_models[i].state_dict()
        weighted = dict_weight(local_dict, adjusted_weights[i])
        if new_model_dict is None:
            new_model_dict = weighted
        else:
            new_model_dict = dict_add(new_model_dict, weighted)
    global_model.load_state_dict(new_model_dict)

#---------------------------------------
# Load Data
def load_data_per_center(label_crop, windowing, augment, roi_size, batch_size):
    centers = ['ukb', 'berlin', 'g']  # 3 centers
    train_loaders, val_loaders = [], []
    client_weights = []  # For aggregation
    discrepancies = []  # For FedDisco
    class_weights_list = []  # Per-center if needed
    train_sizes = []

    for cid, center in enumerate(centers):
        # Load train/val for this center 
        train_df = pd.read_csv(os.path.join(DATA_DIR, f'train_{center}.csv'))
        val_df = pd.read_csv(os.path.join(DATA_DIR, f'val_{center}.csv')).assign(center_id=cid)

        # Create data lists 
        train_data_list = []
        for idx in range(len(train_df)):
            sample = {
                "image": train_df.loc[idx, "path"],
                "seg": train_df.loc[idx, "path_seg"],
                "label": float(train_df.loc[idx, "N Status"]),
                "center": train_df.loc[idx, "center"],
                "metadata": train_df.loc[idx, ['N Status', 'Head_Localization', 'Tail_Localization',
                                               'T_0', 'T_1', 'T_2', 'T_3', 'T_4']].values.astype('float32')
            }
            train_data_list.append(sample)

        val_data_list = []
        for idx in range(len(val_df)):
            sample = {
                "image": val_df.loc[idx, "path"],
                "seg": val_df.loc[idx, "path_seg"],
                "label": float(val_df.loc[idx, "N Status"]),
                "center": val_df.loc[idx, "center"],
                "metadata": val_df.loc[idx, ['N Status', 'Head_Localization', 'Tail_Localization',
                                              'T_0', 'T_1', 'T_2', 'T_3', 'T_4']].values.astype('float32'),
                'center_id': val_df.loc[idx, 'center_id']
            }
            val_data_list.append(sample)

        # Convert labels to tensors
        for sample in train_data_list + val_data_list:
            sample["label"] = torch.tensor(sample["label"], dtype=torch.long)

        # Transforms (your functions)
        train_transform = create_train_transform(label_crop, windowing, augment, roi_size)
        val_transform = create_val_transform(label_crop, windowing, roi_size)

        # Datasets
        train_ds = CacheDataset(data=train_data_list, transform=train_transform, cache_rate=1.0, num_workers=8)
        val_ds = CacheDataset(data=val_data_list, transform=val_transform, cache_rate=1.0, num_workers=4)

        # Loaders
        trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        valloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        train_loaders.append(trainloader)
        train_sizes.append(len(train_ds))

        val_loaders.append(valloader)

        # Client weight: proportional to data size
        client_weights.append(len(train_ds) / sum(len(ld.dataset) for ld in train_loaders))  # Normalize later if needed

        # Discrepancy for FedDisco: from train labels
        train_labels = [int(sample["label"].item()) for sample in train_data_list]
        discrepancies.append(compute_discrepancy(train_labels))

        # Class weights per center (if using weighted loss)
        class_labels = np.unique(train_labels)
        class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=train_labels)
        # Normalize to have mean = 1, preserving ratios
        scaled_weights = class_weights / np.mean(class_weights)
        class_weights_list.append(torch.tensor(scaled_weights, dtype=torch.float))

    total_train = sum(train_sizes)
    if total_train == 0:
        logging.warning("No training data across all centers")
        client_weights = [0.0] * len(centers)
    else:
        client_weights = [s / total_train for s in train_sizes]
    
    total_d = sum(discrepancies)
    if total_d > 0:
        discrepancies = [d/total_d for d in discrepancies]
    else:
        discrepancies = [0.0] * len(discrepancies)

    return train_loaders, val_loaders, client_weights, discrepancies, class_weights_list

#---------------------------------------
# Local Train/Validate (with per-center hyperparams)
# def train_local_epoch(model, dataloader, criterion, optimizer, device, temperature=1.0, gamma=2.0, max_batches=None, grad_clip=1.0, accumulation_steps=1):
def train_local_epoch(model, dataloader, criterion, optimizer, device, temperature=1.0,max_batches=None, grad_clip=1.0, accumulation_steps=1):
    # If using FocalLoss instead of CrossEntropy: criterion = FocalLoss(gamma=gamma)
    # Old script CrossEntropy, so passing criterion as-is (gamma ignored unless switched)
    model.train()
    if len(dataloader) == 0:
        return 0.0, np.array([]), np.array([])
    total_loss = 0.0
    all_labels, all_probs = [], []
    optimizer.zero_grad()
    loop = tqdm(enumerate(dataloader), desc="Local Training", leave=True, position=0)
    for batch_idx, batch in loop:
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        outputs = model(batch['image'])
        if batch_idx % 50 == 0:
            print(f"Logits - Mean: {outputs.mean().item():.2f}, Std: {outputs.std().item():.2f}")
        loss = criterion(outputs, batch['label'].long().view(-1))
        loss = loss / accumulation_steps
        loss.backward()
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item() * accumulation_steps
        probs = torch.softmax(outputs / temperature, dim=1).detach().cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(batch['label'].long().view(-1).cpu().numpy())
        loop.set_postfix(loss=loss.item() * accumulation_steps)
        torch.cuda.empty_cache()
    num_batches = (batch_idx + 1) if max_batches is None else max_batches
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss, np.array(all_labels), np.array(all_probs)

# def validate_local(model, dataloader, criterion, device, temperature=1.0, gamma=2.0):
def validate_local(model, dataloader, criterion, device, temperature=1.0):
    # If using FocalLoss: criterion = FocalLoss(gamma=gamma)
    model.eval()
    if len(dataloader) == 0:
        return 0.0, np.array([]), np.array([])
    total_loss = 0.0
    all_labels, all_probs = [], []
    loop = tqdm(dataloader, desc="Local Validating", leave=True, position=0)
    with torch.no_grad():
        for batch in loop:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch['image'])
            loss = criterion(outputs, batch['label'].long().view(-1))
            total_loss += loss.item()
            probs = torch.softmax(outputs / temperature, dim=1).detach().cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(batch['label'].long().view(-1).cpu().numpy())
            loop.set_postfix(loss=loss.item())
            torch.cuda.empty_cache()
    avg_loss = total_loss / len(dataloader)
    return avg_loss, np.array(all_labels), np.array(all_probs)

#---------------------------------------
# FL Evaluation (Per-Round)
# def evaluate_fl(global_model, val_loaders, criterion, device, temps, gammas, center_names):
def evaluate_fl(global_model, val_loaders, criterion, device, temps, center_names, round_num):
    agg_loss, agg_labels, agg_probs = 0.0, [], []
    per_center_metrics = {}
    per_center_y_true = {}  # Dict for per-center y_true
    per_center_y_pred = {}  # Dict for per-center y_pred
    num_clients = len(val_loaders)
    total_val_size = sum(len(vl.dataset) for vl in val_loaders)  # For optional weighted agg_loss

    for cid in range(num_clients):
        # local_loss, local_labels, local_probs, _ = validate_local(global_model, val_loaders[cid], criterion, device, temperature=temps[cid], gamma=gammas[cid])
        local_loss, local_labels, local_probs = validate_local(global_model, val_loaders[cid], criterion, device, temperature=temps[cid])

        metrics = calculate_metrics(local_labels, local_probs)
        per_center_metrics[center_names[cid]] = metrics
        per_center_y_true[center_names[cid]] = local_labels
        per_center_y_pred[center_names[cid]] = local_probs
        agg_loss += local_loss * (len(val_loaders[cid].dataset) / total_val_size)
        agg_labels.extend(local_labels)
        agg_probs.extend(local_probs)

        # Log per-center 
        wandb.log({
            f"val_loss_{center_names[cid]}": local_loss,
            f"val_auc_{center_names[cid]}": metrics["auc"],
            f"val_f1_{center_names[cid]}": metrics["f1_score"],
            f"val_balacc_{center_names[cid]}": metrics["balanced_accuracy"],
            f"val_precision_{center_names[cid]}": metrics["precision"],
            f"val_recall_{center_names[cid]}": metrics["recall"],
            f"val_dor_{center_names[cid]}": metrics["dor"],
            f"val_auprc_{center_names[cid]}": metrics["auprc"],
            f"val_accuracy_{center_names[cid]}": metrics["accuracy"],
            f"val_specificity_{center_names[cid]}": metrics["specificity"],
            f"val_mcc_{center_names[cid]}": metrics["mcc"],
            "round": round_num + 1,
        })

    global_metrics = calculate_metrics(np.array(agg_labels), np.array(agg_probs))
    return agg_loss, global_metrics, per_center_metrics,np.array(agg_labels), np.array(agg_probs), per_center_y_true, per_center_y_pred

#---------------------------------------
# Main FL Training Loop
def main_fl(pretrained_path=None, use_feddisco=False, a=1.0, b=0.0, gamma = 0.0):
    num_rounds = 20  # Global rounds
    local_epochs = 3  # Per-round local training 
    accumulation_steps = 8
    roi_size = (256, 256, 128)
    dropout_rate = None #0.1 
    lr = 0.001
    weight_decay = 0.01
    data_comb = "labelcrop_clahewindow_commonaug"
    batch_size = 4
    opt_sched = {"optimizer": "AdamW", "scheduler": "CosineAnnealingLR"}
    temperatures = [0.7, 0.7, 0.7]
    center_names = ['UKB', 'Berlin', 'Goe']

    label_crop, windowing, augmentation = parse_data_combination_name(data_comb)
    effective_batch_size = batch_size * accumulation_steps
    run_name = (
        f"1311_fl_bs{batch_size}_ep{num_rounds}_"
        f"{'labelcrop' if label_crop else 'nolabelcrop'}_"
        f"{windowing}window_"
        f"{augmentation}aug_"
        f"dropout{dropout_rate}_lr{lr}_wd{weight_decay}_"
        f"{opt_sched['optimizer']}_{opt_sched['scheduler']}_feda{a}_fedb{b}_gamma{gamma}"
    )

    run_path = os.path.join(run_dir, run_name)
    check_and_create_dir(run_path)
    # Subfolders
    models_dir = os.path.join(run_path, "models")
    plots_dir = os.path.join(run_path, "plots")
    checkpoints_dir = os.path.join(run_path, "checkpoints")
    check_and_create_dir(models_dir)
    check_and_create_dir(plots_dir)
    check_and_create_dir(checkpoints_dir)

    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        config={
            "num_rounds": num_rounds,
            "local_epochs": local_epochs,
            "dropout_rate": dropout_rate,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "data_combination": data_comb,
            "batch_size": batch_size,
            "effective_batch_size": effective_batch_size,
            "optimizer": opt_sched["optimizer"],
            "scheduler": opt_sched["scheduler"],
            "augmentation": augmentation,
            "windowing": windowing,
            "feddisco_enabled": use_feddisco,
            "feddisco_a": a,
            "feddisco_b": b,
            'gamma':gamma
        }
    )

    train_loaders, val_loaders, client_weights, discrepancies, class_weights_list = load_data_per_center(label_crop, windowing, augmentation, roi_size, batch_size)

    num_clients = len(train_loaders)

    pretrained_model = SegResNet.from_pretrained("project-lighter/ct_fm_segresnet")
    encoder = pretrained_model.encoder
    global_model = CTClassificationModel(encoder, dropout_rate=dropout_rate).to(device)

    local_models = [copy.deepcopy(global_model) for _ in range(num_clients)]
    for local in local_models:
        local.to(device)
        
    local_optimizers = []
    for cid in range(num_clients):
        optimizer = optim.AdamW(local_models[cid].parameters(), lr=lr, weight_decay=weight_decay)
        local_optimizers.append(optimizer)

    criterion = nn.CrossEntropyLoss()
    schedulers = [optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_rounds * local_epochs) for opt in local_optimizers]

    # Track best metrics
    best_global_balacc = -1.0
    best_global_auc = -1.0

    # FL Loop
    for round_num in range(num_rounds):
        print(f"\nFL Round {round_num + 1}/{num_rounds}")

        FedUpdate(global_model, local_models)

        # Local Training (collect train_losses for avg)
        train_losses = []
        deltas = []
        cos_sims=[]
        for cid in range(num_clients):
            local_model = local_models[cid]
            optimizer = local_optimizers[cid]
            trainloader = train_loaders[cid]
            temp = temperatures[cid]
            local_train_loss = 0.0
            for local_ep in range(local_epochs):
                train_loss, y_train, y_train_pred = train_local_epoch(local_model, trainloader, criterion, optimizer, device, temperature=temp, accumulation_steps=accumulation_steps)
                local_train_loss += train_loss
                schedulers[cid].step()
            train_losses.append(local_train_loss / local_epochs)
            torch.cuda.empty_cache()

        avg_train_loss = sum(train_losses) / num_clients if train_losses else 0.0

        deltas = [] #Compute deltas (dynamic drift)
        for local_model in local_models:  
            delta_k = compute_weight_drift(local_model, global_model)
            deltas.append(delta_k)
            cos_sim= 1 - delta_k
            cos_sims.append(cos_sim)
        
        logger.info({"delta_mean": np.mean(deltas), "delta_std": np.std(deltas), "delta_min": np.min(deltas),"delta_max": np.max(deltas),"round": round_num}) 
        wandb.log({"delta_mean": np.mean(deltas), "delta_std": np.std(deltas), "delta_min": np.min(deltas),"delta_max": np.max(deltas), "round": round_num}) 
        for i, delta in enumerate(deltas):
            wandb.log({f"delta_client_{i}": delta, "round": round_num})
            logger.info({f"delta_client_{i}": delta, "round": round_num})

        logger.info({"cos_sim_mean": np.mean(cos_sims), "cos_sim_std": np.std(cos_sims), "cos_sim_min": np.min(cos_sims),"cos_sim_max": np.max(cos_sims),"round": round_num}) 
        wandb.log({"cos_sim_mean": np.mean(cos_sims), "cos_sim_std": np.std(cos_sims), "cos_sim_min": np.min(cos_sims),"cos_sim_max": np.max(cos_sims), "round": round_num}) 
        for i, cs in enumerate(cos_sims):
            wandb.log({f"cos_sim_client_{i}": cs, "round": round_num})
            logger.info({f"cos_sim_client_{i}": cs, "round": round_num})

        # Aggregate
        if use_feddisco: #if True
            FedDiscoAvg(global_model, local_models, client_weights, discrepancies, deltas, a=a, b=b, gamma=gamma)
        else:  #if False
            FedAvg(global_model, local_models, client_weights)

        val_loss, global_metrics, per_center_metrics, agg_labels, agg_probs, per_center_y_true, per_center_y_pred = evaluate_fl(global_model, val_loaders, criterion, device, temperatures, center_names, round_num)

        logger.info(f"Round {round_num + 1}: Avg Train Loss: {avg_train_loss:.4f}, Global Val Loss: {val_loss:.4f}, AUC: {global_metrics['auc']:.3f}, F1: {global_metrics['f1_score']:.3f}, Balanced Acc: {global_metrics['balanced_accuracy']:.3f}")
        # Log per-center summaries (bal_acc and AUC for bias)
        for center in center_names:
            metrics = per_center_metrics[center]
            logger.info(f"Round {round_num + 1} - {center}: Bal Acc: {metrics['balanced_accuracy']:.3f}, AUC: {metrics['auc']:.3f}")

        wandb.log({
            "round": round_num + 1,
            "avg_train_loss": avg_train_loss,
            "global_val_loss": val_loss,
            "global_auc": global_metrics['auc'],
            "global_f1": global_metrics['f1_score'],
            "global_balacc": global_metrics['balanced_accuracy'],
            "global_precision": global_metrics["precision"],
            "global_recall": global_metrics["recall"],
            "global_dor": global_metrics["dor"],
            "global_auprc": global_metrics["auprc"],
            "global_accuracy": global_metrics["accuracy"],
            "global_specificity": global_metrics["specificity"],
            "global_mcc": global_metrics["mcc"],
        })

        y_pred_binary = (agg_probs[:, 1] >= 0.5).astype(int)
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=agg_labels,
                preds=y_pred_binary,
                class_names=['N0', 'N1']
            )
        })

        # Saving Checkpoints every 5 rounds
        if (round_num + 1) % 5 == 0:
            checkpoint_path = os.path.join(checkpoints_dir, f"global_checkpoint_round_{round_num + 1}.pth")
            save_model(global_model, checkpoint_path)
            logger.info(f"Round {round_num + 1}: Checkpoint saved at {checkpoint_path}")

        # Saving Best Models
        if global_metrics['balanced_accuracy'] > best_global_balacc:
            best_global_balacc = global_metrics['balanced_accuracy']
            best_model_path = os.path.join(models_dir, f"best_global_balacc_round_{round_num + 1}_{best_global_balacc:.3f}.pth")
            save_model(global_model, best_model_path)
            logger.info(f"Round {round_num + 1}: Best Bal Acc model saved with {best_global_balacc:.3f}")

        if global_metrics['auc'] > best_global_auc:
            best_global_auc = global_metrics['auc']
            best_model_path = os.path.join(models_dir, f"best_global_auc_round_{round_num + 1}_{best_global_auc:.3f}.pth")
            save_model(global_model, best_model_path)
            logger.info(f"Round {round_num + 1}: Best AUC model saved with {best_global_auc:.3f}")
            
        # Plots dir for round
        round_plots_dir = os.path.join(plots_dir, f"round_{round_num + 1}")
        check_and_create_dir(round_plots_dir)

        # Saving Metrics/Plots (every 5 or on best bal_acc)
        save_plots = (round_num + 1) % 5 == 0 or global_metrics['balanced_accuracy'] > best_global_balacc  # Note: best already updated
        if save_plots:
            # Save aggregated val metrics as npz
            np.savez(os.path.join(plots_dir, f"val_metrics_round_{round_num + 1}.npz"), y_true=agg_labels, y_pred=agg_probs)

            # Aggregated ROC/PR
            plot_roc_pr_curves(agg_labels, agg_probs[:, 1], round_plots_dir)

            conf_mat_path = os.path.join(round_plots_dir, f"conf_matrix_round_{round_num + 1}.png")
            plot_confusion_matrix(agg_labels, agg_probs, conf_mat_path)

        # Per-center plots triggered when a center lags significantly behind global performance
        global_balacc = global_metrics['balanced_accuracy']
        for center in center_names:
            center_balacc = per_center_metrics[center]['balanced_accuracy']
            if center_balacc < 0.8 * global_balacc:
                logger.warning(f"Bias detected in {center}: Bal Acc {center_balacc:.3f} << Global {global_balacc:.3f}")
                # Save per-center plots (assume per_center_y_true, per_center_y_pred from evaluate_fl; add dict returns if needed)
                center_dir = os.path.join(round_plots_dir, center)
                if not os.path.exists(center_dir):    
                    check_and_create_dir(center_dir)
                plot_roc_pr_curves(per_center_y_true[center], per_center_y_pred[center][:, 1], center_dir)

                center_y_pred_binary = (per_center_y_pred[center][:, 1] >= 0.5).astype(int)
                wandb.log({
                    f"conf_matrix_{center}": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=per_center_y_true[center],
                        preds=center_y_pred_binary,
                        class_names=['N0', 'N1']
                    )
                })
        torch.cuda.empty_cache()
    wandb.finish()

if __name__ == "__main__":
    a_values = [0.0]
    b_values = [1.0]
    gamma_values = [1.0]
    for a, b, gamma in itertools.product(a_values, b_values, gamma_values):
        main_fl(pretrained_path=None, use_feddisco=True, a=a, b=b, gamma=gamma)  #use_feddisco=False for FedAvg or just use a=b=gamma=0
