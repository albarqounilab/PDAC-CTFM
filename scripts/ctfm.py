#For full fine-tuning

import os
import glob
import random
import numpy as np
import time
import pandas as pd
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from monai import transforms
from monai.data import Dataset, CacheDataset
from monai.utils import set_determinism
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    auc,
    roc_curve,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import wandb

# Pre-trained SegResNet encoder
from lighter_zoo import SegResNet
import timm

from timm.scheduler import CosineLRScheduler
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, CropForegroundd,
    SpatialPadd, ScaleIntensityRanged, ToTensor, RandZoomd, RandGaussianNoised,
    RandScaleIntensityd, RandShiftIntensityd, RandFlipd, RandRotate90d, OneOf, ScaleIntensityd, Lambda, Lambdad, ToTensord, RandSpatialCropd
)
from skimage import exposure
from monai.transforms import MapTransform
from scipy.stats import wasserstein_distance
from sklearn.metrics import matthews_corrcoef
from sklearn.utils.class_weight import compute_class_weight
import torch.nn.functional as F

# Setup and Reproducibility
seed = 42
set_determinism(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"Current CUDA Device: {torch.cuda.current_device()}")
else:
    print(f"Running on: {device}")

roi_size = (256, 256, 128)

# Logging and Directories
# Override any of these with environment variables before running.
DATA_DIR      = os.environ.get("DATA_DIR",      os.path.join(os.path.dirname(__file__), "..", "data", "all_combine_retrain_files"))
save_dir      = os.environ.get("SAVE_DIR",      os.path.join(os.path.dirname(__file__), "..", "results", "ctfm"))
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "pdac-ctfm")
WANDB_ENTITY  = os.environ.get("WANDB_ENTITY",  None)
os.makedirs(save_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join(save_dir, "ctfm_training.log"),
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CT_FM_Training")
writer = SummaryWriter(log_dir=save_dir)

# ------------------------------
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

# ------------------------------
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


# ------------------------------
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

# Model Definition
freeze_params = False
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
        if freeze_params == True:
            with torch.no_grad():
                features = self.encoder(x)
        else:
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

# ------------------------------
# Load Data
def load_data(label_crop, windowing, augment, roi_size, batch_size):
    # Load the CSV files into DataFrames
    train_df1 = pd.read_csv(os.path.join(DATA_DIR, 'train_ukb.csv'))
    train_df2 = pd.read_csv(os.path.join(DATA_DIR, 'train_berlin.csv'))
    train_df3 = pd.read_csv(os.path.join(DATA_DIR, 'train_g.csv'))

    train_df = pd.concat([train_df1, train_df2, train_df3], ignore_index=True)
    train_df.reset_index(drop=True, inplace=True)

    val_df1 = pd.read_csv(os.path.join(DATA_DIR, 'val_ukb.csv')).assign(center_id=0)
    val_df2 = pd.read_csv(os.path.join(DATA_DIR, 'val_berlin.csv')).assign(center_id=1)
    val_df3 = pd.read_csv(os.path.join(DATA_DIR, 'val_g.csv')).assign(center_id=2)
    val_df4 = pd.read_csv(os.path.join(DATA_DIR, 'val_mel.csv')).assign(center_id=3)
    val_df = pd.concat([val_df1, val_df2, val_df3], ignore_index=True)
    val_df.reset_index(drop=True, inplace=True)

    # Create lists for training data (each entry is a dictionary)
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
    
    # Create lists for validation data 
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

    val_data_list4 = []
    for idx in range(len(val_df4)):
        sample = {
            "image": val_df4.loc[idx, "path"],
            "seg": val_df4.loc[idx, "path_seg"],
            "label": float(val_df4.loc[idx, "N Status"]),
            "center": val_df4.loc[idx, "center"],
            "metadata": val_df4.loc[idx, ['N Status', 'Head_Localization', 'Tail_Localization', 
                                          'T_0', 'T_1', 'T_2', 'T_3', 'T_4']].values.astype('float32'),
            'center_id': val_df4.loc[idx, 'center_id']
        }
        val_data_list4.append(sample)

    # Convert the label for each sample to a tensor with shape [1]
    for sample in train_data_list:
        # Convert the label (a float) to a tensor and unsqueeze to add a dimension.
        sample["label"] = torch.tensor(sample["label"], dtype=torch.long)
    for sample in val_data_list:
        sample["label"] = torch.tensor(sample["label"], dtype=torch.long)

    for sample in val_data_list4:
        sample["label"] = torch.tensor(sample["label"], dtype=torch.long)

    train_transform = create_train_transform(label_crop, windowing, augment, roi_size)
    val_transform = create_val_transform(label_crop, windowing, roi_size)

    train_ds = CacheDataset(
        data=train_data_list,
        transform=train_transform,
        cache_rate=1.0,
        num_workers=8
    )

    val_ds = CacheDataset(
        data=val_data_list,
        transform=val_transform,
        cache_rate=1.0,
        num_workers=4
    )

    val_ds4 = CacheDataset(
        data=val_data_list4,
        transform=val_transform,
        cache_rate=1.0,
        num_workers=4
    )

    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    valloader4 = DataLoader(val_ds4, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Extract all training labels
    all_train_labels = [int(sample["label"].item()) for sample in train_data_list]
    class_labels = np.unique(all_train_labels)

    class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=all_train_labels)

    # Normalize to have mean = 1, preserving ratios
    scaled_weights = class_weights / np.mean(class_weights)

    class_weights_f = torch.tensor(scaled_weights, dtype=torch.float)
    return trainloader, valloader, valloader4, class_weights_f


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

def plot_loss_curves(train_losses, val_losses, epoch_plots_dir, epoch):
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label="Train Loss", marker='o')
    plt.plot(epochs, val_losses, label="Val Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curves up to Epoch {epoch}")
    plt.legend()
    plt.grid(True)
    loss_curve_path = os.path.join(epoch_plots_dir, f"loss_curve_epoch_{epoch}.png")
    plt.savefig(loss_curve_path)
    plt.close()

def plot_kde_class_separation(y_true, y_pred, dataset_name, save_dir, epoch=None, center_name=None, save_preds=False):
    """
    Plots KDE of predicted probabilities for class 1, separated by true labels, and computes Wasserstein distance.
    
    Args:
        y_true: np.array of true labels (0 or 1)
        y_pred: np.array of shape [N, 2] (softmax probabilities)
        dataset_name: str, e.g., 'Train', 'Validation'
        save_dir: str, path to save plots
        epoch: int, optional for naming
        center_name: str, optional for per-institute plots
        save_preds: bool, whether to save predictions to .npz file
    
    Returns:
        float, Wasserstein distance between class distributions (or None if insufficient data)
    """
    if len(y_true) == 0:
        print(f"No data for {dataset_name}")
        return None
    
    prob_class1 = y_pred[:, 1]  # Probability of positive class
    probs_neg = prob_class1[y_true == 0]
    probs_pos = prob_class1[y_true == 1]
    
    if len(probs_neg) == 0 or len(probs_pos) == 0:
        print(f"Insufficient data for one class in {dataset_name}")
        return None
    
    # Plot KDE
    plt.figure(figsize=(8, 6))
    sns.kdeplot(probs_neg, fill=True, color='blue', label='True Class 0 (Negative)')
    sns.kdeplot(probs_pos, fill=True, color='red', label='True Class 1 (Positive)')
    plt.xlabel('Predicted Probability for Class 1')
    plt.ylabel('Density')
    title = f'KDE of Class Separation - {dataset_name}'
    if center_name:
        title += f' ({center_name})'
    if epoch is not None:
        title += f' - Epoch {epoch + 1}'
    plt.title(title)
    plt.legend()
    plt.xlim(0, 1)
    plt.tight_layout()
    
    # Save plot
    filename = f'kde_separation_{dataset_name.lower()}'
    if center_name:
        filename += f'_{center_name.lower()}'
    if epoch is not None:
        filename += f'_epoch_{epoch + 1}'
    filepath = os.path.join(save_dir, f'{filename}.png')
    plt.savefig(filepath)
    plt.close()
    
    print(f'KDE plot saved: {filepath}')
    
    # Compute Wasserstein distance
    overlap_metric = wasserstein_distance(probs_neg, probs_pos)
    print(f'Wasserstein Distance for {dataset_name}: {overlap_metric:.4f}')
    
    # Save predictions if requested
    if save_preds:
        save_filename = f'preds_{dataset_name.lower()}'
        if center_name:
            save_filename += f'_{center_name.lower()}'
        if epoch is not None:
            save_filename += f'_epoch_{epoch + 1}'
        save_path = os.path.join(save_dir, f'{save_filename}.npz')
        np.savez(save_path, 
                 y_true=y_true, 
                 y_pred=y_pred, 
                 probs_class1=prob_class1, 
                 probs_neg=probs_neg, 
                 probs_pos=probs_pos)
        print(f'Predictions saved: {save_path}')
    
    return overlap_metric

class EarlyStopping:
    """Early stops the training if validation metric doesn't improve after a given patience."""
    def __init__(self, patience=7, delta=0.001, mode='max', trace_func=print):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            delta (float): Minimum change to qualify as an improvement.
            mode (str): 'max' for metrics that should increase (e.g. balanced_acc, AUC).
                        'min' for metrics that should decrease (e.g. loss).
            trace_func: Function to print messages (default: print).
        """
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.trace_func = trace_func
        self.best = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = -1

        if mode == 'min':
            self.best = np.inf
        else:
            self.best = -np.inf

    def __call__(self, metric, epoch):
        improvement = (metric - self.best) if self.mode == 'max' else (self.best - metric)

        if improvement >= self.delta:
            self.best = metric
            self.counter = 0
            self.best_epoch = epoch
            self.trace_func(f"  EarlyStopping: metric improved ({self.best:.5f})")
            return False   # keep training

        else:
            self.counter += 1
            self.trace_func(
                f"  EarlyStopping: no improvement for {self.counter}/{self.patience} epochs"
            )
            if self.counter >= self.patience:
                self.trace_func(
                    f"  EarlyStopping: stopping early (best {self.mode} = {self.best:.5f} @ epoch {self.best_epoch+1})"
                )
                self.early_stop = True
            return self.early_stop

def train_one_epoch(model, dataloader, criterion, optimizer, device,epoch,temperature=1.0, max_batches=None, grad_clip=1.0, accumulation_steps=1):
    model.train()
    if len(dataloader) == 0:
        return 0.0, np.array([]), np.array([])
    total_loss = 0.0
    all_labels, all_probs = [], []
    optimizer.zero_grad()
    loop = tqdm(enumerate(dataloader), desc="Training", leave=True, position=0)
    for batch_idx, batch in loop:
        # If debugging, process only up to max_batches (if specified)
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        outputs = model(batch['image'])  # Expected shape: [batch, 1]

        # LOGIT MONITORING
        if batch_idx % 50 == 0:
            print(f"Logits - Mean: {outputs.mean().item():.2f}, Std: {outputs.std().item():.2f}")
        
        loss = criterion(outputs, batch['label'].long().view(-1))
        loss = loss / accumulation_steps  # Scale loss
        loss.backward()
        # --- Gradient Norm Tracking ---
        # Compute global grad norm
        grad_norm = 0.0
        has_nan_grad = False
        for p in model.parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any():
                    has_nan_grad = True
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5

        if batch_idx % 10 == 0 or grad_norm < 1e-5 or grad_norm > 10 or has_nan_grad:
            logger.info(f"Epoch {epoch+1}, Batch {batch_idx}: Grad Norm = {grad_norm:.6f}, NaN={has_nan_grad}")
            wandb.log({"grad_norm": grad_norm, "has_nan_grad": int(has_nan_grad), "epoch": epoch, "batch": batch_idx})
            writer.add_scalar("Grad/Norm", grad_norm, global_step=epoch * len(dataloader) + batch_idx)
        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item() * accumulation_steps  # Multiply back for reporting, reverse scaling

        # Apply softmax with temperature for metrics
        probs = torch.softmax(outputs / temperature, dim=1).detach().cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(batch['label'].long().view(-1).cpu().numpy())

        loop.set_postfix(loss=loss.item() * accumulation_steps)
        torch.cuda.empty_cache()

    num_batches = (batch_idx + 1) if max_batches is None else max_batches
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss, np.array(all_labels), np.array(all_probs)

def validate(model, dataloader, criterion, device,temperature=1.0):
    model.eval()
    if len(dataloader) == 0:
        return 0.0, np.array([]), np.array([])

    total_loss = 0.0
    all_labels, all_probs = [], []
    all_centers=[]

    loop = tqdm(dataloader, desc="Validating", leave=True, position=0)
    with torch.no_grad():
        for batch in loop:
            # assert "val" in os.path.normpath(dataloader.dataset.base_dir).lower(), "Validation data path mismatch"
            if hasattr(dataloader.dataset, 'mode'):
                assert dataloader.dataset.mode == 'val', "Expected validation dataset but got training data"
            # assert dataloader.dataset.mode == 'val', "Expected validation dataset but got training data"

            batch = move_batch_to_device(batch, device)
            centers=batch['center_id'].cpu().numpy()
            outputs = model(batch['image'])
            loss = criterion(outputs, batch['label'].long().view(-1))
            total_loss += loss.item()
            probs = torch.softmax(outputs / temperature, dim=1).detach().cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(batch['label'].long().view(-1).cpu().numpy())
            all_centers.extend(centers)
            loop.set_postfix(loss=loss.item())
            torch.cuda.empty_cache()

    avg_loss = total_loss / len(dataloader)
    return avg_loss, np.array(all_labels), np.array(all_probs), np.array(all_centers)

#Training loop with early stopping
def main():
    num_epochs = 100
    accumulation_steps = 8
    roi_size = (256, 256, 128)
    dropout_rates = [0.1,0.2]
    learning_rates = [5e-4]
    weight_decays = [1e-4, 1e-3]
    data_combinations = [
        "labelcrop_clahewindow_commonaug"
    ]

    batch_sizes = [4]

    optimizer_scheduler_configs = [
        {
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR"
        }
    ]
    temperatures = [0.7]
    feature_dim = 512  # Ensure this matches extracted feature dimensions

    # Base directory for the run
    current_date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(save_dir, f"run_{current_date_time}")
    check_and_create_dir(run_dir)

    for data_comb, dropout_rate, lr, weight_decay, batch_size, opt_sched, temperature in itertools.product(
    data_combinations, dropout_rates, learning_rates, weight_decays, batch_sizes, optimizer_scheduler_configs, temperatures):

        label_crop, windowing, augmentation = parse_data_combination_name(data_comb)
        center = "combined3_combined3"

        # Calculate effective batch size
        effective_batch_size = batch_size * accumulation_steps
        run_name = (
            f"511_finetune_ga_center_bs{batch_size}_ep{num_epochs}_"
            f"{'labelcrop' if label_crop else 'nolabelcrop'}_"
            f"{windowing}window_"
            f"{augmentation}aug_"
            f"{center}_"
            f"dropout{dropout_rate}_lr{lr}_wd{weight_decay}_"
            f"{opt_sched['optimizer']}_{opt_sched['scheduler']}_temp{temperature}"
        )

        run_path = os.path.join(run_dir, run_name)
        check_and_create_dir(run_path)
        
        # Subfolders for this run
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
                "num_epochs": num_epochs,
                "dropout_rate": dropout_rate,
                "learning_rate": lr,
                "weight_decay": weight_decay,
                "data_combination": data_comb,
                "batch_size": batch_size,
                "effective_batch_size": effective_batch_size,
                "optimizer": opt_sched["optimizer"],
                "scheduler": opt_sched["scheduler"],
                "temperature": temperature,
                "augmentation": augmentation,
                "windowing": windowing,
                "center": center,
            }
        )

        early_stopping = EarlyStopping(
            patience=20,
            delta=0.0001,          # minimum improvement required
            mode='max',           # balanced accuracy should increase
            trace_func=logger.info
        )

        print(f"\nTraining on {data_comb} with Dropout: {dropout_rate}, LR: {lr}, Weight Decay: {weight_decay}")

        pretrained_model = SegResNet.from_pretrained("project-lighter/ct_fm_segresnet")
        encoder = pretrained_model.encoder
        model = CTClassificationModel(encoder,dropout_rate=dropout_rate).to(device)
        if freeze_params == True:
            model.encoder.requires_grad_(False)

        if freeze_params == True:
            optimizer = optim.AdamW(
                list(model.fc1.parameters()) + list(model.fc2.parameters()),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        criterion = nn.CrossEntropyLoss()

        trainloader, valloader, valloader_mel, class_weights = load_data(label_crop=label_crop, windowing=windowing, augment=augmentation, roi_size=roi_size,batch_size=batch_size)    

        # Track best metrics
        best_val_loss = float("inf")
        best_auc = -1.0
        best_f1 = -1.0
        best_balacc = -1.0
        best_dor = -1.0
        best_val_wass = -float("inf")
        train_results = []
        train_losses = []
        val_losses = []
        best_epoch_metrics = {}

        center_names = {0: 'UKB', 1: 'berlin', 2: 'goe'} 
        best_center_balacc = {cid: 0.0 for cid in center_names}
        best_center_metrics = {cid: None for cid in center_names}


        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            epoch_plots_dir = os.path.join(plots_dir, f"epoch_{epoch + 1}")
            check_and_create_dir(epoch_plots_dir)

            # accumulation_steps = 8
            # train_loss, y_train, y_train_pred, epoch_train_features = train_one_epoch(model, trainloader, criterion, optimizer, device)
            # val_loss, y_val, y_val_pred, epoch_val_features = validate(model, valloader, criterion, device)
            train_loss, y_train, y_train_pred = train_one_epoch(model, trainloader, criterion, optimizer, device,epoch, temperature=temperature, accumulation_steps=accumulation_steps)
            val_loss, y_val, y_val_pred, centers = validate(model, valloader, criterion, device, temperature=temperature)
            val_loss_mel, y_val_mel, y_val_pred_mel, centers_mel = validate(model, valloader_mel, criterion, device, temperature=temperature)

            scheduler.step()  # Update learning rate
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            train_metrics = calculate_metrics(y_train, y_train_pred)
            val_metrics = calculate_metrics(y_val, y_val_pred)
            val_mel_metrics = calculate_metrics(y_val_mel, y_val_pred_mel)

            # Generate KDE plots and compute Wasserstein distances
            train_wass = plot_kde_class_separation(
                y_train, y_train_pred, 'Train', epoch_plots_dir, epoch=epoch,
                save_preds=False  # Save only for best epoch to save space
            )
            val_wass = plot_kde_class_separation(
                y_val, y_val_pred, 'Validation', epoch_plots_dir, epoch=epoch,
                save_preds=False
            )
            val_mel_wass = plot_kde_class_separation(
                y_val_mel, y_val_pred_mel, 'Validation_Mel', epoch_plots_dir, epoch=epoch,
                save_preds=False
            )

            # Per-center KDE and Wasserstein
            center_wass = {}
            for center_id, center_name in center_names.items():
                mask = centers == center_id
                if np.sum(mask) > 0:
                    center_y_true = y_val[mask]
                    center_y_pred = y_val_pred[mask]
                    center_wass[center_name] = plot_kde_class_separation(
                        center_y_true, center_y_pred, 'Validation', epoch_plots_dir,
                        epoch=epoch, center_name=center_name, save_preds=False
                    )

            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_epoch_{epoch + 1}.pth")
                save_model(model, checkpoint_path)
                logger.info(f"Epoch {epoch + 1}: Checkpoint saved at {checkpoint_path}")
        
            # Save best models for each metric (each if-statement evaluated independently)
            if val_metrics["balanced_accuracy"] > best_balacc:
                best_balacc = val_metrics["balanced_accuracy"]
                best_epoch_metrics = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_metrics["accuracy"],
                    "AUC": val_metrics["auc"],
                    "val_auprc": val_metrics["auprc"],
                    "F1": val_metrics["f1_score"],
                    "val_dor": val_metrics["dor"],
                    "balanced_accuracy": val_metrics["balanced_accuracy"],
                    "precision": val_metrics["precision"],
                    "recall": val_metrics["recall"],
                    'specificity': val_metrics["specificity"],
                    'mcc': val_metrics["mcc"],
                }
                model_balacc_path = os.path.join(models_dir, f"best_model_balacc_epoch_{epoch + 1}_{val_metrics['balanced_accuracy']:.3f}.pth")
                save_model(model, model_balacc_path)
                np.savez(model_balacc_path.replace('.pth', '.npz'), y_true=y_val, y_pred=y_val_pred)
                logger.info(f"Epoch {epoch + 1}: Best Balanced Accuracy model saved with balanced_accuracy = {val_metrics['balanced_accuracy']:.3f}")
            
            if early_stopping(val_metrics["balanced_accuracy"], epoch):
                logger.info(f"Early stopping triggered after epoch {epoch+1}")
                last_model_path = os.path.join(models_dir, f"last_model_epoch_{num_epochs}.pth")
                save_model(model, last_model_path)  # Save the final model using a custom function
                logger.info(f"Final model saved at last epoch - early stopping: {last_model_path}")
                break

            if val_metrics["dor"] > best_dor:
                best_dor = val_metrics["dor"]
                model_dor_path = os.path.join(models_dir, f"best_model_dor_epoch_{epoch + 1}_{val_metrics['dor']:.3f}.pth")
                save_model(model, model_dor_path)
                np.savez(model_dor_path.replace('.pth', '.npz'), y_true=y_val, y_pred=y_val_pred)
                logger.info(f"Epoch {epoch + 1}: Best DOR model saved with dor = {val_metrics['dor']:.3f}")
        
            if val_metrics["f1_score"] > best_f1:
                best_f1 = val_metrics["f1_score"]
                model_f1_path = os.path.join(models_dir, f"best_model_f1_epoch_{epoch + 1}_{val_metrics['f1_score']:.3f}.pth")
                save_model(model, model_f1_path)
                np.savez(model_f1_path.replace('.pth', '.npz'), y_true=y_val, y_pred=y_val_pred)
                logger.info(f"Epoch {epoch + 1}: Best F1 model saved with F1 = {val_metrics['f1_score']:.3f}")

            #Save best model based on Wasserstein distance
            if val_wass is not None and val_wass > best_val_wass:
                best_val_wass = val_wass
                model_wass_path = os.path.join(models_dir, f"best_model_wass_epoch_{epoch + 1}_{val_wass:.4f}.pth")
                save_model(model, model_wass_path)
                np.savez(model_wass_path.replace('.pth', '.npz'), y_true=y_val, y_pred=y_val_pred)
                logger.info(f"Epoch {epoch + 1}: Best Wasserstein model saved with val_wass = {val_wass:.4f}")
        
            if val_metrics["auc"] > best_auc:
                best_auc = val_metrics["auc"]
                model_auc_path = os.path.join(models_dir, f"best_model_auc_epoch_{epoch + 1}_{val_metrics['auc']:.3f}.pth")
                save_model(model, model_auc_path)
                np.savez(model_auc_path.replace('.pth', '.npz'), y_true=y_val, y_pred=y_val_pred)
                logger.info(f"Epoch {epoch + 1}: Best AUC model saved with AUC = {val_metrics['auc']:.3f}")
        
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_loss_path = os.path.join(models_dir, f"best_model_loss_epoch_{epoch + 1}_{val_loss:.4f}.pth")
                save_model(model, model_loss_path)
                np.savez(model_loss_path.replace('.pth', '.npz'), y_true=y_val, y_pred=y_val_pred)
                logger.info(f"Epoch {epoch + 1}: Best Loss model saved with val_loss = {val_loss:.4f}")

            for center_id, center_name in center_names.items():
                mask = centers == center_id
                print(f"Center {center_id} count: {(centers == center_id).sum()}")
                center_y_true = y_val[mask]
                center_y_pred = y_val_pred[mask]
                print(center_name, center_y_true.shape, center_y_pred.shape)
                assert y_val.shape[0] == centers.shape[0]
                assert y_val_pred.shape[0] == centers.shape[0]

                center_dir = os.path.join(epoch_plots_dir, center_name)
                check_and_create_dir(center_dir)
                plot_roc_pr_curves(center_y_true, center_y_pred[:, 1], center_dir)
                center_metrics = calculate_metrics(center_y_true,center_y_pred)
                wandb.log({
                    f"val_loss_{center_name}": val_loss,
                    f"val_accuracy_{center_name}": center_metrics["accuracy"],
                    f"precision_{center_name}": center_metrics["precision"],
                    f"recall_{center_name}": center_metrics["recall"],
                    f"val_dor_{center_name}": center_metrics["dor"],
                    f"val_auprc_{center_name}": center_metrics["auprc"],
                    f"val_auc_{center_name}": center_metrics["auc"],
                    f"val_f1_score_{center_name}": center_metrics["f1_score"],
                    f"val_balanced_acc_{center_name}": center_metrics["balanced_accuracy"],
                    f'specificity_{center_name}': center_metrics["specificity"],
                    f'mcc_{center_name}': center_metrics["mcc"],
                    "epoch": epoch
                })

                center_plot_dir = os.path.join(epoch_plots_dir, f"{center_name}")
                check_and_create_dir(center_plot_dir)

                confusion_matrix_path = os.path.join(center_plot_dir, f"conf_matrix_{center_name}_epoch_{epoch + 1}.png")
                plot_confusion_matrix(center_y_true, center_y_pred, confusion_matrix_path)

                plot_roc_pr_curves(center_y_true, center_y_pred[:, 1], center_plot_dir)

            train_results.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **train_metrics,
                **val_metrics
            })

            logger.info(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, AUC: {val_metrics['auc']:.3f}, F1: {val_metrics['f1_score']:.3f}, Balanced Acc: {val_metrics['balanced_accuracy']:.3f}")
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Validation", val_loss, epoch)
            writer.add_scalar("AUC/Validation", val_metrics["auc"], epoch)
            writer.add_scalar("F1/Validation", val_metrics["f1_score"], epoch)
            writer.add_scalar("BalancedAcc/Validation", val_metrics["balanced_accuracy"], epoch)
    
            # Log confusion matrix to wandb
            y_pred_binary = (y_val_pred[:, 1] >= 0.5).astype(int)
            conf_mat = wandb.plot.confusion_matrix(
                probs=None,
                y_true=(y_val >= 0.5).astype(int).flatten(),
                preds=y_pred_binary,
                class_names=['N0', 'NPlus']
            )
            wandb.log({"conf_mat": conf_mat, "epoch": epoch})

        
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_metrics["accuracy"],
                "val_accuracy": val_metrics["accuracy"],
                "precision": val_metrics["precision"],
                "recall": val_metrics["recall"],
                "train_dor": train_metrics["dor"],
                "val_dor": val_metrics["dor"],
                "train_auprc": train_metrics["auprc"],
                "val_auprc": val_metrics["auprc"],
                "train_auc": train_metrics["auc"],
                "val_auc": val_metrics["auc"],
                "train_f1_score": train_metrics["f1_score"],
                "val_f1_score": val_metrics["f1_score"],
                "train_balanced_acc": train_metrics["balanced_accuracy"],
                "val_balanced_acc": val_metrics["balanced_accuracy"],
                'specificity': val_metrics["specificity"],
                'mcc': val_metrics["mcc"],
                "train_wasserstein": train_wass,
                "val_wasserstein": val_wass,
                "val_mel_wasserstein": val_mel_wass,
                **{f"val_wasserstein_{center_name}": wass for center_name, wass in center_wass.items() if wass is not None},
                "epoch": epoch
            })

            wandb.log({
                "val_loss_mel": val_loss_mel,
                "val_accuracy_mel": val_mel_metrics["accuracy"],
                "val_auc_mel": val_mel_metrics["auc"],
                "val_auprc_mel": val_mel_metrics["auprc"],
                "val_f1_score_mel": val_mel_metrics["f1_score"],
                "val_balanced_acc_mel": val_mel_metrics["balanced_accuracy"],
                "precision_mel": val_mel_metrics["precision"],
                "recall_mel": val_mel_metrics["recall"],
                "val_dor_mel": val_mel_metrics["dor"],
                'specificity_mel': val_mel_metrics["specificity"],
                'mcc_mel': val_mel_metrics["mcc"],
                "epoch": epoch
            })

            mel_plot_dir = os.path.join(epoch_plots_dir, "mel")
            check_and_create_dir(mel_plot_dir)
            confusion_matrix_path_mel = os.path.join(mel_plot_dir, f"conf_matrix_mel_epoch_{epoch + 1}.png")
            plot_confusion_matrix(y_val_mel, y_val_pred_mel, confusion_matrix_path_mel)
            plot_roc_pr_curves(y_val_mel, y_val_pred_mel[:, 1], mel_plot_dir)

            print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, AUC: {val_metrics['auc']:.3f}, F1: {val_metrics['f1_score']:.3f}, Balanced Acc: {val_metrics['balanced_accuracy']:.3f}")
    
            confusion_matrix_path = os.path.join(epoch_plots_dir, f"confusion_matrix_epoch_{epoch + 1}.png")
            plot_confusion_matrix(y_val, y_val_pred, confusion_matrix_path)
            plot_roc_pr_curves(y_val, y_val_pred[:, 1], epoch_plots_dir)
            plot_loss_curves(train_losses, val_losses, epoch_plots_dir, epoch + 1)
        
        # Save final model at last epoch
        last_model_path = os.path.join(models_dir, f"last_model_epoch_{num_epochs}.pth")
        save_model(model, last_model_path)
        logger.info(f"Final model saved at last epoch: {last_model_path}")

        # Save metrics and loss curves for later analysis
        metrics_csv_path = os.path.join(run_path, "metrics.csv")
        pd.DataFrame(train_results).to_csv(metrics_csv_path, index=False)
        np.save(os.path.join(run_path, "train_losses.npy"), np.array(train_losses))
        np.save(os.path.join(run_path, "val_losses.npy"), np.array(val_losses))

        # Log the best validation epoch metrics at the end of training
        wandb.log({
            "best_epoch": best_epoch_metrics["epoch"],
            "best_train_loss": best_epoch_metrics["train_loss"],
            "best_val_loss": best_epoch_metrics["val_loss"],
            "best_val_accuracy": best_epoch_metrics["val_accuracy"],
            "best_val_auc": best_epoch_metrics["AUC"],
            "best_val_auprc": best_epoch_metrics["val_auprc"],
            "best_val_f1_score": best_epoch_metrics["F1"],
            "best_val_dor": best_epoch_metrics["val_dor"],
            "best_val_balanced_accuracy": best_epoch_metrics["balanced_accuracy"],
            "best_val_precision": best_epoch_metrics["precision"],
            "best_val_recall": best_epoch_metrics["recall"],
            'best_val_specificity': best_epoch_metrics["specificity"],
            'best_val_mcc': best_epoch_metrics["mcc"],
            "best_val_wass": best_val_wass,
        })

        # Save the best epoch metrics to a CSV
        best_metrics_csv_path = os.path.join(run_path, "best_epoch_metrics.csv")
        pd.DataFrame([best_epoch_metrics]).to_csv(best_metrics_csv_path, index=False)

        writer.close()
        wandb.finish()
        logger.info(f"Training for {run_name} complete.")

if __name__ == "__main__":
    main()
