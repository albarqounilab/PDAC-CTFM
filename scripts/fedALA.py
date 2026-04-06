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
# Adaptive Local Aggregation Module
from typing import List, Tuple

class ALA:
    def __init__(self,
                cid: int,
                loss: nn.Module,
                train_data, 
                batch_size: int, 
                rand_percent: int, 
                layer_idx: int = 0,
                eta: float = 1.0,
                device: str = 'cpu', 
                threshold: float = 0.1,
                num_pre_loss: int = 10) -> None:
        """
        Initialize ALA module adapted for dictionary-based datasets
        """
        self.cid = cid
        self.loss = loss
        self.train_data = train_data
        self.batch_size = batch_size
        self.rand_percent = rand_percent
        self.layer_idx = layer_idx
        self.eta = eta
        self.threshold = threshold
        self.num_pre_loss = num_pre_loss
        self.device = device

        self.weights = None # Learnable local aggregation weights.
        self.start_phase = True


    def adaptive_local_aggregation(self, 
                            global_model: nn.Module,
                            local_model: nn.Module) -> None:
        
        # randomly sample partial local training data
        rand_ratio = self.rand_percent / 100
        rand_num = int(rand_ratio*len(self.train_data))
        
        # FIX for Subset and dictionary dataset
        indices = list(range(len(self.train_data)))
        rand_idx = random.randint(0, len(self.train_data)-rand_num)
        subset_indices = indices[rand_idx:rand_idx+rand_num]
        subset = Subset(self.train_data, subset_indices)
        rand_loader = DataLoader(subset, self.batch_size, drop_last=False)

        # obtain the references of the parameters
        params_g = list(global_model.parameters())
        params = list(local_model.parameters())

        # deactivate ALA at the 1st communication iteration
        if torch.sum(params_g[0] - params[0]) == 0:
            return

        # preserve all the updates in the lower layers
        if self.layer_idx > 0:
            for param, param_g in zip(params[:-self.layer_idx], params_g[:-self.layer_idx]):
                param.data = param_g.data.clone()
        else: # if layer_idx is 0, preserve all layers (which means weight learning on all layers)
            pass

        # temp local model only for weight learning
        model_t = copy.deepcopy(local_model)
        params_t = list(model_t.parameters())

        # only consider higher layers
        if self.layer_idx > 0:
            params_p = params[-self.layer_idx:]
            params_gp = params_g[-self.layer_idx:]
            params_tp = params_t[-self.layer_idx:]
            
            # frozen the lower layers to reduce computational cost in Pytorch
            for param in params_t[:-self.layer_idx]:
                param.requires_grad = False
        else:
            params_p = params
            params_gp = params_g
            params_tp = params_t

        # used to obtain the gradient of higher layers
        # no need to use optimizer.step(), so lr=0
        optimizer = torch.optim.SGD(params_tp, lr=0)

        # initialize the weight to all ones in the beginning
        if self.weights == None:
            self.weights = [torch.ones_like(param.data).to(self.device) for param in params_p]

        # initialize the higher layers in the temp local model
        with torch.no_grad():
            for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp,
                                                    self.weights):
                param_t.data = param + (param_g - param) * weight

        # weight learning
        losses = []  # record losses
        cnt = 0  # weight training iteration counter
        while True:
            for batch in rand_loader:
                x = batch['image'].to(self.device)
                y = batch['label'].long().to(self.device)
                
                optimizer.zero_grad(set_to_none=True)
                output = model_t(x)
                loss_value = self.loss(output, y) # modify according to the local objective
                loss_value.backward()

                # update weight in this batch
                with torch.no_grad():
                    for param_t, param, param_g, weight in zip(params_tp, params_p,
                                                            params_gp, self.weights):
                        weight.data = torch.clamp(
                            weight - self.eta * (param_t.grad * (param_g - param)), 0, 1)

                    # update temp local model in this batch
                    for param_t, param, param_g, weight in zip(params_tp, params_p,
                                                            params_gp, self.weights):
                        param_t.data = param + (param_g - param) * weight

            torch.cuda.empty_cache()
            losses.append(loss_value.item())
            cnt += 1

            # only train one epoch in the subsequent iterations
            if not self.start_phase:
                break

            # train the weight until convergence
            if len(losses) > self.num_pre_loss and np.std(losses[-self.num_pre_loss:]) < self.threshold:
                print('Client:', self.cid, '\tStd:', np.std(losses[-self.num_pre_loss:]),
                    '\tALA epochs:', cnt)
                break

        self.start_phase = False

        # obtain initialized local model
        with torch.no_grad():
            for param, param_t in zip(params_p, params_tp):
                param.data = param_t.data.clone()
            
        del model_t
        torch.cuda.empty_cache()

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
save_dir      = os.environ.get("SAVE_DIR",      os.path.join(os.path.dirname(__file__), "..", "results", "fedala"))
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "pdac-ctfm")
WANDB_ENTITY  = os.environ.get("WANDB_ENTITY",  None)
os.makedirs(save_dir, exist_ok=True)

#Creating per-run directory
current_date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join(save_dir, f"run_{current_date_time}")
os.makedirs(run_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join(run_dir, "fl_adapt2_training.log"),
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
def FedAvg(global_model, local_models, client_weights):
    new_model_dict = None
    for i in range(len(local_models)):
        local_dict = local_models[i].state_dict()
        weighted = {k: v * client_weights[i] for k, v in local_dict.items()}
        if new_model_dict is None:
            new_model_dict = weighted
        else:
            new_model_dict = {k: new_model_dict[k] + weighted[k] for k in new_model_dict}
    global_model.load_state_dict(new_model_dict)

#---------------------------------------
# Load Data
def load_data_per_center(label_crop, windowing, augment, roi_size, batch_size):
    centers = ['ukb', 'berlin', 'g']  # 3 centers
    train_loaders, val_loaders = [], []
    client_weights = []
    class_weights_list = []
    train_sizes = []

    for cid, center in enumerate(centers):
        train_df = pd.read_csv(os.path.join(DATA_DIR, f'train_{center}.csv'))
        val_df = pd.read_csv(os.path.join(DATA_DIR, f'val_{center}.csv')).assign(center_id=cid)

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

        for sample in train_data_list + val_data_list:
            sample["label"] = torch.tensor(sample["label"], dtype=torch.long)

        train_transform = create_train_transform(label_crop, windowing, augment, roi_size)
        val_transform = create_val_transform(label_crop, windowing, roi_size)

        train_ds = CacheDataset(data=train_data_list, transform=train_transform, cache_rate=1.0, num_workers=8)
        val_ds = CacheDataset(data=val_data_list, transform=val_transform, cache_rate=1.0, num_workers=4)

        trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        valloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        train_loaders.append(trainloader)
        train_sizes.append(len(train_ds))
        val_loaders.append(valloader)

        # Client weight normalized after all centers are loaded below
        client_weights.append(len(train_ds) / sum(len(ld.dataset) for ld in train_loaders))

        train_labels = [int(sample["label"].item()) for sample in train_data_list]

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
    
    return train_loaders, val_loaders, client_weights, class_weights_list, train_sizes

#---------------------------------------
# Local Train/Validate
def train_local_epoch(model, dataloader, criterion, optimizer, device, temperature=1.0, max_batches=None, grad_clip=1.0, accumulation_steps=1):
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

def validate_local(model, dataloader, criterion, device, temperature=1.0):
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
def evaluate_fl(models, val_loaders, criterion, device, temps, center_names, round_num):
    agg_loss, agg_labels, agg_probs = 0.0, [], []
    per_center_metrics = {}
    per_center_y_true = {}
    per_center_y_pred = {}
    num_clients = len(val_loaders)
    total_val_size = sum(len(vl.dataset) for vl in val_loaders)

    for cid in range(num_clients):
        eval_model = models[cid] if isinstance(models, list) else models
        local_loss, local_labels, local_probs = validate_local(eval_model, val_loaders[cid], criterion, device, temperature=temps[cid])

        metrics = calculate_metrics(local_labels, local_probs)
        per_center_metrics[center_names[cid]] = metrics
        per_center_y_true[center_names[cid]] = local_labels
        per_center_y_pred[center_names[cid]] = local_probs
        agg_loss += local_loss * (len(val_loaders[cid].dataset) / total_val_size)
        agg_labels.extend(local_labels)
        agg_probs.extend(local_probs)
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
def main_fl(pretrained_path=None,learning_rate=0.001, weight_decay=0.01, num_rounds= 34,local_epochs = 3 ):
    accumulation_steps = 8
    roi_size = (256, 256, 128)
    dropout_rate = None
    data_comb = "labelcrop_clahewindow_commonaug"
    batch_size = 4
    opt_sched = {"optimizer": "AdamW", "scheduler": "CosineAnnealingLR"}
    temperatures = [0.7, 0.7, 0.7]
    center_names = ['UKB', 'Berlin', 'Goe']

    label_crop, windowing, augmentation = parse_data_combination_name(data_comb)
    effective_batch_size = batch_size * accumulation_steps

    run_name = (
        f"3011_fl_bs{batch_size}_rounds{num_rounds}_localepochs_{local_epochs}_"
        f"{'labelcrop' if label_crop else 'nolabelcrop'}_"
        f"{windowing}window_"
        f"{augmentation}aug_"
        # f"{center}_"
        f"dropout{dropout_rate}_lr{learning_rate}_wd{weight_decay}_"
        f"{opt_sched['optimizer']}_{opt_sched['scheduler']}"
    )
    run_path = os.path.join(run_dir, run_name)
    check_and_create_dir(run_path)
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
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "data_combination": data_comb,
            "batch_size": batch_size,
            "effective_batch_size": effective_batch_size,
            "optimizer": opt_sched["optimizer"],
            "scheduler": opt_sched["scheduler"],
            "augmentation": augmentation,
            "windowing": windowing,
        }
    )

    train_loaders, val_loaders, client_weights, class_weights_list, train_sizes = load_data_per_center(label_crop, windowing, augmentation, roi_size, batch_size)

    num_clients = len(train_loaders)

    pretrained_model = SegResNet.from_pretrained("project-lighter/ct_fm_segresnet")
    encoder = pretrained_model.encoder
    global_model = CTClassificationModel(encoder, dropout_rate=dropout_rate).to(device)

    local_models = [copy.deepcopy(global_model) for _ in range(num_clients)]
    for local in local_models:
        local.to(device)
        
    local_optimizers = []
    for cid in range(num_clients):
        optimizer = optim.AdamW(local_models[cid].parameters(), lr=learning_rate, weight_decay=weight_decay)
        local_optimizers.append(optimizer)

    criterion = nn.CrossEntropyLoss()
    schedulers = [optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_rounds * local_epochs) for opt in local_optimizers]

    # Track best metrics
    best_global_balacc = -1.0
    best_global_auc = -1.0

    # Initialize ALA modules
    alas = []
    rand_percent = 80  # Default rand_percent for sampling in ALA
    for cid in range(num_clients):
        client_train_data = train_loaders[cid].dataset
        ala = ALA(cid, criterion, client_train_data, batch_size, rand_percent, device=device)
        alas.append(ala)

    for round_num in range(num_rounds):
        print(f"\nFL Round {round_num + 1}/{num_rounds}")

        for cid in range(num_clients):
            alas[cid].adaptive_local_aggregation(global_model, local_models[cid])

        train_losses = []
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

        FedAvg(global_model, local_models, client_weights)

        val_loss, global_metrics, per_center_metrics, agg_labels, agg_probs, per_center_y_true, per_center_y_pred = evaluate_fl(local_models, val_loaders, criterion, device, temperatures, center_names, round_num)

        logger.info(f"Round {round_num + 1}: Avg Train Loss: {avg_train_loss:.4f}, Global Val Loss: {val_loss:.4f}, AUC: {global_metrics['auc']:.3f}, F1: {global_metrics['f1_score']:.3f}, Balanced Acc: {global_metrics['balanced_accuracy']:.3f}")
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
        
        # ===  Save additionally if it crosses the 0.57 threshold === (Not required always)
        if global_metrics['balanced_accuracy'] >= 0.58:
            good_model_path = os.path.join(
                models_dir, 
                f"good_enough_round_{round_num + 1}_balacc_{global_metrics['balanced_accuracy']:.3f}.pth"
            )
            save_model(global_model, good_model_path)
            logger.info(f"Round {round_num + 1}: GOOD ENOUGH model saved (≥0.58) → {global_metrics['balanced_accuracy']:.3f}")

        if global_metrics['auc'] > best_global_auc:
            best_global_auc = global_metrics['auc']
            best_model_path = os.path.join(models_dir, f"best_global_auc_round_{round_num + 1}_{best_global_auc:.3f}.pth")
            save_model(global_model, best_model_path)
            logger.info(f"Round {round_num + 1}: Best AUC model saved with {best_global_auc:.3f}")
            
        round_plots_dir = os.path.join(plots_dir, f"round_{round_num + 1}")
        check_and_create_dir(round_plots_dir)

        save_plots = (round_num + 1) % 5 == 0 or global_metrics['balanced_accuracy'] > best_global_balacc
        if save_plots:
            np.savez(os.path.join(plots_dir, f"val_metrics_round_{round_num + 1}.npz"), y_true=agg_labels, y_pred=agg_probs)
            plot_roc_pr_curves(agg_labels, agg_probs[:, 1], round_plots_dir)
            conf_mat_path = os.path.join(round_plots_dir, f"conf_matrix_round_{round_num + 1}.png")
            plot_confusion_matrix(agg_labels, agg_probs, conf_mat_path)

        # Per-center plots triggered when a center lags significantly behind global performance
        global_balacc = global_metrics['balanced_accuracy']
        for center in center_names:
            center_balacc = per_center_metrics[center]['balanced_accuracy']
            if center_balacc < 0.8 * global_balacc:
                logger.warning(f"Bias detected in {center}: Bal Acc {center_balacc:.3f} << Global {global_balacc:.3f}")
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
    num_rounds = 20
    combinations = [
        (1e-4, 1e-5, 3),
        (1e-3, 1e-4, 2),
        (1e-3, 1e-4, 3),
        (1e-5, 5e-4, 2),
        (1e-5, 5e-4, 3),
        (2e-5, 1e-4, 2),
        (2e-5, 1e-4, 3),
    ]

    for lr, wd, local_epoch in combinations:
        main_fl(pretrained_path=None, learning_rate=lr, weight_decay=wd, num_rounds=num_rounds, local_epochs=local_epoch)
