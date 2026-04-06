"""
Evaluation Script for New FedBN and FedALA Models
=======================================
Evaluates the FedBN and FedALA models on UKB, Berlin, Göttingen, and Combined test sets,
and saves the `roc_prc_final.npz` files for plotting in the notebook.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from monai import transforms
from monai.data import Dataset
from monai.utils import set_determinism
from monai.transforms import (
    ScaleIntensityRanged, ToTensord, ScaleIntensityd, MapTransform,
)
from lighter_zoo import SegResNet
from skimage import exposure
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, confusion_matrix,
    auc, roc_curve, precision_recall_curve, matthews_corrcoef,
)
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# Config
# Set DATA_DIR to the folder containing test CSV files, and MODEL_DIR to the folder where trained models are saved.
# Override via environment variables DATA_DIR, MODEL_DIR, OUTPUT_BASE_DIR, or edit directly below.
# ──────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(_SCRIPT_DIR, "..", "data", "all_combine_retrain_files"))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(_SCRIPT_DIR, "..", "results"))
OUTPUT_BASE_DIR = os.environ.get("OUTPUT_BASE_DIR", os.path.join(_SCRIPT_DIR, "..", "results", "eval_npz"))

# Update MODELS to point to your trained model checkpoints:
# e.g. "FedBN": os.path.join(MODEL_DIR, "fedbn", "run_XXXXXX", "models", "best_model.pth")
MODELS = {
    # "FedBN":  os.path.join(MODEL_DIR, "fedbn",  "run_XXXXXX", "models", "best_model.pth"),
    # "FedALA": os.path.join(MODEL_DIR, "fedala", "run_XXXXXX", "models", "best_model.pth"),
}

TEST_CENTERS = {
    "UKB":    os.path.join(DATA_DIR, "test_ukb.csv"),
    "BERLIN": os.path.join(DATA_DIR, "test_berlin.csv"),
    "GOE":    os.path.join(DATA_DIR, "test_g.csv"),
}


# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
SEED = 42
set_determinism(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(device)
roi_size = (256, 256, 128)

# ──────────────────────────────────────────────
# Helper Functions & Transforms
# ──────────────────────────────────────────────
def to_plain_tensor(x, device):
    if hasattr(x, "as_tensor"):
        return x.as_tensor().to(device)
    else:
        return x.clone().to(device)

def move_batch_to_device(batch, device):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    return batch

class LabelCrop(transforms.Transform):
    def __init__(self, keys, roi_size):
        super().__init__()
        self.keys = keys
        self.roi_size = roi_size

    def __call__(self, data):
        label = data[self.keys[1]]
        center = np.array(np.where(label == 1)).mean(axis=1).astype(float)
        center = center[1:]
        if center.size == 0:
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
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            if img.ndim == 3:
                img = np.stack([
                    exposure.equalize_adapthist(img[..., i], clip_limit=self.clip_limit)
                    for i in range(img.shape[-1])
                ], axis=-1)
            elif img.ndim == 4:
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

clip_and_norm = ScaleIntensityRanged(
    keys=["image"], a_min=-1000, a_max=2000, b_min=0.0, b_max=1.0, clip=True,
)

def create_val_transform(roi_size):
    return transforms.Compose([
        transforms.LoadImaged(keys=["image", "seg"], image_only=False),
        transforms.EnsureChannelFirstd(keys=["image", "seg"]),
        transforms.Orientationd(keys=["image", "seg"], axcodes="RAS"),
        transforms.Spacingd(keys=["image", "seg"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        transforms.CropForegroundd(keys=["image", "seg"], source_key="image", allow_smaller=True),
        LabelCrop(keys=["image", "seg"], roi_size=roi_size),
        transforms.SpatialPadd(keys=["image", "seg"], spatial_size=roi_size),
        clip_and_norm,
        ApplyCLAHE(keys=["image"], clip_limit=0.03),
        transforms.ToTensord(keys=["image", "seg"])
    ])

# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────
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

# ──────────────────────────────────────────────
# Data loading & Metrics
# ──────────────────────────────────────────────
def load_test_data(csv_path, roi_size, batch_size=4):
    val_df = pd.read_csv(csv_path)
    val_data_list = []
    for idx in range(len(val_df)):
        sample = {
            "image": val_df.loc[idx, "path"],
            "seg": val_df.loc[idx, "path_seg"],
            "label": torch.tensor(float(val_df.loc[idx, "N Status"]), dtype=torch.long)
        }
        val_data_list.append(sample)

    val_transform = create_val_transform(roi_size)
    val_ds = Dataset(data=val_data_list, transform=val_transform)
    return DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

def validate(model, dataloader, device, temperature=0.7):
    model.eval()
    if len(dataloader) == 0:
        return np.array([]), np.array([])
    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = move_batch_to_device(batch, device)
            outputs = model(batch['image'])
            probs = torch.softmax(outputs / temperature, dim=1).detach().cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(batch['label'].long().view(-1).cpu().numpy())
    return np.array(all_labels), np.array(all_probs)

def compute_and_save_roc_prc(y_true, y_pred_prob, save_path):
    print(f"y_true shape: {y_true.shape}, y_pred_prob shape: {y_pred_prob.shape}")
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    auc_score = roc_auc_score(y_true, y_pred_prob)
    
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_prob)
    auprc_score = auc(recall_vals, precision_vals)
    
    np.savez(
        save_path, 
        fpr=fpr, 
        tpr=tpr, 
        auc=auc_score, 
        precision=precision_vals, 
        recall=recall_vals, 
        prc_auc=auprc_score
    )
    print(f"Saved data to {save_path} (AUC: {auc_score:.4f}, AUPRC: {auprc_score:.4f})")

# ──────────────────────────────────────────────
# Main Batch Processing
# ──────────────────────────────────────────────
def main():
    print("Preloading Test Data...")
    dataloaders = {}
    for center_name, csv_path in TEST_CENTERS.items():
        dataloaders[center_name] = load_test_data(csv_path, roi_size=roi_size, batch_size=4)

    # Initialize model backbone once
    pretrained_model = SegResNet.from_pretrained("project-lighter/ct_fm_segresnet")
    encoder = pretrained_model.encoder
    model = CTClassificationModel(encoder, dropout_rate=None).to(device)

    for model_name, model_path in MODELS.items():
        print(f"\nEvaluating Model: {model_name}")
        
        # Load weights
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            print(f"Failed to load {model_path}: {e}")
            continue
            
        all_labels_global = []
        all_probs_global = []
        
        for center_name, valloader in dataloaders.items():
            print(f"  Testing on {center_name}...")
            y_val, y_val_pred = validate(model, valloader, device, temperature=0.7)
            
            # Store for global metrics
            all_labels_global.extend(y_val)
            all_probs_global.extend(y_val_pred)
            
            # Save center .npz
            center_dir = os.path.join(OUTPUT_BASE_DIR, center_name, model_name)
            os.makedirs(center_dir, exist_ok=True)
            npz_path = os.path.join(center_dir, "roc_prc_final.npz")
            compute_and_save_roc_prc(y_val, y_val_pred[:, 1], npz_path)
            
        # Global metrics
        print("  Testing on COMBINED_3...")
        all_labels_global = np.array(all_labels_global)
        all_probs_global = np.array(all_probs_global)
        
        global_dir = os.path.join(OUTPUT_BASE_DIR, "COMBINED_3", model_name)
        os.makedirs(global_dir, exist_ok=True)
        npz_path = os.path.join(global_dir, "roc_prc_final.npz")
        compute_and_save_roc_prc(all_labels_global, all_probs_global[:, 1], npz_path)

if __name__ == "__main__":
    main()
