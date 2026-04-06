"""
Test Script for Federated Learning Models
==========================================
Evaluates any trained FL model checkpoint (FedBN, FedALA, FedDisco, FedAvg, or Centralized)
on all test centers (UKB, Berlin, Göttingen). Computes per-center and aggregated metrics,
plots KDE class-separation curves, confusion matrices, and ROC/PRC curves, and logs to W&B.

Usage:
    python scripts/test_fl_model.py --model_path /path/to/checkpoint.pth
    python scripts/test_fl_model.py --model_path /path/to/checkpoint.pth --data_dir data/all_combine_retrain_files
"""

import os
import random
import itertools
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
from datetime import datetime
from scipy.stats import wasserstein_distance
import wandb

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

# Override any of these with environment variables before running.
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
FEDBN_MODEL_PATH = os.environ.get("MODEL_PATH",    None)
DATA_DIR      = os.environ.get("DATA_DIR",      os.path.join(_SCRIPT_DIR, "..", "data", "all_combine_retrain_files"))
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "pdac-ctfm")
WANDB_ENTITY  = os.environ.get("WANDB_ENTITY",  None)

TEST_CENTERS = {
    "UKB":       os.path.join(DATA_DIR, "test_ukb.csv"),
    "Berlin":    os.path.join(DATA_DIR, "test_berlin.csv"),
    "Göttingen": os.path.join(DATA_DIR, "test_g.csv"),
}

# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
SEED = 42
set_determinism(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
roi_size = (256, 256, 128)

# Logging
save_dir = os.environ.get("SAVE_DIR", os.path.join(_SCRIPT_DIR, "..", "results", "test_fl_model"))
os.makedirs(save_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join(save_dir, "testing.log"),
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FL_Model_Testing")


# ──────────────────────────────────────────────
# Helper Functions
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

def check_and_create_dir(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        except Exception as e:
            raise OSError(f"Unable to create directory {directory}: {e}")
    elif not os.access(directory, os.W_OK):
        raise PermissionError(f"Directory {directory} is not writable.")


# ──────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────
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
# Data loading
# ──────────────────────────────────────────────
def load_test_data(csv_path, roi_size, batch_size=4):
    val_df = pd.read_csv(csv_path)
    val_data_list = []
    for idx in range(len(val_df)):
        sample = {
            "image": val_df.loc[idx, "path"],
            "seg": val_df.loc[idx, "path_seg"],
            "label": torch.tensor(float(val_df.loc[idx, "N Status"]), dtype=torch.long),
            "center": val_df.loc[idx, "center"],
            "metadata": val_df.loc[idx, ['N Status', 'Head_Localization', 'Tail_Localization',
                                          'T_0', 'T_1', 'T_2', 'T_3', 'T_4']].values.astype('float32')
        }
        val_data_list.append(sample)

    val_transform = create_val_transform(roi_size)
    val_ds = Dataset(data=val_data_list, transform=val_transform)
    valloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"  Total test samples: {len(val_df)} | Dataset length: {len(val_ds)} | Dataloader batches: {len(valloader)}")
    return valloader


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────
def calculate_metrics(y_true, y_pred):
    try:
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
        "precision": precision, "recall": recall, "specificity": specificity,
        "f1_score": f1_score, "accuracy": accuracy, "balanced_accuracy": bal_acc,
        "auc": auc_score, "auprc": auprc_score, "dor": dor, "mcc": mcc
    }


def plot_kde_class_separation(y_true, y_pred, dataset_name, save_dir, center_name=None, save_preds=False):
    if len(y_true) == 0:
        print(f"No data for {dataset_name}")
        return None

    prob_class1 = y_pred[:, 1]
    probs_neg = prob_class1[y_true == 0]
    probs_pos = prob_class1[y_true == 1]

    if len(probs_neg) == 0 or len(probs_pos) == 0:
        print(f"Insufficient data for one class in {dataset_name}")
        return None

    plt.figure(figsize=(8, 6))
    sns.kdeplot(probs_neg, fill=True, color='blue', label='True Class 0 (Negative)')
    sns.kdeplot(probs_pos, fill=True, color='red', label='True Class 1 (Positive)')
    plt.xlabel('Predicted Probability for Class 1')
    plt.ylabel('Density')
    title = f'KDE of Class Separation - {dataset_name}'
    if center_name:
        title += f' ({center_name})'
    plt.title(title)
    plt.legend()
    plt.xlim(0, 1)
    plt.tight_layout()

    filename = f'kde_separation_{dataset_name.lower()}'
    if center_name:
        filename += f'_{center_name.lower()}'
    filepath = os.path.join(save_dir, f'{filename}.png')
    plt.savefig(filepath)
    plt.close()
    print(f'KDE plot saved: {filepath}')

    overlap_metric = wasserstein_distance(probs_neg, probs_pos)
    print(f'Wasserstein Distance for {dataset_name}: {overlap_metric:.4f}')

    if save_preds:
        save_path = os.path.join(save_dir, f'preds_{dataset_name.lower()}.npz')
        np.savez(save_path, y_true=y_true, y_pred=y_pred, probs_class1=prob_class1,
                 probs_neg=probs_neg, probs_pos=probs_pos)
        print(f'Predictions saved: {save_path}')

    return overlap_metric


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


# ──────────────────────────────────────────────
# Validate
# ──────────────────────────────────────────────
def validate(model, dataloader, criterion, device, temperature=1.0):
    model.eval()
    if len(dataloader) == 0:
        return 0.0, np.array([]), np.array([])
    total_loss = 0.0
    all_labels, all_probs = [], []
    loop = tqdm(dataloader, desc="Testing", leave=True, position=0)
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


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test any trained FL model checkpoint")
    parser.add_argument("--model_path", type=str, default=FEDBN_MODEL_PATH,
                        help="Path to the trained FL model checkpoint (.pth)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to folder containing test CSV files")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save test results")
    args = parser.parse_args()

    model_path = args.model_path
    if model_path is None:
        raise ValueError("No model checkpoint specified. Use --model_path or set MODEL_PATH env var.")

    global DATA_DIR, TEST_CENTERS, save_dir
    if args.data_dir is not None:
        DATA_DIR = args.data_dir
        TEST_CENTERS = {
            "UKB":       os.path.join(DATA_DIR, "test_ukb.csv"),
            "Berlin":    os.path.join(DATA_DIR, "test_berlin.csv"),
            "Göttingen": os.path.join(DATA_DIR, "test_g.csv"),
        }
    if args.output_dir is not None:
        save_dir = args.output_dir

    temperature = 0.7
    batch_size = 4

    current_date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(save_dir, f"run_{current_date_time}")
    check_and_create_dir(run_dir)

    print(f"\n{'='*60}")
    print(f"FL Model Testing")
    print(f"Model: {model_path}")
    print(f"Output: {run_dir}")
    print(f"{'='*60}\n")

    pretrained_model = SegResNet.from_pretrained("project-lighter/ct_fm_segresnet")
    encoder = pretrained_model.encoder
    model = CTClassificationModel(encoder, dropout_rate=None).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully\n")

    criterion = nn.CrossEntropyLoss()

    # W&B
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=f"test_fl_model_{current_date_time}",
        config={
            "model_path": FEDBN_MODEL_PATH,
            "temperature": temperature,
            "batch_size": batch_size,
            "test_centers": list(TEST_CENTERS.keys()),
        }
    )

    # Aggregated results across all centers
    all_labels_global = []
    all_probs_global = []
    all_metrics = {}

    for center_name, csv_path in TEST_CENTERS.items():
        print(f"\n{'='*60}")
        print(f"Testing on: {center_name}")
        print(f"{'='*60}")

        center_dir = os.path.join(run_dir, center_name)
        check_and_create_dir(center_dir)

        valloader = load_test_data(csv_path, roi_size, batch_size=batch_size)
        val_loss, y_val, y_val_pred = validate(model, valloader, criterion, device, temperature=temperature)

        # KDE plot
        val_wass = plot_kde_class_separation(y_val, y_val_pred, 'Test', center_dir, center_name=center_name, save_preds=True)

        # Metrics
        val_metrics = calculate_metrics(y_val, y_val_pred)
        all_metrics[center_name] = val_metrics

        # Confusion matrix plot
        conf_mat_path = os.path.join(center_dir, "confusion_matrix.png")
        plot_confusion_matrix(y_val, y_val_pred, conf_mat_path)

        # ROC/PR curves
        plot_roc_pr_curves(y_val, y_val_pred[:, 1], center_dir)

        # Print results
        print(f"\n  {center_name} Results:")
        print(f"  Loss:          {val_loss:.4f}")
        print(f"  AUC:           {val_metrics['auc']:.3f}")
        print(f"  F1:            {val_metrics['f1_score']:.3f}")
        print(f"  Balanced Acc:  {val_metrics['balanced_accuracy']:.3f}")
        print(f"  Precision:     {val_metrics['precision']:.3f}")
        print(f"  Recall:        {val_metrics['recall']:.3f}")
        print(f"  Specificity:   {val_metrics['specificity']:.3f}")
        print(f"  AUPRC:         {val_metrics['auprc']:.3f}")
        print(f"  MCC:           {val_metrics['mcc']:.3f}")
        print(f"  DOR:           {val_metrics['dor']:.3f}")
        if val_wass is not None:
            print(f"  Wasserstein:   {val_wass:.4f}")

        # Log to W&B
        log_dict = {f"test_{k}_{center_name}": v for k, v in val_metrics.items()}
        log_dict[f"test_loss_{center_name}"] = val_loss
        if val_wass is not None:
            log_dict[f"test_wasserstein_{center_name}"] = val_wass

        y_pred_binary = (y_val_pred[:, 1] >= 0.5).astype(int)
        log_dict[f"conf_mat_{center_name}"] = wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_val.astype(int).flatten(),
            preds=y_pred_binary,
            class_names=['N0', 'N1']
        )
        wandb.log(log_dict)

        # Accumulate for global metrics
        all_labels_global.extend(y_val)
        all_probs_global.extend(y_val_pred)

        torch.cuda.empty_cache()

    # Global metrics across all centers
    all_labels_global = np.array(all_labels_global)
    all_probs_global = np.array(all_probs_global)

    global_metrics = calculate_metrics(all_labels_global, all_probs_global)
    all_metrics["Global"] = global_metrics

    # Global plots
    global_dir = os.path.join(run_dir, "global")
    check_and_create_dir(global_dir)
    plot_kde_class_separation(all_labels_global, all_probs_global, 'Test', global_dir, center_name='Global', save_preds=True)
    plot_confusion_matrix(all_labels_global, all_probs_global, os.path.join(global_dir, "confusion_matrix.png"))
    plot_roc_pr_curves(all_labels_global, all_probs_global[:, 1], global_dir)

    # Log global metrics to W&B
    global_log = {f"test_{k}_global": v for k, v in global_metrics.items()}
    y_pred_binary_global = (all_probs_global[:, 1] >= 0.5).astype(int)
    global_log["conf_mat_global"] = wandb.plot.confusion_matrix(
        probs=None,
        y_true=all_labels_global.astype(int).flatten(),
        preds=y_pred_binary_global,
        class_names=['N0', 'N1']
    )
    wandb.log(global_log)

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Center':<12} {'AUC':>7} {'F1':>7} {'BalAcc':>8} {'Prec':>7} {'Rec':>7} {'Spec':>7} {'AUPRC':>8} {'MCC':>7} {'DOR':>8}")
    print("-" * 90)
    
    summary_data = []
    for center_name, metrics in all_metrics.items():
        print(f"{center_name:<12} {metrics['auc']:>7.4f} {metrics['f1_score']:>7.4f} {metrics['balanced_accuracy']:>8.4f} "
              f"{metrics['precision']:>7.4f} {metrics['recall']:>7.4f} {metrics['specificity']:>7.4f} "
              f"{metrics['auprc']:>8.4f} {metrics['mcc']:>7.4f} {metrics['dor']:>8.4f}")
        
        row = {"Center": center_name}
        row.update(metrics)
        summary_data.append(row)

    # Save summary to CSV with 4 decimals
    metrics_csv_path = os.path.join(run_dir, "metrics_summary.csv")
    df_summary = pd.DataFrame(summary_data)
    # Round metrics columns to 4 decimals
    cols_to_round = [c for c in df_summary.columns if c != "Center"]
    df_summary[cols_to_round] = df_summary[cols_to_round].round(4)
    df_summary.to_csv(metrics_csv_path, index=False)
    print(f"\nMetrics summary saved to: {metrics_csv_path}")

    wandb.finish()
    logger.info("FL model testing complete.")
    print(f"\nAll results saved to: {run_dir}")


if __name__ == "__main__":
    main()
