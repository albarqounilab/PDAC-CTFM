"""
t-SNE and UMAP Feature Visualization Script
=============================================
Generates t-SNE and UMAP scatter plots of backbone and aligned feature
representations from different FL models, colored by center (site).

This addresses the reviewer comment requesting qualitative evidence that
the proposed method reduces domain shift compared to FedAvg/FedDisco.

Usage:
    python scripts/tsne_umap_visualization.py [--output_dir results/tsne_umap_figures]
"""

import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from monai import transforms
from monai.data import Dataset, CacheDataset
from monai.utils import set_determinism
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    CropForegroundd, SpatialPadd, ScaleIntensityRanged, ToTensord,
    ScaleIntensityd, MapTransform,
)
from lighter_zoo import SegResNet
from skimage import exposure
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

try:
    import umap
    HAS_UMAP = True
except ImportError:
    print("WARNING: umap-learn not installed. UMAP plots will be skipped.")
    HAS_UMAP = False

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROI_SIZE = (256, 256, 128)

# ──────────────────────────────────────────────
# Model paths — update with your trained checkpoint paths
# ──────────────────────────────────────────────
# Set paths to your trained model checkpoints below, or override at runtime via --model_configs_json.
MODEL_CONFIGS = {
    "FedAvg":            {"path": None},   # e.g. "results/feddisco/run_XXXX/models/best_model.pth"
    "FedDisco":          {"path": None},
    "FedBN":             {"path": None},
    "FedALA":            {"path": None},
    "CTFM (Ours)":       {"path": None},
    "PDAC-CTFM (Central)": {"path": None},
}

# ──────────────────────────────────────────────
# Test data CSVs and center info
# ──────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(_SCRIPT_DIR, "..", "data", "all_combine_retrain_files"))

TEST_CENTERS = {
    "UKB":       os.path.join(DATA_DIR, "test_ukb.csv"),
    "Berlin":    os.path.join(DATA_DIR, "test_berlin.csv"),
    "Göttingen": os.path.join(DATA_DIR, "test_g.csv"),
}

CENTER_COLORS = {
    "UKB":        "#E74C3C",   # red
    "Berlin":     "#3498DB",   # blue
    "Göttingen":  "#2ECC71",   # green
}

CLASS_MARKERS = {
    0: "o",  # Circle for Negative
    1: "X",  # Cross for Positive
}

CLASS_LABELS = {
    0: "N0 (Negative)",
    1: "N1 (Positive)",
}


# ──────────────────────────────────────────────
# Helper functions (matching training scripts)
# ──────────────────────────────────────────────
def to_plain_tensor(x, device):
    """Converts a MONAI MetaTensor to a plain tensor."""
    if hasattr(x, "as_tensor"):
        return x.as_tensor().to(device)
    else:
        return x.clone().to(device)


def move_batch_to_device(batch, device):
    """Moves all tensor elements in a batch to the target device."""
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    return batch


# ──────────────────────────────────────────────
# Transforms (matching training scripts)
# ──────────────────────────────────────────────
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
            return data
        center = tuple(center.astype(np.int16).tolist())
        data['center'] = center
        cropper = transforms.SpatialCropd(
            keys=["image", "seg"], roi_center=data['center'], roi_size=self.roi_size
        )
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
    """Validation transform matching training: labelcrop + clahewindow."""
    return Compose([
        LoadImaged(keys=["image", "seg"], image_only=False),
        EnsureChannelFirstd(keys=["image", "seg"]),
        Orientationd(keys=["image", "seg"], axcodes="RAS"),
        Spacingd(keys=["image", "seg"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        CropForegroundd(keys=["image", "seg"], source_key="image", allow_smaller=True),
        LabelCrop(keys=["image", "seg"], roi_size=roi_size),
        SpatialPadd(keys=["image", "seg"], spatial_size=roi_size),
        clip_and_norm,
        ApplyCLAHE(keys=["image"], clip_limit=0.03),
        ToTensord(keys=["image", "seg"]),
    ])


# ──────────────────────────────────────────────
# Model definition (matching training scripts)
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

    def extract_features(self, x):
        """Extract both backbone and aligned features.

        Returns:
            backbone_features: (B, 512) — after encoder + pool + flatten
            aligned_features:  (B, 256) — after fc1 + relu
        """
        features = self.encoder(x)
        x = features[-1] if isinstance(features, (list, tuple)) else features
        device = next(self.parameters()).device
        x = to_plain_tensor(x, device)
        x = self.pool(x)
        x = self.flatten(x)
        backbone_features = to_plain_tensor(x, device)

        x = self.fc1(backbone_features)
        x = self.relu(x)
        aligned_features = x

        return backbone_features, aligned_features


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────
def load_test_data(roi_size, batch_size=4, subset_n=None, num_workers=16):
    """Load test data from all centers into a single dataloader with center labels.

    Args:
        roi_size: size for spatial crop
        batch_size: batch size for dataloader
        subset_n: if provided, only take first N samples per center for quick testing
        num_workers: number of workers for data loading
    """
    val_transform = create_val_transform(roi_size)

    all_data_list = []
    center_labels = []  # parallel list of center names
    class_labels = []   # parallel list of N0/N1 labels

    for center_name, csv_path in TEST_CENTERS.items():
        df = pd.read_csv(csv_path)
        print(f"  {center_name:10}: {len(df):3} samples in total test set")
        
        if subset_n is not None:
            # Balanced sampling (N0 and N1) from each center
            df_0 = df[df["N Status"] == 0]
            df_1 = df[df["N Status"] == 1]
            
            # Select subset_n of each (or all if fewer available)
            n_to_sample = int(subset_n)
            s0 = df_0.sample(n=min(len(df_0), n_to_sample), random_state=SEED).sort_index()
            s1 = df_1.sample(n=min(len(df_1), n_to_sample), random_state=SEED).sort_index()
            
            # Combine consistently (N0 then N1)
            df = pd.concat([s0, s1]).reset_index(drop=True)

        for idx in range(len(df)):
            sample = {
                "image": df.loc[idx, "path"],
                "seg": df.loc[idx, "path_seg"],
                "label": torch.tensor(float(df.loc[idx, "N Status"]), dtype=torch.long),
            }
            all_data_list.append(sample)
            center_labels.append(center_name)
            class_labels.append(int(df.loc[idx, "N Status"]))

    print(f"Total samples being used: {len(all_data_list)} across {len(TEST_CENTERS)} centers")
    
    # Detailed count printing per center
    print("-" * 30)
    for center_name in TEST_CENTERS:
        indices = [i for i, c in enumerate(center_labels) if c == center_name]
        c_labels = [class_labels[i] for i in indices]
        n0 = c_labels.count(0)
        n1 = c_labels.count(1)
        print(f"  {center_name:10}: Total={len(c_labels):3}, N0={n0:3}, N1={n1:3}")
    print("-" * 30)
    
    print("Caching pre-processed images in RAM...")
    dataset = CacheDataset(data=all_data_list, transform=val_transform, cache_rate=1.0, num_workers=num_workers)
    
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )


    return dataloader, center_labels, class_labels


# ──────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────
def extract_features_from_model(model, dataloader):
    """Extract backbone and aligned features for all test samples."""
    model.eval()
    all_backbone = []
    all_aligned = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            batch = move_batch_to_device(batch, DEVICE)
            backbone_feat, aligned_feat = model.extract_features(batch["image"])
            all_backbone.append(backbone_feat.cpu().numpy())
            all_aligned.append(aligned_feat.cpu().numpy())

    return np.concatenate(all_backbone, axis=0), np.concatenate(all_aligned, axis=0)


# ──────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────
def compute_tsne(features, perplexity=30, n_iter=1000):
    """Compute t-SNE embedding."""
    try:
        tsne = TSNE(
            n_components=2, perplexity=min(perplexity, len(features) - 1),
            n_iter=n_iter, random_state=SEED, init='pca', learning_rate='auto',
        )
    except TypeError:
        # Fallback for older scikit-learn versions where n_iter or learning_rate='auto' might not be supported
        tsne = TSNE(
            n_components=2, perplexity=min(perplexity, len(features) - 1),
            random_state=SEED, init='pca'
        )
    return tsne.fit_transform(features)


def compute_umap(features, n_neighbors=15, min_dist=0.1):
    """Compute UMAP embedding."""
    if not HAS_UMAP:
        return None
    reducer = umap.UMAP(
        n_components=2, n_neighbors=min(n_neighbors, len(features) - 1),
        min_dist=min_dist, random_state=SEED, metric='euclidean',
    )
    return reducer.fit_transform(features)


def compute_pca(features):
    """Compute PCA embedding."""
    pca = PCA(n_components=2, random_state=SEED)
    return pca.fit_transform(features)


def plot_embedding(ax, embedding, center_labels, class_labels, title):
    """Plot a single embedding scatter plot on an axis."""
    for center_name in TEST_CENTERS:
        for cls_id in CLASS_MARKERS:
            # Mask for both center and class
            mask = np.logical_and(
                np.array([c == center_name for c in center_labels]),
                np.array([c == cls_id for c in class_labels])
            )
            if mask.sum() == 0:
                continue
            ax.scatter(
                embedding[mask, 0], embedding[mask, 1],
                c=CENTER_COLORS[center_name],
                marker=CLASS_MARKERS[cls_id],
                s=60, alpha=0.75, edgecolors='white', linewidth=0.5,
            )
    ax.set_title(title, fontsize=13, fontweight='bold', pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color('#cccccc')


def generate_figures(all_model_features, center_labels, class_labels, output_dir, feature_type_label, feature_type_tag, method='all'):
    """Generate a combined figure with dimensionality reduction rows for all models."""
    model_names = list(all_model_features.keys())
    n_models = len(model_names)

    # Compute only requested embeddings
    tsne_embeddings = {}
    umap_embeddings = {}
    pca_embeddings = {}
    
    do_pca = method in ['pca', 'all']
    do_tsne = method in ['tsne', 'both', 'all']
    do_umap = method in ['umap', 'both', 'all'] and HAS_UMAP

    for name in model_names:
        if do_pca:
            print(f"  Computing PCA for {name} ({feature_type_tag})...")
            pca_embeddings[name] = compute_pca(all_model_features[name])
        if do_tsne:
            print(f"  Computing t-SNE for {name} ({feature_type_tag})...")
            tsne_embeddings[name] = compute_tsne(all_model_features[name])
        if do_umap:
            print(f"  Computing UMAP for {name} ({feature_type_tag})...")
            umap_embeddings[name] = compute_umap(all_model_features[name])

    rows = []
    if do_pca: rows.append('pca')
    if do_tsne: rows.append('tsne')
    if do_umap: rows.append('umap')
    
    n_rows = len(rows)
    if n_rows == 0:
        print(f"  Warning: No visualizations to generate for {feature_type_tag}")
        return

    fig, axes = plt.subplots(n_rows, n_models, figsize=(5 * n_models, 5 * n_rows), squeeze=False)

    fig.suptitle(f"{feature_type_label}", fontsize=16, fontweight='bold', y=0.98)

    for col_idx, model_name in enumerate(model_names):
        row_idx = 0
        
        # PCA row
        if do_pca:
            plot_embedding(axes[row_idx, col_idx], pca_embeddings[model_name], center_labels, class_labels,
                           f"PCA — {model_name}")
            row_idx += 1

        # t-SNE row
        if do_tsne:
            plot_embedding(axes[row_idx, col_idx], tsne_embeddings[model_name], center_labels, class_labels,
                           f"t-SNE — {model_name}")
            row_idx += 1

        # UMAP row
        if do_umap:
            plot_embedding(axes[row_idx, col_idx], umap_embeddings[model_name], center_labels, class_labels,
                           f"UMAP — {model_name}")
            row_idx += 1

    # Add shared legend for centers
    center_handles = [
        mlines.Line2D([], [], color=CENTER_COLORS[c], marker='o',
                      markersize=8, linestyle='None', label=c, markeredgecolor='white',
                      markeredgewidth=0.5)
        for c in TEST_CENTERS
    ]
    
    # Add shared legend for classes
    class_handles = [
        mlines.Line2D([], [], color='gray', marker=CLASS_MARKERS[cls],
                      markersize=8, linestyle='None', label=CLASS_LABELS[cls], markeredgecolor='white',
                      markeredgewidth=0.5)
        for cls in CLASS_MARKERS
    ]
    
    # Combine legends into two columns or a single row depending on space
    fig.legend(handles=center_handles + class_handles, loc='lower center', ncol=len(TEST_CENTERS) + len(CLASS_MARKERS),
               fontsize=11, frameon=True, fancybox=True, shadow=False,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])

    # Save
    for ext in ['png', 'pdf']:
        save_path = os.path.join(output_dir, f"tsne_umap_{feature_type_tag}.{ext}")
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {save_path}")

    plt.close(fig)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="t-SNE/UMAP feature visualization")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to folder containing test CSV files (default: ../data/all_combine_retrain_files)")
    parser.add_argument("--output_dir", type=str,
                        default="results/tsne_umap_figures",
                        help="Directory to save output figures")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for inference (increase for speed)")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of workers for data loading")
    parser.add_argument("--subset_n", type=int, default=None,
                        help="Number of samples to take per center for quick testing")
    parser.add_argument("--method", type=str, default="all", choices=["tsne", "umap", "pca", "both", "all"],
                        help="Visualization method to run")
    args = parser.parse_args()

    if args.data_dir is not None:
        global DATA_DIR, TEST_CENTERS
        DATA_DIR = args.data_dir
        TEST_CENTERS = {
            "UKB":       os.path.join(DATA_DIR, "test_ukb.csv"),
            "Berlin":    os.path.join(DATA_DIR, "test_berlin.csv"),
            "Göttingen": os.path.join(DATA_DIR, "test_g.csv"),
        }

    if args.method == "umap" and not HAS_UMAP:
        print("ERROR: UMAP selected but umap-learn is not installed.")
        return

    # Resolve output dir relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load test data once
    print("\n" + "=" * 60)
    print(f"Loading test data (subset_n={args.subset_n})...")
    print("=" * 60)
    dataloader, center_labels, class_labels = load_test_data(
        ROI_SIZE, batch_size=args.batch_size, subset_n=args.subset_n, num_workers=args.num_workers
    )

    # Extract features from each model
    all_backbone_features = {}
    all_aligned_features = {}

    for model_name, config in MODEL_CONFIGS.items():
        print(f"\n{'=' * 60}")
        print(f"Processing model: {model_name}")
        print(f"  Path: {config['path']}")
        print(f"{'=' * 60}")

        # Initialize fresh model
        pretrained_model = SegResNet.from_pretrained("project-lighter/ct_fm_segresnet")
        encoder = pretrained_model.encoder
        model = CTClassificationModel(encoder, dropout_rate=None).to(DEVICE)

        # Load trained weights
        state_dict = torch.load(config["path"], map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"  Model loaded successfully")

        # Extract features
        backbone, aligned = extract_features_from_model(model, dataloader)
        all_backbone_features[model_name] = backbone
        all_aligned_features[model_name] = aligned
        print(f"  Backbone features shape: {backbone.shape}")
        print(f"  Aligned features shape:  {aligned.shape}")

        # Free memory
        del model, encoder, pretrained_model, state_dict
        torch.cuda.empty_cache()

    # Generate visualization figures
    print(f"\n{'=' * 60}")
    print("Generating visualizations...")
    print(f"{'=' * 60}")

    print("\n--- Backbone Features ---")
    generate_figures(
        all_backbone_features, center_labels, class_labels, output_dir,
        feature_type_label="Backbone Features (512-d)",
        feature_type_tag="backbone",
        method=args.method
    )

    print("\n--- Aligned Features ---")
    generate_figures(
        all_aligned_features, center_labels, class_labels, output_dir,
        feature_type_label="Aligned Features (256-d)",
        feature_type_tag="aligned",
        method=args.method
    )

    # Also save individual per-model plots for flexibility
    print(f"\n{'=' * 60}")
    print("Generating individual model plots...")
    print(f"{'=' * 60}")
    individual_dir = os.path.join(output_dir, "individual")
    os.makedirs(individual_dir, exist_ok=True)

    for model_name in MODEL_CONFIGS:
        for feat_tag, feat_label, feat_dict in [
            ("backbone", "Backbone (512-d)", all_backbone_features),
            ("aligned", "Aligned (256-d)", all_aligned_features),
        ]:
            features = feat_dict[model_name]
            safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")

            # PCA
            if args.method in ['pca', 'all']:
                pca_emb = compute_pca(features)
                fig, ax = plt.subplots(1, 1, figsize=(6, 5))
                plot_embedding(ax, pca_emb, center_labels, class_labels,
                               f"PCA — {model_name}\n{feat_label}")
                               
                # Single plot legend
                center_handles = [mlines.Line2D([], [], color=CENTER_COLORS[c], marker='o', markersize=6, linestyle='None', label=c) for c in TEST_CENTERS]
                class_handles = [mlines.Line2D([], [], color='gray', marker=CLASS_MARKERS[cls], markersize=6, linestyle='None', label=CLASS_LABELS[cls]) for cls in CLASS_MARKERS]
                ax.legend(handles=center_handles + class_handles, fontsize=8, loc='best', frameon=True, fancybox=True)
                
                plt.tight_layout()
                for ext in ['png', 'pdf']:
                    path = os.path.join(individual_dir, f"pca_{feat_tag}_{safe_name}.{ext}")
                    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close(fig)

            # t-SNE
            if args.method in ['tsne', 'both', 'all']:
                tsne_emb = compute_tsne(features)
                fig, ax = plt.subplots(1, 1, figsize=(6, 5))
                plot_embedding(ax, tsne_emb, center_labels, class_labels,
                               f"t-SNE — {model_name}\n{feat_label}")
                               
                # Single plot legend
                center_handles = [mlines.Line2D([], [], color=CENTER_COLORS[c], marker='o', markersize=6, linestyle='None', label=c) for c in TEST_CENTERS]
                class_handles = [mlines.Line2D([], [], color='gray', marker=CLASS_MARKERS[cls], markersize=6, linestyle='None', label=CLASS_LABELS[cls]) for cls in CLASS_MARKERS]
                ax.legend(handles=center_handles + class_handles, fontsize=8, loc='best', frameon=True, fancybox=True)
                
                plt.tight_layout()
                for ext in ['png', 'pdf']:
                    path = os.path.join(individual_dir, f"tsne_{feat_tag}_{safe_name}.{ext}")
                    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close(fig)

            # UMAP
            if (args.method in ['umap', 'both', 'all']) and HAS_UMAP:
                umap_emb = compute_umap(features)
                fig, ax = plt.subplots(1, 1, figsize=(6, 5))
                plot_embedding(ax, umap_emb, center_labels, class_labels,
                               f"UMAP — {model_name}\n{feat_label}")
                               
                # Single plot legend
                center_handles = [mlines.Line2D([], [], color=CENTER_COLORS[c], marker='o', markersize=6, linestyle='None', label=c) for c in TEST_CENTERS]
                class_handles = [mlines.Line2D([], [], color='gray', marker=CLASS_MARKERS[cls], markersize=6, linestyle='None', label=CLASS_LABELS[cls]) for cls in CLASS_MARKERS]
                ax.legend(handles=center_handles + class_handles, fontsize=8, loc='best', frameon=True, fancybox=True)
                
                plt.tight_layout()
                for ext in ['png', 'pdf']:
                    path = os.path.join(individual_dir, f"umap_{feat_tag}_{safe_name}.{ext}")
                    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close(fig)

    print(f"\n{'=' * 60}")
    print("Done! All figures saved to:")
    print(f"  Combined:   {output_dir}/tsne_umap_backbone.{{png,pdf}}")
    print(f"              {output_dir}/tsne_umap_aligned.{{png,pdf}}")
    print(f"  Individual: {individual_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
