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
from tqdm import tqdm

# Reproducibility
SEED = 42
set_determinism(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROI_SIZE = (256, 256, 128)

# DATA_DIR: path to folder containing test CSV files. Override with --data_dir argument or DATA_DIR env var.
_DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "all_combine_retrain_files")
DATA_DIR = os.environ.get("DATA_DIR", _DEFAULT_DATA_DIR)

TEST_CENTERS = {
    "UKB":       os.path.join(DATA_DIR, "test_ukb.csv"),
    "Berlin":    os.path.join(DATA_DIR, "test_berlin.csv"),
    "Göttingen": os.path.join(DATA_DIR, "test_g.csv"),
}

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

def load_test_data(roi_size, batch_size=4, subset_n=None, num_workers=4):
    val_transform = create_val_transform(roi_size)

    all_data_list = []
    center_labels = []  
    class_labels = []   

    for center_name, csv_path in TEST_CENTERS.items():
        df = pd.read_csv(csv_path)
        print(f"  {center_name}: {len(df)} samples in test set")
        if subset_n is not None:
            df = df.head(subset_n)

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
    print("Caching pre-processed images in RAM...")
    dataset = CacheDataset(data=all_data_list, transform=val_transform, cache_rate=1.0, num_workers=num_workers)
    
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return dataloader, center_labels, class_labels

def extract_features_from_model(model, dataloader):
    model.eval()
    all_backbone = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            batch = move_batch_to_device(batch, DEVICE)
            backbone_feat, _ = model.extract_features(batch["image"])
            all_backbone.append(backbone_feat.cpu().numpy())

    return np.concatenate(all_backbone, axis=0)

def main():
    parser = argparse.ArgumentParser(description="Extract features for visualization")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to folder containing test CSV files (default: ../data/all_combine_retrain_files)")
    parser.add_argument("--output_dir", type=str,
                        default="results/extracted_features",
                        help="Directory to save extracted features")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for inference (increase for speed)")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of workers for data loading")
    parser.add_argument("--subset_n", type=int, default=None,
                        help="Number of samples to take per center for quick testing")
    args = parser.parse_args()

    if args.data_dir is not None:
        global DATA_DIR, TEST_CENTERS
        DATA_DIR = args.data_dir
        TEST_CENTERS = {
            "UKB":       os.path.join(DATA_DIR, "test_ukb.csv"),
            "Berlin":    os.path.join(DATA_DIR, "test_berlin.csv"),
            "Göttingen": os.path.join(DATA_DIR, "test_g.csv"),
        }

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    print("\n" + "=" * 60)
    print(f"Loading test data (subset_n={args.subset_n})...")
    print("=" * 60)
    dataloader, center_labels, class_labels = load_test_data(
        ROI_SIZE, batch_size=args.batch_size, subset_n=args.subset_n, num_workers=args.num_workers
    )

    pretrained_model = SegResNet.from_pretrained("project-lighter/ct_fm_segresnet")
    encoder = pretrained_model.encoder
    model = CTClassificationModel(encoder, dropout_rate=None).to(DEVICE)
    model.eval()

    backbone = extract_features_from_model(model, dataloader)

    backbone_path = os.path.join(output_dir, "pretrained_raw_features.npz")
    np.savez_compressed(backbone_path, **{"PreTrained_SegResNet": backbone})
    print(f"  Saved: {backbone_path}")


if __name__ == "__main__":
    main()
