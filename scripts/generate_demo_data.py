"""
Generate synthetic demo data for PDAC-CTFM.

Creates small random NIfTI volumes and matching CSV files so the full
pipeline can run end-to-end without access to real patient data.
The synthetic volumes are tiny (64x64x32) to keep generation fast and
storage minimal.  Results will be meaningless — this is only to prove
the code executes correctly.

Usage
-----
    python scripts/generate_demo_data.py            # default output
    python scripts/generate_demo_data.py --out_dir data/demo   # custom path
"""

import argparse
import os
import random

import nibabel as nib
import numpy as np
import pandas as pd

# ── Configuration ────────────────────────────────────────────────────
CENTERS = {
    "UKB":       {"n_train": 8, "n_val": 2, "n_test": 3},
    "Berlin":    {"n_train": 8, "n_val": 2, "n_test": 3},
    "Goettingen": {"n_train": 8, "n_val": 2, "n_test": 3},
}

CENTER_CSV_MAP = {
    "UKB":        "ukb",
    "Berlin":     "berlin",
    "Goettingen": "g",
}

VOLUME_SHAPE = (64, 64, 32)       # Small volumes — fast to generate
AFFINE = np.eye(4)                # 1 mm isotropic identity affine

SEED = 42

# ── Helpers ──────────────────────────────────────────────────────────

def _random_ct_volume(shape):
    """Return a random volume with HU-like values in [-1000, 2000]."""
    return np.random.uniform(-1000, 2000, size=shape).astype(np.float32)


def _random_seg_mask(shape, blob_radius=8):
    """Return a binary mask with a small spherical blob near the center."""
    mask = np.zeros(shape, dtype=np.uint8)
    cx, cy, cz = [s // 2 for s in shape]
    # Add jitter
    cx += np.random.randint(-4, 5)
    cy += np.random.randint(-4, 5)
    cz += np.random.randint(-2, 3)
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2 + (zz - cz)**2)
    mask[dist <= blob_radius] = 1
    return mask


def _random_clinical_row(patient_id, center, n_status):
    """Build one CSV row with random but schema-valid clinical features."""
    g_value = random.choice([2.0, 3.0])
    head = random.randint(0, 1)
    tail = 1 - head
    t_stage = [0] * 5
    t_stage[random.randint(0, 4)] = 1
    return {
        "ID": patient_id,
        "N Status": n_status,
        "G_value": g_value,
        "Head_Localization": head,
        "Tail_Localization": tail,
        "T_0": t_stage[0],
        "T_1": t_stage[1],
        "T_2": t_stage[2],
        "T_3": t_stage[3],
        "T_4": t_stage[4],
        "center": center,
    }


# ── Main ─────────────────────────────────────────────────────────────

def generate(out_dir: str):
    random.seed(SEED)
    np.random.seed(SEED)

    nifti_dir = os.path.join(out_dir, "nifti")
    csv_dir   = os.path.join(out_dir, "csv")
    os.makedirs(nifti_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    patient_counter = 1000

    for center, splits in CENTERS.items():
        csv_tag = CENTER_CSV_MAP[center]

        for split, n_samples in [("train", splits["n_train"]),
                                  ("val",   splits["n_val"]),
                                  ("test",  splits["n_test"])]:
            rows = []
            for _ in range(n_samples):
                pid = patient_counter
                patient_counter += 1
                n_status = random.randint(0, 1)

                # --- write NIfTI volume ---
                vol = _random_ct_volume(VOLUME_SHAPE)
                vol_path = os.path.join(nifti_dir, f"{pid}.nii")
                nib.save(nib.Nifti1Image(vol, AFFINE), vol_path)

                # --- write segmentation mask ---
                seg = _random_seg_mask(VOLUME_SHAPE)
                seg_path = os.path.join(nifti_dir, f"{pid}_seg.nii.gz")
                nib.save(nib.Nifti1Image(seg, AFFINE), seg_path)

                row = _random_clinical_row(pid, center, n_status)
                row["path"]     = os.path.abspath(vol_path)
                row["path_seg"] = os.path.abspath(seg_path)
                rows.append(row)

            df = pd.DataFrame(rows)
            csv_path = os.path.join(csv_dir, f"{split}_{csv_tag}.csv")
            df.to_csv(csv_path, index=False)
            print(f"  {csv_path}  ({len(df)} samples)")

    # The pipeline also loads val_mel.csv — create a minimal one
    mel_rows = []
    for _ in range(2):
        pid = patient_counter
        patient_counter += 1
        n_status = random.randint(0, 1)
        vol = _random_ct_volume(VOLUME_SHAPE)
        vol_path = os.path.join(nifti_dir, f"{pid}.nii")
        nib.save(nib.Nifti1Image(vol, AFFINE), vol_path)
        seg = _random_seg_mask(VOLUME_SHAPE)
        seg_path = os.path.join(nifti_dir, f"{pid}_seg.nii.gz")
        nib.save(nib.Nifti1Image(seg, AFFINE), seg_path)
        row = _random_clinical_row(pid, "Mel", n_status)
        row["path"]     = os.path.abspath(vol_path)
        row["path_seg"] = os.path.abspath(seg_path)
        mel_rows.append(row)

    mel_csv = os.path.join(csv_dir, "val_mel.csv")
    pd.DataFrame(mel_rows).to_csv(mel_csv, index=False)
    print(f"  {mel_csv}  ({len(mel_rows)} samples)")

    # Also create test_mel.csv for completeness
    test_mel_csv = os.path.join(csv_dir, "test_mel.csv")
    pd.DataFrame(mel_rows).to_csv(test_mel_csv, index=False)
    print(f"  {test_mel_csv}  ({len(mel_rows)} samples)")

    print(f"\nDone. Demo data written to: {os.path.abspath(out_dir)}")
    print(f"Total NIfTI files: {(patient_counter - 1000) * 2}")
    print(f"\nTo run the pipeline with demo data:")
    print(f"  export DATA_DIR={os.path.abspath(csv_dir)}")
    print(f"  python scripts/ctfm.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic demo data for PDAC-CTFM")
    parser.add_argument(
        "--out_dir",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "demo"),
        help="Output directory for demo data (default: data/demo)",
    )
    args = parser.parse_args()
    generate(args.out_dir)
