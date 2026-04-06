# Federated CT Foundation Models for Multi-Center Detection of Lymph Node Metastasis in Pancreatic Cancer

**Official implementation** of the paper:

> **Federated CT Foundation Models for Multi-Center Detection of Lymph Node Metastasis in Pancreatic Cancer**  
> Parinishtha Bhalla, David Dueñas Gaviria, Patrick Kupczyk, Ali Seif Amir Hosseini, Lena Conradi, Uli Fehrenbach, Matthaeus Felsenstein, Dou Ma, Alexander Semaan, and Shadi Albarqouni  
> *Scientific Reports* (accepted)  

## Overview

This work introduces a privacy-preserving federated deep learning framework for preoperative lymph node metastasis (LNM) detection in pancreatic ductal adenocarcinoma (PDAC). We integrate large-scale CT foundation model pre-training with a heterogeneity-aware federated aggregation strategy to enable collaborative training across three German institutions without sharing raw patient data.

**Key contributions:**
- Fine-tuning of the [CT Vision Foundation Model (CT-FM)](https://huggingface.co/project-lighter/ct_fm_segresnet) on multi-center PDAC CT data for LNM classification.
- A heterogeneity-aware federated aggregation strategy (PDAC-CTFM<sub>Ours</sub>) that jointly accounts for label-distribution imbalance and representation-level divergence across clients.
- Multi-center evaluation on 546 patients from three institutions (UKB Bonn, Charité Berlin, UMG Göttingen).

**Main results (aggregated test set):**

| Model | Balanced Acc. | DOR | AUPRC |
|---|---|---|---|
| PDAC-CTFM<sub>FedAvg</sub> | 0.4607 | 0.2500 | 0.6968 |
| PDAC-CTFM<sub>Ours</sub> | **0.5866** | **2.1010** | 0.6752 |
| PDAC-CTFM<sub>Central</sub> (upper bound) | 0.6010 | 3.4541 | 0.7376 |

Our federated method outperforms all FL baselines (FedAvg, FedDisco, FedBN, FedALA) by **+12.6% balanced accuracy** over FedAvg while preserving data privacy.

## Repository Structure

```
pdac-ctfm-scientific-journal/
├── scripts/
│   ├── ctfm.py                          # Centralized fine-tuning (upper-bound reference)
│   ├── feddisco_modified.py             # Proposed method: Heterogeneity-Aware FedDisco
│   ├── fedBN.py                         # FL baseline: FedBN
│   ├── fedALA.py                        # FL baseline: FedALA
│   ├── ala_module.py                    # Adaptive Layer Alignment (ALA) module for FedALA
│   ├── extract_pretrained_features.py   # Extract CT-FM features for classical ML baselines
│   ├── eval_and_save_npz.py             # Evaluate trained models and save predictions
│   ├── test_fl_model.py                 # Test any FL model checkpoint and generate metrics
│   ├── tsne_umap_visualization.py       # t-SNE/UMAP feature space visualization
│   ├── plot_features_visualization.py   # Plot extracted feature distributions
│   └── plot_backbone_vs_ctfm.py         # Compare backbone vs. fine-tuned representations
├── requirements/
│   └── environment.yml                  # Full conda environment specification
├── requirements.txt                     # Minimal pip requirements
└── .gitignore
```

## Setup

### Prerequisites

- Python 3.10.9
- CUDA-capable GPU (experiments run on NVIDIA A100)
- [Conda](https://docs.conda.io/) (recommended)

### Installation

**Option 1 — Conda (recommended):**
```bash
conda env create -f requirements/environment.yml
conda activate pdac-ctfm
conda install pip
```

**Option 2 — pip:**
```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

Key dependencies: PyTorch, [MONAI](https://monai.io/), [lighter-zoo](https://github.com/project-lighter/lighter) (for the CT-FM backbone), scikit-learn, timm, wandb.

## Data

The dataset consists of **546 patients** from three German institutions diagnosed with PDAC:
- **UKB** — University Hospital Bonn (*n* = 180)
- **Berlin** — Charité – University Medicine Berlin (*n* = 163)
- **Göttingen** — University Medicine Göttingen (*n* = 203)

All CT volumes are contrast-enhanced abdominal scans acquired in the portal-venous phase. LNM labels are derived from postoperative histopathology and/or multidisciplinary tumor board consensus.

**Data availability:** Patient data cannot be shared publicly due to privacy regulations. 

**Updating image paths:** Before running any script, update the `path` column in the CSV files to match the location of the CT volumes on your system. Each CSV (e.g., `train_ukb.csv`, `test_berlin.csv`) references one center's split.

## Configuring Paths

All scripts default to reading CSVs from `data/all_combine_retrain_files/` relative to the repository root. Override via:

```bash
# Environment variable
export DATA_DIR=/path/to/your/csv/folder
export SAVE_DIR=/path/to/save/results

# Or pass directly (where supported)
python scripts/test_fl_model.py --data_dir /path/to/csv/folder --model_path /path/to/model.pth
```

## Usage

### 1. Centralized Fine-Tuning (upper-bound reference)

Fine-tunes the CT-FM backbone end-to-end on pooled multi-center data:

```bash
python scripts/ctfm.py
```

Trains with AdamW (lr=1e-3, weight decay=0.01), cosine annealing, batch size 4 (with gradient accumulation, effective batch size 32), for 100 epochs.

### 2. Federated Learning — Proposed Method (Heterogeneity-Aware FedDisco)

```bash
python scripts/feddisco_modified.py
```

Runs PDAC-CTFM<sub>Ours</sub> with default hyperparameters (*a*=0.1, *b*=0.1, *γ*=10). To sweep hyperparameters, edit the `a_values`, `b_values`, `gamma_values` lists at the bottom of the script:

```python
# In feddisco_modified.py, at __main__:
a_values = [0.1]
b_values = [0.1]
gamma_values = [10]
```

- `a` — sensitivity to label-distribution imbalance (KL divergence weight)
- `b` — bias term for aggregation weights
- `γ` — sensitivity to representation-level divergence (cosine distance weight)
- Set `use_feddisco=False` to run vanilla **FedAvg** (a=b=γ=0)

### 3. Federated Learning — Baselines

**FedBN** (siloed batch normalization):
```bash
python scripts/fedBN.py
```

**FedALA** (adaptive local aggregation):
```bash
python scripts/fedALA.py
```

### 4. Classical ML Baselines (CT-FM Features)

Extract frozen CT-FM features, then fit classical classifiers (Logistic Regression, LDA, SVM, Gradient Boosting, Random Forest, MLP):

```bash
python scripts/extract_pretrained_features.py \
    --data_dir data/all_combine_retrain_files \
    --output_dir results/extracted_features
```

### 5. Evaluation

Evaluate a trained model and save predictions as `.npz` for downstream plotting:

```bash
python scripts/eval_and_save_npz.py
```

Update `MODELS` dict in the script with paths to your trained checkpoints before running.

Test any trained FL model (FedBN, FedALA, FedDisco, FedAvg, Centralized) with per-center metrics:

```bash
python scripts/test_fl_model.py \
    --model_path results/fedbn/run_XXXX/models/best_model.pth \
    --data_dir data/all_combine_retrain_files \
    --output_dir results/test_fl_model
```

### 6. Visualization

**t-SNE / UMAP feature space:**
```bash
python scripts/tsne_umap_visualization.py \
    --data_dir data/all_combine_retrain_files \
    --output_dir results/tsne_umap_figures \
    --method all
```

Update `MODEL_CONFIGS` in the script with checkpoint paths before running.

**Feature distribution plots:**
```bash
python scripts/plot_features_visualization.py \
    --input_dir results/extracted_features \
    --output_dir results/plots
```

**Backbone vs. fine-tuned comparison:**
```bash
python scripts/plot_backbone_vs_ctfm.py --plot_mode 2D
```

## Model Architecture

All models use the **CT Vision Foundation Model (CT-FM)** as the backbone encoder:
- Architecture: [SegResNet](https://docs.monai.io/en/stable/networks.html#segresnet), pre-trained via contrastive self-supervised learning on 148,000 volumetric CT scans from the [Imaging Data Commons (IDC)](https://imaging.datacommons.cancer.gov/).
- Loaded automatically from HuggingFace: `project-lighter/ct_fm_segresnet`
- A two-layer classification head (512 → 256 → 2) with ReLU activations and optional dropout is appended for the binary LNM prediction task.

**Input preprocessing:**
- Resampled to 1 mm isotropic voxel spacing
- Intensity clipped to [−1000, 2000] HU, z-score normalized per patient
- ROI: 256 × 256 × 128 voxels centered on the pancreas (via TotalSegmentator masks)
- Augmentations: random flips, zooming (0.9–1.2), Gaussian noise (σ=0.01), intensity scaling (±0.3), random 90° rotations

## Federated Learning Setup

Training follows a standard FL protocol:
- **3 clients** (one per institution), each holding local data
- **Communication rounds:** 20–40
- **Local epochs per round:** 1–3
- Only model **weights** are transmitted; no raw images or patient data leave the institution
- Global model initialized from the pre-trained CT-FM backbone

The proposed aggregation rule (PDAC-CTFM<sub>Ours</sub>) computes per-client weights as:

$$p_k \propto \text{ReLU}(n_k - a \cdot d_k - \gamma \Delta_k + b)$$

where $d_k$ = KL divergence of client label distribution from uniform, $\Delta_k$ = cosine distance between client and global classifier weights, and $n_k$ = number of training samples.

## Citation

If you use this code, please cite:

```bibtex
@article{bhalla2026federated,
  title   = {Federated {CT} Foundation Models for Multi-Center Detection of Lymph Node Metastasis in Pancreatic Cancer},
  author  = {Bhalla, Parinishtha and Due{\~n}as Gaviria, David and Kupczyk, Patrick and Hosseini, Ali Seif Amir and Conradi, Lena and Fehrenbach, Uli and Felsenstein, Matthaeus and Ma, Dou and Semaan, Alexander and Albarqouni, Shadi},
  journal = {Scientific Reports},
  year    = {2026}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgements

- CT Foundation Model: [project-lighter/lighter-zoo](https://github.com/project-lighter/lighter-zoo)
- FedALA implementation adapted from [TsingZ0/FedALA](https://github.com/TsingZ0/FedALA)
- Medical image processing: [MONAI](https://monai.io/)
- Experiment tracking: [Weights & Biases](https://wandb.ai/)
