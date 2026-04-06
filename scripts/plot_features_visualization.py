"""
Feature Visualization Script
============================
Loads extracted features (from extract_features_visualization.py)
and generates t-SNE and UMAP scatter plots. This decoupled approach
allows for rapid iteration on visual settings without recomputing inference.
"""

import os
import argparse
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

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
random.seed(SEED)
np.random.seed(SEED)

# ──────────────────────────────────────────────
# Center config
# ──────────────────────────────────────────────
TEST_CENTERS = ["UKB", "Berlin", "Göttingen"]

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
# Visualization Definitions
# ──────────────────────────────────────────────
def compute_tsne(features, perplexity=30, n_iter=1000):
    """Compute t-SNE embedding."""
    try:
        tsne = TSNE(
            n_components=2, perplexity=min(perplexity, len(features) - 1),
            n_iter=n_iter, random_state=SEED, init='pca', learning_rate='auto',
        )
    except TypeError:
        # Fallback for older scikit-learn versions
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
    parser = argparse.ArgumentParser(description="Plot t-SNE/UMAP of extracted features")
    parser.add_argument("--input_dir", type=str,
                        default="results/extracted_features",
                        help="Directory containing extracted features and labels")
    parser.add_argument("--output_dir", type=str,
                        default="results/tsne_umap_figures",
                        help="Directory to save output figures")
    parser.add_argument("--method", type=str, default="all", choices=["tsne", "umap", "pca", "both", "all"],
                        help="Visualization method to run")
    parser.add_argument("--subset_n", type=int, default=10,
                        help="Number of samples per class (N0/N1) to use per center (e.g. 10)")
    args = parser.parse_args()

    if args.method == "umap" and not HAS_UMAP:
        print("ERROR: UMAP selected but umap-learn is not installed.")
        return

    # Resolve dirs relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_dir = os.path.join(project_root, args.input_dir)
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    labels_path = os.path.join(input_dir, "center_labels.npy")
    class_labels_path = os.path.join(input_dir, "class_labels.npy")
    backbone_path = os.path.join(input_dir, "all_backbone_features.npz")
    aligned_path = os.path.join(input_dir, "all_aligned_features.npz")

    if not os.path.exists(labels_path) or not os.path.exists(class_labels_path) or not os.path.exists(backbone_path) or not os.path.exists(aligned_path):
        print(f"Error: Could not find extracted features or labels in {input_dir}.")
        print("Make sure you run extract_features_visualization.py first.")
        return

    print(f"Loading extracted features from {input_dir}...")
    center_labels = np.load(labels_path).tolist()
    class_labels = np.load(class_labels_path).tolist()

    # Features will be loaded first, then subsetted, then counts printed

    all_backbone_features = {}
    with np.load(backbone_path) as data:
        for k in data.files:
            all_backbone_features[k] = data[k]

    all_aligned_features = {}
    with np.load(aligned_path) as data:
        for k in data.files:
            all_aligned_features[k] = data[k]
            
    if args.subset_n is not None:
        print(f"Subsetting to {args.subset_n} N0 and {args.subset_n} N1 samples per center...")
        
        # Create a temporary dataframe for sampling indices
        df_meta = pd.DataFrame({
            'center': center_labels,
            'label': class_labels,
            'orig_idx': np.arange(len(center_labels))
        })
        
        final_indices = []
        for center_name in TEST_CENTERS:
            df_c = df_meta[df_meta['center'] == center_name]
            df_c0 = df_c[df_c['label'] == 0]
            df_c1 = df_c[df_c['label'] == 1]
            
            # Deterministic sampling consistent with extraction script
            s0 = df_c0.sample(n=min(len(df_c0), args.subset_n), random_state=SEED).sort_index()
            s1 = df_c1.sample(n=min(len(df_c1), args.subset_n), random_state=SEED).sort_index()
            
            final_indices.extend(s0['orig_idx'].tolist())
            final_indices.extend(s1['orig_idx'].tolist())
        
        # Apply subsetting to all data structures
        all_indices = final_indices # Keep the sampled order (UKB_N0, UKB_N1, Berlin_N0, ...)
        
        center_labels = [center_labels[i] for i in all_indices]
        class_labels = [class_labels[i] for i in all_indices]
        for k in all_backbone_features:
            all_backbone_features[k] = all_backbone_features[k][all_indices]
        for k in all_aligned_features:
            all_aligned_features[k] = all_aligned_features[k][all_indices]
            
    # Detailed count printing per center (Final counts after subsetting)
    print("\n--- Final Data Counts per Center (used for plotting) ---")
    print("-" * 30)
    for center_name in TEST_CENTERS:
        indices = [i for i, c in enumerate(center_labels) if c == center_name]
        c_labels = [class_labels[i] for i in indices]
        n0 = c_labels.count(0)
        n1 = c_labels.count(1)
        print(f"  {center_name:10}: Total={len(c_labels):3}, N0={n0:3}, N1={n1:3}")
    print("-" * 30)
            
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

    model_names = list(all_backbone_features.keys())
    for model_name in model_names:
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
