import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.manifold import TSNE
import umap
import argparse

# Input parameters
input_dir = 'results/extracted_features'
output_dir = 'results/plots'
os.makedirs(output_dir, exist_ok=True)
subset_n = 10
seed = 0

random.seed(seed)
np.random.seed(seed)

# Load labels
center_labels_path = os.path.join(input_dir, 'center_labels.npy')
class_labels_path = os.path.join(input_dir, 'class_labels.npy')

if not os.path.exists(center_labels_path):
    raise FileNotFoundError(f"Missing {center_labels_path}. Please run feature extraction first.")

center_labels = np.load(center_labels_path).tolist()
class_labels = np.load(class_labels_path).tolist()

# Load aligned features
all_aligned_features = {}
aligned_path = os.path.join(input_dir, 'all_aligned_features.npz')
if os.path.exists(aligned_path):
    with np.load(aligned_path) as data:
        for k in data.files:
            all_aligned_features[k] = data[k]
else:
    raise FileNotFoundError(f"Missing {aligned_path}. Please run feature extraction first.")

# Load pretrained raw features
pretrained_features = {}
pretrained_path = os.path.join(input_dir, 'pretrained_raw_features.npz')
if os.path.exists(pretrained_path):
    with np.load(pretrained_path) as data:
        pretrained_features = data['PreTrained_SegResNet']
else:
    raise FileNotFoundError(f"Missing {pretrained_path}. Please run pretrained extraction first.")

TEST_CENTERS = ['UKB', 'Berlin', 'Göttingen']
CLASS_LABELS = {0: 'N0', 1: 'N1'}

if subset_n > 0:
    print(f'Subsetting to {subset_n} N0 and {subset_n} N1 samples per center...')
    df_meta = pd.DataFrame({'center': center_labels, 'label': class_labels, 'orig_idx': np.arange(len(center_labels))})
    final_indices = []
    
    for center_name in TEST_CENTERS:
        df_c = df_meta[df_meta['center'] == center_name]
        df_c0 = df_c[df_c['label'] == 0]
        df_c1 = df_c[df_c['label'] == 1]
        
        s0 = df_c0.sample(n=min(len(df_c0), subset_n), random_state=seed).sort_index()
        s1 = df_c1.sample(n=min(len(df_c1), subset_n), random_state=seed).sort_index()
        
        final_indices.extend(s0['orig_idx'].tolist())
        final_indices.extend(s1['orig_idx'].tolist())
        
    all_indices = final_indices
    center_labels = [center_labels[i] for i in all_indices]
    class_labels = [class_labels[i] for i in all_indices]
    
    for k in all_aligned_features:
        all_aligned_features[k] = all_aligned_features[k][all_indices]
    pretrained_features = pretrained_features[all_indices]


# Plotting
CENTER_COLORS = {'UKB': 'blue', 'Berlin': 'green', 'Göttingen': 'red'}
CLASS_MARKERS = {0: 'o', 1: 'X'}

def plot_embedding(ax, embedding, center_labels, class_labels, title):
    for center in TEST_CENTERS:
        for cls in [0, 1]:
            mask = np.array([c == center and cl == cls for c, cl in zip(center_labels, class_labels)])
            if np.any(mask):
                ax.scatter(embedding[mask, 0], embedding[mask, 1], c=CENTER_COLORS[center], marker=CLASS_MARKERS[cls], s=50, alpha=0.7)
    ax.set_title(title, fontsize=12)
    ax.axis('off')
    ax.grid(True, linestyle='--', alpha=0.5)

def plot_embedding_3d(ax, embedding, center_labels, class_labels, title):
    for center in TEST_CENTERS:
        for cls in [0, 1]:
            mask = np.array([c == center and cl == cls for c, cl in zip(center_labels, class_labels)])
            if np.any(mask):
                ax.scatter(embedding[mask, 0], embedding[mask, 1], embedding[mask, 2], c=CENTER_COLORS[center], marker=CLASS_MARKERS[cls], s=50, alpha=0.7)
    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_mode', type=str, default='2D', choices=['2D', '3D'])
    parser.add_argument('--tsne_perplexity', type=int, default=30)
    parser.add_argument('--tsne_max_iter', type=int, default=250)
    parser.add_argument('--umap_n_neighbors', type=int, default=15)
    parser.add_argument('--umap_min_dist', type=float, default=0.17)
    args = parser.parse_args()

    n_dims = 3 if args.plot_mode == '3D' else 2

    # To match the user's logic, compare PreTrained Features vs Aligned Baselines
    baselines = ['FedAvg', 'FedBN', 'CTFM (Ours)']
    fig = plt.figure(figsize=(18, 16), dpi=300)
    fig.suptitle(f'{args.plot_mode} Feature Comparison: Untrained Backbone vs Aligned Baselines', fontsize=18, y=0.98)

    def compute_and_plot(ax, data, method, title):
        if method == 't-SNE':
            reducer = TSNE(n_components=n_dims, perplexity=min(args.tsne_perplexity, len(data)-1), max_iter=args.tsne_max_iter, random_state=seed, init='pca', learning_rate='auto')
        else:
            reducer = umap.UMAP(n_components=n_dims, n_neighbors=min(args.umap_n_neighbors, len(data)-1), min_dist=args.umap_min_dist, random_state=seed)
        
        emb = reducer.fit_transform(data)
        if n_dims == 3:
            plot_embedding_3d(ax, emb, center_labels, class_labels, title)
        else:
            plot_embedding(ax, emb, center_labels, class_labels, title)

    for i, baseline in enumerate(baselines):
        ax1 = fig.add_subplot(4, len(baselines), i + 1, projection='3d' if n_dims==3 else None)
        compute_and_plot(ax1, pretrained_features, 't-SNE', f'Pretrained Raw (t-SNE)')
        
        ax2 = fig.add_subplot(4, len(baselines), len(baselines) + i + 1, projection='3d' if n_dims==3 else None)
        compute_and_plot(ax2, all_aligned_features.get(baseline, []), 't-SNE', f'{baseline} Aligned (t-SNE)')
        
        ax3 = fig.add_subplot(4, len(baselines), 2*len(baselines) + i + 1, projection='3d' if n_dims==3 else None)
        compute_and_plot(ax3, pretrained_features, 'UMAP', f'Pretrained Raw (UMAP)')
        
        ax4 = fig.add_subplot(4, len(baselines), 3*len(baselines) + i + 1, projection='3d' if n_dims==3 else None)
        compute_and_plot(ax4, all_aligned_features.get(baseline, []), 'UMAP', f'{baseline} Aligned (UMAP)')

    center_handles = [mlines.Line2D([], [], color=CENTER_COLORS[c], marker='o', markersize=8, linestyle='None', label=c) for c in TEST_CENTERS]
    class_handles = [mlines.Line2D([], [], color='gray', marker=CLASS_MARKERS[cls], markersize=8, linestyle='None', label=CLASS_LABELS[cls]) for cls in CLASS_MARKERS]
    fig.legend(handles=center_handles + class_handles, loc='lower center', ncol=len(TEST_CENTERS) + len(CLASS_MARKERS), bbox_to_anchor=(0.5, 0.02))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save the figure
    output_path = os.path.join(output_dir, f'pretrained_vs_aligned_comparison_{args.plot_mode}.png')
    plt.savefig(output_path, bbox_inches='tight')
    print(f"\nPlot successfully saved to: {output_path}")

if __name__ == "__main__":
    main()
