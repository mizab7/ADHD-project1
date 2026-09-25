import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

results_dir = "./results"
os.makedirs(results_dir, exist_ok=True)

# 1. Plot Top Influential Brain Regions Bar Chart
rois_csv = os.path.join(results_dir, "fast_demo_top_rois.csv")
if os.path.exists(rois_csv):
    df_rois = pd.read_csv(rois_csv)
    
    plt.figure(figsize=(10, 6), dpi=150)
    sns.set_theme(style="whitegrid")
    
    palette = sns.color_palette("mako", len(df_rois))
    ax = sns.barplot(
        x="importance_score", 
        y="roi_name", 
        data=df_rois, 
        palette=palette,
        hue="roi_name",
        legend=False
    )
    
    plt.title("Top AAL Brain Regions Influencing ADHD Diagnosis (Grad-CAM)", fontsize=14, pad=15, fontweight="bold")
    plt.xlabel("Importance Score (Grad-CAM Weight)", fontsize=12)
    plt.ylabel("AAL Anatomical Brain Region", fontsize=12)
    
    # Annotate bar values
    for p in ax.patches:
        width = p.get_width()
        ax.annotate(f"{width:.1f}", (width + 1, p.get_y() + p.get_height() / 2),
                    ha="left", va="center", fontsize=10, color="#333333", fontweight="semibold")
                    
    plt.tight_layout()
    chart_path = os.path.join(results_dir, "top_brain_regions.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"[SUCCESS] Saved brain region chart to: {chart_path}")

# 2. Plot Dynamic Connectivity Heatmap for a Sample Window
sample_ts_path = "./data/cached_features/Peking_1_sub-1038415_aal116.npy"
if os.path.exists(sample_ts_path):
    ts = np.load(sample_ts_path)
    # Compute correlation matrix for first 30 timepoints
    w = ts[:30, :]
    stds = np.std(w, axis=0, keepdims=True)
    stds[stds == 0] = 1e-8
    norm = (w - np.mean(w, axis=0, keepdims=True)) / stds
    corr = (norm.T @ norm) / (30 - 1)
    np.fill_diagonal(corr, 0.0)
    
    plt.figure(figsize=(9, 8), dpi=150)
    sns.heatmap(
        corr, 
        cmap="coolwarm", 
        center=0.0, 
        vmin=-0.8, 
        vmax=0.8,
        cbar_kws={'label': 'Functional Correlation (Sync)'}
    )
    plt.title("Brain Functional Connectivity Matrix (116 x 116 AAL Regions)\nPatient: sub-1038415 (Peking Hospital)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("AAL Brain Region Index (0 to 115)", fontsize=11)
    plt.ylabel("AAL Brain Region Index (0 to 115)", fontsize=11)
    
    plt.tight_layout()
    heatmap_path = os.path.join(results_dir, "brain_connectivity_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"[SUCCESS] Saved connectivity heatmap to: {heatmap_path}")
