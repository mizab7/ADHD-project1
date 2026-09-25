import os
import sys
import torch
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.loso_trainer import LOSOTrainer
from models.conv_lstm import ConvLSTMClassifier
from models.connectivity import DynamicConnectivityGenerator
from explainability.grad_cam import ConvLSTMGradCAM

def run_fast_demo():
    print("=" * 70, flush=True)
    print(" ADHD-200 FAST DEMONSTRATION RUN (~60 seconds)", flush=True)
    print("=" * 70, flush=True)

    cache_dir = "./data/cached_features"
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)

    manifest_path = os.path.join(cache_dir, "dataset_manifest.csv")
    if not os.path.exists(manifest_path):
        print("[ERROR] Dataset manifest not found.", flush=True)
        return

    print("\n[1/3] Loading cached dataset manifest...", flush=True)
    manifest_df = pd.read_csv(manifest_path)
    print(f"       Total available subjects: {len(manifest_df)}", flush=True)

    # Train and evaluate 1 held-out hospital fold (Peking_1: 45 subjects, 23 controls / 22 ADHD)
    test_site = "Peking_1"
    print(f"\n[2/3] Training and evaluating on held-out hospital [{test_site}]...", flush=True)
    print("       - Noise reduction: 1D Denoising Autoencoder (DAE)", flush=True)
    print("       - Harmonization: Zero-leakage ComBat", flush=True)
    print("       - Deep Learning: ConvLSTM running on Apple Silicon GPU (mps)", flush=True)
    
    trainer = LOSOTrainer(manifest_df, device="mps")
    res = trainer.train_and_eval_fold(
        test_site=test_site,
        model_type="conv_lstm",
        use_dae=True,
        use_combat=True,
        connectivity_mode="dynamic",
        epochs=12
    )

    if res is None:
        print("[ERROR] Fold evaluation failed.", flush=True)
        return

    metrics, trained_model, X_test_t = res
    res_df = pd.DataFrame([metrics])
    res_df['site'] = test_site
    res_df.to_csv(os.path.join(results_dir, "fast_demo_metrics.csv"), index=False)

    # 3. Grad-CAM Explainability using the freshly trained model
    print("\n[3/3] Generating Grad-CAM Explainability Report for a subject...", flush=True)
    sample_sub_path = manifest_df[manifest_df['site'] == test_site].iloc[0]['feature_path']
    ts_sample = np.load(sample_sub_path)
    conn_gen = DynamicConnectivityGenerator(window_length=30, stride=10)
    dyn_conn = conn_gen.generate_dynamic_connectivity(ts_sample)

    max_T = 12
    if dyn_conn.shape[0] >= max_T:
        dyn_conn_fixed = dyn_conn[:max_T]
    else:
        pad = np.tile(dyn_conn[-1:], (max_T - dyn_conn.shape[0], 1, 1))
        dyn_conn_fixed = np.vstack([dyn_conn, pad])

    input_tensor = torch.from_numpy(dyn_conn_fixed).float().unsqueeze(0).unsqueeze(2)

    explainer = ConvLSTMGradCAM(trained_model, device=trainer.device)
    cam_results = explainer.generate_cam(input_tensor)

    print("\n" + "=" * 55, flush=True)
    print(" GRAD-CAM BRAIN EXPLAINABILITY REPORT", flush=True)
    print("=" * 55, flush=True)
    print(f" Subject: {manifest_df[manifest_df['site'] == test_site].iloc[0]['subject_id']} (Hospital: {test_site})", flush=True)
    print(f" Diagnosis: {cam_results['predicted_label']} (P(ADHD) = {cam_results['adhd_probability']:.4f})", flush=True)
    
    print("\n Top 5 Influential AAL Brain Regions for Prediction:", flush=True)
    for r in cam_results['top_rois'][:5]:
        print(f"   #{r['rank']}: {r['roi_name']} (Importance Score: {r['importance_score']:.2f})", flush=True)

    print("\n Top 5 Influential Functional Connectivity Edges:", flush=True)
    for e in cam_results['top_edges'][:5]:
        print(f"   #{e['rank']}: {e['roi1_name']} <---> {e['roi2_name']} (Weight: {e['edge_weight']:.4f})", flush=True)

    pd.DataFrame(cam_results['top_rois']).to_csv(os.path.join(results_dir, "fast_demo_top_rois.csv"), index=False)
    pd.DataFrame(cam_results['top_edges']).to_csv(os.path.join(results_dir, "fast_demo_top_edges.csv"), index=False)

    # 4. Generate updated visual charts automatically
    print("\n[4/4] Updating visual charts (PNG images)...", flush=True)
    import subprocess
    subprocess.run([sys.executable, "visualize_results.py"], check=False)

    print("\n" + "=" * 70, flush=True)
    print(" [DONE] Fast demo finished successfully! Outputs saved in ./results/", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    run_fast_demo()
