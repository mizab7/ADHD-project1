import os
import sys
import time
import argparse
import torch
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.conv_lstm import ConvLSTMClassifier
from models.connectivity import DynamicConnectivityGenerator
from explainability.grad_cam import ConvLSTMGradCAM

def predict_patient(subject_id=None, random_pick=False):
    start_time = time.time()
    
    weights_path = "./results/trained_convlstm.pt"
    manifest_path = "./data/cached_features/dataset_manifest.csv"
    
    if not os.path.exists(weights_path):
        print(f"[ERROR] Trained weights not found at {weights_path}.")
        print("Please run `python train_and_save.py` first to save the model.")
        return
        
    manifest_df = pd.read_csv(manifest_path)
    
    # Select subject
    if random_pick:
        row = manifest_df.sample(1).iloc[0]
    elif subject_id:
        matches = manifest_df[manifest_df['subject_id'] == subject_id]
        if len(matches) == 0:
            print(f"[ERROR] Subject '{subject_id}' not found in dataset manifest.")
            return
        row = matches.iloc[0]
    else:
        # Default to a sample subject
        row = manifest_df.iloc[0]
        
    sid = row['subject_id']
    site = row['site']
    true_label = "ADHD" if row['label'] == 1 else "Control"
    age = row['age'] if pd.notnull(row.get('age')) else "N/A"
    feat_path = row['feature_path']
    
    # 1. Load trained model weights instantly
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = ConvLSTMClassifier(conv_channels=[16, 32], lstm_hidden_dim=64, dropout=0.3)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 2. Load patient brain scan time series & build dynamic connectivity
    ts = np.load(feat_path) # (T, 116)
    conn_gen = DynamicConnectivityGenerator(window_length=30, stride=10)
    dyn_conn = conn_gen.generate_dynamic_connectivity(ts) # (Tw, 116, 116)
    
    # Pad to fixed sequence length 12
    max_T = 12
    if dyn_conn.shape[0] >= max_T:
        dyn_conn_fixed = dyn_conn[:max_T]
    else:
        pad = np.tile(dyn_conn[-1:], (max_T - dyn_conn.shape[0], 1, 1))
        dyn_conn_fixed = np.vstack([dyn_conn, pad])
        
    input_tensor = torch.from_numpy(dyn_conn_fixed).float().unsqueeze(0).unsqueeze(2).to(device)
    
    # 3. Model forward pass (instant inference)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=-1)[0]
        prob_control = probs[0].item()
        prob_adhd = probs[1].item()
        
    predicted_class = "ADHD" if prob_adhd >= 0.50 else "Control"
    confidence = max(prob_adhd, prob_control) * 100.0
    
    # 4. Grad-CAM brain explainability
    explainer = ConvLSTMGradCAM(model, device=device)
    cam_results = explainer.generate_cam(input_tensor)
    
    elapsed = time.time() - start_time
    
    # 5. Print Clean Clinical AI Report
    print("\n" + "=" * 62)
    print("       🏥 ADHD-200 INSTANT AI DIAGNOSTIC REPORT")
    print("=" * 62)
    print(f" Patient ID:       {sid}")
    print(f" Hospital/Site:    {site}")
    print(f" Age:              {age}")
    print(f" Ground Truth:     {true_label}")
    print("-" * 62)
    status_icon = "🟢" if predicted_class == true_label else "🟡"
    print(f" {status_icon} AI Verdict:     {predicted_class.upper()} ({confidence:.1f}% Confidence)")
    print(f"    • P(ADHD):     {prob_adhd * 100:.1f}%")
    print(f"    • P(Control):  {prob_control * 100:.1f}%")
    print("-" * 62)
    print(" 🧠 Top 3 Influential Brain Regions for Diagnosis:")
    for r in cam_results['top_rois'][:3]:
        print(f"    #{r['rank']}: {r['roi_name']} (Weight: {r['importance_score']:.1f})")
        
    print("\n 🔗 Top 3 Influential Functional Neural Circuits:")
    for e in cam_results['top_edges'][:3]:
        print(f"    #{e['rank']}: {e['roi1_name']} <---> {e['roi2_name']}")
    print("=" * 62)
    print(f" ⚡ Diagnostic Speed: {elapsed:.2f} seconds (0.0s spent training!)")
    print("=" * 62 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Instant Patient ADHD Diagnosis")
    parser.add_argument("pos_subject", nargs="?", default=None, help="Subject ID (e.g. 1038415 or sub-1038415)")
    parser.add_argument("--subject", type=str, default=None, help="Subject ID (e.g. sub-1038415)")
    parser.add_argument("--random", action="store_true", help="Pick a random patient from the dataset")
    args = parser.parse_args()
    
    target_id = args.subject or args.pos_subject
    if target_id and not target_id.startswith("sub-"):
        target_id = f"sub-{target_id}"
        
    predict_patient(subject_id=target_id, random_pick=args.random)
