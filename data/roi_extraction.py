import os
import glob
import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
import joblib
import sys

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import scan_bids_subjects

def extract_aal_roi_time_series(bids_root_dir, cache_dir="./data/cached_features", max_subjects=None):
    """
    Parcellates functional fMRI scans into 116 AAL atlas region time series.
    Saves extracted time series and dataset metadata manifest.
    """
    os.makedirs(cache_dir, exist_ok=True)
    manifest_path = os.path.join(cache_dir, "dataset_manifest.csv")
    
    # 1. Fetch AAL SPM12 atlas (116 ROIs)
    print("[INFO] Fetching AAL atlas (116 regions)...")
    atlas = datasets.fetch_atlas_aal(version='SPM12')
    atlas_filename = atlas.maps
    labels = atlas.labels  # 116 ROI labels + background
    
    # 2. Instantiate NiftiLabelsMasker with standard fMRI preprocessing parameters
    masker = NiftiLabelsMasker(
        labels_img=atlas_filename,
        standardize="zscore_sample",
        detrend=True,
        high_pass=0.01,
        low_pass=0.1,
        t_r=2.0,
        verbose=0
    )

    # 3. Scan dataset
    df_subjects = scan_bids_subjects(bids_root_dir)
    if max_subjects is not None and max_subjects > 0:
        df_subjects = df_subjects.head(max_subjects).copy()
        
    print(f"[INFO] Processing {len(df_subjects)} valid subjects...")
    
    processed_records = []
    
    for idx, row in df_subjects.iterrows():
        sub_id = row['subject_id']
        site = row['site']
        func_path = row['func_path']
        label = row['label']
        
        save_path = os.path.join(cache_dir, f"{site}_{sub_id}_aal116.npy")
        
        # Check if already cached
        if os.path.exists(save_path):
            try:
                ts = np.load(save_path)
                if ts.ndim == 2 and ts.shape[1] == 116:
                    processed_records.append({
                        "subject_id": sub_id,
                        "site": site,
                        "label": label,
                        "feature_path": save_path,
                        "timepoints": ts.shape[0],
                        "num_rois": ts.shape[1],
                        "age": row['age'],
                        "gender": row['gender']
                    })
                    continue
            except Exception:
                pass
                
        # Extract time series
        try:
            time_series = masker.fit_transform(func_path)
            # Ensure shape is (T, 116)
            if time_series.shape[1] > 116:
                time_series = time_series[:, :116]
            elif time_series.shape[1] < 116:
                print(f"[WARN] Subject {sub_id} has {time_series.shape[1]} ROIs (expected 116). Skipping.")
                continue
                
            np.save(save_path, time_series)
            processed_records.append({
                "subject_id": sub_id,
                "site": site,
                "label": label,
                "feature_path": save_path,
                "timepoints": time_series.shape[0],
                "num_rois": time_series.shape[1],
                "age": row['age'],
                "gender": row['gender']
            })
            if (len(processed_records)) % 10 == 0 or len(processed_records) == len(df_subjects):
                print(f"  Processed [{len(processed_records)}/{len(df_subjects)}] {sub_id} ({site}) -> shape {time_series.shape}")
        except Exception as e:
            print(f"[ERROR] Failed to extract ROI time series for {sub_id} ({func_path}): {e}")
            
    df_manifest = pd.DataFrame(processed_records)
    df_manifest.to_csv(manifest_path, index=False)
    print(f"\n[SUCCESS] Extracted and saved AAL time series for {len(df_manifest)} subjects.")
    print(f"Manifest saved to: {manifest_path}")
    return df_manifest

if __name__ == "__main__":
    bids_path = "/Volumes/Lexar/ADHD200_RawDataBIDS"
    cache_dir = "./data/cached_features"
    manifest_path = os.path.join(cache_dir, "dataset_manifest.csv")
    
    if not os.path.exists(bids_path):
        print(f"[INFO] Raw BIDS drive path '{bids_path}' is not mounted.")
        if os.path.exists(manifest_path):
            df_manifest = pd.read_csv(manifest_path)
            print(f"[SUCCESS] Pre-extracted cached features found ({len(df_manifest)} subjects) in '{cache_dir}'.")
            print("[INFO] Re-extraction is not needed to train models or run demonstrations.")
        else:
            print("[ERROR] Neither raw BIDS directory nor cached manifest was found.")
    else:
        # Execute extraction for subjects
        df_manifest = extract_aal_roi_time_series(bids_path, cache_dir=cache_dir)
