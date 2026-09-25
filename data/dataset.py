import os
import glob
import re
import pandas as pd
import numpy as np

def load_all_phenotypic_data(bids_root_dir):
    """
    Parses all phenotypic CSV files in the ADHD-200 BIDS root directory.
    Returns a unified pandas DataFrame indexed by subject ID.
    """
    csv_files = glob.glob(os.path.join(bids_root_dir, "*phenotypic.csv"))
    dfs = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, encoding='utf-8', on_bad_lines='skip')
        except Exception:
            df = pd.read_csv(csv_file, encoding='latin1', on_bad_lines='skip')
        
        # Standardize subject ID column name
        if 'ScanDir ID' in df.columns:
            df.rename(columns={'ScanDir ID': 'subject_id_raw'}, inplace=True)
        elif 'ID' in df.columns:
            df.rename(columns={'ID': 'subject_id_raw'}, inplace=True)
        else:
            continue
            
        # Clean subject IDs
        df['subject_id_raw'] = df['subject_id_raw'].astype(str).str.strip()
        # Drop rows where subject_id is invalid or missing
        df = df[df['subject_id_raw'].str.isdigit()]
        
        dfs.append(df)
        
    if not dfs:
        raise ValueError(f"No phenotypic CSV files found or parsed in {bids_root_dir}")
        
    combined_df = pd.concat(dfs, ignore_index=True)
    # Standardize column names
    combined_df.columns = [c.strip() for c in combined_df.columns]
    
    # Process DX (Diagnostic Label): 0 = Typically Developing / Control, 1/2/3 = ADHD
    # DX values: 0 -> 0 (Control), 1,2,3 -> 1 (ADHD). Drop pending/missing/'N/A'/'-999'
    combined_df['DX_clean'] = pd.to_numeric(combined_df['DX'], errors='coerce')
    combined_df = combined_df.dropna(subset=['DX_clean'])
    combined_df['DX_clean'] = combined_df['DX_clean'].astype(int)
    
    # Filter valid labels: 0 for Control, 1 for ADHD (DX in 1,2,3)
    combined_df = combined_df[combined_df['DX_clean'].isin([0, 1, 2, 3])].copy()
    combined_df['label'] = (combined_df['DX_clean'] > 0).astype(int)
    
    return combined_df

def scan_bids_subjects(bids_root_dir):
    """
    Scans the BIDS directory for all subject folders across all site subdirectories.
    Returns a list of dicts with subject metadata and path to scan files.
    """
    sites = ["Brown", "KKI", "NYU", "NeuroIMAGE", "OHSU", "Peking_1", "Peking_2", "Peking_3", "Pittsburgh", "WashU"]
    subject_records = []
    
    pheno_df = load_all_phenotypic_data(bids_root_dir)
    
    # Create lookup map from integer string subject ID -> row dict
    pheno_map = {}
    for _, row in pheno_df.iterrows():
        row_dict = row.to_dict()
        sid_str = str(row['subject_id_raw']).zfill(7)
        pheno_map[sid_str] = row_dict
        # Also map unpadded
        pheno_map[str(row['subject_id_raw'])] = row_dict
        
    for site in sites:
        site_dir = os.path.join(bids_root_dir, site)
        if not os.path.isdir(site_dir):
            continue
            
        sub_dirs = glob.glob(os.path.join(site_dir, "sub-*"))
        for sub_dir in sub_dirs:
            if not os.path.isdir(sub_dir):
                continue
                
            folder_name = os.path.basename(sub_dir) # e.g. sub-0015001 or sub-1000804
            raw_id = folder_name.replace("sub-", "").lstrip("0")
            raw_id_padded = raw_id.zfill(7) if raw_id.isdigit() else raw_id
            
            # Find subject phenotypic entry
            match = pheno_map.get(raw_id_padded)
            if match is None:
                match = pheno_map.get(raw_id)
            if match is None:
                # Try finding numeric match
                for k in pheno_map:
                    if k.lstrip("0") == raw_id:
                        match = pheno_map[k]
                        break
                        
            if match is None:
                continue
                
            label = int(match['label'])
            
            # Search for functional bold NIfTI files (*bold.nii.gz) inside sub_dir
            bold_files = glob.glob(os.path.join(sub_dir, "**", "*_bold.nii.gz"), recursive=True)
            # Filter out macOS AppleDouble files (starting with ._)
            bold_files = [f for f in bold_files if not os.path.basename(f).startswith("._")]
            
            if not bold_files:
                continue
                
            # Use primary functional scan
            func_file = bold_files[0]
            
            subject_records.append({
                "subject_id": folder_name,
                "site": site,
                "func_path": func_file,
                "label": label,  # 0: Control, 1: ADHD
                "dx_detail": int(match['DX_clean']),
                "age": float(match['Age']) if pd.notnull(match.get('Age')) and str(match.get('Age')).replace('.','',1).isdigit() else np.nan,
                "gender": int(match['Gender']) if pd.notnull(match.get('Gender')) and str(match.get('Gender')).isdigit() else np.nan
            })
            
    df_subjects = pd.DataFrame(subject_records)
    return df_subjects

if __name__ == "__main__":
    bids_path = "/Volumes/Lexar/ADHD200_RawDataBIDS"
    if not os.path.exists(bids_path):
        print(f"[INFO] Raw BIDS drive path '{bids_path}' is not mounted.")
        print("[INFO] Checking for cached preprocessed features in './data/cached_features'...")
        manifest_path = "./data/cached_features/dataset_manifest.csv"
        if os.path.exists(manifest_path):
            df = pd.read_csv(manifest_path)
            print(f"[SUCCESS] Found pre-extracted dataset manifest with {len(df)} subjects.")
            print("\nSubject count by site and label (0: Control, 1: ADHD):")
            print(pd.crosstab(df['site'], df['label'], margins=True))
        else:
            print("[ERROR] Neither raw BIDS directory nor cached manifest was found.")
    else:
        df = scan_bids_subjects(bids_path)
        print(f"Total valid subjects found with functional scans & phenotypic labels: {len(df)}")
        print("\nSubject count by site and label (0: Control, 1: ADHD):")
        print(pd.crosstab(df['site'], df['label'], margins=True))
