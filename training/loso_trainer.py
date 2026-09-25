import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import scan_bids_subjects
from models.connectivity import DynamicConnectivityGenerator
from models.dae import DenoisingAutoencoder, train_dae_model
from harmonization.combat import LeakageSafeComBat
from models.conv_lstm import ConvLSTMClassifier
from models.baselines import BaselineMachineLearningClassifier, SpatialCNNBaseline
from evaluation.metrics import compute_classification_metrics, print_metrics_summary

class LOSOTrainer:
    """
    Leave-One-Site-Out (LOSO) Cross-Validation Benchmark Suite.
    Guarantees strict zero data leakage across held-out sites.
    """
    def __init__(self, manifest_df, config=None, device="mps"):
        self.manifest_df = manifest_df
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() and device=="mps" else "cpu"))
        print(f"[INFO] LOSO Trainer initialized using compute device: {self.device}")

    def _load_subject_data(self, df_sub, use_dae=True, dae_model=None, use_combat=True, combat_model=None, is_training=False, connectivity_mode="dynamic"):
        """
        Loads cached AAL ROI time series, applies optional DAE denoising,
        optional leakage-safe ComBat harmonization, and computes connectivity matrices.
        """
        conn_gen = DynamicConnectivityGenerator(window_length=30, stride=10)
        X_conn_list = []
        y_list = []
        site_list = []
        
        # 1. Load raw ROI time series for all fold subjects
        ts_records = []
        for _, row in df_sub.iterrows():
            ts = np.load(row['feature_path']) # shape (T, 116)
            ts_records.append(ts)
            y_list.append(row['label'])
            site_list.append(row['site'])
            
        y_array = np.array(y_list, dtype=int)
        site_array = np.array(site_list)
        
        # 2. DAE Denoising (Optional)
        if use_dae:
            if is_training and dae_model is None:
                # Concatenate all training time points to fit DAE
                all_ts_flat = torch.from_numpy(np.vstack(ts_records)).float()
                dae_model = DenoisingAutoencoder(input_dim=116, latent_dim=32)
                dae_model = train_dae_model(dae_model, all_ts_flat, epochs=20, batch_size=64, device=self.device)
                
            # Apply DAE to denoise time series
            if dae_model is not None:
                denoised_ts = []
                for ts in ts_records:
                    ts_tensor = torch.from_numpy(ts).float().to(self.device)
                    clean_ts = dae_model.denoise(ts_tensor).cpu().numpy()
                    denoised_ts.append(clean_ts)
                ts_records = denoised_ts
                
        # 3. Dynamic or Static Connectivity Generation
        conn_matrices = []
        for ts in ts_records:
            if connectivity_mode == "dynamic":
                c = conn_gen.generate_dynamic_connectivity(ts) # (T_w, 116, 116)
            else:
                c = conn_gen.generate_static_connectivity(ts)  # (1, 116, 116)
            conn_matrices.append(c)
            
        # 4. Leakage-Safe ComBat Harmonization (Optional)
        # Flatten connectivity matrices per subject to 2D for ComBat
        sample_shapes = [c.shape for c in conn_matrices]
        flat_conn = [c.reshape(c.shape[0], -1) for c in conn_matrices]  # list of (T_w, 116*116)
        
        if use_combat:
            # Average across time windows per subject for ComBat site effect fitting
            sub_mean_features = np.array([np.mean(f, axis=0) for f in flat_conn])  # (N_sub, 116*116)
            
            if is_training:
                combat_model = LeakageSafeComBat()
                sub_mean_harm = combat_model.fit_transform(sub_mean_features, site_array)
            else:
                if combat_model is not None:
                    sub_mean_harm = combat_model.transform(sub_mean_features, site_array)
                else:
                    sub_mean_harm = sub_mean_features
                    
            # Apply additive shift delta back to time-window matrices
            harmonized_conn = []
            for i, c in enumerate(conn_matrices):
                delta_shift = sub_mean_harm[i] - sub_mean_features[i]
                c_flat = flat_conn[i] + delta_shift[None, :]
                c_harm = c_flat.reshape(sample_shapes[i])
                harmonized_conn.append(c_harm)
            conn_matrices = harmonized_conn
            
        return conn_matrices, y_array, site_array, dae_model, combat_model

    def train_and_eval_fold(self, test_site, model_type="conv_lstm", use_dae=True, use_combat=True, connectivity_mode="dynamic", epochs=35):
        """
        Executes a single LOSO fold holding out test_site.
        """
        train_df = self.manifest_df[self.manifest_df['site'] != test_site].copy()
        test_df = self.manifest_df[self.manifest_df['site'] == test_site].copy()
        
        # Require test fold to have both ADHD and Control subjects if evaluating ROC-AUC
        if len(test_df) == 0 or len(np.unique(test_df['label'])) < 2:
            print(f"[WARN] Skipping test site {test_site}: insufficient class diversity or empty.")
            return None
            
        print(f"\n=======================================================")
        print(f" LOSO Fold: Train on {len(train_df)} subs | Test on [{test_site}] ({len(test_df)} subs)")
        print(f"=======================================================")
        
        # Load & harmonize training fold
        X_train_conn, y_train, sites_train, dae_model, combat_model = self._load_subject_data(
            train_df, use_dae=use_dae, is_training=True, use_combat=use_combat, connectivity_mode=connectivity_mode
        )
        
        # Load & harmonize test fold using training parameters
        X_test_conn, y_test, sites_test, _, _ = self._load_subject_data(
            test_df, use_dae=use_dae, dae_model=dae_model, use_combat=use_combat, combat_model=combat_model, is_training=False, connectivity_mode=connectivity_mode
        )
        
        # Conventional ML Baseline
        if model_type in ["svm", "rf", "logreg"]:
            clf = BaselineMachineLearningClassifier(model_type=model_type)
            clf.fit(X_train_conn, y_train)
            train_probs = clf.predict_proba(X_train_conn)[:, 1]
            
            # Calibrate threshold on training fold (zero leakage)
            best_thresh = 0.5
            best_score = -1.0
            for th in np.linspace(0.25, 0.75, 51):
                th_preds = (train_probs >= th).astype(int)
                sens = np.sum((y_train == 1) & (th_preds == 1)) / max(np.sum(y_train == 1), 1)
                spec = np.sum((y_train == 0) & (th_preds == 0)) / max(np.sum(y_train == 0), 1)
                bal = (sens + spec) / 2.0
                if bal > best_score:
                    best_score = bal
                    best_thresh = th
                    
            probs = clf.predict_proba(X_test_conn)
            preds = (probs[:, 1] >= best_thresh).astype(int)
            metrics = compute_classification_metrics(y_test, preds, y_probs=probs)
            print_metrics_summary(test_site, metrics)
            return metrics, clf, None
            
        # PyTorch Neural Networks (ConvLSTM or SpatialCNN)
        # Standardize max time windows sequence across batch via padding/truncating to fixed T_w=12
        max_T = 12
        def pad_truncate_conn(conn_list):
            padded = []
            for c in conn_list:
                T, H, W = c.shape
                if T >= max_T:
                    c_fixed = c[:max_T]
                else:
                    pad = np.tile(c[-1:], (max_T - T, 1, 1))
                    c_fixed = np.vstack([c, pad])
                padded.append(c_fixed)
            return torch.from_numpy(np.array(padded, dtype=np.float32)).unsqueeze(2) # (B, max_T, 1, 116, 116)

        X_train_t = pad_truncate_conn(X_train_conn)
        y_train_t = torch.from_numpy(y_train).long()
        X_test_t = pad_truncate_conn(X_test_conn)
        y_test_t = torch.from_numpy(y_test).long()
        
        # Balanced class weighting for CrossEntropyLoss to handle ADHD vs Control imbalance
        n_pos = torch.sum(y_train_t == 1).item()
        n_neg = torch.sum(y_train_t == 0).item()
        total_samples = float(len(y_train_t))
        w0 = total_samples / (2.0 * max(n_neg, 1))
        w1 = total_samples / (2.0 * max(n_pos, 1))
        class_weights = torch.tensor([w0, w1], dtype=torch.float32).to(self.device)
        
        if model_type == "conv_lstm":
            model = ConvLSTMClassifier(conv_channels=[16, 32], lstm_hidden_dim=64, dropout=0.3).to(self.device)
        elif model_type == "spatial_cnn":
            model = SpatialCNNBaseline(conv_channels=[16, 32], dropout=0.3).to(self.device)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
            
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        
        train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
        
        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                logits = model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * bx.size(0)
                
            scheduler.step()
            if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
                print(f"  [Train Epoch {epoch+1}/{epochs}] Loss: {total_loss/len(train_dataset):.4f}", flush=True)
                
        # Evaluate on held-out test fold
        model.eval()
        with torch.no_grad():
            train_logits = model(X_train_t.to(self.device))
            train_probs = torch.softmax(train_logits, dim=-1).cpu().numpy()[:, 1]
            test_logits = model(X_test_t.to(self.device))
            probs = torch.softmax(test_logits, dim=-1).cpu().numpy()
            
        # Calibrate optimal decision threshold strictly on training set (zero test leakage)
        best_thresh = 0.5
        best_score = -1.0
        for th in np.linspace(0.25, 0.75, 51):
            th_preds = (train_probs >= th).astype(int)
            sens = np.sum((y_train == 1) & (th_preds == 1)) / max(np.sum(y_train == 1), 1)
            spec = np.sum((y_train == 0) & (th_preds == 0)) / max(np.sum(y_train == 0), 1)
            bal = (sens + spec) / 2.0
            if bal > best_score:
                best_score = bal
                best_thresh = th
                
        preds = (probs[:, 1] >= best_thresh).astype(int)
        metrics = compute_classification_metrics(y_test, preds, y_probs=probs)
        print_metrics_summary(test_site, metrics)
        return metrics, model, X_test_t

    def run_loso_cross_validation(self, model_type="conv_lstm", use_dae=True, use_combat=True, connectivity_mode="dynamic", epochs=30):
        """
        Runs full Leave-One-Site-Out benchmark across all eligible sites.
        """
        unique_sites = self.manifest_df['site'].unique()
        all_fold_metrics = []
        
        for site in unique_sites:
            res = self.train_and_eval_fold(
                test_site=site,
                model_type=model_type,
                use_dae=use_dae,
                use_combat=use_combat,
                connectivity_mode=connectivity_mode,
                epochs=epochs
            )
            if res is not None:
                metrics, model, test_data = res
                metrics['site'] = site
                all_fold_metrics.append(metrics)
                self.last_trained_model = model
                
        df_results = pd.DataFrame(all_fold_metrics)
        print("\n=======================================================")
        print(f" FINAL LOSO CROSS-VALIDATION SUMMARY [{model_type.upper()}]")
        print("=======================================================")
        print(df_results[['site', 'accuracy', 'precision', 'recall_sensitivity', 'specificity', 'f1_score', 'roc_auc']].to_string(index=False))
        
        mean_acc = df_results['accuracy'].mean()
        mean_auc = df_results['roc_auc'].mean()
        mean_f1 = df_results['f1_score'].mean()
        print(f"\n  Mean Accuracy across sites: {mean_acc*100:.2f}%")
        print(f"  Mean ROC-AUC across sites:  {mean_auc:.4f}")
        print(f"  Mean F1-Score across sites: {mean_f1:.4f}")
        print("=======================================================\n")
        return df_results
