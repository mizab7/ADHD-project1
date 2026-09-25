import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.loso_trainer import LOSOTrainer

def run_ablation_experiments(manifest_df, output_dir="./results", epochs=25):
    """
    Executes automated ablation study comparing all key modular components.
    """
    os.makedirs(output_dir, exist_ok=True)
    trainer = LOSOTrainer(manifest_df)
    
    experiments = [
        {"name": "Full Proposed ConvLSTM (DAE + ComBat + Dynamic)", "model_type": "conv_lstm", "use_dae": True, "use_combat": True, "connectivity_mode": "dynamic"},
        {"name": "Ablation: Without DAE", "model_type": "conv_lstm", "use_dae": False, "use_combat": True, "connectivity_mode": "dynamic"},
        {"name": "Ablation: Without ComBat", "model_type": "conv_lstm", "use_dae": True, "use_combat": False, "connectivity_mode": "dynamic"},
        {"name": "Ablation: Static Connectivity", "model_type": "conv_lstm", "use_dae": True, "use_combat": True, "connectivity_mode": "static"},
        {"name": "Baseline: Spatial CNN", "model_type": "spatial_cnn", "use_dae": True, "use_combat": True, "connectivity_mode": "dynamic"},
        {"name": "Baseline: SVM Classifier", "model_type": "svm", "use_dae": True, "use_combat": True, "connectivity_mode": "dynamic"},
        {"name": "Baseline: Random Forest", "model_type": "rf", "use_dae": True, "use_combat": True, "connectivity_mode": "dynamic"}
    ]
    
    ablation_summary = []
    
    for exp in experiments:
        print(f"\n=======================================================")
        print(f" RUNNING EXPERIMENT: {exp['name']}")
        print(f"=======================================================")
        
        df_exp = trainer.run_loso_cross_validation(
            model_type=exp['model_type'],
            use_dae=exp['use_dae'],
            use_combat=exp['use_combat'],
            connectivity_mode=exp['connectivity_mode'],
            epochs=epochs
        )
        
        mean_acc = df_exp['accuracy'].mean()
        mean_auc = df_exp['roc_auc'].mean()
        mean_f1 = df_exp['f1_score'].mean()
        mean_rec = df_exp['recall_sensitivity'].mean()
        mean_spec = df_exp['specificity'].mean()
        
        ablation_summary.append({
            "Experiment Name": exp['name'],
            "Model Architecture": exp['model_type'],
            "DAE Denoising": exp['use_dae'],
            "ComBat Harmonization": exp['use_combat'],
            "Connectivity Mode": exp['connectivity_mode'],
            "Mean Accuracy": f"{mean_acc*100:.2f}%",
            "Mean ROC-AUC": f"{mean_auc:.4f}",
            "Mean F1-Score": f"{mean_f1:.4f}",
            "Mean Sensitivity": f"{mean_rec:.4f}",
            "Mean Specificity": f"{mean_spec:.4f}"
        })
        
    df_summary = pd.DataFrame(ablation_summary)
    csv_path = os.path.join(output_dir, "ablation_experiments_summary.csv")
    df_summary.to_csv(csv_path, index=False)
    
    print("\n=======================================================")
    print(" ABLATION BENCHMARK FINAL SUMMARY COMPARISON TABLE")
    print("=======================================================")
    print(df_summary.to_string(index=False))
    print(f"\nAblation summary saved to: {csv_path}\n")
    return df_summary

if __name__ == "__main__":
    manifest_path = "./data/cached_features/dataset_manifest.csv"
    if os.path.exists(manifest_path):
        manifest_df = pd.read_csv(manifest_path)
        run_ablation_experiments(manifest_df, epochs=15)
