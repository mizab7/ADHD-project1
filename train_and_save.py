import os
import sys
import torch
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from training.loso_trainer import LOSOTrainer

def train_and_save(epochs=160):
    print(f"[1/2] Training model for {epochs} epochs to save deep weights...", flush=True)
    manifest_path = "./data/cached_features/dataset_manifest.csv"
    manifest_df = pd.read_csv(manifest_path)
    
    trainer = LOSOTrainer(manifest_df, device="mps")
    # Train on training fold (Peking_1 held out as test)
    res = trainer.train_and_eval_fold(
        test_site="Peking_1",
        model_type="conv_lstm",
        use_dae=True,
        use_combat=True,
        connectivity_mode="dynamic",
        epochs=epochs
    )
    
    metrics, trained_model, _ = res
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    
    weights_path = os.path.join(results_dir, "trained_convlstm.pt")
    torch.save(trained_model.state_dict(), weights_path)
    print(f"[2/2] Saved trained model weights to: {weights_path}", flush=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train and save model weights")
    parser.add_argument("--epochs", type=int, default=160, help="Number of training epochs (default: 160)")
    args = parser.parse_args()
    
    train_and_save(epochs=args.epochs)
