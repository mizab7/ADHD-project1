import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
import numpy as np
from nilearn import datasets

def get_aal116_labels():
    """
    Returns the 116 anatomical region label names for AAL SPM12 atlas.
    """
    try:
        atlas = datasets.fetch_atlas_aal(version='SPM12')
        labels = atlas.labels  # Includes background or 116 labels
        if len(labels) == 117:
            return labels[1:]  # Exclude background index 0
        return labels[:116]
    except Exception:
        return [f"ROI_{i+1}" for i in range(116)]

class ConvLSTMGradCAM:
    """
    Grad-CAM interpreter for ConvLSTMClassifier.
    Computes spatial activation gradients over 116 x 116 connectivity matrices
    and maps them back to anatomical AAL ROIs and ROI-to-ROI edge pairs.
    """
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device
        self.model.eval()
        self.aal_labels = get_aal116_labels()

    def generate_cam(self, input_tensor, target_class=None):
        """
        input_tensor: Single subject or batch tensor of shape (1, T_windows, 1, 116, 116).
        Returns:
            cam_matrix: (116, 116) edge importance matrix.
            top_rois: list of dicts with top ROI indices, names, and importance scores.
            top_edges: list of dicts with top ROI pairs, names, and connection importance scores.
            pred_class: int (0 or 1)
            prob: float
        """
        self.model.zero_grad()
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True
        
        # Forward pass
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=-1)[0]
        
        if target_class is None:
            target_class = torch.argmax(probs).item()
            
        target_score = logits[0, target_class]
        target_score.backward()
        
        # Retrieve gradients and activations from Conv2DSpatialEncoder hook
        grads = self.model.spatial_encoder.gradients  # (B * T, 64, H_out, W_out)
        acts = self.model.spatial_encoder.activations  # (B * T, 64, H_out, W_out)
        
        if grads is None or acts is None:
            # Fallback to direct input gradients if hook missing
            input_grad = input_tensor.grad.data  # (1, T, 1, 116, 116)
            cam_matrix = torch.mean(torch.abs(input_grad), dim=(0, 1, 2)).cpu().numpy()
        else:
            # Grad-CAM weight calculation: channel-wise mean gradient
            weights = torch.mean(grads, dim=(2, 3), keepdim=True)  # (B*T, 64, 1, 1)
            cam = torch.sum(weights * acts, dim=1, keepdim=True)    # (B*T, 1, H_out, W_out)
            cam = F.relu(cam)
            
            # Upsample CAM back to 116 x 116 connectivity resolution
            cam_upsampled = F.interpolate(cam, size=(116, 116), mode='bilinear', align_corners=False)
            
            # Mean across time windows
            cam_matrix = torch.mean(cam_upsampled, dim=0).squeeze().detach().cpu().numpy()
            
        # Make CAM symmetric (since connectivity matrix is N x N)
        cam_matrix = 0.5 * (cam_matrix + cam_matrix.T)
        np.fill_diagonal(cam_matrix, 0.0)
        
        # Standardize CAM matrix to [0, 1]
        c_min, c_max = np.min(cam_matrix), np.max(cam_matrix)
        if c_max > c_min:
            cam_matrix = (cam_matrix - c_min) / (c_max - c_min)
            
        # Top ROIs (sum of importance weights per ROI node)
        roi_importance = np.sum(cam_matrix, axis=1)
        top_roi_indices = np.argsort(roi_importance)[::-1]
        
        top_rois = []
        for rank, idx in enumerate(top_roi_indices[:10]):
            name = self.aal_labels[idx] if idx < len(self.aal_labels) else f"ROI_{idx+1}"
            top_rois.append({
                "rank": rank + 1,
                "roi_index": int(idx),
                "roi_name": name,
                "importance_score": float(roi_importance[idx])
            })
            
        # Top Connectivity Edges (highest weights between ROI pairs)
        triu_i, triu_j = np.triu_indices(116, k=1)
        edge_weights = cam_matrix[triu_i, triu_j]
        top_edge_indices = np.argsort(edge_weights)[::-1]
        
        top_edges = []
        for rank, edge_idx in enumerate(top_edge_indices[:20]):
            i, j = triu_i[edge_idx], triu_j[edge_idx]
            name1 = self.aal_labels[i] if i < len(self.aal_labels) else f"ROI_{i+1}"
            name2 = self.aal_labels[j] if j < len(self.aal_labels) else f"ROI_{j+1}"
            top_edges.append({
                "rank": rank + 1,
                "roi1_index": int(i),
                "roi1_name": name1,
                "roi2_index": int(j),
                "roi2_name": name2,
                "edge_weight": float(edge_weights[edge_idx])
            })
            
        return {
            "predicted_class": int(target_class),
            "predicted_label": "ADHD" if target_class == 1 else "Control",
            "adhd_probability": float(probs[1].item()),
            "control_probability": float(probs[0].item()),
            "cam_matrix": cam_matrix,
            "top_rois": top_rois,
            "top_edges": top_edges
        }

if __name__ == "__main__":
    from models.conv_lstm import ConvLSTMClassifier
    model = ConvLSTMClassifier()
    explainer = ConvLSTMGradCAM(model)
    dummy_input = torch.randn(1, 10, 1, 116, 116)
    explanation = explainer.generate_cam(dummy_input)
    print(f"Prediction: {explanation['predicted_label']} (ADHD prob: {explanation['adhd_probability']:.4f})")
    print(f"\nTop 5 Important AAL Brain ROIs:")
    for r in explanation['top_rois'][:5]:
        print(f"  Rank {r['rank']}: {r['roi_name']} (score: {r['importance_score']:.4f})")
    print(f"\nTop 3 Important Connectivity Edges:")
    for e in explanation['top_edges'][:3]:
        print(f"  Rank {e['rank']}: {e['roi1_name']} <---> {e['roi2_name']} (weight: {e['edge_weight']:.4f})")
