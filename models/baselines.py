import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def extract_upper_triangular_features(connectivity_matrices):
    """
    Extracts vectorized upper-triangular edge features from connectivity matrices:
    - List of 2D/3D matrices per subject: [(T_w, N, N), ...] or [(N, N), ...]
    - 3D array (B, N, N)
    - 4D array (B, T, N, N)
    Returns feature matrix of shape (B, N*(N-1)/2).
    """
    if isinstance(connectivity_matrices, list):
        processed = []
        for m in connectivity_matrices:
            m = np.asarray(m)
            if m.ndim == 3:
                processed.append(np.mean(m, axis=0))
            elif m.ndim == 2:
                processed.append(m)
            else:
                raise ValueError(f"Expected 2D or 3D connectivity matrix per subject, got ndim={m.ndim}")
        connectivity_matrices = np.array(processed)
    elif hasattr(connectivity_matrices, 'ndim') and connectivity_matrices.ndim == 4:
        # Average over time windows to get static mean connectivity
        connectivity_matrices = np.mean(connectivity_matrices, axis=1)
        
    B, N, _ = connectivity_matrices.shape
    triu_indices = np.triu_indices(N, k=1)
    
    features = []
    for i in range(B):
        matrix = connectivity_matrices[i]
        edge_vector = matrix[triu_indices]
        features.append(edge_vector)
        
    return np.array(features, dtype=np.float32)

class BaselineMachineLearningClassifier:
    """
    Conventional Machine Learning Baseline (SVM, RandomForest, or LogisticRegression)
    operating on vectorized static connectivity features.
    """
    def __init__(self, model_type="svm"):
        self.model_type = model_type
        if model_type == "svm":
            self.model = SVC(kernel="rbf", probability=True, C=1.0)
        elif model_type == "rf":
            self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        elif model_type == "logreg":
            self.model = LogisticRegression(max_iter=500, C=1.0)
        else:
            raise ValueError(f"Unknown ML model type: {model_type}")

    def fit(self, X_conn, y):
        features = extract_upper_triangular_features(X_conn)
        self.model.fit(features, y)
        return self

    def predict(self, X_conn):
        features = extract_upper_triangular_features(X_conn)
        return self.model.predict(features)

    def predict_proba(self, X_conn):
        features = extract_upper_triangular_features(X_conn)
        return self.model.predict_proba(features)

class SpatialCNNBaseline(nn.Module):
    """
    2D Spatial CNN Baseline operating on Static 116x116 Functional Connectivity Matrix.
    (No temporal LSTM dimension).
    """
    def __init__(self, in_channels=1, conv_channels=[32, 64], num_classes=2, dropout=0.3):
        super(SpatialCNNBaseline, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, conv_channels[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_channels[0])
        self.pool1 = nn.MaxPool2d(2, 2)  # 116x116 -> 58x58
        
        self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_channels[1])
        self.pool2 = nn.MaxPool2d(2, 2)  # 58x58 -> 29x29
        self.pool3 = nn.AvgPool2d(4, 4)  # 29x29 -> 7x7
        
        feature_dim = conv_channels[1] * 7 * 7
        
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (B, 1, 116, 116) or (B, T, 1, 116, 116) -> mean over T if 5D
        if x.ndim == 5:
            x = torch.mean(x, dim=1)
        elif x.ndim == 3:
            x = x.unsqueeze(1)
            
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.pool3(x)
        
        flat_x = x.view(x.size(0), -1)
        logits = self.fc(flat_x)
        return logits

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)

if __name__ == "__main__":
    dummy_conn = np.random.randn(20, 10, 116, 116).astype(np.float32)
    y = np.random.randint(0, 2, size=20)
    
    svm_baseline = BaselineMachineLearningClassifier("svm")
    svm_baseline.fit(dummy_conn, y)
    svm_preds = svm_baseline.predict_proba(dummy_conn)
    print(f"SVM Baseline predictions shape: {svm_preds.shape}")
    
    cnn_baseline = SpatialCNNBaseline()
    dummy_tensor = torch.from_numpy(dummy_conn).unsqueeze(2)  # (20, 10, 1, 116, 116)
    cnn_preds = cnn_baseline.predict_proba(dummy_tensor)
    print(f"Spatial CNN Baseline predictions shape: {cnn_preds.shape}")
