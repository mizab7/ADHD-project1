# Explainable & Cross-Hospital Generalizable ADHD Classification from fMRI Using Denoised Spatiotemporal Deep Learning

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch 2.14](https://img.shields.io/badge/PyTorch-2.14-orange.svg)](https://pytorch.org/)
[![Nilearn](https://img.shields.io/badge/Nilearn-AAL116-green.svg)](https://nilearn.github.io/)
[![Dataset: ADHD--200](https://img.shields.io/badge/Dataset-ADHD--200-purple.svg)](http://fcon_1000.projects.nitrc.org/indi/adhd200/)

An end-to-end, publication-grade deep-learning framework designed to classify **ADHD vs. Typically Developing / Control** subjects using resting-state functional MRI (fMRI) data from the **ADHD-200** dataset. 

---

## 🎯 Key Capabilities & Research Goals

1. **Noise Reduction (Denoising Autoencoder):** 1D Denoising Autoencoder (DAE) removes high-frequency scanner noise and movement artifacts from preprocessed fMRI ROI time series.
2. **Cross-Hospital Generalization (Leakage-Safe ComBat):** Multi-site scanner harmonization using empirical Bayes ComBat, fitted **strictly on training fold data** during Leave-One-Site-Out (LOSO) cross-validation to guarantee **zero data leakage**.
3. **Geometrically Sound Spatial Representation:** Avoids arbitrary 2D grid reshaping of 116 AAL ROIs (e.g. $10 \times 12$). Uses dynamic ROI-to-ROI functional connectivity matrices ($116 \times 116$) across temporal sliding windows.
4. **Spatiotemporal Deep Learning (ConvLSTM):** 2D CNN layers capture spatial interaction topological networks, while LSTM cells learn temporal trajectory transitions across consecutive connectivity windows.
5. **Brain Region Interpretability (Grad-CAM):** Gradient-based activation mapping back to human-readable AAL atlas brain regions (e.g., `Frontal_Sup_Medial_R`, `Precuneus_R`) and functional connectivity edges.

---

## 🏗️ Architecture Pipeline

$$\text{Raw BIDS fMRI} \longrightarrow \text{Nilearn Confound Regression} \longrightarrow \text{1D DAE} \longrightarrow \text{AAL Atlas (116 ROIs)} \longrightarrow \text{Dynamic Connectivity} \longrightarrow \text{Leakage-Safe ComBat} \longrightarrow \text{ConvLSTM} \longrightarrow \text{P(ADHD)} \longrightarrow \text{Grad-CAM}$$

```
+-----------------------------+
| Raw fMRI (BIDS Format)      |
+--------------+--------------+
               |
               v
+-----------------------------+
| Nilearn Confound Regression |
| & Temporal Filtering        |
+--------------+--------------+
               |
               v
+-----------------------------+
| Denoising Autoencoder (DAE) |
+--------------+--------------+
               |
               v
+-----------------------------+
| AAL Atlas 116 ROI Extraction|
+--------------+--------------+
               |
               v
+-----------------------------+
| Dynamic Sliding-Window      |
| Connectivity (116x116 x Tw) |
+--------------+--------------+
               |
               v
+-----------------------------+
| Leakage-Safe ComBat         |
| (Fitted on Train Fold ONLY) |
+--------------+--------------+
               |
               v
+-----------------------------+
| ConvLSTM Spatiotemporal Model|
+--------------+--------------+
               |
               v
+-----------------------------+
| P(ADHD) vs P(Control)       |
+--------------+--------------+
               |
               v
+-----------------------------+
| Grad-CAM Explainability     |
| (Top AAL ROIs & Edges)      |
+-----------------------------+
```

---

## 📂 Project Structure

```
.
├── configs/
│   └── config.yaml               # Master hyperparameters, data paths, and model settings
├── data/
│   ├── dataset.py                # ADHD-200 phenotypic metadata & subject BIDS parser
│   ├── roi_extraction.py         # Nilearn AAL 116 ROI time-series extractor & caching engine
│   └── cached_features/          # Extracted AAL time series & dataset manifest
├── models/
│   ├── dae.py                    # 1D Denoising Autoencoder for fMRI ROI signals
│   ├── connectivity.py           # Dynamic sliding-window Pearson connectivity matrix generator
│   ├── conv_lstm.py              # ConvLSTM 2D spatial + temporal network architecture
│   └── baselines.py              # SVM, RandomForest, and 2D Spatial CNN baselines
├── harmonization/
│   └── combat.py                 # Leakage-Safe ComBat harmonizer (fit_transform on train fold only)
├── training/
│   ├── loso_trainer.py           # Leave-One-Site-Out (LOSO) cross-validation benchmark engine
│   └── utils.py                  # Training loss functions, batch collators & helpers
├── explainability/
│   └── grad_cam.py               # ConvLSTM Grad-CAM mapped to AAL ROIs & connectivity edges
├── evaluation/
│   ├── metrics.py                # Accuracy, Precision, Recall, Specificity, F1, ROC-AUC
│   ├── ablations.py              # Automated ablation benchmark suite (±DAE, ±ComBat, etc.)
│   └── failure_analysis.py       # Error breakdown by site, age, gender, and scan duration
├── results/                      # Saved metrics CSVs, ablation summary, and explainability reports
├── main.py                       # Master CLI runner
├── requirements.txt              # Project Python dependencies
└── README.md                     # Documentation
```

---

## 🚀 Getting Started

### 1. Requirements Installation
```bash
pip install -r requirements.txt
```

### 2. Dataset Configuration
The pipeline directly reads raw BIDS fMRI data from your external drive at `/Volumes/Lexar/ADHD200_RawDataBIDS`. If your path differs, update `configs/config.yaml`:
```yaml
dataset:
  raw_data_dir: "/Volumes/Lexar/ADHD200_RawDataBIDS"
```

### 3. Run End-to-End Pipeline
Execute the master runner to perform AAL ROI extraction, LOSO cross-validation, Grad-CAM explainability, pipeline ablations, and failure analysis:
```bash
python3 main.py
```

### 4. Run Individual Modules
- **Extract & Cache AAL ROI Time Series:**
  ```bash
  python3 data/roi_extraction.py
  ```
- **Test Dynamic Connectivity Matrix Generation:**
  ```bash
  python3 models/connectivity.py
  ```
- **Run Denoising Autoencoder (DAE) Test:**
  ```bash
  python3 models/dae.py
  ```
- **Run ConvLSTM Architecture & Grad-CAM Test:**
  ```bash
  python3 explainability/grad_cam.py
  ```
- **Execute Automated Ablation Benchmarks:**
  ```bash
  python3 evaluation/ablations.py
  ```

---

## 📊 Scientific Constraints & Evaluation Rigor

1. **Leave-One-Site-Out (LOSO):** Models are evaluated by holding out an entire imaging site per fold (e.g., Train on NYU + KKI + OHSU + Peking, Test on WashU).
2. **Zero Test Data Leakage:** ComBat harmonization parameters and feature scalers are fitted **strictly inside each LOSO training fold**. The test site remains completely unseen during fitting.
3. **No Arbitrary Reshaping:** ROI vectors are represented as $116 \times 116$ functional connectivity matrices, preserving meaningful anatomical pairwise relationships.
4. **Metrics Reported:** Accuracy, Precision, Recall/Sensitivity, Specificity, F1-Score, ROC-AUC, and Confusion Matrix.
