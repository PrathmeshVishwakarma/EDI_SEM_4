# MotorSense - Research Benchmark Results

## 1. Overlap Sweep - Accuracy vs RPM Overlap

| RPM Overlap | Accuracy | F1 Score |
|------------|----------|----------|
| 0.0 | 1.0000+/-0.0000 | 1.0000+/-0.0000 |
| 0.2 | 1.0000+/-0.0000 | 1.0000+/-0.0000 |
| 0.4 | 0.9778+/-0.0208 | 0.9778+/-0.0208 |
| 0.6 | 0.9167+/-0.0491 | 0.9167+/-0.0487 |
| 0.8 | 0.7667+/-0.0943 | 0.7650+/-0.0976 |
| 1.0 | 0.3944+/-0.0550 | 0.3822+/-0.0636 |

## 2. Baseline Comparisons

| Method | overlap=0.0 | overlap=0.4 | overlap=1.0 |
|--------|------------|------------|------------|
| RMS Threshold (2s) | 0.1667 | 0.1500 | 0.1833 |
| RandomForest Only | 1.0000 | 0.9667 | 0.5000 |
| OC-SVM Only | 0.7500 | 0.7333 | 0.1500 |
| Full Ensemble (Ours) | 1.0000 | 0.9667 | 0.3833 |

## 3. Feature Ablation

Base accuracy (all 28 features): 0.9889

| Rank | Feature | Importance (d-acc) |
|------|---------|------------------|
| 1 | Skewness | 0.1267 |
| 2 | Peak Freq 3 | 0.0000 |
| 3 | Peak Freq 2 | 0.0000 |
| 4 | Peak-to-Peak | 0.0000 |
| 5 | Variance | 0.0000 |
| 6 | Kurtosis | 0.0000 |
| 7 | Crest Factor | 0.0000 |
| 8 | Shape Factor | 0.0000 |
| 9 | Zero-Crossing Rate | 0.0000 |
| 10 | Spectral Band 1 | 0.0000 |

**Group Importance:**

- Time Domain (0-7): 0.1267
- Spectral Bands (8-17): -0.0111
- Frequency Stats (18-27): -0.0111

## 4. Data Scaling

| Samples/Company | Training Size | Accuracy |
|----------------|--------------|----------|
| 10 | 70 | 1.0000 |
| 20 | 140 | 0.9667 |
| 40 | 280 | 0.9667 |
| 60 | 420 | 0.9778 |
| 80 | 560 | 0.9667 |

## 5. Transfer Learning (CWRU to JNU)

- Zero-shot accuracy: 0.0000
- Fine-tuned (+10% target): 1.0000
- Delta: +1.0000

## 6. SoTA Classifier Comparison (10 methods)

| Method | overlap=0.0 | overlap=0.4 | overlap=1.0 | Avg Rank |
|--------|------------|------------|------------|----------|
| LogisticRegression | 0.9833 | 0.9833 | 0.3833 | |
| KNN (k=5) | 1.0 | 0.9667 | 0.3 | |
| KNN (k=21) | 1.0 | 0.9167 | 0.3667 | |
| SVC (rbf) | 1.0 | 0.9833 | 0.4167 | |
| DecisionTree | 0.9833 | 0.9667 | 0.5167 | |
| RandomForest | 1.0 | 0.9667 | 0.5 | |
| ExtraTrees | 1.0 | 0.9833 | 0.3667 | |
| GradientBoosting | 0.9833 | 0.9333 | 0.3833 | |
| MLP (64,32) | 1.0 | 0.9833 | 0.3667 | |
| MLP (128,64,32) | 1.0 | 0.9833 | 0.3667 | |
| OC-SVM (1vsRest) | 0.1667 | 0.1667 | 0.1667 | |
| Ours (RF+IF+SVM) | 1.0 | 0.9667 | 0.3833 | |

**Top-3 by average accuracy:** DecisionTree (0.8222), RandomForest (0.8222), SVC (rbf) (0.8000)

## 7. Anomaly Detection Benchmark (6 methods)

| Method | Unbalanced F1 | Bearing F1 | Misalignment F1 | Avg F1 |
|--------|--------------|------------|-----------------|--------|
| IsolationForest | 0.9836 | 0.9836 | 0.9836 | 0.9836 |
| OC-SVM | 0.9266 | 0.9266 | 0.9231 | 0.9254 |
| EllipticEnvelope | 0.9756 | 0.9756 | 0.9756 | 0.9756 |
| LOF | 0.9756 | 0.9756 | 0.9756 | 0.9756 |
| PCA (99% var) | 0.1818 | 0.1818 | 0.1818 | 0.1818 |
| Autoencoder (MLP) | 0.1818 | 0.1818 | 0.1818 | 0.1818 |