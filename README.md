# 🎓 Playground Series S6E1: Advanced EDA & Blended Ensemble Solution

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-Powered-orange?style=for-the-badge&logo=xgboost)
![LightGBM](https://img.shields.io/badge/LightGBM-Powered-green?style=for-the-badge)
![CatBoost](https://img.shields.io/badge/CatBoost-Powered-yellow?style=for-the-badge)
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?style=for-the-badge&logo=kaggle)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A two-stage solution: deep exploratory analysis driving feature engineering, followed by a GPU-accelerated XGBoost + LightGBM + CatBoost blended ensemble with automated weighting.**

[📊 View EDA Notebook on Kaggle](https://www.kaggle.com/code/rishabhkannaujiya/s6e1-advanced-eda-xgboost-competition-starter?scriptVersionId=289630377)

[📊 View Solution Notebook on Kaggle](https://www.kaggle.com/code/rishabhkannaujiya/s6e1-xgb-lgbm-catboost-auto-blend?scriptVersionId=294993011)

</div>

---

## 🎯 Overview

This repository contains a **production-grade solution** for the Kaggle Playground Series Season 6 Episode 1 competition, predicting student exam scores based on various academic and lifestyle factors. The solution is split across two notebooks that work together as a deliberate pipeline.

| Notebook | Role |
|---|---|
| `s6e1-advanced-eda-xgboost-competition-starter.ipynb` | Deep EDA (19 sections) → Feature discovery → Single XGBoost baseline |
| `s6e1-xgb-lgbm-catboost-auto-blend.ipynb` | Production pipeline → 45-feature engineering → 3-model ensemble → Auto-blended submission |

### 🔍 Problem Statement
Predict student exam scores using features like:
- Study hours & class attendance
- Sleep quality & patterns
- Study methods & facilities
- Demographic information

### 🏆 Competition Goal
Minimize **RMSE (Root Mean Squared Error)** on test predictions

---

## 📊 Results

| Model | OOF RMSE | Notes |
|---|---|---|
| Ridge (Stacking Layer) | 8.8923 | 10-Fold, Target Encoded, used as a meta-feature |
| XGBoost (3-Seed Avg) | 8.72596 | 10-Fold, GPU, augmented with original dataset |
| LightGBM (3-Seed Avg) | 8.72474 | 10-Fold, GPU, best individual model |
| CatBoost (3-Seed Avg) | 8.77007 | 10-Fold, GPU, native categorical handling |
| **Final Blend** | **8.72101** | Auto-weighted: XGB 46.5% / LGBM 53.5% / CAT 0.0% |

> The automated blender assigned zero weight to CatBoost, converging on an XGBoost + LightGBM blend as the optimal combination.

---

## ✨ Key Features

### 🔬 Comprehensive EDA (Notebook 1)
- **19 analytical sections** covering every aspect of the data
- **Adversarial Validation** (AUC ≈ 0.50 confirms no train/test drift)
- **KS-Test drift analysis** across all numeric and categorical features
- **UMAP dimensionality reduction** reveals two structural clusters driven by `gender` and `exam_difficulty`
- **Heteroscedasticity detection** via Breusch-Pagan test → justified log-target transformation
- **Residual-based feature discovery**: linear model on `study_hours` exposes `class_attendance` and `sleep_hours` as high-value additions
- **Synthetic artifact inspection**: precision mismatch (4 vs 3 decimals) detected between train and test
- **Lasso reverse-engineering** to approximate the data generation formula
- **Quantile trend analysis** shows studying reduces score variance (risk mitigation)

### 🎯 Robust Modeling Pipeline (Notebook 2)

**Stage 1 — Ridge Stacking Layer**
- 10-Fold CV with Target Encoding on 7 categorical features
- Training data augmented with the original (non-synthetic) Kaggle dataset
- Ridge predictions added as a meta-feature (`feature_lr_pred`) for all three tree models

**Stage 2 — Three-Model Ensemble with Seed Averaging**
- Every fold trains 3 models (seeds: 42, 2024, 123) and averages predictions to reduce variance
- All models augmented with original dataset during training
- All models use GPU acceleration

| Model | Key Hyperparameters |
|---|---|
| XGBoost | `lr=0.004`, `max_depth=9`, `n_est=20000`, `colsample_bytree=0.55`, early stop @ 100 |
| LightGBM | `lr=0.008`, `num_leaves=100`, `n_est=12000`, `colsample_bytree=0.6`, early stop @ 100 |
| CatBoost | `lr=0.02`, `depth=6`, `iterations=8000`, native categorical, early stop @ 100 |

**Stage 3 — Automated Ensemble Weighting**
- Non-negative `LinearRegression` (no intercept) fitted on OOF predictions
- Weights normalized to sum to 1.0
- Result: **XGBoost 46.5% + LightGBM 53.5%** (CatBoost zeroed out by the optimizer)

---

## 🏗️ Pipeline Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Notebook 1: EDA & Discovery                            │
│  ├─ Target & drift analysis (19 sections)               │
│  ├─ UMAP cluster identification                         │
│  ├─ Residual-guided feature discovery                   │
│  ├─ Heteroscedasticity → log-target decision            │
│  └─ Single XGBoost baseline (5-Fold, log target)        │
└─────────────────────┬───────────────────────────────────┘
                      │  findings feed into
                      ▼
┌─────────────────────────────────────────────────────────┐
│  Notebook 2: Production Ensemble                        │
│  ├─ 45-feature engineering (EDA-informed)               │
│  ├─ Stage 1: Ridge stacking layer (10-Fold, TE)         │
│  ├─ Stage 2: XGB + LGBM + CAT (3-seed avg, 10-Fold)     │
│  ├─ Stage 3: Auto-weighted blend                        │
│  └─ Final submission (clipped 0–100)                    │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 Visualizations (Notebook 1)

The EDA notebook includes **20+ professional visualizations**:

- 📈 Target distribution with KDE & skewness/kurtosis metrics
- 🗺️ UMAP dimensionality reduction (colored by score, gender, difficulty, study method, course)
- 🔥 Correlation heatmaps (Pearson & Spearman hierarchical clustering)
- 📦 Residual boxplots by categorical feature (error attribution)
- 🌳 Decision tree explaining the two UMAP clusters
- 📊 Quantile trend analysis (10th / 50th / 90th percentile curves)
- 🔍 Interaction heatmap (study hours × attendance → average score grid)
- ⚠️ Anomaly detection scatter (Isolation Forest, 1% contamination)
- 📐 Heteroscedasticity residual fan plots (Raw vs Log vs Box-Cox)
- 📏 KS-test drift plots for all numeric features
- 📊 Categorical drift bar charts
- 🎯 Mutual Information scores & Permutation Importance (Random Forest)
- 🧪 Lasso coefficient plot (reverse-engineered formula weights)

---

## 📁 Repository Structure

```
S6E1-Predicting-Student-Test-Scores:
   ├── README.md                                             # This file
   ├── s6e1-advanced-eda-xgboost-competition-starter.ipynb   # EDA & baseline
   ├── s6e1-xgb-lgbm-catboost-auto-blend.ipynb               # Production ensemble
   └── LICENSE                                               # MIT License
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

### ⭐ Star this repository if you found it helpful!

**Made with ❤️**

[⬆ Back to Top](#-playground-series-s6e1-advanced-eda--blended-ensemble-solution)

</div>
