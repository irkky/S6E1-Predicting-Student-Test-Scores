# 🎓 Playground Series S6E1: Advanced EDA & XGBoost Solution

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-Powered-orange?style=for-the-badge&logo=xgboost)
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?style=for-the-badge&logo=kaggle)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A comprehensive machine learning solution featuring advanced EDA techniques and optimized XGBoost modeling**

[📊 View Notebook on Kaggle](https://www.kaggle.com/code/rishabhkannaujiya/s6e1-advanced-eda-xgboost-competition-starter?scriptVersionId=289630377)

</div>

---

## 🎯 Overview

This repository contains a **production-grade solution** for the Kaggle Playground Series Season 6 Episode 1 competition, predicting student exam scores based on various academic and lifestyle factors.

### 🔍 Problem Statement
Predict student exam scores using features like:
- Study hours & class attendance
- Sleep quality & patterns
- Study methods & facilities
- Demographic information

### 🏆 Competition Goal
Minimize **RMSE (Root Mean Squared Error)** on test predictions

---

## ✨ Key Features

### 🔬 Comprehensive EDA
- **19 analytical sections** covering every aspect of the data
- Statistical tests for drift detection (KS-Test, Adversarial Validation)
- Visual analysis with UMAP dimensionality reduction
- Heteroscedasticity detection using Breusch-Pagan test

### 🧠 Advanced Feature Engineering
- **Multiplicative interaction features** (`study_hours × class_attendance`)
- Polynomial transformations for diminishing returns patterns
- K-Means clustering for latent pattern extraction
- Residual-based feature discovery

### 🎯 Robust Modeling
- XGBoost with **GPU acceleration**
- 5-Fold Cross-Validation
- Log-transformation for variance stabilization
- Early stopping to prevent overfitting

---


## 📊 Visualizations

The notebook includes **20+ professional visualizations**:

- 📈 Distribution plots with KDE
- 🗺️ UMAP dimensionality reduction
- 🔥 Correlation heatmaps
- 📦 Residual boxplots
- 🌳 Decision tree interpretability
- 📊 Quantile trend analysis

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

### ⭐ Star this repository if you found it helpful!

**Made with ❤️ and lots of ☕**

[⬆ Back to Top](#-playground-series-s6e1-advanced-eda--xgboost-solution)

</div>
