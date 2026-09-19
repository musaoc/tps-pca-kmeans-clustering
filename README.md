# Unsupervised Learning with PCA & K-Means Clustering — TPS Jul 2022

An unsupervised machine learning project leveraging Principal Component Analysis for dimensional compression and K-Means/GMM algorithms for cluster discovery in competitive tabular data.

[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/lazer999/tps-pca-k-means-simplified-4-everyone)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Field](https://img.shields.io/badge/Field-Unsupervised%20Learning%20/%20Clustering-brightgreen)](#)

---

## Table of Contents
- [Project Overview](#project-overview)
- [Key Highlights & Results](#key-highlights--results)
- [System Architecture & Workflow](#system-architecture--workflow)
- [Repository Structure](#repository-structure)
- [Quickstart & Reproduction](#quickstart--reproduction)
- [Dataset Details](#dataset-details)
- [Author & Acknowledgments](#author--acknowledgments)

---

## Project Overview

This repository provides the complete, production-structured implementation of the **[Unsupervised Learning with PCA & K-Means Clustering — TPS Jul 2022](https://www.kaggle.com/code/lazer999/tps-pca-k-means-simplified-4-everyone)** project originally published on Kaggle. 

The primary focus of this work is translating complex data into actionable machine learning solutions using disciplined data engineering, rigorous validation strategies, and clean, leak-free preprocessing pipelines.

---

## Key Highlights & Results

- Preprocessed multi-dimensional continuous tabular feature matrices using standard scaling.
- Applied Principal Component Analysis (PCA) to extract dominant variance vectors and reduce noise.
- Determined optimal cluster quantity (K) using Yellowbrick's `KElbowVisualizer` (distortion/silhouette scores).
- Benchmarked deterministic K-Means partitioning against probabilistic Gaussian Mixture Models (GMM).

---

## System Architecture & Workflow

The pipeline follows a structured, modular execution path:

```mermaid
flowchart LR
    A[Tabular Feature Matrix] --> B[StandardScaler Normalization]
    B --> C[PCA Dimensionality Reduction]
    C --> D[Elbow Method / Distortion Analysis]
    D --> E[K-Means Clustering]
    D --> F[Gaussian Mixture Models]
    E --> G[Cluster Assignments & Visual Diagnostics]
    F --> G
```

---

## Repository Structure

```plaintext
tps-pca-kmeans-clustering/
├── notebooks/
│   └── tps-pca-kmeans-clustering.ipynb      # Original Jupyter notebook with full exploratory visuals
├── src/
│   └── main.py                # Modular, executable Python pipeline
├── .gitignore                 # Standard Python/Jupyter ignores
├── LICENSE                    # MIT License
├── README.md                  # Human-friendly documentation
└── requirements.txt           # Verified Python dependencies
```

---

## Quickstart & Reproduction

### 1. Clone the Repository
```bash
git clone https://github.com/musaoc/tps-pca-kmeans-clustering.git
cd tps-pca-kmeans-clustering
```

### 2. Set Up a Virtual Environment
```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Pipeline
You can run the end-to-end script directly:
```bash
python src/main.py
```

Or open and run the interactive notebook:
```bash
jupyter lab notebooks/tps-pca-kmeans-clustering.ipynb
```

---

## Dataset Details

- **Dataset / Competition**: [Kaggle Tabular Playground Series (Jul 2022)](https://www.kaggle.com/c/tabular-playground-series-jul-2022)
- **Origin Platform**: Kaggle
- For automated dataset downloading via Kaggle CLI:
  ```bash
  kaggle competitions download -c tabular-playground-series-jul-2022
  ```

---

## Author & Acknowledgments

- **Author**: **Muhammad Musa Khan** (Kaggle Master)
- **Kaggle Profile**: [@lazer999](https://www.kaggle.com/lazer999)
- **GitHub**: [@musaoc](https://github.com/musaoc)
- **Original Kaggle Solution**: [Unsupervised Learning with PCA & K-Means Clustering — TPS Jul 2022](https://www.kaggle.com/code/lazer999/tps-pca-k-means-simplified-4-everyone)

If you found this project helpful or insightful, please consider starring the repository ⭐!
