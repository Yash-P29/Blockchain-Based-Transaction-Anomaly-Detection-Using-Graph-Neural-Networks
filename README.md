# Blockchain-Based Transaction Anomaly Detection Using Graph Neural Networks

## 1. Problem Statement
Illicit activities on blockchain networks, such as money laundering, scams, and ransomware payments, pose significant challenges to financial security. Detecting anomalous transactions is difficult because malicious actors obscure their tracks using complex transaction patterns and mixing services. Traditional tabular machine learning approaches evaluate transactions in isolation, missing the structural graph context inherent in blockchain ledgers.

## 2. Research Motivation & Core Innovation
Graph Neural Networks (GNNs) natively capture topological transaction graph context by aggregating features from neighboring nodes. In this project, we propose a **Temporal-Structural Graph Neural Network (Temporal-GNN)** architecture that combines:
1. **GraphSAGE GNN Encoder** for dynamic topological representation learning
2. **Temporal & Structural MLP** for encoding node degree dynamics, neighborhood density, and timestep statistics
3. **Feature Fusion Layer** with a skip-connection preserving raw local transaction features
4. **Weighted Cross-Entropy** to handle extreme class imbalance (~10% illicit transactions)

---

## 3. Comprehensive Experimental Results

Evaluation is performed on the **Elliptic Bitcoin Transaction Dataset** under strict chronological splitting (Train: timesteps 1–30, Val: timesteps 31–34, Test: timesteps 35–49) to prevent temporal data leakage.

### Benchmark & Ablation Results (Test Set: 16,670 Labeled Transactions)

| Model | Category | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| Logistic Regression | Baseline (Linear) | 0.6349 | 0.1447 | 0.9409 | 0.2509 | 0.8572 | 0.2157 |
| Random Forest | Baseline (Tree) | 0.9775 | 0.9903 | 0.6602 | 0.7922 | 0.9184 | 0.7807 |
| **XGBoost** | **Baseline (Tree)** | **0.9621** | **0.6973** | **0.7359** | **0.7161** | **0.9329** | **0.7956** |
| GCN | Baseline (GNN) | 0.7459 | 0.1645 | 0.7138 | 0.2674 | 0.8046 | 0.2477 |
| GAT | Baseline (GNN) | 0.5506 | 0.1224 | 0.9594 | 0.2172 | 0.8742 | 0.3260 |
| GraphSAGE (Ablation A) | GNN Baseline | 0.7665 | 0.1904 | 0.7978 | 0.3074 | 0.8891 | 0.5629 |
| GraphSAGE + Temp (Ablation B) | Phase 3 Ablation | 0.7878 | 0.2036 | 0.7784 | 0.3228 | 0.8837 | 0.5107 |
| **Temporal-GNN (Model C)** | **Proposed Model** | **0.7623** | **0.1878** | **0.7996** | **0.3041** | **0.8860** | **0.5276** |

---

## 4. Technology Stack
- **Python 3.11.9**
- **PyTorch 2.2.1** & **PyTorch Geometric (PyG) 2.5.0**
- **Pandas** & **NumPy (<2.0)**
- **Scikit-learn** & **XGBoost**
- **Streamlit** (Interactive Demo Application)
- **Pytest** (Automated Test Suite)

---

## 5. Dataset Setup & Structure
The **Elliptic Bitcoin Transaction Dataset** consists of:
- `elliptic_txs_features.csv`: 203,769 transactions with 166 local/aggregated features + timestep (1–49)
- `elliptic_txs_edgelist.csv`: 234,355 directed edges representing transaction flow
- `elliptic_txs_classes.csv`: Labels (1 = illicit, 2 = licit, unknown = unlabeled)

---

## 6. Installation & Usage

### 1. Environment Setup
Activate your virtual environment and verify dependencies:
```bash
venv\Scripts\activate
```

### 2. Preprocessing & Graph Construction
```bash
python -m src.data.preprocessing
```

### 3. Run Classical & Baseline Model Training (Phase 2)
```bash
python -m src.training.train_baselines
```

### 4. Run Proposed Model & Ablation Study Training (Phase 3)
```bash
python -m src.training.train_proposed
```

### 5. Compute Feature Importance & Explainability
```bash
python -m src.training.compute_importance
```

### 6. Run Automated Test Suite
```bash
pytest tests/ -v
```

### 7. Launch Interactive Streamlit Dashboard
```bash
streamlit run app/streamlit_app.py
```

---

## 7. Streamlit Dashboard Features
The 6-tab Streamlit dashboard provides:
1. **Dataset Overview**: Detailed node/edge stats, timestep distribution, degree breakdown
2. **Baseline Comparison**: Metrics and PR-AUC bar charts comparing classical ML & baseline GNNs
3. **Proposed Model**: Model architecture breakdown, training loss curves, actual experiment results
4. **Transaction Prediction**: Interactive lookup by transaction ID or timestep with real-time model probability scoring
5. **Explainability**: Permutation-based feature importance chart and confusion matrix
6. **Research Results**: Comprehensive model comparison table, PR/ROC curves, ablation summary

---

## 8. Documentation Artifacts
- [`docs/PROPOSED_MODEL.md`](docs/PROPOSED_MODEL.md) — Comprehensive technical overview of model architecture, feature design, and training strategy.
- [`docs/FINAL_RESULTS.md`](docs/FINAL_RESULTS.md) — Detailed final research evaluation report, ablation study analysis, limitations, and future work.
