# Final Research Results & Evaluation Report

**Project Title:** Blockchain-Based Transaction Anomaly Detection Using Graph Neural Networks  
**Dataset:** Elliptic Bitcoin Transaction Dataset  
**Date:** September 2026  

---

## 1. Executive Summary

This report documents the experimental evaluation of classical machine learning models, baseline Graph Neural Networks (GNNs), and the proposed **Temporal-Structural Graph Neural Network (Temporal-GNN)** for anomaly detection in Bitcoin transactions.

The objective was to evaluate whether augmenting GraphSAGE embeddings with handcrafted temporal and structural features (such as graph degree statistics and timestep metrics) improves the detection of illicit Bitcoin transactions under strict chronological evaluation constraints (train: timesteps 1–30, val: 31–34, test: 35–49).

---

## 2. Complete Model Comparison Table

All metrics below are measured on the held-out test set (16,670 labeled transactions across timesteps 35–49). Class imbalance is preserved (~10% illicit), making **PR-AUC** the primary benchmark metric.

| Model | Model Category | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Training Time (s) | Inference Time (ms) |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | Baseline (Classical) | 0.6349 | 0.1447 | 0.9409 | 0.2509 | 0.8572 | 0.2157 | — | — |
| Random Forest | Baseline (Classical) | 0.9775 | 0.9903 | 0.6602 | 0.7922 | 0.9184 | 0.7807 | — | — |
| **XGBoost** | **Baseline (Classical)** | **0.9621** | **0.6973** | **0.7359** | **0.7161** | **0.9329** | **0.7956** | — | — |
| GCN | Baseline (GNN) | 0.7459 | 0.1645 | 0.7138 | 0.2674 | 0.8046 | 0.2477 | — | — |
| GAT | Baseline (GNN) | 0.5506 | 0.1224 | 0.9594 | 0.2172 | 0.8742 | 0.3260 | — | — |
| GraphSAGE (P2 Baseline) | Baseline (GNN) | 0.8677 | 0.2898 | 0.7147 | 0.4124 | 0.8846 | 0.5292 | — | — |
| GraphSAGE (Ablation A) | Phase 3 GNN | 0.7665 | 0.1904 | 0.7978 | 0.3074 | 0.8891 | 0.5629 | 41.97 | 133.55 |
| GraphSAGE + Temp (Ablation B) | Phase 3 Proposed | 0.7878 | 0.2036 | 0.7784 | 0.3228 | 0.8837 | 0.5107 | 55.29 | 416.64 |
| **Temporal-GNN (Model C)** | **Phase 3 Proposed** | **0.7623** | **0.1878** | **0.7996** | **0.3041** | **0.8860** | **0.5276** | **73.70** | **382.06** |

---

## 3. Ablation Study Analysis

To isolate the contributions of different architectural components, we evaluated three variants:

1. **Variant A (GraphSAGE Baseline):** Standard 2-layer GraphSAGE encoder operating solely on original node features and graph structure. Achieved **PR-AUC 0.5629** and **ROC-AUC 0.8891**.
2. **Variant B (GraphSAGE + Temporal Features):** GraphSAGE combined directly with the 9-dimensional temporal/structural MLP embedding without a skip connection. Achieved **PR-AUC 0.5107**.
3. **Variant C (Full Temporal-GNN):** GraphSAGE + temporal/structural MLP + skip connection fusing raw node features into the classification head. Achieved **PR-AUC 0.5276** and **Recall 0.7996** (highest among GNN variants).

### Key Findings
* **GNN vs. Tree-Based Performance:** Gradient-boosted decision trees (XGBoost PR-AUC 0.7956) and Random Forest (PR-AUC 0.7807) significantly outperform all static GNN architectures on the Elliptic dataset. Tree-based models leverage exact split thresholds on tabular node features directly.
* **Temporal Feature Contribution:** Incorporating basic static/global graph degree and normalized timestep features into the GNN did not improve overall PR-AUC over standard GraphSAGE (0.5276 vs 0.5629). This indicates that global structural summaries do not replace dynamic local graph structural shifts across timesteps.
* **High Recall in GNNs:** The proposed Temporal-GNN achieved a high recall of **79.96%**, capturing 4 out of 5 illicit transactions, making it suitable as a high-recall candidate generator in a multi-stage anomaly detection pipeline.

---

## 4. Explainability & Feature Contributions

Permutation-based feature importance on the test set identified the top temporal/structural contributors to the model's predictions:

1. **`local_neighbor_count`** (PR-AUC drop: +0.0035) — Structural metric measuring local neighborhood density.
2. **`total_degree`** (PR-AUC drop: +0.0019) — Node degree reflecting transaction connectivity.
3. **`out_degree`** (PR-AUC drop: +0.0008) — Fan-out behavior characteristic of mixing or dispersion transactions.

---

## 5. Research Limitations

1. **Static PyG Graph Representation:** The Elliptic dataset consists of 49 disconnected subgraphs representing individual timesteps. Aggregating them into a single graph with static SAGEConv operations restricts cross-timestep temporal propagation.
2. **Anonymized Features:** 166 of the 167 features are local/aggregated transaction features without public semantic labels, limiting custom domain feature engineering.
3. **Imbalance Sensitivity:** Standard weighted cross-entropy helps recall but introduces precision trade-offs in GNN outputs compared to decision tree boundaries.

---

## 6. Conclusion & Future Work

The experiments demonstrate:
* strong performance from tree-based models on the anonymized transaction features
* lower performance from vanilla static GNNs
* the tested temporal/structural augmentation did not improve PR-AUC over the GraphSAGE baseline in this experiment
* this indicates that simply adding temporal/structural features is insufficient and provides a meaningful direction for future research
