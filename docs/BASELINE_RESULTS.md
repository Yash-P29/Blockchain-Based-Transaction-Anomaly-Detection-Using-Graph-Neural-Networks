# Phase 2 Baseline Results
## Blockchain-Based Transaction Anomaly Detection Using Graph Neural Networks

> **Dataset**: Elliptic Bitcoin Transaction Dataset  
> **Task**: Binary classification — Illicit (anomaly) vs. Licit (normal) transactions  
> **Evaluation Split**: Chronological (no data leakage)

---

## Data Split Summary

| Split      | Timesteps | Labeled Samples |
|------------|-----------|-----------------|
| Training   | 1 – 30    | 26,905          |
| Validation | 31 – 34   | 2,989           |
| Test       | 35 – 49   | 16,670          |

> **Note**: Unknown-label samples (`class = unknown`) are excluded from all splits. Only supervised training is performed with labeled samples.

---

## Experiment Settings

| Setting              | Value                                         |
|----------------------|-----------------------------------------------|
| Random seed          | 42                                            |
| Class weighting      | `count_normal / count_anomaly` on train set   |
| Data split policy    | Chronological only — no random shuffling      |
| GNN hidden channels  | 64                                            |
| GNN optimizer        | Adam (lr=0.01, weight_decay=5e-4)             |
| GNN loss             | Weighted CrossEntropyLoss                     |
| GNN max epochs       | 200                                           |
| GNN early stopping   | Patience = 20 (on validation loss)            |

---

## Results: All Baseline Models

| Model               | Accuracy | Precision | Recall   | F1       | ROC-AUC  | PR-AUC    |
|---------------------|----------|-----------|----------|----------|----------|-----------|
| Logistic Regression | 0.6349   | 0.1447    | 0.9409   | 0.2509   | 0.8572   | 0.2157    |
| Random Forest       | 0.9775   | 0.9903    | 0.6602   | 0.7922   | 0.9184   | 0.7807    |
| XGBoost             | 0.9621   | 0.6973    | 0.7359   | 0.7161   | 0.9329   | **0.7956**|
| GCN                 | 0.7459   | 0.1645    | 0.7138   | 0.2674   | 0.8046   | 0.2477    |
| GraphSAGE           | 0.8677   | 0.2898    | 0.7147   | 0.4124   | 0.8846   | 0.5292    |
| GAT                 | 0.5506   | 0.1224    | 0.9594   | 0.2172   | 0.8742   | 0.3260    |

> **Primary metric**: PR-AUC (Precision-Recall AUC), since the dataset is highly imbalanced (~10% illicit transactions).

---

## Analysis & Key Observations

### 1. Tree-based Models Lead the Baselines
- **XGBoost** achieves the highest PR-AUC (0.7956), closely followed by **Random Forest** (0.7807).
- Random Forest has near-perfect precision (0.99) but low recall (0.66) — almost zero false alarms but misses ~34% of illicit transactions.
- XGBoost strikes a better precision-recall balance (F1 = 0.716).

### 2. Logistic Regression Collapses to High-Recall / Low-Precision
- LR achieves near-perfect recall (0.94) but very poor precision (0.14), essentially flagging most transactions as anomalies.
- A failure mode caused by linear decision boundaries being insufficient for this feature space.

### 3. Vanilla GNNs Struggle Without Temporal Awareness
- GCN (PR-AUC 0.248) and GAT (PR-AUC 0.326) perform closer to LR than to tree models.
- Both GCN and GAT converge to high-recall/low-precision strategies — not practically useful.
- **GraphSAGE** is the strongest GNN baseline (PR-AUC 0.529), benefiting from neighbourhood sampling that captures local transaction structure.

### 4. The Static Graph Alone Is Not Enough (Yet)
- The gap between tree models (~0.79 PR-AUC) and GNNs (~0.25-0.53 PR-AUC) highlights that static message-passing without temporal context is insufficient.
- This directly motivates the Phase 3 temporal GNN approach.

---

## Early Stopping Epochs (GNNs)

| Model      | Stopped at Epoch |
|------------|-----------------|
| GCN        | 131 / 200        |
| GraphSAGE  | 149 / 200        |
| GAT        | 185 / 200        |

---

## Saved Artifacts

| Artifact                            | Path                                         |
|-------------------------------------|----------------------------------------------|
| Baseline metrics CSV                | `results/metrics/baseline_results.csv`       |
| Logistic Regression model           | `results/models/logistic_regression.joblib`  |
| Random Forest model                 | `results/models/random_forest.joblib`        |
| XGBoost model                       | `results/models/xgboost.joblib`              |
| GCN checkpoint                      | `results/models/gcn.pt`                      |
| GraphSAGE checkpoint                | `results/models/graphsage.pt`                |
| GAT checkpoint                      | `results/models/gat.pt`                      |
| Confusion matrices, ROC & PR curves | `results/figures/`                           |
| GNN training loss curves            | `results/figures/*_loss.png`                 |

---

## Conclusion & Path to Phase 3

Baseline experiments confirm that **tree-based ensemble methods (XGBoost, Random Forest) significantly outperform vanilla GNNs** in a static graph setting. However, GNN baselines are not designed to exploit the temporal dynamics intrinsic to the Elliptic dataset.

**Phase 3 targets:**
1. **Temporal GNN** - incorporating the 49 time-step evolution of the transaction graph.
2. **Improved graph construction** - edge features (transaction volume/value).
3. **GNN Explainability (XAI)** - identifying subgraph structures that predict illicit behaviour.

The baseline PR-AUC benchmarks to beat:
- **XGBoost: 0.7956**
- **GraphSAGE (best GNN): 0.5292**
