# Proposed Temporal-Structural GNN Model

## Blockchain-Based Transaction Anomaly Detection Using Graph Neural Networks

---

## 1. Problem

The Elliptic Bitcoin Transaction Dataset contains 203,769 transactions (nodes) connected by 234,355 directed fund-flow edges, observed across 49 discrete timesteps. The task is to classify each labeled transaction as **illicit** (anomalous) or **licit** (normal), under:

- Severe class imbalance (~10 % illicit in labeled set)
- Temporal ordering that must be respected to avoid data leakage

Phase 2 baselines demonstrated that **tree-based models significantly outperform vanilla GNNs** in a static graph setting (XGBoost PR-AUC = 0.7956 vs. GraphSAGE PR-AUC = 0.5292). The gap motivates introducing temporal and structural context into the GNN.

---

## 2. Baseline Limitations

| Model | PR-AUC | Primary Limitation |
|-------|--------|--------------------|
| Logistic Regression | 0.2157 | Linear boundary insufficient |
| Random Forest | 0.7807 | Ignores graph structure |
| XGBoost | 0.7956 | Ignores graph structure |
| GCN | 0.2477 | Spectral, no temporal context |
| GraphSAGE | 0.5292 | No temporal context |
| GAT | 0.3260 | No temporal context |

Vanilla GNNs use only raw node features and static message-passing. They do not exploit the sequential timestep structure of the dataset, which encodes behavioural patterns of illicit actors over time.

---

## 3. Motivation

The Elliptic dataset's 49 timesteps represent roughly two-week periods in real Bitcoin transaction history. Illicit activity often clusters in specific temporal windows and exhibits distinct structural signatures (e.g., high out-degree "cash-out" nodes). Combining:

1. **Graph structural signals** — local topology, message-passing neighbourhood
2. **Temporal signals** — timestep position, per-timestep activity statistics

…into a unified model may improve discrimination beyond what either component achieves alone.

---

## 4. Proposed Architecture

```
[Node Features x ∈ R^{166}]          [Temporal/Structural Features x_temp ∈ R^9]
         │                                            │
    ┌────┴────┐                                ┌──────┴──────┐
    │ SAGEConv │                                │  Linear(32) │
    │  (→ 64) │                                │    ReLU     │
    │   ReLU  │                                │  Dropout    │
    │ Dropout │                                │  Linear(32) │
    │ SAGEConv│                                │    ReLU     │
    │  (→ 64) │                                └──────┬──────┘
    └────┬────┘                                       │
         │  gnn_emb (64)               temp_emb (32)  │
         │                                            │
         └──────────────────────┬─────────────────────┘
                                │  + skip: x (166)
                     [cat: 64 + 32 + 166 = 262]
                                │
                        Linear(262 → 64)
                              ReLU
                            Dropout
                        Linear(64 → 2)
                                │
                          [Logits: licit / illicit]
```

---

## 5. Graph Component

- **Encoder**: Two-layer GraphSAGE (`SAGEConv`) — mean aggregation
- **Layer 1**: `in_channels → 64`, ReLU, Dropout(0.5)
- **Layer 2**: `64 → 64`
- **Rationale**: GraphSAGE was the strongest GNN baseline (PR-AUC 0.5292) and is inductively sound (can generalise to unseen nodes)

---

## 6. Temporal Component

A lightweight two-layer MLP encodes 9 handcrafted temporal and structural features:

| Feature | Description |
|---------|-------------|
| `timestep_norm` | Timestep normalised to [0, 1] |
| `in_degree` | Incoming edge count |
| `out_degree` | Outgoing edge count |
| `total_degree` | in + out degree |
| `degree_ratio` | in / (total + ε), asymmetry indicator |
| `local_neighbor_count` | Distinct undirected neighbours |
| `timestep_degree_interact` | `timestep_norm × log(1 + total_degree)` |
| `ts_mean_degree` | Mean degree of all nodes at same timestep |
| `ts_illicit_rate` | Illicit fraction at timestep (**training steps only** — no leakage) |

**No transaction value or amount features are used** — the Elliptic dataset's 166 features are anonymised and their exact semantics are not publicly disclosed.

**MLP**: `9 → 32 → ReLU → Dropout(0.3) → 32 → ReLU`

---

## 7. Feature Fusion

Three information streams are concatenated:

```
fused = [gnn_emb(64) | temp_emb(32) | raw_features(166)]  # dim = 262
```

A two-layer fusion MLP then produces the final classification logit:

```
Linear(262 → 64) → ReLU → Dropout(0.5) → Linear(64 → 2)
```

The skip connection over raw features preserves the discriminative power of the original 166 node features demonstrated by XGBoost.

---

## 8. Training Methodology

| Setting | Value |
|---------|-------|
| Optimizer | Adam (lr=0.01, weight_decay=5e-4) |
| Loss | `CrossEntropyLoss(weight=[1.0, N_licit/N_illicit])` |
| Max epochs | 200 |
| Early stopping | Patience 20 (on validation loss) |
| Dropout | 0.5 (GNN), 0.3 (temporal MLP) |

**Class weights** are computed from training labels only. The test set is not touched during training or weight computation.

---

## 9. Evaluation Methodology

**Data split** (chronological, no shuffling):
- Training: timesteps 1–30
- Validation: timesteps 31–34
- Test: timesteps 35–49

**Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC  
**Primary metric**: PR-AUC (dataset is imbalanced; ~10 % illicit in labeled set)

**Ablations**:
- A. GraphSAGE baseline (re-evaluated for consistency)
- B. GraphSAGE + temporal/structural features only
- C. Full proposed model (A + B + feature fusion with skip connection)

---

## 10. Limitations

1. **Temporal modelling is MLP-based** — a GRU/LSTM applied per-node across timesteps would require constructing per-node temporal sequences, which is architecturally complex. The MLP approach encodes temporal position implicitly.
2. **Timestep granularity** — 49 timesteps at ~two-week resolution may miss intra-week patterns.
3. **Unlabeled nodes** — ~21 % of nodes have `class = unknown` and are excluded from supervised training. Semi-supervised approaches could leverage these.
4. **Static graph** — edges across all timesteps are merged into one graph. A dynamic graph model (e.g., TGN) could encode edge timing explicitly.
5. **`ts_illicit_rate` feature** — this encodes the empirical label rate at each training timestep, which is a form of aggregated label information. It is computed exclusively from training steps (1–30) to avoid leakage, but it does encode supervision signal indirectly.
6. **CPU-only training** — the current environment uses `torch 2.2.1+cpu`. GPU acceleration would significantly reduce training time.
