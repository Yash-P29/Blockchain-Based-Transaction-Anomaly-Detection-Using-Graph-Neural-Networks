# Experiment Configuration (Phase 2)

## Environment
- **Python version**: 3.11+
- **PyTorch version**: 2.2.1
- **PyTorch Geometric version**: 2.5.0
- **Scikit-Learn version**: 1.4.1
- **XGBoost version**: 3.2.0

## Dataset
- **Source**: Elliptic Bitcoin Transaction Dataset (Kaggle)
- **Features**: 166 features (164 node features, 1 timestep, 1 id)
- **Time-step split**:
  - Training: Timesteps 1 to 30
  - Validation: Timesteps 31 to 34
  - Test: Timesteps 35 to 49

## Methodology & Settings
- **Random Seed**: 42 (used across PyTorch, Scikit-Learn, and XGBoost)
- **Class Weights**: Calculated from the training data only. Weight of negative class (0/Normal) = 1.0, Weight of positive class (1/Anomaly) = count_normal / count_anomaly.

## Model Architectures & Hyperparameters

### Classical Models
1. **Logistic Regression**: `max_iter=1000`, balanced class weights
2. **Random Forest**: `n_estimators=100`, balanced class weights, `n_jobs=-1`
3. **XGBoost**: `n_estimators=100`, `scale_pos_weight` set to ratio of negative to positive samples, `eval_metric='logloss'`

### Graph Neural Networks (GNNs)
- **GCN**:
  - Hidden dimensions: 64
  - Optimizer: Adam (lr=0.01, weight_decay=5e-4)
  - Dropout: 0.5
- **GraphSAGE**:
  - Hidden dimensions: 64
  - Optimizer: Adam (lr=0.01, weight_decay=5e-4)
  - Dropout: 0.5
- **GAT**:
  - Hidden dimensions: 64
  - Heads: 4 (layer 1), 1 (layer 2)
  - Optimizer: Adam (lr=0.01, weight_decay=5e-4)
  - Dropout: 0.5

### Training Loop (GNNs)
- **Loss Function**: Weighted CrossEntropyLoss (to handle class imbalance)
- **Epochs**: 200
- **Early Stopping**: Patience of 20 epochs based on validation loss. Model checkpoints saved at best validation epoch.
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC. (PR-AUC is prioritized due to high class imbalance).
