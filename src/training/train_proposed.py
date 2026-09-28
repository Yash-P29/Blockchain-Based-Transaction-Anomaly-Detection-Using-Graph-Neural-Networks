"""
Training pipeline for the Phase 3 proposed Temporal-Structural GNN model.

Also trains two ablation variants:
  A. GraphSAGE baseline (Phase 2 model, re-evaluated for consistency)
  B. GraphSAGE + temporal/structural features only (no fusion with orig features)
  C. Full proposed model (GraphSAGE + temporal MLP + feature fusion)

Results are saved to:
  results/metrics/proposed_results.csv
  results/models/temporal_gnn.pt
  results/models/temporal_gnn_ablation_B.pt
"""

import time
import logging
import torch
import torch.nn.functional as F
import numpy as np

from src.utils import config
from src.data.loader import load_processed_data
from src.data.temporal_features import compute_temporal_structural_features, TEMPORAL_FEATURE_COLS
from src.graph.graph_builder import build_pyg_graph
from src.models.graphsage import GraphSAGE
from src.models.temporal_gnn import TemporalGNN, GraphSAGEWithTemp
from src.training.evaluation import evaluate_model, save_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Helper: build augmented PyG data object
# ------------------------------------------------------------------ #

def build_augmented_data(nodes_df, edges_df):
    """
    Build a PyG Data object that also carries temporal feature tensor.
    Temporal features are appended as data.x_temp.
    """
    from torch_geometric.data import Data

    # Add temporal features to the DataFrame
    nodes_aug = compute_temporal_structural_features(nodes_df, edges_df)

    # Build the standard PyG object (uses feat_* and degree cols for x)
    data = build_pyg_graph(nodes_aug, edges_df)

    # Build temporal feature tensor aligned to the same node ordering
    unique_nodes = nodes_aug['txId'].unique()
    # nodes are ordered as in the PyG graph (same order as build_pyg_graph uses)
    temp_arr = nodes_aug[TEMPORAL_FEATURE_COLS].astype(float).values
    x_temp = torch.tensor(temp_arr, dtype=torch.float)

    data.x_temp = x_temp
    return data, nodes_aug


# ------------------------------------------------------------------ #
# Training loop helpers
# ------------------------------------------------------------------ #

def compute_class_weights(data):
    num_normal = (data.y[data.train_mask] == 0).sum().float()
    num_anomaly = (data.y[data.train_mask] == 1).sum().float()
    pos_weight = num_normal / num_anomaly if num_anomaly > 0 else torch.tensor(1.0)
    return torch.tensor([1.0, pos_weight.item()])


def train_step(model, data, optimizer, criterion, use_temp=True):
    model.train()
    optimizer.zero_grad()
    if use_temp:
        out = model(data.x, data.edge_index, data.x_temp)
    else:
        out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def val_step(model, data, criterion, use_temp=True):
    model.eval()
    if use_temp:
        out = model(data.x, data.edge_index, data.x_temp)
    else:
        out = model(data.x, data.edge_index)
    return criterion(out[data.val_mask], data.y[data.val_mask]).item()


def run_training_loop(model, data, model_name, epochs=200, lr=0.01,
                      weight_decay=5e-4, patience=20, use_temp=True):
    criterion = torch.nn.CrossEntropyLoss(weight=compute_class_weights(data))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []
    save_path = config.MODELS_DIR / f"{model_name.lower().replace(' ', '_').replace('+', 'plus')}.pt"

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        loss = train_step(model, data, optimizer, criterion, use_temp)
        vl = val_step(model, data, criterion, use_temp)
        train_losses.append(loss)
        val_losses.append(vl)

        if vl < best_val_loss:
            best_val_loss = vl
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"[{model_name}] Early stopping at epoch {epoch}")
            break

    train_time = time.time() - t0

    # Plot loss curve
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.legend()
        plt.title(f"{model_name} Training Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(config.FIGURES_DIR / f"{model_name.lower().replace(' ', '_').replace('+', 'plus')}_loss.png")
        plt.close()
    except Exception as plot_err:
        logger.warning(f"Could not save loss plot: {plot_err}")

    # Load best checkpoint and evaluate
    model.load_state_dict(torch.load(save_path))
    model.eval()

    t_inf_start = time.time()
    with torch.no_grad():
        if use_temp:
            out = model(data.x, data.edge_index, data.x_temp)
        else:
            out = model(data.x, data.edge_index)
    inf_time = time.time() - t_inf_start

    probs = F.softmax(out[data.test_mask], dim=1)[:, 1].numpy()
    preds = out[data.test_mask].argmax(dim=1).numpy()
    y_test = data.y[data.test_mask].numpy()

    metrics = evaluate_model(y_test, probs, preds, model_name)
    metrics['TrainTime_s'] = round(train_time, 2)
    metrics['InfTime_ms'] = round(inf_time * 1000, 2)
    logger.info(
        f"[{model_name}] PR-AUC={metrics['PR-AUC']:.4f} "
        f"F1={metrics['F1']:.4f} "
        f"ROC-AUC={metrics['ROC-AUC']:.4f} "
        f"Train={train_time:.1f}s"
    )
    return metrics


# ------------------------------------------------------------------ #
# Ablation A: GraphSAGE baseline (re-evaluated on same data)
# ------------------------------------------------------------------ #

def run_ablation_A(data, in_channels):
    """GraphSAGE baseline — identical to Phase 2 model."""
    logger.info("=== Ablation A: GraphSAGE Baseline ===")
    model = GraphSAGE(in_channels, 64, 2)
    return run_training_loop(model, data, "GraphSAGE_Baseline", use_temp=False)



def run_ablation_B(data, in_channels, temporal_feat_dim):
    """GraphSAGE + temporal features only."""
    logger.info("=== Ablation B: GraphSAGE + Temporal Features ===")
    model = GraphSAGEWithTemp(in_channels, temporal_feat_dim)
    return run_training_loop(model, data, "GraphSAGE_plus_Temporal", use_temp=True)


# ------------------------------------------------------------------ #
# Full proposed model (Ablation C)
# ------------------------------------------------------------------ #

def run_proposed_model(data, in_channels, temporal_feat_dim):
    """Full proposed Temporal-Structural GNN."""
    logger.info("=== Proposed Model: Temporal-Structural GNN ===")
    model = TemporalGNN(
        node_feat_dim=in_channels,
        temporal_feat_dim=temporal_feat_dim,
        gnn_hidden=64,
        gnn_embed_dim=64,
        temp_hidden=32,
        temp_embed_dim=32,
        fusion_hidden=64,
        dropout=0.5,
    )
    return run_training_loop(model, data, "Temporal_GNN", use_temp=True)


# ------------------------------------------------------------------ #
# Main entry point
# ------------------------------------------------------------------ #

def run_phase3_training():
    logger.info("Loading processed data...")
    nodes_df, edges_df = load_processed_data()

    if nodes_df is None or edges_df is None:
        logger.error("Processed data not found. Run: python -m src.data.preprocessing")
        return []

    logger.info("Building augmented PyG graph with temporal features...")
    data, nodes_aug = build_augmented_data(nodes_df, edges_df)

    in_channels = data.x.size(1)
    temporal_feat_dim = data.x_temp.size(1)
    logger.info(f"Node features: {in_channels}, Temporal features: {temporal_feat_dim}")

    results = []

    # Ablation A
    try:
        m_a = run_ablation_A(data, in_channels)
        results.append(m_a)
    except Exception as e:
        logger.error(f"Ablation A failed: {e}")

    # Ablation B
    try:
        m_b = run_ablation_B(data, in_channels, temporal_feat_dim)
        results.append(m_b)
    except Exception as e:
        logger.error(f"Ablation B failed: {e}")

    # Full model (C)
    try:
        m_c = run_proposed_model(data, in_channels, temporal_feat_dim)
        results.append(m_c)
    except Exception as e:
        logger.error(f"Proposed model failed: {e}")

    if results:
        save_metrics(results, "proposed_results.csv")
        logger.info("Phase 3 results saved to results/metrics/proposed_results.csv")

    return results


if __name__ == "__main__":
    run_phase3_training()
