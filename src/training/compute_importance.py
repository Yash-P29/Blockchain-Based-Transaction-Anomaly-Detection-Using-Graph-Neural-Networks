import json
import logging
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc

from src.utils import config
from src.data.loader import load_processed_data
from src.data.temporal_features import compute_temporal_structural_features, TEMPORAL_FEATURE_COLS
from src.graph.graph_builder import build_pyg_graph
from src.models.temporal_gnn import TemporalGNN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_pr_auc(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return float(auc(recall, precision))

def main():
    logger.info("Loading processed data and computing temporal features...")
    nodes_df, edges_df = load_processed_data()
    nodes_aug = compute_temporal_structural_features(nodes_df, edges_df)
    data = build_pyg_graph(nodes_aug, edges_df)

    feature_cols = [c for c in nodes_aug.columns if c.startswith('feat_') or c.endswith('_degree')]
    in_ch = len(feature_cols)
    temp_ch = len(TEMPORAL_FEATURE_COLS)

    x_temp = torch.tensor(nodes_aug[TEMPORAL_FEATURE_COLS].astype(float).values, dtype=torch.float)
    data.x_temp = x_temp

    model_path = config.MODELS_DIR / "temporal_gnn.pt"
    if not model_path.exists():
        logger.error(f"Model checkpoint not found at {model_path}")
        return

    logger.info("Loading trained TemporalGNN model...")
    model = TemporalGNN(node_feat_dim=in_ch, temporal_feat_dim=temp_ch)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    test_mask = data.test_mask
    y_true = data.y[test_mask].cpu().numpy()

    with torch.no_grad():
        out = model(data.x, data.edge_index, data.x_temp)
        probs = F.softmax(out, dim=1)[test_mask, 1].cpu().numpy()

    base_pr_auc = compute_pr_auc(y_true, probs)
    logger.info(f"Base Test PR-AUC: {base_pr_auc:.4f}")

    importance_results = []
    np.random.seed(42)

    for i, col_name in enumerate(TEMPORAL_FEATURE_COLS):
        drops = []
        for _ in range(5):  # 5 permutations
            permuted_x_temp = data.x_temp.clone()
            # permute column i for test nodes
            perm = torch.randperm(test_mask.sum())
            permuted_x_temp[test_mask, i] = permuted_x_temp[test_mask, i][perm]

            with torch.no_grad():
                out_p = model(data.x, data.edge_index, permuted_x_temp)
                probs_p = F.softmax(out_p, dim=1)[test_mask, 1].cpu().numpy()

            p_pr_auc = compute_pr_auc(y_true, probs_p)
            drop = base_pr_auc - p_pr_auc
            drops.append(drop)

        mean_drop = float(np.mean(drops))
        std_drop = float(np.std(drops))

        importance_results.append({
            "feature": col_name,
            "importance": max(0.0, mean_drop),
            "std": std_drop
        })
        logger.info(f"Feature: {col_name:25s} | Importance (PR-AUC Drop): {mean_drop:.4f} +/- {std_drop:.4f}")

    out_file = config.METRICS_DIR / "feature_importance.json"
    with open(out_file, "w") as f:
        json.dump(importance_results, f, indent=2)

    logger.info(f"Saved feature importance results to {out_file}")

if __name__ == "__main__":
    main()
