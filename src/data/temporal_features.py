"""
Temporal and structural feature engineering for Phase 3.

Computes features derived exclusively from available dataset attributes:
  - transaction timestep
  - graph topology (degrees computed from edgelist)

No synthetic or invented features are introduced.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_temporal_structural_features(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes temporal and structural features for every node.

    Features derived:
      - timestep              : raw time step (already present)
      - timestep_norm         : timestep normalised to [0, 1]
      - in_degree             : number of incoming edges (already computed in preprocessing,
                                but recomputed here if missing)
      - out_degree            : number of outgoing edges
      - total_degree          : in_degree + out_degree
      - degree_ratio          : in_degree / (total_degree + 1e-6)   — asymmetry indicator
      - local_neighbor_count  : distinct neighbours (union of sources and targets)
      - timestep_degree_interact : timestep_norm * log1p(total_degree)

    Temporal activity statistics (computed per timestep from training data,
    then applied to val/test without leakage):
      - ts_mean_degree        : mean total degree of all nodes at the same timestep
      - ts_illicit_rate       : fraction of *known-labeled* training nodes at same timestep
                                that are illicit.  Encoded only from timesteps seen in training.
                                For unseen timesteps it is filled with the global training rate.

    NOTE: ts_illicit_rate encodes label information only from training nodes; it is
          therefore computed exclusively from train-set timesteps (steps 1-30) and
          then looked up (not refit) for val/test timesteps — no leakage.
    """
    logger.info("Computing temporal and structural features...")
    df = nodes_df.copy()

    # ------------------------------------------------------------------ #
    # 1. Degree features (recompute to guarantee correctness)
    # ------------------------------------------------------------------ #
    in_deg = edges_df.groupby('txId2').size().rename('in_degree').reset_index().rename(columns={'txId2': 'txId'})
    out_deg = edges_df.groupby('txId1').size().rename('out_degree').reset_index().rename(columns={'txId1': 'txId'})

    # Drop existing degree cols to avoid merge duplication
    for c in ['in_degree', 'out_degree', 'total_degree']:
        if c in df.columns:
            df = df.drop(columns=[c])

    df = df.merge(in_deg, on='txId', how='left')
    df = df.merge(out_deg, on='txId', how='left')
    df['in_degree'] = df['in_degree'].fillna(0).astype(float)
    df['out_degree'] = df['out_degree'].fillna(0).astype(float)
    df['total_degree'] = df['in_degree'] + df['out_degree']

    # ------------------------------------------------------------------ #
    # 2. Derived structural features
    # ------------------------------------------------------------------ #
    df['degree_ratio'] = df['in_degree'] / (df['total_degree'] + 1e-6)

    # Local neighbor count: number of distinct neighbors (undirected view)
    srcs = edges_df[['txId1', 'txId2']].rename(columns={'txId1': 'txId', 'txId2': 'neighbor'})
    dsts = edges_df[['txId1', 'txId2']].rename(columns={'txId2': 'txId', 'txId1': 'neighbor'})
    all_nb = pd.concat([srcs, dsts], ignore_index=True)
    local_nb = all_nb.groupby('txId')['neighbor'].nunique().rename('local_neighbor_count').reset_index()
    df = df.merge(local_nb, on='txId', how='left')
    df['local_neighbor_count'] = df['local_neighbor_count'].fillna(0).astype(float)

    # ------------------------------------------------------------------ #
    # 3. Timestep normalisation
    # ------------------------------------------------------------------ #
    ts_min = df['timestep'].min()
    ts_max = df['timestep'].max()
    df['timestep_norm'] = (df['timestep'] - ts_min) / (ts_max - ts_min + 1e-6)

    # ------------------------------------------------------------------ #
    # 4. Interaction feature
    # ------------------------------------------------------------------ #
    df['timestep_degree_interact'] = df['timestep_norm'] * np.log1p(df['total_degree'])

    # ------------------------------------------------------------------ #
    # 5. Per-timestep mean degree  (computed over ALL nodes — no label info)
    # ------------------------------------------------------------------ #
    ts_mean_deg = df.groupby('timestep')['total_degree'].mean().rename('ts_mean_degree')
    df = df.join(ts_mean_deg, on='timestep')

    # ------------------------------------------------------------------ #
    # 6. Illicit rate per timestep — training-only encoding
    #    (only labeled nodes in steps 1-30 contribute)
    # ------------------------------------------------------------------ #
    from src.utils.config import TRAIN_END_STEP
    train_labeled = df[(df['timestep'] <= TRAIN_END_STEP) & (df['label'] != -1)].copy()
    ts_illicit_rate = (
        train_labeled.groupby('timestep')['label'].mean()
        .rename('ts_illicit_rate')
    )
    global_rate = train_labeled['label'].mean() if len(train_labeled) > 0 else 0.1
    df['ts_illicit_rate'] = df['timestep'].map(ts_illicit_rate).fillna(global_rate)

    logger.info(
        f"Temporal features computed. New columns: "
        f"timestep_norm, degree_ratio, local_neighbor_count, "
        f"timestep_degree_interact, ts_mean_degree, ts_illicit_rate"
    )
    return df


TEMPORAL_FEATURE_COLS = [
    'timestep_norm',
    'in_degree',
    'out_degree',
    'total_degree',
    'degree_ratio',
    'local_neighbor_count',
    'timestep_degree_interact',
    'ts_mean_degree',
    'ts_illicit_rate',
]
