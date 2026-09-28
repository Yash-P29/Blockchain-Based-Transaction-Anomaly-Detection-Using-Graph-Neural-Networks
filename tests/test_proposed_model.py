"""
Essential tests for Phase 3 proposed model components.

Tests verify:
  - TemporalGNN model initialization
  - Forward pass (correct output shape)
  - Output dimensionality
  - Temporal feature generation
  - Ablation model forward pass
  - Prediction pipeline (end-to-end smoke test)
"""

import pytest
import numpy as np
import torch
import pandas as pd
from src.models.temporal_gnn import TemporalGNN, GraphSAGEEncoder, TemporalMLP
from src.data.temporal_features import compute_temporal_structural_features, TEMPORAL_FEATURE_COLS


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def sample_nodes_df():
    return pd.DataFrame({
        'txId':     [101, 102, 103, 104, 105],
        'timestep': [5,   10,  20,  31,  40],
        'label':    [0,   1,   0,   -1,  1],
        'feat_1':   [0.1, 0.2, 0.3, 0.4, 0.5],
        'feat_2':   [1.0, 2.0, 3.0, 4.0, 5.0],
    })

@pytest.fixture
def sample_edges_df():
    return pd.DataFrame({
        'txId1': [101, 102, 103, 104],
        'txId2': [102, 103, 104, 105],
    })

@pytest.fixture
def small_graph():
    """Returns (x, edge_index, x_temp) tensors for a 5-node graph."""
    N, node_feat, temp_feat = 5, 10, 9
    x = torch.randn(N, node_feat)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    x_temp = torch.randn(N, temp_feat)
    return x, edge_index, x_temp


# ------------------------------------------------------------------ #
# 1. Model initialization
# ------------------------------------------------------------------ #

def test_temporal_gnn_init():
    model = TemporalGNN(node_feat_dim=10, temporal_feat_dim=9)
    assert isinstance(model, TemporalGNN)
    assert isinstance(model.gnn, GraphSAGEEncoder)
    assert isinstance(model.temporal_mlp, TemporalMLP)
    assert model.fusion is not None


# ------------------------------------------------------------------ #
# 2. Forward pass shape
# ------------------------------------------------------------------ #

def test_temporal_gnn_forward_shape(small_graph):
    x, edge_index, x_temp = small_graph
    model = TemporalGNN(node_feat_dim=10, temporal_feat_dim=9)
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index, x_temp)
    assert out.shape == (5, 2), f"Expected (5,2) got {out.shape}"


# ------------------------------------------------------------------ #
# 3. Output dimension consistency
# ------------------------------------------------------------------ #

def test_temporal_gnn_different_sizes():
    for N in [10, 50, 100]:
        x = torch.randn(N, 15)
        edge_index = torch.zeros(2, 0, dtype=torch.long)  # no edges
        x_temp = torch.randn(N, 9)
        model = TemporalGNN(node_feat_dim=15, temporal_feat_dim=9)
        model.eval()
        with torch.no_grad():
            out = model(x, edge_index, x_temp)
        assert out.shape == (N, 2)


# ------------------------------------------------------------------ #
# 4. GraphSAGE encoder sub-module
# ------------------------------------------------------------------ #

def test_graphsage_encoder_forward():
    enc = GraphSAGEEncoder(in_channels=10, hidden_channels=16, embed_dim=8)
    x = torch.randn(5, 10)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    enc.eval()
    with torch.no_grad():
        out = enc(x, edge_index)
    assert out.shape == (5, 8)


# ------------------------------------------------------------------ #
# 5. Temporal MLP sub-module
# ------------------------------------------------------------------ #

def test_temporal_mlp_forward():
    mlp = TemporalMLP(in_dim=9, hidden_dim=16, out_dim=8)
    x = torch.randn(5, 9)
    with torch.no_grad():
        out = mlp(x)
    assert out.shape == (5, 8)


# ------------------------------------------------------------------ #
# 6. Temporal feature generation
# ------------------------------------------------------------------ #

def test_temporal_feature_generation(sample_nodes_df, sample_edges_df):
    result = compute_temporal_structural_features(sample_nodes_df, sample_edges_df)

    # All expected columns should be present
    for col in TEMPORAL_FEATURE_COLS:
        assert col in result.columns, f"Missing column: {col}"

    # No NaN in temporal feature columns
    for col in TEMPORAL_FEATURE_COLS:
        assert not result[col].isna().any(), f"NaN found in column: {col}"

    # timestep_norm should be in [0, 1]
    assert result['timestep_norm'].min() >= 0.0
    assert result['timestep_norm'].max() <= 1.0 + 1e-6


# ------------------------------------------------------------------ #
# 7. Temporal features count matches TEMPORAL_FEATURE_COLS
# ------------------------------------------------------------------ #

def test_temporal_feature_col_count():
    assert len(TEMPORAL_FEATURE_COLS) == 9  # update if list changes


# ------------------------------------------------------------------ #
# 8. Prediction pipeline: model produces valid probabilities
# ------------------------------------------------------------------ #

def test_model_output_probabilities(small_graph):
    x, edge_index, x_temp = small_graph
    model = TemporalGNN(node_feat_dim=10, temporal_feat_dim=9)
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index, x_temp)
        probs = torch.softmax(logits, dim=1)
    # Probs should sum to 1 per node
    assert torch.allclose(probs.sum(dim=1), torch.ones(5), atol=1e-5)
    # All probs should be in [0, 1]
    assert (probs >= 0).all() and (probs <= 1).all()


# ------------------------------------------------------------------ #
# 9. get_embeddings method
# ------------------------------------------------------------------ #

def test_get_embeddings(small_graph):
    x, edge_index, x_temp = small_graph
    model = TemporalGNN(node_feat_dim=10, temporal_feat_dim=9, gnn_embed_dim=8, temp_embed_dim=4)
    model.eval()
    with torch.no_grad():
        embs = model.get_embeddings(x, edge_index, x_temp)
    assert 'gnn_embedding' in embs
    assert 'temporal_embedding' in embs
    assert embs['gnn_embedding'].shape == (5, 8)
    assert embs['temporal_embedding'].shape == (5, 4)


# ------------------------------------------------------------------ #
# 10. Ablation B model (GraphSAGEWithTemp)
# ------------------------------------------------------------------ #

def test_ablation_b_model():
    from src.models.temporal_gnn import GraphSAGEWithTemp
    model = GraphSAGEWithTemp(node_feat_dim=10, temporal_feat_dim=9)
    x = torch.randn(5, 10)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    x_temp = torch.randn(5, 9)
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index, x_temp)
    assert out.shape == (5, 2)
