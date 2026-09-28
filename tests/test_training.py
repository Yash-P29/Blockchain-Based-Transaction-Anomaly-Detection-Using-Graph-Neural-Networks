import pytest
import numpy as np
import torch
import pandas as pd
from src.utils import config
from src.graph.graph_builder import build_pyg_graph
from src.models.gcn import GCN
from src.models.graphsage import GraphSAGE
from src.models.gat import GAT
from src.training.evaluation import evaluate_model

def test_chronological_split_and_masks():
    # Mock data
    nodes_df = pd.DataFrame({
        'txId': [1, 2, 3, 4, 5],
        'timestep': [10, 20, 31, 35, 40],
        'label': [0, 1, 0, -1, 1], # 4 has unknown label
        'feat_1': [0.1, 0.2, 0.3, 0.4, 0.5]
    })
    edges_df = pd.DataFrame({
        'txId1': [1, 2, 3],
        'txId2': [2, 3, 4]
    })
    
    data = build_pyg_graph(nodes_df, edges_df)
    
    train_mask = data.train_mask.numpy()
    val_mask = data.val_mask.numpy()
    test_mask = data.test_mask.numpy()
    
    # Train mask: <=30 and label != -1 -> [True, True, False, False, False]
    assert np.array_equal(train_mask, [True, True, False, False, False])
    
    # Val mask: >30 and <=34 and label != -1 -> [False, False, True, False, False]
    assert np.array_equal(val_mask, [False, False, True, False, False])
    
    # Test mask: >34 and label != -1 -> [False, False, False, False, True]
    assert np.array_equal(test_mask, [False, False, False, False, True])

def test_model_forward_passes():
    in_channels = 10
    hidden_channels = 16
    out_channels = 2
    
    x = torch.randn(5, in_channels)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    
    gcn = GCN(in_channels, hidden_channels, out_channels)
    out_gcn = gcn(x, edge_index)
    assert out_gcn.shape == (5, 2)
    
    sage = GraphSAGE(in_channels, hidden_channels, out_channels)
    out_sage = sage(x, edge_index)
    assert out_sage.shape == (5, 2)
    
    gat = GAT(in_channels, hidden_channels, out_channels)
    out_gat = gat(x, edge_index)
    assert out_gat.shape == (5, 2)

def test_evaluation_metrics():
    y_true = np.array([0, 1, 0, 1, 1])
    y_prob = np.array([0.1, 0.9, 0.8, 0.6, 0.4])
    y_pred = np.array([0, 1, 1, 1, 0])
    
    metrics = evaluate_model(y_true, y_prob, y_pred, "TestModel")
    
    assert metrics['Model'] == "TestModel"
    assert 'Accuracy' in metrics
    assert 'Precision' in metrics
    assert 'Recall' in metrics
    assert 'F1' in metrics
    assert 'ROC-AUC' in metrics
    assert 'PR-AUC' in metrics
