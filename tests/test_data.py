import pytest
import pandas as pd
from src.data.preprocessing import clean_and_merge_data
from src.data.feature_engineering import compute_basic_node_features
from src.utils import config

@pytest.fixture
def sample_data():
    features = pd.DataFrame({
        'txId': [1, 2, 3],
        'timestep': [1, 1, 2],
        'feat_1': [0.1, 0.2, 0.3],
        'feat_2': [0.5, 0.6, 0.7]
    })
    
    classes = pd.DataFrame({
        'txId': [1, 2, 3],
        'class': ['1', '2', 'unknown']
    })
    
    edges = pd.DataFrame({
        'txId1': [1, 1],
        'txId2': [2, 3]
    })
    
    return features, edges, classes

def test_clean_and_merge_data(sample_data):
    features, edges, classes = sample_data
    nodes_df = clean_and_merge_data(features, edges, classes)
    
    assert len(nodes_df) == 3
    assert 'label' in nodes_df.columns
    assert nodes_df[nodes_df['txId'] == 1]['label'].iloc[0] == 1  # 1 -> 1
    assert nodes_df[nodes_df['txId'] == 2]['label'].iloc[0] == 0  # 2 -> 0
    assert nodes_df[nodes_df['txId'] == 3]['label'].iloc[0] == -1 # unknown -> -1

def test_compute_basic_node_features(sample_data):
    features, edges, classes = sample_data
    nodes_df = clean_and_merge_data(features, edges, classes)
    nodes_df = compute_basic_node_features(nodes_df, edges)
    
    # Node 1: out_degree=2, in_degree=0
    assert nodes_df[nodes_df['txId'] == 1]['out_degree'].iloc[0] == 2
    assert nodes_df[nodes_df['txId'] == 1]['in_degree'].iloc[0] == 0
    
    # Node 2: out_degree=0, in_degree=1
    assert nodes_df[nodes_df['txId'] == 2]['out_degree'].iloc[0] == 0
    assert nodes_df[nodes_df['txId'] == 2]['in_degree'].iloc[0] == 1
