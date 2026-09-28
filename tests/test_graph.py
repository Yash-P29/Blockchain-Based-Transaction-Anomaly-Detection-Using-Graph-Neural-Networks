import pytest
import pandas as pd
from src.graph.graph_builder import build_networkx_graph, build_pyg_graph

@pytest.fixture
def processed_data():
    nodes_df = pd.DataFrame({
        'txId': [1, 2, 3],
        'timestep': [1, 1, 2],
        'label': [1, 0, -1],
        'feat_1': [0.1, 0.2, 0.3],
        'in_degree': [0, 1, 1],
        'out_degree': [2, 0, 0]
    })
    
    edges_df = pd.DataFrame({
        'txId1': [1, 1],
        'txId2': [2, 3]
    })
    
    return nodes_df, edges_df

def test_build_networkx_graph(processed_data):
    nodes_df, edges_df = processed_data
    G = build_networkx_graph(nodes_df, edges_df)
    
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 2
    assert G.has_edge(1, 2)
    assert G.nodes[1]['label'] == 1

def test_build_pyg_graph(processed_data):
    nodes_df, edges_df = processed_data
    
    try:
        import torch_geometric
    except ImportError:
        pytest.skip("PyTorch Geometric not installed")
        
    data = build_pyg_graph(nodes_df, edges_df)
    
    assert data.num_nodes == 3
    assert data.num_edges == 2
    assert list(data.edge_index.shape) == [2, 2]
    # x should contain feat_1, in_degree, out_degree
    assert list(data.x.shape) == [3, 3] 
