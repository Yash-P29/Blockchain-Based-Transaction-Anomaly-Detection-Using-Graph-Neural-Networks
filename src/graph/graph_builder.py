import networkx as nx
import numpy as np
import pandas as pd
import torch
try:
    from torch_geometric.data import Data
except ImportError:
    Data = None
import logging
from src.utils import config

logger = logging.getLogger(__name__)

def build_networkx_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.DiGraph:
    """
    Constructs a NetworkX Directed Graph from node and edge DataFrames.
    Useful for exploration, centrality measures, and visualization.
    """
    logger.info("Building NetworkX graph...")
    G = nx.DiGraph()
    
    # Add nodes with attributes
    # We only take a subset of attributes to keep memory usage reasonable,
    # primarily label and timestep.
    node_attr = nodes_df[['txId', 'timestep', 'label']].set_index('txId').to_dict('index')
    G.add_nodes_from(node_attr.items())
    
    # Add edges
    edge_list = list(zip(edges_df['txId1'], edges_df['txId2']))
    G.add_edges_from(edge_list)
    
    logger.info(f"NetworkX Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

def build_pyg_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> 'Data':
    """
    Constructs a PyTorch Geometric Data object for GNN training.
    """
    if Data is None:
        raise ImportError("torch_geometric is not installed. Please install it to build PyG graphs.")
        
    logger.info("Building PyTorch Geometric graph...")
    
    # Map txIds to contiguous integers (0 to N-1) for PyG
    unique_nodes = nodes_df['txId'].unique()
    tx_to_idx = {tx: idx for idx, tx in enumerate(unique_nodes)}
    
    # Node features (excluding txId, timestep, label)
    feature_cols = [col for col in nodes_df.columns if col.startswith('feat_') or col.endswith('_degree')]
    x = torch.tensor(nodes_df[feature_cols].astype(float).values, dtype=torch.float)
    
    # Labels
    y = torch.tensor(nodes_df['label'].astype(int).values, dtype=torch.long)
    
    # Timesteps for temporal splits
    time_steps = torch.tensor(nodes_df['timestep'].astype(int).values, dtype=torch.long)
    
    # Edge index
    # We must filter edges to only include nodes that exist in nodes_df
    valid_edges = edges_df[edges_df['txId1'].isin(tx_to_idx) & edges_df['txId2'].isin(tx_to_idx)]
    
    source_nodes = valid_edges['txId1'].map(tx_to_idx).values
    target_nodes = valid_edges['txId2'].map(tx_to_idx).values
    
    edge_array = np.asarray([source_nodes, target_nodes], dtype=np.int64)
    edge_index = torch.from_numpy(edge_array)
    
    # Create masks based on chronological split and known labels
    is_known = (y != -1)
    train_mask = is_known & (time_steps <= config.TRAIN_END_STEP)
    val_mask = is_known & (time_steps > config.TRAIN_END_STEP) & (time_steps <= config.VAL_END_STEP)
    test_mask = is_known & (time_steps > config.VAL_END_STEP)
    
    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index, y=y, time_steps=time_steps,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    
    logger.info(f"PyG Graph created: {data}")
    return data
