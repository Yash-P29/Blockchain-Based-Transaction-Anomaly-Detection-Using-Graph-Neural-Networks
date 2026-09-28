import pandas as pd
import numpy as np

def compute_basic_node_features(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes basic topological features for nodes (in-degree, out-degree) using pandas.
    This avoids data leakage because it purely structural over the whole graph, 
    but strictly speaking, temporal features should be windowed. 
    For Phase 1, we compute global degrees.
    """
    # In-degree
    in_degrees = edges_df.groupby('txId2').size().reset_index(name='in_degree')
    in_degrees.rename(columns={'txId2': 'txId'}, inplace=True)
    
    # Out-degree
    out_degrees = edges_df.groupby('txId1').size().reset_index(name='out_degree')
    out_degrees.rename(columns={'txId1': 'txId'}, inplace=True)
    
    # Merge with nodes
    nodes_df = nodes_df.merge(in_degrees, on='txId', how='left')
    nodes_df = nodes_df.merge(out_degrees, on='txId', how='left')
    
    # Fill NaN with 0 for nodes with no incoming or outgoing edges
    nodes_df['in_degree'] = nodes_df['in_degree'].fillna(0)
    nodes_df['out_degree'] = nodes_df['out_degree'].fillna(0)
    
    # Total degree
    nodes_df['total_degree'] = nodes_df['in_degree'] + nodes_df['out_degree']
    
    return nodes_df
