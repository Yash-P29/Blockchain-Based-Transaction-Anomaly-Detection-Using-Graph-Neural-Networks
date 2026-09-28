import pandas as pd
import os
from typing import Tuple, Optional
from src.utils import config

def load_raw_data() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Loads raw transaction features, edges, and classes from the dataset.
    Returns:
        Tuple of DataFrames: (features_df, edges_df, classes_df)
        Returns None for any file that cannot be found.
    """
    features_df = None
    edges_df = None
    classes_df = None

    if os.path.exists(config.FEATURES_FILE):
        features_df = pd.read_csv(config.FEATURES_FILE, header=None)
        col_names = ['txId', 'timestep'] + [f'feat_{i}' for i in range(1, features_df.shape[1] - 1)]
        features_df.columns = col_names
    
    if os.path.exists(config.EDGES_FILE):
        edges_df = pd.read_csv(config.EDGES_FILE)
    
    if os.path.exists(config.CLASSES_FILE):
        classes_df = pd.read_csv(config.CLASSES_FILE)
        
    return features_df, edges_df, classes_df

def load_processed_data() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Loads processed node features (with labels) and edges.
    Returns:
        Tuple of DataFrames: (nodes_df, edges_df)
    """
    nodes_df = None
    edges_df = None
    
    if os.path.exists(config.PROCESSED_FEATURES_FILE):
        nodes_df = pd.read_csv(config.PROCESSED_FEATURES_FILE)
        
    if os.path.exists(config.PROCESSED_EDGES_FILE):
        edges_df = pd.read_csv(config.PROCESSED_EDGES_FILE)
        
    return nodes_df, edges_df
