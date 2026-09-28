import pandas as pd
import logging
from src.utils import config
from src.data.loader import load_raw_data
from src.data.feature_engineering import compute_basic_node_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_and_merge_data(features_df: pd.DataFrame, edges_df: pd.DataFrame, classes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges features and classes, applies label mapping, and handles missing values.
    """
    logger.info("Merging features and classes...")
    
    # Merge features with classes on txId
    nodes_df = features_df.merge(classes_df, on='txId', how='inner')
    
    # Map classes: 1 -> 1 (Illicit/Anomaly), 2 -> 0 (Licit/Normal), 'unknown' -> -1
    nodes_df['label'] = nodes_df['class'].map(config.CLASS_MAPPING)
    
    # Drop the original class column
    nodes_df = nodes_df.drop(columns=['class'])
    
    # Handle missing values (if any, although Elliptic is usually clean)
    nodes_df = nodes_df.fillna(0)
    
    return nodes_df

def run_preprocessing() -> None:
    """
    Main preprocessing pipeline.
    """
    logger.info("Loading raw data...")
    features_df, edges_df, classes_df = load_raw_data()
    
    if features_df is None or edges_df is None or classes_df is None:
        logger.error("Raw data files are missing. Please check DATASET_SETUP.md instructions.")
        return
        
    nodes_df = clean_and_merge_data(features_df, edges_df, classes_df)
    nodes_df = compute_basic_node_features(nodes_df, edges_df)
    
    # Save processed data
    logger.info("Saving processed data...")
    # Ensure processed dir exists
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    nodes_df.to_csv(config.PROCESSED_FEATURES_FILE, index=False)
    edges_df.to_csv(config.PROCESSED_EDGES_FILE, index=False)
    
    logger.info("Preprocessing complete.")

if __name__ == "__main__":
    run_preprocessing()
