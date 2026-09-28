import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Elliptic Dataset File Names
FEATURES_FILE = RAW_DATA_DIR / "elliptic_txs_features.csv"
EDGES_FILE = RAW_DATA_DIR / "elliptic_txs_edgelist.csv"
CLASSES_FILE = RAW_DATA_DIR / "elliptic_txs_classes.csv"

# Preprocessed File Names
PROCESSED_FEATURES_FILE = PROCESSED_DATA_DIR / "processed_features.csv"
PROCESSED_EDGES_FILE = PROCESSED_DATA_DIR / "processed_edges.csv"
PROCESSED_CLASSES_FILE = PROCESSED_DATA_DIR / "processed_classes.csv"

# Result Directories
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = RESULTS_DIR / "models"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"

# Create directories if they don't exist
for d in [RESULTS_DIR, MODELS_DIR, METRICS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Random Seed for Reproducibility
RANDOM_SEED = 42

# Label mapping
# In Elliptic dataset: '1' is illicit, '2' is licit, 'unknown' is unlabelled.
# We will map them for ML purposes: 1 -> 1 (Anomaly), 2 -> 0 (Normal), 'unknown' -> -1
CLASS_MAPPING = {
    '1': 1,
    '2': 0,
    'unknown': -1
}

# Chronological Data Split (Time steps 1-49)
TRAIN_END_STEP = 30
VAL_END_STEP = 34
