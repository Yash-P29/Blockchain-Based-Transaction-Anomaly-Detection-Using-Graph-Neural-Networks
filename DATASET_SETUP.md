# Dataset Setup Instructions

For this research project, we will use the **Elliptic Bitcoin Transaction Dataset** (or an equivalent public blockchain transaction dataset for anomaly detection).

## 1. Downloading the Dataset
The Elliptic dataset is publicly available on Kaggle.

1. Go to the Kaggle dataset page: [Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
2. Download the archive.
3. Extract the contents of the archive. You should see three primary CSV files:
   - `elliptic_txs_features.csv`
   - `elliptic_txs_edgelist.csv`
   - `elliptic_txs_classes.csv`

## 2. Placing the Files
Place the extracted CSV files into the `data/raw/` directory of this project so the pipeline can find them.

Your directory structure should look like this:

```
blockchain-anomaly-gnn/
└── data/
    ├── raw/
    │   ├── elliptic_txs_features.csv
    │   ├── elliptic_txs_edgelist.csv
    │   └── elliptic_txs_classes.csv
    └── processed/
```

## 3. Dataset Schema
- **Features (`elliptic_txs_features.csv`)**: Contains 166 features for each transaction. The first column is the transaction ID, the second column is the time step (1 to 49). The remaining 164 columns are anonymized features (local and aggregated).
- **Edges (`elliptic_txs_edgelist.csv`)**: Contains the source (`txId1`) and destination (`txId2`) transactions, representing the flow of Bitcoin.
- **Classes (`elliptic_txs_classes.csv`)**: Contains the labels for the transactions. `1` means illicit, `2` means licit, and `unknown` means unlabelled.

## 4. Running the Pipeline
Once the dataset is placed in the `data/raw/` folder, you can run the preprocessing pipeline using:

```bash
python -m src.data.preprocessing
```

Alternatively, you can explore the data using the Jupyter notebooks in `notebooks/`.
