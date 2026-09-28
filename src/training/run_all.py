import logging
from src.training.train_classical import run_classical_training
from src.training.train_gnn import run_gnn_training
from src.training.evaluation import save_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Classical Model Training...")
    lr_metrics, rf_metrics, xgb_metrics = run_classical_training()
    
    logger.info("Starting GNN Model Training...")
    gcn_metrics, sage_metrics, gat_metrics = run_gnn_training()
    
    all_metrics = [lr_metrics, rf_metrics, xgb_metrics, gcn_metrics, sage_metrics, gat_metrics]
    all_metrics = [m for m in all_metrics if m is not None]
    
    logger.info("Saving results...")
    save_metrics(all_metrics)
    logger.info("Training pipeline complete.")

if __name__ == "__main__":
    main()
