import pandas as pd
import numpy as np
import logging
import joblib
from sklearn.preprocessing import StandardScaler
from src.utils import config
from src.data.loader import load_processed_data
from src.graph.graph_builder import build_pyg_graph
from src.models.classical import get_logistic_regression, get_random_forest, get_xgboost
from src.training.evaluation import evaluate_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_classical_training():
    logger.info("Loading processed data...")
    nodes_df, edges_df = load_processed_data()
    data = build_pyg_graph(nodes_df, edges_df)
    
    X = data.x.numpy()
    y = data.y.numpy()
    
    train_mask = data.train_mask.numpy()
    test_mask = data.test_mask.numpy()
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    num_normal = (y_train == 0).sum()
    num_anomaly = (y_train == 1).sum()
    scale_pos_weight = num_normal / num_anomaly if num_anomaly > 0 else 1.0
    class_weight = {0: 1.0, 1: scale_pos_weight}
    
    logger.info("Training Logistic Regression...")
    lr = get_logistic_regression(class_weight=class_weight)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_prob = lr.predict_proba(X_test)[:, 1]
    lr_metrics = evaluate_model(y_test, lr_prob, lr_pred, "Logistic Regression")
    joblib.dump(lr, config.MODELS_DIR / "logistic_regression.joblib")
    
    logger.info("Training Random Forest...")
    rf = get_random_forest(class_weight=class_weight)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    rf_metrics = evaluate_model(y_test, rf_prob, rf_pred, "Random Forest")
    joblib.dump(rf, config.MODELS_DIR / "random_forest.joblib")
    
    logger.info("Training XGBoost...")
    xgb = get_xgboost(scale_pos_weight=scale_pos_weight)
    if xgb is not None:
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_test)
        xgb_prob = xgb.predict_proba(X_test)[:, 1]
        xgb_metrics = evaluate_model(y_test, xgb_prob, xgb_pred, "XGBoost")
        joblib.dump(xgb, config.MODELS_DIR / "xgboost.joblib")
    else:
        xgb_metrics = None
        
    return lr_metrics, rf_metrics, xgb_metrics

if __name__ == "__main__":
    run_classical_training()
