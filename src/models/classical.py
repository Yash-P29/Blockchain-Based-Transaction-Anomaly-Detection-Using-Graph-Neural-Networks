from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
import logging

logger = logging.getLogger(__name__)

def get_logistic_regression(class_weight=None, random_state=42):
    return LogisticRegression(class_weight=class_weight, random_state=random_state, max_iter=1000)

def get_random_forest(class_weight=None, random_state=42):
    return RandomForestClassifier(n_estimators=100, class_weight=class_weight, random_state=random_state, n_jobs=-1)

def get_xgboost(scale_pos_weight=1.0, random_state=42):
    if XGBClassifier is None:
        logger.warning("XGBoost is not installed.")
        return None
    return XGBClassifier(n_estimators=100, scale_pos_weight=scale_pos_weight, random_state=random_state, n_jobs=-1, eval_metric='logloss')
