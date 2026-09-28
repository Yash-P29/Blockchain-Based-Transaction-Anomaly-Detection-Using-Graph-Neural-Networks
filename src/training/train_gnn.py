import torch
import torch.nn.functional as F
import logging
from src.utils import config
from src.data.loader import load_processed_data
from src.graph.graph_builder import build_pyg_graph
from src.models.gcn import GCN
from src.models.graphsage import GraphSAGE
from src.models.gat import GAT
from src.training.evaluation import evaluate_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_gnn(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def validate_gnn(model, data, criterion):
    model.eval()
    out = model(data.x, data.edge_index)
    val_loss = criterion(out[data.val_mask], data.y[data.val_mask]).item()
    return val_loss

def run_gnn_model(model, model_name, data, epochs=200, lr=0.01, weight_decay=5e-4):
    logger.info(f"Training {model_name}...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    num_normal = (data.y[data.train_mask] == 0).sum().float()
    num_anomaly = (data.y[data.train_mask] == 1).sum().float()
    pos_weight = num_normal / num_anomaly if num_anomaly > 0 else torch.tensor(1.0)
    
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight]))
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience = 20
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    for epoch in range(1, epochs + 1):
        loss = train_gnn(model, data, optimizer, criterion)
        val_loss = validate_gnn(model, data, criterion)
        
        train_losses.append(loss)
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), config.MODELS_DIR / f"{model_name.lower()}.pt")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
            
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title(f"{model_name} Training Curve")
    plt.savefig(config.FIGURES_DIR / f"{model_name.lower()}_loss.png")
    plt.close()
    
    model.load_state_dict(torch.load(config.MODELS_DIR / f"{model_name.lower()}.pt"))
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = F.softmax(out[data.test_mask], dim=1)[:, 1].numpy()
        preds = out[data.test_mask].argmax(dim=1).numpy()
        y_test = data.y[data.test_mask].numpy()
        
    metrics = evaluate_model(y_test, probs, preds, model_name)
    return metrics

def run_gnn_training():
    nodes_df, edges_df = load_processed_data()
    data = build_pyg_graph(nodes_df, edges_df)
    
    in_channels = data.x.size(1)
    hidden_channels = 64
    out_channels = 2
    
    gcn = GCN(in_channels, hidden_channels, out_channels)
    gcn_metrics = run_gnn_model(gcn, "GCN", data)
    
    sage = GraphSAGE(in_channels, hidden_channels, out_channels)
    sage_metrics = run_gnn_model(sage, "GraphSAGE", data)
    
    gat = GAT(in_channels, hidden_channels, out_channels)
    gat_metrics = run_gnn_model(gat, "GAT", data)
    
    return gcn_metrics, sage_metrics, gat_metrics

if __name__ == "__main__":
    run_gnn_training()
