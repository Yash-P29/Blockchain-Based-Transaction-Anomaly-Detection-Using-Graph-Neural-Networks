"""
Temporal-Structural Transaction Anomaly Detection Model (Phase 3).

Architecture:
  1. GraphSAGE encoder  : two SAGEConv layers produce node embeddings from
                          the transaction graph.
  2. Temporal MLP       : a two-layer MLP encodes handcrafted temporal and
                          structural features (timestep, degrees, etc.).
  3. Feature fusion     : GNN embedding + temporal embedding + original
                          transaction features concatenated, then projected
                          through a classifier head.

Output: logits over [licit, illicit].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGEEncoder(nn.Module):
    """Two-layer GraphSAGE producing node-level embeddings (no classifier)."""

    def __init__(self, in_channels: int, hidden_channels: int, embed_dim: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, embed_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x  # shape: (N, embed_dim)


class TemporalMLP(nn.Module):
    """Lightweight MLP to encode temporal/structural features."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalGNN(nn.Module):
    """
    Temporal-Structural GNN model for transaction anomaly detection.

    Parameters
    ----------
    node_feat_dim   : dimensionality of original node features (feat_*)
    temporal_feat_dim : dimensionality of temporal/structural features
    gnn_hidden      : hidden size for GraphSAGE layers
    gnn_embed_dim   : output embedding size of the GNN encoder
    temp_hidden     : hidden size of the temporal MLP
    temp_embed_dim  : output embedding size of the temporal MLP
    fusion_hidden   : hidden size of the fusion MLP
    dropout         : dropout probability
    """

    def __init__(
        self,
        node_feat_dim: int,
        temporal_feat_dim: int,
        gnn_hidden: int = 64,
        gnn_embed_dim: int = 64,
        temp_hidden: int = 32,
        temp_embed_dim: int = 32,
        fusion_hidden: int = 64,
        dropout: float = 0.5,
    ):
        super().__init__()

        # 1. GNN encoder on raw node features
        self.gnn = GraphSAGEEncoder(node_feat_dim, gnn_hidden, gnn_embed_dim, dropout)

        # 2. Temporal MLP
        self.temporal_mlp = TemporalMLP(temporal_feat_dim, temp_hidden, temp_embed_dim, dropout=0.3)

        # 3. Fusion: gnn_embed + temp_embed + node_feat (skip connection)
        fusion_in = gnn_embed_dim + temp_embed_dim + node_feat_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 2),  # binary classification
        )

    def forward(
        self,
        x: torch.Tensor,          # raw node features,  shape (N, node_feat_dim)
        edge_index: torch.Tensor,  # graph edges,         shape (2, E)
        x_temp: torch.Tensor,      # temporal features,   shape (N, temporal_feat_dim)
    ) -> torch.Tensor:
        # GNN embedding
        gnn_emb = self.gnn(x, edge_index)          # (N, gnn_embed_dim)

        # Temporal embedding
        temp_emb = self.temporal_mlp(x_temp)        # (N, temp_embed_dim)

        # Concatenate: GNN + temporal + original features (skip)
        fused = torch.cat([gnn_emb, temp_emb, x], dim=-1)   # (N, fusion_in)
        logits = self.fusion(fused)                          # (N, 2)
        return logits

    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        x_temp: torch.Tensor,
    ) -> dict:
        """Return intermediate representations for analysis."""
        gnn_emb = self.gnn(x, edge_index)
        temp_emb = self.temporal_mlp(x_temp)
        fused = torch.cat([gnn_emb, temp_emb, x], dim=-1)
        return {
            'gnn_embedding': gnn_emb,
            'temporal_embedding': temp_emb,
            'fused': fused,
        }


class GraphSAGEWithTemp(torch.nn.Module):
    """
    Ablation B model: GraphSAGE encoder + temporal features concatenated before
    the classifier. No skip connection over original features.
    """

    def __init__(self, node_feat_dim: int, temporal_feat_dim: int,
                 hidden: int = 64, embed_dim: int = 64, dropout: float = 0.5):
        super().__init__()
        self.conv1 = SAGEConv(node_feat_dim, hidden)
        self.conv2 = SAGEConv(hidden, embed_dim)
        self.dropout = dropout
        self.classifier = nn.Linear(embed_dim + temporal_feat_dim, 2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                x_temp: torch.Tensor = None) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        if x_temp is not None:
            h = torch.cat([h, x_temp], dim=-1)
        return self.classifier(h)

