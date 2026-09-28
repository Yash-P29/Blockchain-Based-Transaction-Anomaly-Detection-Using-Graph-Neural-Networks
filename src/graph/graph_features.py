import networkx as nx
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def compute_centrality_features(G: nx.DiGraph, nodes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes centrality measures (PageRank, Betweenness) for graph nodes.
    Note: Betweenness centrality on large graphs is extremely slow, so it's skipped by default
    unless running on a very small subgraph.
    """
    logger.info("Computing PageRank...")
    pagerank = nx.pagerank(G, alpha=0.85)
    
    # Map to DataFrame
    pr_df = pd.DataFrame(list(pagerank.items()), columns=['txId', 'pagerank'])
    
    nodes_df = nodes_df.merge(pr_df, on='txId', how='left')
    nodes_df['pagerank'] = nodes_df['pagerank'].fillna(0)
    
    return nodes_df
