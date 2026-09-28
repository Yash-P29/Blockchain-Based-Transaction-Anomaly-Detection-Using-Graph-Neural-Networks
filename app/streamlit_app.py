"""
Streamlit Research Demo — Phase 3
Blockchain-Based Transaction Anomaly Detection Using Graph Neural Networks
"""

import os
import sys
import json
import time
import logging

import pandas as pd
import numpy as np
import streamlit as st

from src.utils import config
from src.data.loader import load_processed_data, load_raw_data

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Blockchain Anomaly Detection | GNN Research Demo",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/46/Bitcoin.svg", width=60)
    st.title("Research Demo")
    st.markdown("**Phase 3 — Temporal-Structural GNN**")
    st.markdown("---")
    st.markdown(
        "Dataset: [Elliptic Bitcoin](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)  \n"
        "Framework: PyTorch Geometric  \n"
        "Submission: 2026"
    )
    st.markdown("---")
    st.caption("Run `python -m src.training.train_proposed` to train the proposed model.")

# ─────────────────────────────────────────────────────────────────────────────
# Load data (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    nodes_df, edges_df = load_processed_data()
    return nodes_df, edges_df


@st.cache_data(show_spinner=False)
def load_metrics(filename):
    path = config.METRICS_DIR / filename
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


with st.spinner("Loading dataset…"):
    nodes_df, edges_df = load_data()

baseline_metrics = load_metrics("baseline_results.csv")
proposed_metrics = load_metrics("proposed_results.csv")

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dataset Overview",
    "📈 Baseline Comparison",
    "🧠 Proposed Model",
    "🔍 Transaction Prediction",
    "💡 Explainability",
    "📋 Research Results",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Dataset Overview
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("📊 Elliptic Bitcoin Transaction Dataset")

    if nodes_df is not None and edges_df is not None:
        labeled = nodes_df[nodes_df['label'] != -1]
        illicit_count = int((labeled['label'] == 1).sum())
        licit_count   = int((labeled['label'] == 0).sum())

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Total Transactions", f"{len(nodes_df):,}")
        c2.metric("Labeled Transactions", f"{len(labeled):,}")
        c3.metric("Illicit (Anomaly)", f"{illicit_count:,}")
        c4.metric("Licit (Normal)", f"{licit_count:,}")
        c5.metric("Timesteps", f"{nodes_df['timestep'].nunique()}")
        c6.metric("Graph Edges", f"{len(edges_df):,}")

        st.markdown("---")

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Class Distribution (Labeled Nodes)")
            dist_df = pd.DataFrame({
                'Class': ['Licit (Normal)', 'Illicit (Anomaly)'],
                'Count': [licit_count, illicit_count],
            })
            st.bar_chart(dist_df.set_index('Class'))
            ratio = licit_count / illicit_count if illicit_count > 0 else float('inf')
            st.info(f"Imbalance ratio (Normal : Anomaly) = **{ratio:.1f} : 1**")

        with col_b:
            st.subheader("Transactions per Timestep")
            ts_counts = nodes_df.groupby('timestep').size().reset_index(name='count')
            st.line_chart(ts_counts.set_index('timestep'))

        st.subheader("Structural Statistics")
        if 'in_degree' in nodes_df.columns and 'out_degree' in nodes_df.columns:
            dc1, dc2, dc3 = st.columns(3)
            dc1.metric("Max In-Degree",  int(nodes_df['in_degree'].max()))
            dc2.metric("Max Out-Degree", int(nodes_df['out_degree'].max()))
            dc3.metric("Avg Total Degree", f"{nodes_df['total_degree'].mean():.2f}"
                       if 'total_degree' in nodes_df.columns else "N/A")

            st.markdown("**Degree distribution (labeled nodes):**")
            deg_df = labeled[['in_degree', 'out_degree']].copy()
            deg_summary = deg_df.describe().T
            st.dataframe(deg_summary.style.format(precision=2))
        else:
            st.info("Run preprocessing to see degree statistics.")
    else:
        st.error("Processed data not found. Run `python -m src.data.preprocessing` first.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Baseline Comparison
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("📈 Phase 2 Baseline Model Comparison")

    # Hard-coded Phase 2 results (from BASELINE_RESULTS.md, actual experimental output)
    phase2_hardcoded = pd.DataFrame([
        {"Model": "Logistic Regression", "Accuracy": 0.6349, "Precision": 0.1447, "Recall": 0.9409, "F1": 0.2509, "ROC-AUC": 0.8572, "PR-AUC": 0.2157},
        {"Model": "Random Forest",       "Accuracy": 0.9775, "Precision": 0.9903, "Recall": 0.6602, "F1": 0.7922, "ROC-AUC": 0.9184, "PR-AUC": 0.7807},
        {"Model": "XGBoost",             "Accuracy": 0.9621, "Precision": 0.6973, "Recall": 0.7359, "F1": 0.7161, "ROC-AUC": 0.9329, "PR-AUC": 0.7956},
        {"Model": "GCN",                 "Accuracy": 0.7459, "Precision": 0.1645, "Recall": 0.7138, "F1": 0.2674, "ROC-AUC": 0.8046, "PR-AUC": 0.2477},
        {"Model": "GraphSAGE",           "Accuracy": 0.8677, "Precision": 0.2898, "Recall": 0.7147, "F1": 0.4124, "ROC-AUC": 0.8846, "PR-AUC": 0.5292},
        {"Model": "GAT",                 "Accuracy": 0.5506, "Precision": 0.1224, "Recall": 0.9594, "F1": 0.2172, "ROC-AUC": 0.8742, "PR-AUC": 0.3260},
    ])

    display_df = baseline_metrics if baseline_metrics is not None else phase2_hardcoded

    st.subheader("Metrics Table")
    st.dataframe(
        display_df.style.format({
            col: "{:.4f}" for col in display_df.columns if col != "Model"
        }).highlight_max(
            subset=[c for c in display_df.columns if c not in ("Model",)],
            color="#d4edda",
        ),
        use_container_width=True,
    )

    st.subheader("PR-AUC Comparison (Primary Metric)")
    pr_chart = display_df.set_index("Model")[["PR-AUC"]]
    st.bar_chart(pr_chart)

    st.subheader("ROC-AUC Comparison")
    roc_chart = display_df.set_index("Model")[["ROC-AUC"]]
    st.bar_chart(roc_chart)

    st.markdown("---")
    st.subheader("Saved Figures")
    fig_cols = st.columns(3)
    for i, model in enumerate(display_df['Model']):
        slug = model.lower().replace(' ', '_')
        img_path = config.FIGURES_DIR / f"{slug}_cm.png"
        if os.path.exists(img_path):
            fig_cols[i % 3].image(str(img_path), caption=f"{model} Confusion Matrix")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Proposed Model
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("🧠 Phase 3: Temporal-Structural GNN (Proposed Model)")

    st.markdown("""
    **Architecture Summary:**
    - **Graph Encoder**: 2-layer GraphSAGE (hidden=64, embed=64)
    - **Temporal MLP**: encodes 9 handcrafted temporal/structural features (hidden=32, embed=32)
    - **Fusion**: concatenate GNN embedding + temporal embedding + raw features → classify
    - **Loss**: Weighted CrossEntropyLoss (weight = N_licit / N_illicit)
    """)

    st.code("""
Node Features (166-d)   Temporal Features (9-d)
        │                        │
  SAGEConv (→64)          Linear (→32)
   ReLU / Dropout           ReLU / Dropout
  SAGEConv (→64)          Linear (→32)
        │                    ReLU
        └──── cat ──── cat ────┘
               + raw features (166-d)     [total = 262-d]
                      │
               Linear (→64) → ReLU → Dropout
               Linear (→2)   [logits]
    """, language="text")

    if proposed_metrics is not None:
        st.subheader("Actual Experiment Results")
        st.dataframe(
            proposed_metrics.style.format({
                col: "{:.4f}" for col in proposed_metrics.columns
                if col not in ("Model", "TrainTime_s", "InfTime_ms")
            }),
            use_container_width=True,
        )

        # Compare proposed vs baselines
        st.subheader("Comparison vs. Phase 2 Baselines")
        phase2 = pd.DataFrame([
            {"Model": "Logistic Regression", "PR-AUC": 0.2157},
            {"Model": "Random Forest",       "PR-AUC": 0.7807},
            {"Model": "XGBoost",             "PR-AUC": 0.7956},
            {"Model": "GCN",                 "PR-AUC": 0.2477},
            {"Model": "GraphSAGE (P2)",      "PR-AUC": 0.5292},
            {"Model": "GAT",                 "PR-AUC": 0.3260},
        ])
        proposed_pr = proposed_metrics[["Model", "PR-AUC"]].copy()
        combined = pd.concat([phase2, proposed_pr], ignore_index=True)
        st.bar_chart(combined.set_index("Model")[["PR-AUC"]])
    else:
        st.warning(
            "Proposed model results not found.  \n"
            "Run: `python -m src.training.train_proposed`"
        )

    st.subheader("Training Curves")
    for model_slug in ["temporal_gnn", "graphsage_baseline", "graphsage_plus_temporal"]:
        img_path = config.FIGURES_DIR / f"{model_slug}_loss.png"
        if os.path.exists(img_path):
            st.image(str(img_path), caption=f"{model_slug.replace('_', ' ').title()} Loss Curve")


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Transaction Prediction
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("🔍 Transaction Prediction")

    if nodes_df is None:
        st.error("Data not loaded.")
    else:
        # Check if proposed model is available
        model_path = config.MODELS_DIR / "temporal_gnn.pt"
        proposed_available = os.path.exists(model_path)

        st.markdown("Select a transaction from the dataset to see its features and model prediction.")

        labeled_nodes = nodes_df[nodes_df['label'] != -1].copy()

        col_filter, col_select = st.columns([1, 2])
        with col_filter:
            ts_options = sorted(labeled_nodes['timestep'].unique())
            selected_ts = st.selectbox("Filter by Timestep", options=ts_options, index=min(34, len(ts_options)-1))

        ts_nodes = labeled_nodes[labeled_nodes['timestep'] == selected_ts]

        with col_select:
            tx_options = ts_nodes['txId'].tolist()
            selected_tx = st.selectbox("Select Transaction ID", options=tx_options[:200])

        if selected_tx:
            row = nodes_df[nodes_df['txId'] == selected_tx].iloc[0]
            true_label = int(row['label'])

            st.markdown("---")
            info_cols = st.columns(4)
            info_cols[0].metric("Transaction ID", str(selected_tx))
            info_cols[1].metric("Timestep", str(int(row['timestep'])))
            info_cols[2].metric(
                "True Label",
                "🚨 ILLICIT" if true_label == 1 else "✅ LICIT"
            )
            in_deg = int(row.get('in_degree', 0))
            out_deg = int(row.get('out_degree', 0))
            info_cols[3].metric("Degree", f"in={in_deg} out={out_deg}")

            if proposed_available:
                st.subheader("Model Prediction")
                try:
                    import torch
                    import torch.nn.functional as F
                    from src.models.temporal_gnn import TemporalGNN
                    from src.data.temporal_features import compute_temporal_structural_features, TEMPORAL_FEATURE_COLS
                    from src.graph.graph_builder import build_pyg_graph

                    @st.cache_resource
                    def load_model_and_data():
                        n_df, e_df = load_processed_data()
                        from src.data.temporal_features import compute_temporal_structural_features
                        nodes_aug = compute_temporal_structural_features(n_df, e_df)
                        data = build_pyg_graph(nodes_aug, e_df)

                        feature_cols = [c for c in nodes_aug.columns if c.startswith('feat_') or c.endswith('_degree')]
                        in_ch = len(feature_cols)
                        temp_ch = len(TEMPORAL_FEATURE_COLS)

                        model = TemporalGNN(node_feat_dim=in_ch, temporal_feat_dim=temp_ch)
                        model.load_state_dict(torch.load(str(config.MODELS_DIR / "temporal_gnn.pt"), map_location='cpu'))
                        model.eval()

                        import torch
                        x_temp = torch.tensor(nodes_aug[TEMPORAL_FEATURE_COLS].astype(float).values, dtype=torch.float)
                        data.x_temp = x_temp

                        # Build txId → node index map
                        unique_nodes = nodes_aug['txId'].unique()
                        tx_to_idx = {tx: idx for idx, tx in enumerate(unique_nodes)}
                        return model, data, tx_to_idx, nodes_aug

                    model, data, tx_to_idx, nodes_aug = load_model_and_data()

                    if selected_tx in tx_to_idx:
                        node_idx = tx_to_idx[selected_tx]
                        with torch.no_grad():
                            out = model(data.x, data.edge_index, data.x_temp)
                            probs = F.softmax(out, dim=1)
                            illicit_prob = float(probs[node_idx, 1])
                            pred_label = int(out[node_idx].argmax())

                        pred_text = "🚨 ILLICIT" if pred_label == 1 else "✅ LICIT"
                        match = pred_label == true_label

                        p1, p2, p3 = st.columns(3)
                        p1.metric("Prediction", pred_text)
                        p2.metric("Anomaly Probability", f"{illicit_prob:.4f}")
                        p3.metric("Correct?", "✔ Yes" if match else "✘ No")

                        st.progress(illicit_prob, text=f"Illicit probability: {illicit_prob:.1%}")
                    else:
                        st.warning("Transaction index not found in model graph.")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
            else:
                st.info("Proposed model not trained yet. Run `python -m src.training.train_proposed` first.")

            # Show transaction features
            with st.expander("View Transaction Features"):
                feat_cols = [c for c in nodes_df.columns if c.startswith('feat_')]
                st.dataframe(
                    row[feat_cols[:20]].to_frame().T.style.format(precision=4),
                    use_container_width=True
                )


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — Explainability
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("💡 Model Explainability")

    model_path = config.MODELS_DIR / "temporal_gnn.pt"
    proposed_available = os.path.exists(model_path)

    st.markdown("""
    **Methodology:** Feature importance is estimated using permutation importance on the
    test set. For each feature, we measure the drop in PR-AUC when that feature is
    randomly permuted. A larger drop indicates higher importance.

    This approach is model-agnostic and avoids the known instability of SHAP
    for graph-based models where node representations are interdependent.
    """)

    importance_path = config.METRICS_DIR / "feature_importance.json"

    if os.path.exists(importance_path):
        with open(importance_path) as f:
            imp_data = json.load(f)

        imp_df = pd.DataFrame(imp_data).sort_values("importance", ascending=False)
        st.subheader("Feature Importance (Permutation-Based)")
        st.bar_chart(imp_df.set_index("feature")["importance"])
        st.dataframe(imp_df.style.format({"importance": "{:.4f}", "std": "{:.4f}"}), use_container_width=True)
    else:
        st.info(
            "Feature importance data not yet computed.  \n"
            "Run `python -m src.training.train_proposed` and the explainability "
            "script to generate this data."
        )
        st.markdown("""
        **What you will see here after training:**
        - Importance scores for each of the 9 temporal/structural features
        - Feature names such as `timestep_norm`, `total_degree`, `ts_illicit_rate`
        - Sorted bar chart of contributions to PR-AUC
        """)

        # Show expected feature list as preview
        st.subheader("Temporal/Structural Features (Preview)")
        preview = pd.DataFrame({
            "Feature": [
                "ts_illicit_rate", "timestep_norm", "total_degree",
                "timestep_degree_interact", "ts_mean_degree",
                "local_neighbor_count", "in_degree", "out_degree", "degree_ratio",
            ],
            "Description": [
                "Fraction of illicit transactions at this timestep (train only)",
                "Normalised timestep position [0,1]",
                "Total node degree (in + out)",
                "timestep_norm × log(1 + total_degree)",
                "Mean degree of all nodes at this timestep",
                "Count of distinct neighbours",
                "Number of incoming edges",
                "Number of outgoing edges",
                "in_degree / (total_degree + ε)",
            ]
        })
        st.table(preview)

    st.markdown("---")
    st.subheader("Confusion Matrix (Proposed Model)")
    cm_path = config.FIGURES_DIR / "temporal_gnn_cm.png"
    if os.path.exists(cm_path):
        st.image(str(cm_path), width=450)
    else:
        st.info("Train the proposed model to see its confusion matrix here.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — Research Results
# ════════════════════════════════════════════════════════════════════════════
with tab6:
    st.header("📋 Research Results")

    # Full comparison table
    st.subheader("Complete Model Comparison")
    phase2_rows = [
        {"Model": "Logistic Regression", "Accuracy": 0.6349, "Precision": 0.1447, "Recall": 0.9409, "F1": 0.2509, "ROC-AUC": 0.8572, "PR-AUC": 0.2157, "Phase": "2 (Baseline)"},
        {"Model": "Random Forest",       "Accuracy": 0.9775, "Precision": 0.9903, "Recall": 0.6602, "F1": 0.7922, "ROC-AUC": 0.9184, "PR-AUC": 0.7807, "Phase": "2 (Baseline)"},
        {"Model": "XGBoost",             "Accuracy": 0.9621, "Precision": 0.6973, "Recall": 0.7359, "F1": 0.7161, "ROC-AUC": 0.9329, "PR-AUC": 0.7956, "Phase": "2 (Baseline)"},
        {"Model": "GCN",                 "Accuracy": 0.7459, "Precision": 0.1645, "Recall": 0.7138, "F1": 0.2674, "ROC-AUC": 0.8046, "PR-AUC": 0.2477, "Phase": "2 (Baseline)"},
        {"Model": "GraphSAGE",           "Accuracy": 0.8677, "Precision": 0.2898, "Recall": 0.7147, "F1": 0.4124, "ROC-AUC": 0.8846, "PR-AUC": 0.5292, "Phase": "2 (Baseline)"},
        {"Model": "GAT",                 "Accuracy": 0.5506, "Precision": 0.1224, "Recall": 0.9594, "F1": 0.2172, "ROC-AUC": 0.8742, "PR-AUC": 0.3260, "Phase": "2 (Baseline)"},
    ]
    phase2_df = pd.DataFrame(phase2_rows)

    if proposed_metrics is not None:
        prop_display = proposed_metrics.copy()
        prop_display["Phase"] = "3 (Proposed)"
        for col in ["TrainTime_s", "InfTime_ms"]:
            if col in prop_display.columns:
                prop_display = prop_display.drop(columns=[col])
        combined = pd.concat([phase2_df, prop_display], ignore_index=True)
    else:
        combined = phase2_df.copy()
        st.warning("Proposed model results not yet available. Showing Phase 2 baselines only.")

    fmt_cols = {c: "{:.4f}" for c in combined.columns if c not in ("Model", "Phase")}
    st.dataframe(combined.style.format(fmt_cols), use_container_width=True)

    # PR and ROC curves
    st.markdown("---")
    col_pr, col_roc = st.columns(2)
    with col_pr:
        st.subheader("Precision-Recall Curves")
        for slug in ["temporal_gnn", "graphsage_baseline", "xgboost", "graphsage"]:
            pr_path = config.FIGURES_DIR / f"{slug}_pr.png"
            if os.path.exists(pr_path):
                st.image(str(pr_path), caption=slug.replace("_", " ").title())

    with col_roc:
        st.subheader("ROC Curves")
        for slug in ["temporal_gnn", "graphsage_baseline", "xgboost", "graphsage"]:
            roc_path = config.FIGURES_DIR / f"{slug}_roc.png"
            if os.path.exists(roc_path):
                st.image(str(roc_path), caption=slug.replace("_", " ").title())

    # Ablation table
    st.markdown("---")
    st.subheader("Ablation Study")
    st.markdown("""
    | Variant | Description | PR-AUC |
    |---------|-------------|--------|
    | A | GraphSAGE baseline (re-evaluated) | — |
    | B | GraphSAGE + temporal/structural features | — |
    | C | Full proposed model (A + B + skip connection) | — |

    > Results populated after training. Run `python -m src.training.train_proposed`.
    """)
    if proposed_metrics is not None:
        ablation_preview = proposed_metrics[["Model", "PR-AUC", "F1", "ROC-AUC"]].copy()
        st.dataframe(ablation_preview.style.format({
            "PR-AUC": "{:.4f}", "F1": "{:.4f}", "ROC-AUC": "{:.4f}"
        }), use_container_width=True)

    # Training/inference time
    if proposed_metrics is not None and "TrainTime_s" in proposed_metrics.columns:
        st.markdown("---")
        st.subheader("Training & Inference Time")
        time_df = proposed_metrics[["Model", "TrainTime_s", "InfTime_ms"]]
        st.dataframe(time_df, use_container_width=True)

    st.markdown("---")
    st.subheader("Research Contribution")
    st.markdown("""
    We propose a **temporal-structural GNN framework** for illicit Bitcoin transaction
    detection. The framework combines:

    1. A **GraphSAGE graph encoder** that aggregates neighbourhood information
    2. A **temporal MLP** encoding handcrafted temporal and structural features
       derived from the dataset's 49-timestep structure
    3. A **feature fusion layer** with a skip connection over the original node features

    The experiments investigate whether incorporating temporal and structural context
    improves over the GraphSAGE baseline on the Elliptic Bitcoin Transaction Dataset.
    The results are reported on a held-out chronological test set (timesteps 35–49)
    to ensure temporal integrity.
    """)
