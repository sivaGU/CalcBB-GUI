import streamlit as st


st.set_page_config(
    page_title="BBB Permeability Studio",
    page_icon="🧠",
    layout="wide",
    menu_items={
        "Report a bug": "https://github.com/your-org/bbb-gui/issues",
        "About": "Sparse-label multi-task learning workflow for BBB permeability modelling.",
    },
)

st.title("Blood–Brain Barrier (BBB) Permeability Studio")
st.caption(
    "Prototype interface for the sparse-label multi-task ensemble described in the BBB manuscript."
)

st.sidebar.header("Project Snapshot")
st.sidebar.markdown(
    """
    - **Model focus:** Calibrated BBB permeability classification  
    - **Architecture:** Masked multi-task ensemble blended with a single-task baseline  
    - **External validation:** BBBP and out-of-source (OOS) panels  
    - **Status:** Home & documentation pages in place; ligand submission tab coming soon
    """
)
st.sidebar.success("Next milestone: interactive ligand screening tab (in development).")

st.markdown(
    """
    ## Why this app exists
    Drug discovery teams struggle to predict whether small molecules cross the blood–brain barrier.
    The manuscript’s fourth tab introduces a sparse-label multi-task (MT) learning workflow that blends
    auxiliary ADME tasks (PAMPA, PPB, efflux) with a calibrated single-task (ST) baseline. The blended
    predictor improves both external generalization and probability calibration, addressing two recurring
    issues in BBB screening campaigns.
    """
)

st.markdown(
    """
    ### Model highlights from Tab 4
    - **Sparse-label MT training:** Each auxiliary task contributes signal only where assays exist, avoiding label deletion or imputation bias.
    - **Stacked calibration:** MT logits are linearly blended with the ST baseline before post-hoc calibration selected on the development fold.
    - **Reproducibility guardrails:** All tables/figures originate from `results/metrics_clean_fixed.json`, with scripted pipelines and stratified bootstraps (B = 2000).
    """
)

st.divider()

st.markdown("## Performance at a glance (Tab 4 metrics)")

internal_metrics = {"PR-AUC": 0.915, "ROC-AUC": 0.864, "ΔPR-AUC vs ST": "+0.102"}
external_metrics = [
    {
        "dataset": "BBBP",
        "PR-AUC": 0.950,
        "ΔPR-AUC vs ST": "+0.155",
        "p-value": "< 0.001",
    },
    {
        "dataset": "Out-of-source (OOS)",
        "PR-AUC": 0.944,
        "ΔPR-AUC vs ST": "+0.185",
        "p-value": "< 0.001",
    },
]

col_internal, col_ext_1, col_ext_2 = st.columns(3)
with col_internal:
    st.metric("Internal PR-AUC", internal_metrics["PR-AUC"], internal_metrics["ΔPR-AUC vs ST"])
    st.metric("Internal ROC-AUC", internal_metrics["ROC-AUC"])

with col_ext_1:
    st.metric("BBBP PR-AUC", external_metrics[0]["PR-AUC"], external_metrics[0]["ΔPR-AUC vs ST"])
    st.caption(f"One-sided ΔPR-AUC p-value {external_metrics[0]['p-value']}")

with col_ext_2:
    st.metric(
        "OOS PR-AUC",
        external_metrics[1]["PR-AUC"],
        external_metrics[1]["ΔPR-AUC vs ST"],
    )
    st.caption(f"One-sided ΔPR-AUC p-value {external_metrics[1]['p-value']}")

st.markdown(
    """
    Calibration improves alongside discrimination: the blended model reports lower Brier score and
    expected calibration error (ECE) than the single-task baseline, with reliability diagrams approaching
    the identity line across internal and external datasets.
    """
)

st.divider()

st.markdown("## From Tab 5: evaluation protocol & upcoming assets")

tab5_col1, tab5_col2 = st.columns(2)
with tab5_col1:
    st.subheader("Evaluation blueprint")
    st.markdown(
        """
        - **Primary metric:** Precision–recall AUC (PR-AUC); ROC-AUC reported as a secondary view.  
        - **Uncertainty:** Stratified bootstrap (B = 2000, seed = 42) yields 95% confidence intervals and ΔPR-AUC hypothesis tests.  
        - **Calibration checks:** Brier score, ECE, and reliability diagrams with equal-mass binning; Platt vs isotonic selected on the development fold.  
        - **Applicability domain:** Coverage vs precision curves using ensemble variance or representation distance thresholds.  
        """
    )

with tab5_col2:
    st.subheader("Assets in progress")
    st.markdown(
        """
        - External & internal ROC/PR curves with confidence bands  
        - Calibration dashboards (reliability diagrams, ΔECE summaries)  
        - Confusion matrices at 0.5 and Youden thresholds  
        - Feature attribution (SHAP) views for top ADME descriptors  
        - Applicability domain plots showing precision vs coverage trade-offs  
        """
    )

st.info(
    "The ligand submission workspace will surface alongside these assets. For now, use the Documentation page for setup instructions and repository structure."
)

st.markdown(
    """
    ---
    ### Roadmap
    1. **Current** – Communication spine: home and documentation pages summarizing the Tab 4–5 manuscript content.  
    2. **Next** – Ligand intake tab with structure upload, descriptor generation, and model scoring.  
    3. **Later** – Calibration overlay for user-submitted batches and automated report exports (PDF/CSV).  
    """
)


