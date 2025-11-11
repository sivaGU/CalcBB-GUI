import streamlit as st


st.set_page_config(page_title="Documentation", page_icon="📚")

st.title("Documentation & Runbook")
st.caption("Reference material derived from Tabs 4–5 of the BBB manuscript.")

st.markdown(
    """
    ## Purpose
    This application packages the manuscript’s sparse-label multi-task (MT) modelling workflow into a
    Streamlit interface. The current release focuses on communication: summarising study context,
    evaluation methodology, and planned visual assets before the ligand submission module comes online.
    """
)

st.markdown(
    """
    ## Repository structure
    ```
    .
    ├── streamlit_app.py         # Home page (Tab 4–5 narrative + roadmap)
    ├── pages/
    │   └── 1_Documentation.py   # You are here
    ├── BBB Manuscript.docx      # Source manuscript (reference only)
    └── requirements.txt         # Streamlit dependency pin
    ```
    """
)

st.markdown(
    """
    ## Local setup
    1. Create and activate a virtual environment (conda, venv, or poetry).  
    2. Install dependencies: `pip install -r requirements.txt`.  
    3. Launch the app: `streamlit run streamlit_app.py`.  
    4. Streamlit will open at `http://localhost:8501`. Use the sidebar to switch between Home and Documentation.
    """
)

st.markdown(
    """
    ## Model overview (Tab 4 recap)
    - **Training data:** BBB permeability labels plus auxiliary ADME assays (PAMPA, PPB, efflux).  
    - **Learning strategy:** Masked MT ensemble blended with an ST baseline; losses applied only where task labels exist.  
    - **Calibration:** Platt vs isotonic assessed on the development fold; chosen calibrator reused for internal/external evaluations.  
    - **Reproducibility:** All metrics/figures regenerate from `results/metrics_clean_fixed.json`; stratified bootstrap (B = 2000) underpins confidence intervals and ΔPR-AUC tests.
    """
)

st.markdown(
    """
    ## Evaluation protocol (Tab 5 recap)
    - **Primary metric:** PR-AUC (robust to class imbalance); ROC-AUC reported for context.  
    - **Operational thresholds:** Summaries at 0.5 and Youden (≈0.793) include accuracy, sensitivity, specificity, F1, and MCC.  
    - **Calibration diagnostics:** Brier score, expected calibration error (ECE), reliability diagrams with equal-mass bins.  
    - **Applicability domain:** Precision vs coverage curves thresholded on ensemble variance or representation distance.  
    - **Feature interpretation:** SHAP beeswarm/waterfall plots planned for top descriptors (LightGBM head).
    """
)

st.markdown(
    """
    ## Roadmap
    - **To-do:** Build ligand intake tab supporting SMILES/PDBQT uploads, descriptor generation, and scoring.  
    - **Planned visual assets:** External/internal ROC & PR with CI bands, calibration dashboards, confusion matrices, SHAP explorer, AD curves.  
    - **Reporting:** Automated PDF/CSV exports once ligand scoring is active.
    """
)

st.info(
    "Need to contribute? Fork the GitHub repository, branch from `main`, and submit a pull request. "
    "Include before/after screenshots when adding new widgets to keep the review focused."
)

st.success("Questions? Open an issue via the menu or tag the modelling team on Slack.")


