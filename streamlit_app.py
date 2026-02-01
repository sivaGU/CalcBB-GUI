"""
MechBBB Streamlit GUI — Two-stage mechanistically augmented BBB permeability classifier (Model C).

Run from this folder (project root):
  streamlit run streamlit_app.py
"""
import sys
from pathlib import Path

# Ensure project root (this folder) is on path for src.mechbbb
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

HANDOFF_DIR = PROJECT_ROOT

import streamlit as st
import pandas as pd

from src.mechbbb.predict import predict_single, predict_batch, load_predictor

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="MechBBB — BBB Permeability Studio",
    page_icon="🧪",
    layout="wide",
    menu_items={
        "Report a bug": "https://github.com/your-org/mechbbb-gui/issues",
        "About": "Two-stage mechanistically augmented BBB permeability classifier (Model C).",
    },
)

# Inject custom CSS for color palette - blue-teal scale
st.markdown("""
<style>
    /* Blue-teal palette: light powder -> midnight azure */
    :root {
        --light-powder-blue: #E0F4F8;
        --soft-sky-blue: #B3E5F0;
        --light-azure: #80D4E8;
        --medium-steel-blue: #4DB8D0;
        --deep-cerulean: #2A9DB5;
        --rich-teal-blue: #1E7A8C;
        --midnight-azure: #0D4F5C;
    }
    
    .stApp {
        background-color: #ffffff;
    }
    
    section.main,
    .main,
    [data-testid="stAppViewContainer"] > div:not([data-testid="stSidebar"]) {
        background-color: #ffffff !important;
    }
    
    div[data-testid="stAppViewContainer"] > div > div:not([data-testid="stSidebar"]) {
        background-color: #ffffff !important;
    }
    
    .main .block-container,
    section.main .block-container {
        background-color: #ffffff !important;
        padding: 2rem 3rem;
        margin: 2rem auto;
        max-width: 1400px;
        border-radius: 8px;
        box-shadow: 0 2px 12px rgba(13, 79, 92, 0.08);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D4F5C 0%, #0A3D47 100%);
        color: #ffffff;
        min-width: 200px !important;
        max-width: 280px !important;
        width: 280px !important;
    }
    
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 280px !important;
        min-width: 200px !important;
        max-width: 280px !important;
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        background-color: #0A3D47;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #4DB8D0 0%, #2A9DB5 100%);
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(77, 184, 208, 0.35);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2A9DB5 0%, #1E7A8C 100%);
        box-shadow: 0 4px 8px rgba(42, 157, 181, 0.4);
        transform: translateY(-1px);
    }
    
    .stButton > button:focus {
        background: linear-gradient(135deg, #1E7A8C 0%, #0D4F5C 100%);
        box-shadow: 0 0 0 0.3rem rgba(77, 184, 208, 0.35);
    }
    
    .stDownloadButton > button {
        background: linear-gradient(135deg, #2A9DB5 0%, #1E7A8C 100%);
        color: white;
        border-radius: 6px;
        font-weight: 500;
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #1E7A8C 0%, #0D4F5C 100%);
    }
    
    h1, h2, h3 {
        color: #1E7A8C;
        font-weight: 700;
    }
    
    a {
        color: #1E7A8C;
        text-decoration: none;
    }
    
    a:hover {
        color: #2A9DB5;
        text-decoration: underline;
    }
    
    [data-testid="stMetricValue"] {
        color: #1E7A8C;
        font-weight: 600;
    }
    
    .stSuccess {
        background: linear-gradient(90deg, #E0F4F8 0%, #B3E5F0 100%);
        border-left: 4px solid #4DB8D0;
        color: #1a1a1a;
        border-radius: 4px;
    }
    
    .stInfo {
        background: linear-gradient(90deg, #E0F4F8 0%, #B3E5F0 100%);
        border-left: 4px solid #4DB8D0;
        color: #1a1a1a;
        border-radius: 4px;
    }
    
    .stWarning {
        background: linear-gradient(90deg, #B3E5F0 0%, #80D4E8 100%);
        border-left: 4px solid #2A9DB5;
        color: #1a1a1a;
        border-radius: 4px;
    }
    
    .stError {
        background: linear-gradient(90deg, #B3E5F0 0%, #80D4E8 100%);
        border-left: 4px solid #1E7A8C;
        color: #1a1a1a;
        border-radius: 4px;
    }
    
    .stRadio > label,
    .stSelectbox > label,
    .stTextInput > label,
    .stSlider > label,
    .stFileUploader > label {
        color: #1E7A8C;
        font-weight: 500;
    }
    
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #E0F4F8 0%, #B3E5F0 100%);
        color: #1E7A8C;
        border-radius: 4px;
        font-weight: 500;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(90deg, #B3E5F0 0%, #80D4E8 100%);
    }
    
    .stDataFrame {
        border: 2px solid #4DB8D0;
        border-radius: 4px;
    }
    
    hr {
        border-color: #4DB8D0;
        border-width: 2px;
    }
    
    .stSlider .stSlider > div > div {
        background-color: #4DB8D0;
    }
    
    [data-testid="stSidebar"] .stButton {
        margin-bottom: 0.5rem;
    }
    
    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #2A9DB5 0%, #1E7A8C 100%) !important;
        color: white !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(135deg, #1E7A8C 0%, #0D4F5C 100%) !important;
    }
    
    [data-testid="stSidebar"] h3 {
        color: #ffffff;
        font-weight: 600;
        margin-top: 0;
        margin-bottom: 0.5rem;
    }
    
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] li {
        color: rgba(255, 255, 255, 0.9);
    }
    
    [data-testid="stSidebar"] .stSuccess {
        background: linear-gradient(90deg, rgba(77, 184, 208, 0.3) 0%, rgba(42, 157, 181, 0.25) 100%);
        border-left: 4px solid #4DB8D0;
        color: #ffffff;
    }
    
    [data-testid="stSidebar"] .stInfo {
        background: linear-gradient(90deg, rgba(77, 184, 208, 0.25) 0%, rgba(42, 157, 181, 0.2) 100%);
        border-left: 4px solid #4DB8D0;
        color: #ffffff;
    }
    
    [data-testid="stSidebar"] hr {
        margin: 1rem 0;
        border-color: rgba(255, 255, 255, 0.25);
    }
    
    .main .block-container > div {
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# PREDICTOR (cached)
# ============================================================================

@st.cache_resource
def get_predictor():
    return load_predictor(HANDOFF_DIR)

DEFAULT_THRESHOLD = 0.35

# ============================================================================
# PAGES
# ============================================================================

def render_home_page():
    """Render the home/dashboard page."""
    st.title("🧪 MechBBB — Blood-Brain Barrier Permeability Studio")
    st.caption(
        "Two-stage mechanistically augmented BBB permeability classifier (Model C)."
    )

    st.sidebar.markdown("### Project Snapshot")
    st.sidebar.markdown(
        """
        - **Model focus:** BBB permeability classification (Model C)
        - **Architecture:** Stage-1 (efflux/influx/PAMPA) + Stage-2 (PhysChem+ECFP+mech)
        - **Threshold:** 0.35 (MCC-optimal on BBBP validation)
        - **Status:** All pages available
        """
    )
    st.sidebar.success("Interactive ligand screening available!")

    st.markdown(
        """
        ## Why this app exists
        Drug discovery teams struggle to predict whether small molecules cross the blood-brain barrier.
        MechBBB (Model C) is a two-stage mechanistically augmented classifier that first predicts
        auxiliary ADME properties (efflux, influx, PAMPA) and then combines them with physicochemical
        and fingerprint features to predict BBB permeability. This approach improves both
        external generalization and interpretability.
        """
    )

    st.markdown(
        """
        ### Model highlights
        - **Stage-1:** LightGBM models trained on auxiliary mechanistic datasets (BBBP excluded) → p_efflux, p_influx, p_pampa.
        - **Stage-2:** Model C = PhysChem + ECFP4 + mechanistic probs; ensemble of 5 seeds; threshold 0.35.
        - **No tuning on external data** — threshold selected on BBBP validation set only.
        """
    )

    st.divider()

    st.markdown("## Quick start")

    st.info(
        "**Ready to predict!** Use the **CalcBB Prediction** page in the sidebar to enter SMILES strings or upload a CSV file and get BBB permeability predictions with mechanistic probabilities."
    )

    st.markdown(
        """
        ---
        ### Navigation
        - **Home** — This overview
        - **Documentation** — Setup, model details, and usage
        - **CalcBB Prediction** — Run predictions (Single SMILES or Batch CSV)
        """
    )


def render_documentation_page():
    """Render the documentation page."""
    st.title("Documentation & Runbook")
    st.caption("Reference material for the MechBBB Model C classifier.")

    st.markdown(
        """
        ## Purpose
        This application provides a Streamlit interface for the MechBBB two-stage mechanistically augmented
        BBB permeability classifier (Model C). It supports single SMILES prediction and batch CSV processing.
        """
    )

    st.markdown(
        """
        ## Repository structure
        ```
        .
        ├── streamlit_app.py       # Main application
        ├── requirements.txt      # Dependencies
        ├── src/mechbbb/          # Prediction module
        │   ├── predict.py        # predict_single, predict_batch, load_predictor
        │   └── cli.py            # Command-line interface
        └── artifacts/            # Model artifacts
            ├── stage1_efflux.joblib, stage1_influx.joblib, stage1_pampa.joblib
            ├── stage2_modelC/    # model_seed0.pkl … model_seed4.pkl
            ├── threshold.json
            └── feature_config.json
        ```
        """
    )

    st.markdown(
        """
        ## Local setup
        1. Create and activate a virtual environment (conda, venv, or poetry).
        2. Install dependencies: `pip install -r requirements.txt`.
        3. Launch the app: `streamlit run streamlit_app.py`.
        4. Streamlit will open at `http://localhost:8501`. Use the sidebar to switch between pages.
        """
    )

    st.markdown(
        """
        ## Model overview (Model C)
        - **Stage-1:** LightGBM models on PhysChem + ECFP4 → p_efflux, p_influx, p_pampa.
        - **Stage-2:** 5-model ensemble on PhysChem + ECFP4 + mechanistic probs → P(BBB+).
        - **Threshold:** 0.35 (MCC-optimal on BBBP validation set).
        - **Features:** 10 physicochemical descriptors + 2048-bit ECFP4 + 3 mechanistic probabilities = 2061 total.
        """
    )

    st.markdown(
        """
        ## CLI usage
        From the project folder:
        ```bash
        python -m src.mechbbb.cli --smiles "CCO" "c1ccccc1" --output out.csv
        python -m src.mechbbb.cli --input example_inputs.csv --output out.csv
        ```
        Output columns: `smiles`, `canonical_smiles`, `prob_BBB+`, `BBB_class`, `p_efflux`, `p_influx`, `p_pampa`, `threshold`, `error`.
        """
    )

    st.success("Questions? Contact: Dr. Sivanesan Dakshanamurthy — sd233@georgetown.edu")


def render_calcbb_prediction_page():
    """Render the CalcBB / MechBBB prediction page."""
    st.title("BBB Permeability Prediction")
    st.markdown(
        """
        Predict BBB permeability using MechBBB (Model C). Enter a SMILES string or upload a CSV file.
        The model outputs P(BBB+), mechanistic probabilities (p_efflux, p_influx, p_pampa), and classification.
        
        **Input modes:** Single SMILES | Batch (CSV with smiles/SMILES column)
        """
    )

    try:
        predictor = get_predictor()
    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.info(
            "Ensure the **artifacts/** folder contains:\n"
            "- stage1_efflux.joblib, stage1_influx.joblib, stage1_pampa.joblib\n"
            "- stage2_modelC/ with model_seed0.pkl … model_seed4.pkl\n"
            "- threshold.json"
        )
        return

    st.sidebar.markdown("### Settings")
    threshold = st.sidebar.slider(
        "Classification threshold", 0.0, 1.0, DEFAULT_THRESHOLD, 0.01
    )
    st.sidebar.info(
        "**MechBBB (Model C)** · Default threshold 0.35 = MCC-optimal on BBBP validation."
    )

    st.divider()

    input_mode = st.radio(
        "Input mode",
        ["Single SMILES", "Batch (CSV)"],
        horizontal=True,
        key="input_mode",
    )

    if input_mode == "Single SMILES":
        smiles_input = st.text_input(
            "SMILES",
            placeholder="e.g. CCO, c1ccccc1",
            key="smiles_input",
        )
        if st.button("Predict", type="primary", key="btn_single"):
            if smiles_input and smiles_input.strip():
                result = predict_single(
                    smiles_input.strip(),
                    threshold=threshold,
                    predictor=predictor,
                )
                if result.is_valid:
                    st.success("✅ Valid SMILES")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("P(BBB+)", f"{result.prob:.4f}")
                    with col2:
                        st.metric("Prediction", result.bbb_class)
                    with col3:
                        st.metric("Threshold", f"{threshold:.2f}")

                    st.subheader("Mechanistic probabilities")
                    mcol1, mcol2, mcol3 = st.columns(3)
                    with mcol1:
                        st.metric("p_efflux", f"{result.p_efflux:.4f}")
                    with mcol2:
                        st.metric("p_influx", f"{result.p_influx:.4f}")
                    with mcol3:
                        st.metric("p_pampa", f"{result.p_pampa:.4f}")
                else:
                    st.error(f"❌ {result.error}")
            else:
                st.warning("Please enter a SMILES string.")

    else:
        uploaded_file = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            key="csv_upload",
        )
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            col = next(
                (
                    c
                    for c in df.columns
                    if c.lower() in ("smiles", "canonical_smiles", "smi") or c == "SMILES"
                ),
                None,
            )
            if col is None:
                st.error("CSV must have a SMILES column (smiles, SMILES, canonical_smiles, or smi).")
                st.info(f"Available columns: {', '.join(df.columns)}")
            else:
                if st.button("Predict batch", type="primary", key="btn_batch"):
                    smiles_list = df[col].astype(str).tolist()
                    results = predict_batch(
                        smiles_list,
                        threshold=threshold,
                        predictor=predictor,
                    )
                    df_out = df.copy()
                    df_out["prob_BBB+"] = [r.prob for r in results]
                    df_out["BBB_class"] = [r.bbb_class for r in results]
                    df_out["p_efflux"] = [r.p_efflux for r in results]
                    df_out["p_influx"] = [r.p_influx for r in results]
                    df_out["p_pampa"] = [r.p_pampa for r in results]

                    st.subheader("Results")
                    st.dataframe(df_out, use_container_width=True)

                    st.subheader("Download results")
                    st.download_button(
                        "Download CSV",
                        df_out.to_csv(index=False),
                        "mechbbb_predictions.csv",
                        "text/csv",
                        key="download_csv",
                    )
        else:
            st.info("Upload a CSV file with a SMILES column to run batch predictions.")

    st.divider()
    st.caption(
        "MechBBB (Model C). Stage-1: efflux/influx/PAMPA · Stage-2: PhysChem+ECFP+mech."
    )


# ============================================================================
# MAIN — NAVIGATION
# ============================================================================

def main():
    """Main app entry point with navigation."""
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Home"

    st.sidebar.markdown("### Navigation")
    st.sidebar.markdown("")

    if st.sidebar.button("Home", use_container_width=True, key="nav_home"):
        st.session_state.current_page = "Home"

    if st.sidebar.button("Documentation", use_container_width=True, key="nav_docs"):
        st.session_state.current_page = "Documentation"

    if st.sidebar.button("CalcBB Prediction", use_container_width=True, key="nav_prediction"):
        st.session_state.current_page = "CalcBB Prediction"

    st.sidebar.markdown("---")

    if st.session_state.current_page == "Home":
        render_home_page()
    elif st.session_state.current_page == "Documentation":
        render_documentation_page()
    elif st.session_state.current_page == "CalcBB Prediction":
        render_calcbb_prediction_page()


if __name__ == "__main__":
    main()
