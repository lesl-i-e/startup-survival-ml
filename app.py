import streamlit as st

st.set_page_config(
    page_title="Startup Survival Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #0D1B3E; }
    [data-testid="stSidebar"] * { color: #FFFFFF !important; }
    [data-testid="stSidebar"] .stSelectbox label { color: #AAC4CE !important; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #F4F6F8;
        border-radius: 10px;
        padding: 12px 16px;
        border-left: 4px solid #0F7B8C;
    }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #0D1B3E;
        border-bottom: 2px solid #0F7B8C;
        padding-bottom: 6px;
        margin-bottom: 16px;
    }

    /* Verdict boxes */
    .verdict-safe {
        background: #C6F6D5; border-left: 5px solid #276749;
        border-radius: 8px; padding: 16px; margin: 12px 0;
    }
    .verdict-risk {
        background: #FED7D7; border-left: 5px solid #9B2C2C;
        border-radius: 8px; padding: 16px; margin: 12px 0;
    }
    .verdict-uncertain {
        background: #FEEBC8; border-left: 5px solid #C05621;
        border-radius: 8px; padding: 16px; margin: 12px 0;
    }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.markdown("## 🚀 Startup Survival")
st.sidebar.markdown("**Predicting Startup Survival Using ML**")
st.sidebar.markdown("---")

pages = {
    "🏠  Home & Overview":           "pages/1_Home.py",
    "🔮  Survival Predictor":         "pages/2_Predictor.py",
    "📊  Model Performance":          "pages/3_Model_Performance.py",
    "🧠  Feature Importance (SHAP)":  "pages/4_SHAP.py",
    "🌍  Africa & Geographic Analysis":"pages/5_Africa.py",
}

st.sidebar.markdown("### Navigation")
selection = st.sidebar.radio("Go to", list(pages.keys()), label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Project Details**
- Student: Gedion Leslie Kweya
- Reg: SCT213-C002-0062/2022
- Supervisor: Mr. Adhola Samuel
- Institution: JKUAT
""")
st.sidebar.markdown("---")
st.sidebar.caption("BIT 2303 / SDS 2406 · April 2026")

# ── Route to selected page ────────────────────────────────────────────────────
import importlib.util, sys, os

page_file = pages[selection]
spec = importlib.util.spec_from_file_location("page", page_file)
mod  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)