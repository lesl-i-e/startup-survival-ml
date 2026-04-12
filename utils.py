"""
utils.py — shared model loading, constants, and helper functions.
Loaded once and cached so the app stays fast across page switches.
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "data")

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES   = ["operating", "closed", "acquired", "ipo"]
CLASS_COLOURS = {
    "operating": "#2196F3",
    "closed":    "#F44336",
    "acquired":  "#4CAF50",
    "ipo":       "#FF9800",
}
MODEL_FEATURES = [
    "log_funding", "funding_rounds", "funding_velocity",
    "startup_age_days", "funding_duration_days",
    "days_since_last_funding", "founded_year",
    "num_categories", "is_africa",
]
FEATURE_LABELS = {
    "log_funding":             "Log(1 + Funding USD)",
    "funding_rounds":          "Number of Funding Rounds",
    "funding_velocity":        "Funding Velocity (USD/round)",
    "startup_age_days":        "Startup Age at Last Funding (days)",
    "funding_duration_days":   "Funding Duration (days)",
    "days_since_last_funding": "Days Since Last Funding",
    "founded_year":            "Year Founded",
    "num_categories":          "Number of Sectors",
    "is_africa":               "African Startup (flag)",
}

AFRICA_CODES = {
    "KEN","NGA","ZAF","EGY","GHA","TZA","ETH","RWA",
    "UGA","SEN","MAR","TUN","CIV","AGO","MOZ","ZWE",
    "ZMB","BWA","NAM","MWI"
}

TOP_SECTORS = [
    "Software", "Biotechnology", "Mobile", "E-Commerce",
    "Health Care", "Enterprise Software", "SaaS", "Education",
    "Advertising", "Games", "Analytics", "FinTech",
    "Social Media", "Clean Technology", "Other",
]

# ── Cached loaders — run once per session ────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_models():
    return {
        "LR":  joblib.load(os.path.join(MODELS_DIR, "model_logistic_regression.pkl")),
        "RF":  joblib.load(os.path.join(MODELS_DIR, "model_random_forest.pkl")),
        "XGB": joblib.load(os.path.join(MODELS_DIR, "model_xgboost.pkl")),
        "XGB_tuned": joblib.load(os.path.join(MODELS_DIR, "model_xgboost_tuned.pkl")),
        "scaler":    joblib.load(os.path.join(MODELS_DIR, "scaler.pkl")),
    }

@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    path = os.path.join(DATA_DIR, "startup_model_ready.csv")
    return pd.read_csv(path)

@st.cache_data(show_spinner="Loading results…")
def load_results():
    path = os.path.join(DATA_DIR, "phase2_final_results.csv")
    return pd.read_csv(path)

# ── Prediction helper ─────────────────────────────────────────────────────────
def predict_survival(models, input_features: dict) -> dict:
    """
    Takes a dict of feature_name → value, returns probabilities
    from the tuned XGBoost model.
    """
    X = pd.DataFrame([input_features])[MODEL_FEATURES]
    best = models["XGB_tuned"]
    probs = best.predict_proba(X)[0]
    pred_idx = int(np.argmax(probs))
    return {
        "probs":      {cls: float(p) for cls, p in zip(CLASS_NAMES, probs)},
        "prediction": CLASS_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
    }

# ── Shared chart style helper ─────────────────────────────────────────────────
def style_metric(label, value, colour="#0F7B8C"):
    return f"""
    <div style="background:#F4F6F8;border-left:4px solid {colour};
                border-radius:8px;padding:12px 16px;margin:4px 0;">
        <div style="font-size:0.78rem;color:#64748B;">{label}</div>
        <div style="font-size:1.5rem;font-weight:700;color:#0D1B3E;">{value}</div>
    </div>
    """