"""Page 2 — Interactive Survival Predictor"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (load_models, predict_survival, MODEL_FEATURES,
                   CLASS_NAMES, CLASS_COLOURS, TOP_SECTORS, style_metric)

st.title("🔮 Startup Survival Predictor")
st.markdown("""
Enter a startup's profile below and the model will estimate its probability of
**operating**, **closing**, being **acquired**, or reaching an **IPO**.
All predictions come from the tuned XGBoost model (ROC-AUC 0.8078).
""")
st.markdown("---")

models = load_models()

# ── Input form ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📝 Startup Profile</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Funding Information**")
    funding_usd = st.number_input(
        "Total Funding Raised (USD)",
        min_value=0, max_value=1_000_000_000,
        value=2_000_000, step=100_000,
        help="Total USD raised across all rounds."
    )
    funding_rounds = st.slider(
        "Number of Funding Rounds",
        min_value=1, max_value=15, value=2,
        help="More rounds = sustained investor interest."
    )
    days_since_last = st.slider(
        "Days Since Last Funding",
        min_value=0, max_value=3000, value=365,
        help="0 = very recently funded. Higher = longer without follow-on investment."
    )

with col2:
    st.markdown("**Startup Profile**")
    founded_year = st.slider(
        "Year Founded",
        min_value=1990, max_value=2024, value=2012,
    )
    startup_age_years = st.slider(
        "Startup Age at Last Funding (years)",
        min_value=0, max_value=30, value=3,
        help="Age in years from founding to the last recorded funding date."
    )
    sector = st.selectbox("Primary Sector", TOP_SECTORS, index=0)
    is_africa = st.checkbox("African Startup", value=False,
                            help="Tick if the startup is headquartered in an African country.")

funding_duration_years = st.slider(
    "Funding Duration (years between first and last round)",
    min_value=0, max_value=15, value=2,
    help="0 = single round. Higher = sustained multi-year funding activity."
)
num_categories = st.slider(
    "Number of Sectors / Categories",
    min_value=1, max_value=10, value=2,
    help="How many industry categories the startup operates in."
)

# ── Compute derived features ──────────────────────────────────────────────────
log_funding       = float(np.log1p(funding_usd))
funding_velocity  = float(funding_usd / max(funding_rounds, 1))
startup_age_days  = float(startup_age_years * 365)
funding_dur_days  = float(funding_duration_years * 365)

input_features = {
    "log_funding":             log_funding,
    "funding_rounds":          float(funding_rounds),
    "funding_velocity":        funding_velocity,
    "startup_age_days":        startup_age_days,
    "funding_duration_days":   funding_dur_days,
    "days_since_last_funding": float(days_since_last),
    "founded_year":            float(founded_year),
    "num_categories":          float(num_categories),
    "is_africa":               float(int(is_africa)),
}

st.markdown("---")

# ── Predict button ────────────────────────────────────────────────────────────
predict_btn = st.button("🔮  Predict Survival", type="primary", use_container_width=True)

if predict_btn:
    with st.spinner("Running prediction…"):
        result = predict_survival(models, input_features)

    probs      = result["probs"]
    prediction = result["prediction"]
    confidence = result["confidence"]

    st.markdown("---")
    st.markdown('<div class="section-header">📈 Prediction Results</div>', unsafe_allow_html=True)

    # ── Verdict banner ────────────────────────────────────────────────────────
    if prediction == "operating":
        st.markdown(f"""
        <div class="verdict-safe">
            <b>✅ Likely to Continue Operating</b><br/>
            Confidence: <b>{confidence*100:.1f}%</b> — This startup profile resembles
            those that sustain operations in the historical data.
        </div>
        """, unsafe_allow_html=True)
    elif prediction == "closed":
        st.markdown(f"""
        <div class="verdict-risk">
            <b>⚠️ Elevated Closure Risk</b><br/>
            Confidence: <b>{confidence*100:.1f}%</b> — This profile shows patterns
            associated with startups that closed in the historical data.
        </div>
        """, unsafe_allow_html=True)
    elif prediction == "acquired":
        st.markdown(f"""
        <div class="verdict-safe">
            <b>🤝 Acquisition Candidate</b><br/>
            Confidence: <b>{confidence*100:.1f}%</b> — This profile resembles
            startups that were acquired by larger companies.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-uncertain">
            <b>📈 IPO Potential</b><br/>
            Confidence: <b>{confidence*100:.1f}%</b> — This profile shows
            characteristics associated with IPO-bound startups (rare outcome).
        </div>
        """, unsafe_allow_html=True)

    # ── Probability breakdown ─────────────────────────────────────────────────
    col_prob, col_bar = st.columns([1, 1.4])

    with col_prob:
        st.markdown("**Probability by Outcome**")
        for cls in CLASS_NAMES:
            p = probs[cls]
            bar_width = int(p * 100)
            color = CLASS_COLOURS[cls]
            marker = " ◀ predicted" if cls == prediction else ""
            st.markdown(f"""
            <div style="margin:6px 0;">
                <div style="display:flex;justify-content:space-between;margin-bottom:2px;">
                    <span style="font-weight:{'700' if cls==prediction else '400'};
                                 color:{'#0D1B3E' if cls==prediction else '#475569'};">
                        {cls.capitalize()}{marker}
                    </span>
                    <span style="font-weight:700;color:{color};">{p*100:.1f}%</span>
                </div>
                <div style="background:#E2E8F0;border-radius:4px;height:10px;">
                    <div style="background:{color};width:{bar_width}%;
                                height:10px;border-radius:4px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_bar:
        st.markdown("**Probability Chart**")
        fig, ax = plt.subplots(figsize=(5, 3))
        colors = [CLASS_COLOURS[c] for c in CLASS_NAMES]
        prob_vals = [probs[c] for c in CLASS_NAMES]
        bars = ax.bar(CLASS_NAMES, prob_vals, color=colors, edgecolor="none", width=0.5)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title("Survival Probability by Class")
        for bar, val in zip(bars, prob_vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.015,
                    f"{val*100:.1f}%", ha="center", fontsize=9, fontweight="bold")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        # Highlight predicted class
        pred_idx = CLASS_NAMES.index(prediction)
        bars[pred_idx].set_edgecolor("#0D1B3E")
        bars[pred_idx].set_linewidth(2)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Key drivers for this prediction ──────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">🧠 Key Drivers of This Prediction</div>',
                unsafe_allow_html=True)
    st.caption("Based on global SHAP feature importance from the tuned XGBoost model.")

    shap_ranks = [
        ("Days Since Last Funding", days_since_last,
         "days" , "high recency risk" if days_since_last > 730 else "recent activity"),
        ("Founded Year",            founded_year,
         "",      "early-era startup" if founded_year < 2005 else "modern startup"),
        ("Startup Age at Last Funding", startup_age_years,
         "years", "mature" if startup_age_years > 5 else "early stage"),
        ("Funding Duration",        funding_duration_years,
         "years", "sustained multi-round" if funding_duration_years > 2 else "single-round"),
        ("Log Funding (derived)",   round(log_funding, 2),
         "",      f"from ${funding_usd:,.0f} raised"),
    ]

    for feat, val, unit, note in shap_ranks:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;padding:8px 0;
                    border-bottom:1px solid #E2E8F0;">
            <div style="width:220px;font-weight:600;color:#0D1B3E;">{feat}</div>
            <div style="color:#0F7B8C;font-weight:700;">{val} {unit}</div>
            <div style="color:#64748B;font-size:0.9rem;">→ {note}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.info("""
    **⚠️ Important:** This tool provides probabilistic estimates based on historical patterns in 
    Crunchbase data (circa 2015–2016). Predictions are **not** causal or deterministic. 
    They should be used as one input among many in entrepreneurial decision-making — not as 
    a definitive verdict. The model was trained on US-dominated data and may have limited 
    accuracy for startups in underrepresented regions.
    """)