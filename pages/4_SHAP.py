"""Page 4 — Feature Importance (SHAP)"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import MODEL_FEATURES, FEATURE_LABELS, CLASS_NAMES, CLASS_COLOURS

st.title("🧠 Feature Importance — SHAP Analysis")
st.markdown("""
**SHAP (SHapley Additive exPlanations)** explains *why* the model makes each prediction
by measuring how much each feature pushes the output up or down.
Results are from the tuned XGBoost model evaluated on 13,274 test records.
""")

with st.expander("📖 How to read SHAP values"):
    st.markdown("""
    - **Mean |SHAP|** = average absolute impact of a feature across all predictions.
      Higher = more important overall.
    - **Positive SHAP** for a class = the feature *increases* the probability of that class.
    - **Negative SHAP** for a class = the feature *decreases* the probability of that class.
    - In the dot plots: **red dots** = high feature value, **blue dots** = low feature value.
      Their position on the x-axis shows the direction and magnitude of impact.
    """)

st.markdown("---")

# ── Global importance bar chart ───────────────────────────────────────────────
st.markdown('<div class="section-header">🌐 Global Feature Importance (All Classes)</div>',
            unsafe_allow_html=True)

shap_data = {
    "Feature": [
        "Days Since Last Funding", "Founded Year", "Startup Age (days)",
        "Funding Duration (days)", "Log Funding", "Funding Rounds",
        "Funding Velocity", "Num Categories", "Is Africa",
    ],
    "Mean |SHAP|": [0.4163, 0.3890, 0.3705, 0.3045, 0.2138, 0.2109, 0.2047, 0.2041, 0.0031],
    "Group": ["Temporal","Temporal","Temporal","Temporal",
              "Funding","Funding","Funding","Structural","Geographic"],
}
shap_df = pd.DataFrame(shap_data).sort_values("Mean |SHAP|", ascending=True)

group_colors = {
    "Temporal":    "#0F7B8C",
    "Funding":     "#E8A838",
    "Structural":  "#64748B",
    "Geographic":  "#94A3B8",
}

col_chart, col_insight = st.columns([1.5, 1])

with col_chart:
    fig, ax = plt.subplots(figsize=(7, 5))
    bar_colors = [group_colors[g] for g in shap_df["Group"]]
    ax.barh(shap_df["Feature"], shap_df["Mean |SHAP|"], color=bar_colors, edgecolor="none")
    ax.set_xlabel("Mean |SHAP value| (global importance)")
    ax.set_title("Global Feature Importance — XGBoost (SHAP)", fontsize=11)
    # Legend
    handles = [plt.Rectangle((0,0),1,1, color=c, label=l)
               for l, c in group_colors.items()]
    ax.legend(handles=handles, fontsize=9, loc="lower right")
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_insight:
    st.markdown("""
    **Key finding — temporal features dominate:**

    The **top 4 most important features** are all time-based signals
    engineered in Phase 1 — not raw columns from the dataset.

    | Rank | Feature | SHAP |
    |------|---------|------|
    | 🥇 1 | Days since last funding | 0.416 |
    | 🥈 2 | Founded year | 0.389 |
    | 🥉 3 | Startup age | 0.371 |
    | 4 | Funding duration | 0.305 |
    | 5 | Log funding | 0.214 |

    This directly validates **Hypothesis H2** — incorporating
    temporal features that prior literature underutilised
    produces a meaningful improvement in prediction.

    **Is Africa** has near-zero importance (0.003), reflecting
    the severe underrepresentation of African records (only
    262 out of 66,368) rather than Africa being unimportant
    as a context.
    """)

st.markdown("---")

# ── Per-class SHAP interpretation (text-based, no saved images needed) ───────
st.markdown('<div class="section-header">📐 Per-Class Feature Impact</div>',
            unsafe_allow_html=True)
st.markdown("What each feature means for each outcome class:")

tab_op, tab_cl, tab_ac, tab_ipo = st.tabs(
    ["🔵 Operating", "🔴 Closed", "🟢 Acquired", "🟠 IPO"]
)

with tab_op:
    st.markdown("""
    **Features that push toward "Operating":**
    - **Low `days_since_last_funding`** — recently funded startups are far more likely to be operating
    - **Higher `funding_rounds`** — multiple rounds signal sustained investor confidence
    - **Later `founded_year`** — more recently founded startups are still in growth phase
    - **Higher `log_funding`** — larger raises correlate with operational stability

    **Features that push away from "Operating":**
    - **Very high `days_since_last_funding`** — long gaps without new funding are a warning sign
    - **Very early `founded_year`** — startups from pre-2000 era are more likely closed or acquired by now
    """)

with tab_cl:
    st.markdown("""
    **Features that push toward "Closed":**
    - **High `days_since_last_funding`** — the single strongest closure signal; if a startup
      has gone 2+ years without follow-on investment, closure risk is elevated sharply
    - **Low `funding_rounds`** (1 round only) — single-round startups close at much higher rates
    - **Low `log_funding`** — startups that raised very little total funding
    - **Short `funding_duration_days`** — no sustained multi-round activity

    **Practical insight for investors:**
    Monitor portfolio startups' time since last funding closely — this is the earliest-available
    signal of financial distress, preceding formal closure by months or years.
    """)

with tab_ac:
    st.markdown("""
    **Features that push toward "Acquired":**
    - **Moderate-to-high `log_funding`** — acquired startups typically raised meaningful capital
      (making them attractive targets) but not IPO-scale amounts
    - **Moderate `funding_rounds`** (2–4 rounds) — enough validation without over-capitalisation
    - **Specific `founded_year` range** — acquisitions peaked for 2008–2014 cohorts
    - **Moderate `startup_age_days`** — acquired startups tend to be neither too new nor too old

    **Note:** Acquisition is hard to separate from "operating" in this dataset because many
    acquisitions are not separately tracked — some "operating" labels may be acquired startups
    whose profiles were not updated.
    """)

with tab_ipo:
    st.markdown("""
    **Features that push toward "IPO":**
    - **Very high `log_funding`** — IPO-bound startups raised significantly more capital
      (median well above $10M)
    - **High `funding_rounds`** — IPO companies typically went through 5+ rounds
    - **High `funding_velocity`** — large raises per round signal institutional-grade investment
    - **Long `funding_duration_days`** — multi-year sustained funding campaigns

    **Note:** IPO is the rarest class (2.3% of dataset) and the hardest to predict reliably.
    The model achieves recall of 0.59 for IPO with the tuned XGBoost, meaning it catches
    roughly 6 in 10 actual IPO cases — but with low precision (0.14), meaning many false positives.
    This is an inherent limitation of the dataset rather than a modelling failure.
    """)

st.markdown("---")

# ── Feature engineering credit ────────────────────────────────────────────────
st.markdown('<div class="section-header">⚙️ How the Top Features Were Created</div>',
            unsafe_allow_html=True)

eng_data = pd.DataFrame({
    "Feature": [
        "days_since_last_funding",
        "startup_age_days",
        "funding_duration_days",
        "funding_velocity",
        "log_funding",
    ],
    "Formula": [
        "Reference date (2016-01-01) − last_funding_at",
        "last_funding_at − founded_at",
        "last_funding_at − first_funding_at",
        "funding_total_usd ÷ funding_rounds",
        "log(1 + funding_total_usd)",
    ],
    "Why": [
        "Recency of investment — most powerful survival signal",
        "Maturity at last funding — older = more stable",
        "Length of sustained funding activity",
        "Average raise per round — size of institutional bets",
        "Compresses extreme right skew in funding amounts",
    ],
})
st.dataframe(eng_data, use_container_width=True, hide_index=True)
st.caption("""
All five of these features were derived from existing date and funding columns in Phase 1.
None of them existed in the raw Crunchbase dataset — they are all engineered signals.
The fact that they dominate the SHAP ranking validates the feature engineering approach.
""")