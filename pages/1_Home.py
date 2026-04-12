"""Page 1 — Home & Overview"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_data, load_results, CLASS_COLOURS, style_metric

st.title("🚀 Predicting Startup Survival Using Machine Learning")
st.markdown("""
> **What this tool does:** Using a Crunchbase-derived dataset of 66,368 global startup records,
> we trained machine learning models to predict whether a startup will continue operating,
> close down, get acquired, or reach an IPO — based on its funding history, age, and sector.
> The interactive predictor lets you explore survival probabilities for any startup profile.
""")

st.markdown("---")

# ── Key numbers ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📌 Dataset at a Glance</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records",   "66,368",  "Crunchbase-derived")
col2.metric("Countries",       "137",     "Global coverage")
col3.metric("Best ROC-AUC",    "0.8078",  "Tuned XGBoost ✓")
col4.metric("Top Predictor",   "Days Since\nLast Funding", "SHAP #1 feature")

st.markdown("---")

# ── Status distribution ───────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Startup Status Distribution</div>', unsafe_allow_html=True)

col_chart, col_text = st.columns([1.4, 1])

with col_chart:
    try:
        df = load_data()
        counts = df["status"].value_counts()
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

        # Bar
        colors = [CLASS_COLOURS[s] for s in counts.index]
        axes[0].bar(counts.index, counts.values, color=colors, edgecolor="none", width=0.55)
        axes[0].set_title("Count by Status", fontsize=11)
        axes[0].set_ylabel("Count")
        for i, v in enumerate(counts.values):
            axes[0].text(i, v + 100, f"{v:,}\n({v/len(df)*100:.1f}%)",
                         ha="center", fontsize=8)
        for spine in ["top", "right"]:
            axes[0].spines[spine].set_visible(False)

        # Pie
        axes[1].pie(
            counts.values,
            labels=counts.index,
            colors=colors,
            autopct="%1.1f%%",
            startangle=140,
            pctdistance=0.78,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        )
        axes[1].set_title("Proportional Share", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.info(f"Chart unavailable: {e}")

with col_text:
    st.markdown("""
    **Key observations:**

    - **79.9%** of startups are still operating — severe class imbalance that the model must handle carefully.
    - Only **9.4%** of startups are confirmed closed, making closure the critical minority class.
    - **IPO** is the rarest outcome at 2.3%, making it the hardest class to predict reliably.

    **What this means for modelling:**
    SMOTE oversampling was applied to the training set to balance classes before training.
    ROC-AUC is used as the primary metric — it is robust to class imbalance.
    """)

st.markdown("---")

# ── Model results summary ─────────────────────────────────────────────────────
st.markdown('<div class="section-header">🏆 Model Performance Summary</div>', unsafe_allow_html=True)

try:
    results = load_results()
    results["Target met"] = results["ROC-AUC"].apply(lambda x: "✓" if x >= 0.80 else "✗")

    def highlight(row):
        if "tuned" in str(row["Model"]).lower():
            return ["background-color: #E8F4F7"] * len(row)
        return [""] * len(row)

    st.dataframe(
        results.style.apply(highlight, axis=1).format({"ROC-AUC": "{:.4f}", "F1-macro": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("★ Tuned XGBoost is the model used in the Survival Predictor. Target: ROC-AUC ≥ 0.80.")
except Exception as e:
    st.info(f"Results table unavailable: {e}")

st.markdown("---")

# ── SHAP top features teaser ──────────────────────────────────────────────────
st.markdown('<div class="section-header">🧠 What Predicts Survival?</div>', unsafe_allow_html=True)

shap_data = {
    "Feature":     ["Days Since Last Funding", "Founded Year", "Startup Age (days)",
                    "Funding Duration (days)", "Log Funding", "Funding Rounds",
                    "Funding Velocity", "Num Categories", "Is Africa"],
    "Mean |SHAP|": [0.4163, 0.3890, 0.3705, 0.3045, 0.2138, 0.2109, 0.2047, 0.2041, 0.0031],
    "Rank":        ["#1 🥇", "#2 🥈", "#3 🥉", "#4", "#5", "#6", "#7", "#8", "#9"],
}
shap_df = pd.DataFrame(shap_data)

col_shap, col_note = st.columns([1.5, 1])
with col_shap:
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#0F7B8C" if i < 3 else "#E8A838" if i < 6 else "#94A3B8"
              for i in range(len(shap_df))]
    ax.barh(shap_df["Feature"][::-1], shap_df["Mean |SHAP|"][::-1],
            color=colors[::-1], edgecolor="none")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global Feature Importance (SHAP)", fontsize=11)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_note:
    st.markdown("""
    **The three most important predictors are all temporal features** — engineered from dates rather than taken directly from the dataset.

    This confirms the project's key hypothesis (**H2**):
    incorporating temporal funding signals significantly improves predictive performance over raw structural features alone.

    - **Days since last funding** — how recently a startup received investment is the single strongest survival signal.
    - **Founded year** — the economic era a startup was born in matters.
    - **Startup age** — older startups at the time of last funding are more stable.

    Raw funding amount ranks only 5th.
    """)

st.markdown("---")
st.caption("Data source: Crunchbase-derived dataset via Kaggle · Model: XGBoost (tuned, ROC-AUC 0.8078) · JKUAT Final Year Project 2026")