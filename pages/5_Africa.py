"""Page 5 — Africa & Geographic Analysis"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_data, CLASS_COLOURS, CLASS_NAMES

st.title("🌍 Africa & Geographic Analysis")
st.markdown("""
A core research contribution of this project is evaluating whether a model trained on
globally-dominated data generalises to **African startup ecosystems** — particularly Kenya.
This page presents the geographic subgroup analysis from Phase 2.
""")
st.markdown("---")

# ── Global distribution ───────────────────────────────────────────────────────
st.markdown('<div class="section-header">🗺️ Geographic Distribution of the Dataset</div>',
            unsafe_allow_html=True)

country_data = {
    "Country": ["USA", "GBR", "CAN", "IND", "CHN",
                "FRA", "DEU", "ISR", "ESP", "AUS",
                "NGA", "ZAF", "KEN", "EGY", "GHA", "Other"],
    "Count":   [37601, 3688, 1925, 1596, 1544,
                1135, 1082, 965, 746, 503,
                57, 92, 39, 36, 14, 12445],
    "Region":  ["North America","Europe","North America","Asia","Asia",
                "Europe","Europe","Middle East","Europe","Oceania",
                "Africa","Africa","Africa","Africa","Africa","Various"],
}
country_df = pd.DataFrame(country_data)

col_bar, col_pie = st.columns([1.5, 1])

with col_bar:
    top15 = country_df.head(15)
    colors = ["#EF5350" if r=="Africa" else "#5C6BC0" if r=="North America"
              else "#2196F3" if r=="Europe" else "#FF9800"
              for r in top15["Region"]]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top15["Country"][::-1], top15["Count"][::-1],
            color=colors[::-1], edgecolor="none")
    ax.set_xlabel("Number of Startups")
    ax.set_title("Top Countries by Startup Count\n(Red = African countries)", fontsize=11)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_pie:
    region_counts = {
        "North America": 37601+1925,
        "Europe": 3688+1135+1082+965+746,
        "Asia": 1596+1544,
        "Africa": 57+92+39+36+14,
        "Other": 503+12445,
    }
    fig, ax = plt.subplots(figsize=(5, 4))
    region_colors = ["#5C6BC0","#2196F3","#FF9800","#EF5350","#94A3B8"]
    ax.pie(
        list(region_counts.values()),
        labels=list(region_counts.keys()),
        colors=region_colors,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.78,
        wedgeprops={"edgecolor":"white","linewidth":1.5},
    )
    ax.set_title("Dataset by Region", fontsize=11)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.caption("Africa = 0.4% of dataset. USA alone = 56.7%.")

st.markdown("---")

# ── Africa vs Global performance ─────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Model Performance — Global vs Africa</div>',
            unsafe_allow_html=True)

col_metrics, col_chart = st.columns([1, 1.4])

with col_metrics:
    st.markdown("**Tuned XGBoost on test set:**")

    perf_data = pd.DataFrame({
        "Subset":        ["Global (non-Africa)", "Africa", "All (combined)"],
        "n records":     [13223, 51, 13274],
        "ROC-AUC":       ["0.8075", "N/A *", "0.8078"],
        "F1 (macro)":    ["0.4205", "0.3162", "0.4206"],
    })
    st.dataframe(perf_data, use_container_width=True, hide_index=True)
    st.caption("""
    * ROC-AUC could not be computed for Africa because the 51-record
    African test subset contains only 2 of 4 classes (no acquired, no IPO records).
    One-vs-rest ROC-AUC requires all classes to be present.
    """)

    st.markdown("""
    **F1 gap:** Global = 0.4205 vs Africa = 0.3162

    A difference of **~0.10** in macro-F1 suggests the model
    underperforms on African records relative to global performance.
    However, with only **51 test records**, this gap is not
    statistically robust and should be interpreted cautiously.
    """)

with col_chart:
    fig, ax = plt.subplots(figsize=(6, 4))
    subsets = ["Global\n(non-Africa)", "Africa", "All\n(combined)"]
    f1_vals = [0.4205, 0.3162, 0.4206]
    colors  = ["#0F7B8C", "#EF5350", "#E8A838"]
    bars = ax.bar(subsets, f1_vals, color=colors, edgecolor="none", width=0.45)
    ax.set_ylim(0, 0.6)
    ax.set_ylabel("F1 Score (macro)")
    ax.set_title("F1 Score by Geographic Subset", fontsize=11)
    ax.axhline(0.42, color="#999", linestyle="--", linewidth=0.8, label="Global baseline")
    ax.legend(fontsize=9)
    for bar, val in zip(bars, f1_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.008,
                f"{val:.4f}", ha="center", fontsize=10, fontweight="bold")
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# ── Country-level accuracy ────────────────────────────────────────────────────
st.markdown('<div class="section-header">🇰🇪 African Country Breakdown</div>',
            unsafe_allow_html=True)

country_acc = pd.DataFrame({
    "Country":  ["ZAF (South Africa)", "NGA (Nigeria)", "KEN (Kenya)", "EGY (Egypt)",
                 "BWA (Botswana)", "GHA (Ghana)", "MAR (Morocco)", "TUN (Tunisia)",
                 "UGA (Uganda)", "ZWE (Zimbabwe)"],
    "Accuracy": [0.8235, 0.8571, 1.0000, 1.0000,
                 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
    "Test records": [17, 14, 8, 6, 1, 1, 1, 1, 1, 1],
})

col_table, col_note = st.columns([1, 1])
with col_table:
    def color_acc(val):
        if isinstance(val, float):
            if val >= 0.95: return "color: #276749; font-weight: bold"
            if val >= 0.80: return "color: #C05621; font-weight: bold"
            return "color: #9B2C2C;"
        return ""
    st.dataframe(
        country_acc.style.map(color_acc, subset=["Accuracy"])
                   .format({"Accuracy": "{:.1%}"}),
        use_container_width=True, hide_index=True,
    )

with col_note:
    st.markdown("""
    **Kenya result: 100% accuracy (8 records)**

    This is encouraging but must be interpreted carefully.
    With only 8 Kenyan test records — all of which are "operating"
    startups — predicting "operating" for every one would also
    yield 100% accuracy. The sample is far too small to draw
    meaningful conclusions about the model's Kenya-specific
    performance.

    **South Africa and Nigeria** have the most African records
    (17 and 14 respectively) and show more realistic accuracy
    rates of 82–86%, suggesting reasonable generalisation for
    the larger African ecosystems in the dataset.
    """)

st.markdown("---")

# ── Research finding ──────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📋 Research Finding — Hypothesis H4</div>',
            unsafe_allow_html=True)

col_h4a, col_h4b = st.columns(2)
with col_h4a:
    st.markdown("""
    **H4 (original):**
    *Models trained globally will exhibit measurable performance
    degradation when evaluated on African records alone,
    confirming a geographic generalizability gap.*

    **Verdict: Inconclusive — not falsified**

    The F1 gap (0.42 → 0.32) suggests degradation, but the
    ROC-AUC could not be computed (insufficient class diversity
    in the 51-record African test subset). The hypothesis
    is directionally supported but cannot be confirmed
    statistically with this sample size.
    """)

with col_h4b:
    st.markdown("""
    **Why this still matters academically:**

    The `nan` ROC-AUC is itself a finding — it shows that the
    African subset contains only "operating" and "closed" startups
    in the test set, with no acquired or IPO examples. This reflects
    the structural reality of East African startup ecosystems:
    acquisition and IPO pathways are far less common than in the US.

    **Recommendation for future work:**
    Integrate Africa-specific datasets (e.g., Disrupt Africa,
    Partech Africa reports) to enable robust generalisation testing.
    A dedicated African model trained on local data could meaningfully
    outperform the globally-trained model presented here.
    """)

st.markdown("---")

# ── Kenya context ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🇰🇪 Kenya Context</div>', unsafe_allow_html=True)

col_k1, col_k2, col_k3 = st.columns(3)
col_k1.metric("Kenyan records in dataset", "39", delta=None)
col_k2.metric("Kenyan test records", "8", delta=None)
col_k3.metric("East Africa VC share", "~10%", delta=None)
st.caption("0.06% of total dataset · All 8 test records predicted correctly · East Africa receives ~10% of continental VC flows")

st.markdown("""
Kenya's startup ecosystem — anchored by innovations like M-Pesa and supported by
initiatives such as Konza Technopolis, iHub, and Kenya Vision 2030 — has grown
substantially. Yet it remains critically underrepresented in global startup databases.

This project demonstrates both the **potential** and the **limitations** of applying
globally-trained ML models to the Kenyan context. The survival probability tool on the
Predictor page is available for use with Kenyan startup profiles — but outputs should
be interpreted as informed estimates rather than definitive predictions, given the
training data limitations described here.
""")

st.caption("African startup data sourced from Crunchbase-derived dataset. Country-level figures are from the test subset (20% of 66,368 records, stratified split).")
