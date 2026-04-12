# Predicting Startup Survival Using Machine Learning

**BIT 2303 / SDS 2406 — Final Year Project**  
**Student:** Gedion Leslie Kweya Odera · SCT213-C002-0062/2022  
**Supervisor:** Mr. Adhola Samuel  
**Institution:** Jomo Kenyatta University of Agriculture and Technology (JKUAT)  

---

## 🚀 Live Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

> Replace the link above with your Streamlit Community Cloud URL after deployment.

---

## 📋 Project Overview

This project applies supervised machine learning to predict startup survival outcomes
using a Crunchbase-derived dataset of **66,368 global startup records**.

The system classifies startups into four outcomes:
- **Operating** — still active
- **Closed** — ceased operations  
- **Acquired** — bought by another company
- **IPO** — went public

A core contribution is the **geographic generalizability analysis**, evaluating whether
globally-trained models apply to African and Kenyan startup ecosystems.

---

## 📊 Results

| Model | ROC-AUC | F1 (macro) | Target met |
|-------|---------|------------|------------|
| Logistic Regression | 0.7711 | 0.3822 | ✗ |
| Random Forest | 0.8007 | 0.4237 | ✓ |
| XGBoost (initial) | 0.8025 | 0.4219 | ✓ |
| **XGBoost (tuned) ★** | **0.8078** | **0.4206** | **✓** |

**Top SHAP predictors:** Days since last funding · Founded year · Startup age

---

## 🗂️ Repository Structure

```
startup-survival-ml/
│
├── app.py                          # Streamlit entry point
├── utils.py                        # Shared model loading & helpers
├── requirements.txt                # Python dependencies
├── README.md
│
├── pages/
│   ├── 1_Home.py                   # Overview & dataset stats
│   ├── 2_Predictor.py              # Interactive survival predictor
│   ├── 3_Model_Performance.py      # Evaluation results
│   ├── 4_SHAP.py                   # Feature importance
│   └── 5_Africa.py                 # Geographic analysis
│
├── models/                         # Trained model files (pkl)
│   ├── model_logistic_regression.pkl
│   ├── model_random_forest.pkl
│   ├── model_xgboost.pkl
│   ├── model_xgboost_tuned.pkl
│   └── scaler.pkl
│
├── data/
│   ├── startup_model_ready.csv     # Cleaned dataset (Phase 1 output)
│   └── phase2_final_results.csv    # Model comparison results
│
└── notebooks/
    ├── Phase1_EDA_and_Data_Preprocessing.ipynb
    └── Phase2_Model_Training_and_Evaluation.ipynb
```

---

## ⚙️ Local Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/startup-survival-ml.git
cd startup-survival-ml

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

---

## ☁️ Deploy to Streamlit Community Cloud

1. Push this repository to GitHub (public)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set `app.py` as the entry point
4. Click **Deploy**

The app will install requirements automatically on first load.
Subsequent loads are faster because packages are cached.

---

## 📚 Key References

- Kim, Y., Lee, J., & Ahn, J. (2023). Predicting start-up success using ML. *Sustainability, 15*(4), 3279.
- Żbikowski, K., & Antosiuk, P. (2021). A machine learning, bias-free approach. *Information Processing & Management, 58*(4).
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS 30*.

---

## ⚠️ Disclaimer

Predictions are probabilistic estimates based on historical data (circa 2015–2016).
They are not causal or deterministic and should not be used as the sole basis for
investment or business decisions.

---

*JKUAT · School of Computing and Information Technology · April 2026*
