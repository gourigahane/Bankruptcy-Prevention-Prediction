

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "Bankruptcy_Prevention.pkl"

FEATURES = [
    "industrial_risk",
    "management_risk",
    "financial_flexibility",
    "credibility",
    "competitiveness",
    "operating_risk",
]

FEATURE_LABELS = {
    "industrial_risk": "Industrial Risk",
    "management_risk": "Management Risk",
    "financial_flexibility": "Financial Flexibility",
    "credibility": "Credibility",
    "competitiveness": "Competitiveness",
    "operating_risk": "Operating Risk",
}

# Each feature in the original dataset only takes these three values.
# Map a clean display label -> the actual float value the model expects.
LEVEL_OPTIONS = {"0": 0.0, "0.5": 0.5, "1": 1.0}


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(
            f"Could not find '{MODEL_PATH.name}'. Run `python train_model.py` "
            "first, or place the pickle file next to app.py."
        )
        st.stop()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def main():
    st.set_page_config(page_title="Bankruptcy Prevention Prediction", page_icon="🏦", layout="centered")

    st.title("🏦 Bankruptcy Prevention Prediction")
    st.write(
        "Enter the required business risk factors below to predict whether "
        "the company is likely to be bankrupt."
    )

    model = load_model()

    inputs = {}
    for feat in FEATURES:
        choice = st.selectbox(
            FEATURE_LABELS[feat],
            options=list(LEVEL_OPTIONS.keys()),
            index=0,
            key=feat,
        )
        inputs[feat] = LEVEL_OPTIONS[choice]

    st.write("")
    if st.button("Predict", type="primary"):
        X = pd.DataFrame([inputs], columns=FEATURES)
        prediction = model.predict(X)[0]

        st.divider()
        if prediction == "bankruptcy":
            st.error("### ⚠️ The company is likely to go BANKRUPT")
        else:
            st.success("### ✅ The company is likely to be NON-BANKRUPT")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            classes = list(model.classes_)
            prob_df = pd.DataFrame({"Class": classes, "Probability": proba}).set_index("Class")
            st.bar_chart(prob_df)


if __name__ == "__main__":
    main()
