import streamlit as st
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE_DIR / "toxic_nontoxic_model.pkl")
    vectorizer = joblib.load(BASE_DIR / "toxic_nontoxic_vectorizer.pkl")
    category_names = [
        "severe_toxic", "obscene", "threat", "insult", "identity_hate"
    ]
    category_models = {}
    for name in category_names:
        model_path = BASE_DIR / f"{name}_model.pkl"
        vectorizer_path = BASE_DIR / f"{name}_vectorizer.pkl"
        if model_path.exists() and vectorizer_path.exists():
            category_models[name] = (
                joblib.load(model_path),
                joblib.load(vectorizer_path)
            )
    return model, vectorizer, category_models

st.set_page_config(page_title="Toxic Comment Detection", page_icon="🛡️")
st.title("🛡️ Toxic Comment Detection")
st.write("Enter a comment to classify it as **Toxic** or **Non-Toxic**.")

try:
    model, vectorizer, category_models = load_artifacts()
except Exception as e:
    st.error("Could not load the model files. Check that all required .pkl files are in the same folder as app.py.")
    st.code(str(e))
    st.stop()

comment = st.text_area("Enter your comment:", height=160,
                       placeholder="Type or paste a comment here...")

if st.button("Analyze comment"):
    if not comment.strip():
        st.warning("Please enter a comment first.")
    else:
        features = vectorizer.transform([comment])
        prediction = int(model.predict(features)[0])
        if prediction == 1:
            st.error("⚠️ Prediction: TOXIC")
        else:
            st.success("✅ Prediction: NON-TOXIC")

        if category_models:
            st.subheader("Toxic categories")
            cols = st.columns(2)
            for idx, (name, (cat_model, cat_vectorizer)) in enumerate(category_models.items()):
                cat_features = cat_vectorizer.transform([comment])
                cat_prediction = int(cat_model.predict(cat_features)[0])
                label = "Yes" if cat_prediction == 1 else "No"
                cols[idx % 2].write(f"**{name.replace('_', ' ').title()}:** {label}")

st.caption("Educational prototype. Predictions may be incorrect; review context before acting.")
