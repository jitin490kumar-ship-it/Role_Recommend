import streamlit as st
import joblib
import numpy as np

# ---------------- Page Settings ----------------
st.set_page_config(
    page_title="Career Recommendation System",
    page_icon="💼",
    layout="wide"
)

# ---------------- Load Model ----------------
try:
    model = joblib.load("job_role_model.pkl")
    encoder = joblib.load("label_encoder.pkl")
except Exception as e:
    st.error(f"Error Loading Model: {e}")
    st.stop()

st.title("💼 AI Career Recommendation System")
st.write("Select your skill level and click **Predict Career**.")

# ---------------- Feature Names ----------------
features = [
    "Database Fundamentals",
    "Computer Architecture",
    "Distributed Computing Systems",
    "Cyber Security",
    "Networking",
    "Software Development",
    "Programming Skills",
    "Project Management",
    "Computer Forensics Fundamentals",
    "Technical Communication",
    "AI ML",
    "Software Engineering",
    "Business Analysis",
    "Communication Skills",
    "Data Science",
    "Troubleshooting Skills",
    "Graphics Designing"
]

# ---------------- Input ----------------
col1, col2 = st.columns(2)

values = []

for i, feature in enumerate(features):

    if i % 2 == 0:
        value = col1.slider(feature, 0, 100, 50)
    else:
        value = col2.slider(feature, 0, 100, 50)

    values.append(value)

features_array = np.array([values])

# ---------------- Prediction ----------------
if st.button("🚀 Predict Career"):

    prediction = model.predict(features_array)

    role = encoder.inverse_transform(prediction)[0]

    st.success(f"### 🎯 Recommended Career: {role}")

    # Confidence Score
    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(features_array)[0]

        confidence = np.max(probabilities) * 100

        st.metric("Confidence Score", f"{confidence:.2f}%")

        # Top 3 Career Suggestions
        st.subheader("🏆 Top 3 Career Suggestions")

        top3 = np.argsort(probabilities)[-3:][::-1]

        for i, idx in enumerate(top3):

            career = encoder.inverse_transform([idx])[0]

            score = probabilities[idx] * 100

            st.write(f"**{i+1}. {career}** — {score:.2f}%")

    else:
        st.warning("This model does not support confidence score.")
