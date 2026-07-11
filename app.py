
import streamlit as st
import joblib
import numpy as np

# Load Model
model = joblib.load("job_role_model.pkl")
encoder = joblib.load("label_encoder.pkl")

st.title("💼 Job Role Prediction")
st.write("Adjust the sliders to predict the suitable job role.")

# Input Sliders
db = st.slider("Database Fundamentals", 0, 100, 50)
ca = st.slider("Computer Architecture", 0, 100, 50)
dcs = st.slider("Distributed Computing Systems", 0, 100, 50)
cs = st.slider("Cyber Security", 0, 100, 50)
net = st.slider("Networking", 0, 100, 50)
sd = st.slider("Software Development", 0, 100, 50)
ps = st.slider("Programming Skills", 0, 100, 50)
pm = st.slider("Project Management", 0, 100, 50)
cff = st.slider("Computer Forensics Fundamentals", 0, 100, 50)
tc = st.slider("Technical Communication", 0, 100, 50)
aiml = st.slider("AI ML", 0, 100, 50)
se = st.slider("Software Engineering", 0, 100, 50)
ba = st.slider("Business Analysis", 0, 100, 50)
comm = st.slider("Communication Skills", 0, 100, 50)
ds = st.slider("Data Science", 0, 100, 50)
ts = st.slider("Troubleshooting Skills", 0, 100, 50)
gd = st.slider("Graphics Designing", 0, 100, 50)

# Create Feature Array
features = np.array([[db, ca, dcs, cs, net, sd, ps, pm,
                      cff, tc, aiml, se, ba, comm, ds, ts, gd]])

# Prediction
if st.button("Predict"):
    prediction = model.predict(features)
    role = encoder.inverse_transform(prediction)

    st.success(f"🎯 Predicted Job Role: {role[0]}")
