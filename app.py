import streamlit as st
import pandas as pd
import joblib
from sklearn.datasets import load_iris

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("saved_model/iris_knn_model.pkl")
scaler = joblib.load("saved_model/scaler.pkl")
iris = load_iris()

# -----------------------------
# Title
# -----------------------------
st.title("🌸 Iris Flower Classification")

st.write(
    "Predict the species of an Iris flower using a "
    "**K-Nearest Neighbors (KNN)** Machine Learning model."
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Project Details")

st.sidebar.write("**Algorithm:** KNN")
st.sidebar.write("**Dataset:** Iris")
st.sidebar.write("**Features:** 4")
st.sidebar.write("**Classes:** 3")

st.sidebar.subheader("📊 Model Performance")
st.sidebar.metric("Accuracy", "100%")

# -----------------------------
# User Inputs
# -----------------------------
sl = st.number_input("Sepal Length (cm)", value=5.1)
sw = st.number_input("Sepal Width (cm)", value=3.5)
pl = st.number_input("Petal Length (cm)", value=1.4)
pw = st.number_input("Petal Width (cm)", value=0.2)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Flower"):

    sample = pd.DataFrame(
        [[sl, sw, pl, pw]],
        columns=iris.feature_names
    )

    sample_scaled = scaler.transform(sample)

    prediction = model.predict(sample_scaled)

    probability = model.predict_proba(sample_scaled)

    confidence = probability.max() * 100

    flower = iris.target_names[prediction][0]

    st.success(f"🌼 Predicted Flower: **{flower.capitalize()}**")

    st.info(f"🎯 Confidence: **{confidence:.2f}%**")

    flower_info = {
        "setosa": "🌸 Setosa is known for its short petals and is easily distinguishable from the other two species.",
        "versicolor": "🌼 Versicolor has medium-sized petals and shares characteristics with both Setosa and Virginica.",
        "virginica": "🌺 Virginica has the largest petals and sepals among the three Iris species."
    }

    st.subheader("About the Predicted Flower")
    st.write(flower_info[flower])

# -----------------------------
# Footer
# -----------------------------
st.divider()

st.caption("Developed by Srinivas Vardholu | DecodeLabs AI Internship")