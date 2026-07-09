import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "student_performance_model.pkl"
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()



st.title("🎓 Student Performance Prediction System")

st.markdown(
    """
    Predict a student's expected final exam score
    using academic and personal factors.
    """
)

st.divider()




with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            options=[
                "Male",
                "Female"
            ]
        )

        study_hours = st.number_input(
            "Study Hours per Week",
            min_value=0.0,
            max_value=100.0,
            value=20.0,
            step=1.0
        )

        attendance = st.slider(
            "Attendance Rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
            step=0.5
        )

        past_scores = st.slider(
            "Past Exam Score",
            min_value=0,
            max_value=100,
            value=70
        )

    with col2:

        parent_education = st.selectbox(
            "Parental Education Level",
            options=[
                "High School",
                "Bachelor",
                "Master",
                "PhD"
            ]
        )

        internet_access = st.selectbox(
            "Internet Access at Home",
            options=[
                "Yes",
                "No"
            ]
        )

        extracurricular = st.selectbox(
            "Extracurricular Activities",
            options=[
                "Yes",
                "No"
            ]
        )

    submitted = st.form_submit_button(
        "Predict Final Exam Score",
        use_container_width=True
    )


# -----------------------------
# Prediction
# -----------------------------

if submitted:

    input_data = pd.DataFrame({
        "Gender": [gender],
        "Study_Hours_per_Week": [study_hours],
        "Attendance_Rate": [attendance],
        "Past_Exam_Scores": [past_scores],
        "Parental_Education_Level": [
            parent_education
        ],
        "Internet_Access_at_Home": [
            internet_access
        ],
        "Extracurricular_Activities": [
            extracurricular
        ]
    })

    raw_prediction = model.predict(
        input_data
    )[0]

    # Keep displayed score in valid range
    predicted_score = max(
        0.0,
        min(100.0, raw_prediction)
    )

    st.divider()

    st.subheader("Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.metric(
            label="Predicted Final Exam Score",
            value=f"{predicted_score:.2f} / 100"
        )

    with result_col2:

        if predicted_score >= 85:
            performance = "Excellent"

        elif predicted_score >= 70:
            performance = "Good"

        elif predicted_score >= 50:
            performance = "Average"

        else:
            performance = "Needs Support"

        st.metric(
            label="Performance Level",
            value=performance
        )

    st.progress(
        int(round(predicted_score))
    )

    if predicted_score >= 85:

        st.success(
            "Excellent predicted academic performance."
        )

    elif predicted_score >= 70:

        st.info(
            "Good predicted academic performance."
        )

    elif predicted_score >= 50:

        st.warning(
            "Average predicted performance. "
            "Additional academic improvement may help."
        )

    else:

        st.error(
            "The student may benefit from "
            "additional academic support."
        )


# -----------------------------
# Project information
# -----------------------------

st.divider()

with st.expander("About this project"):

    st.markdown(
        """
        This machine learning application predicts
        student final exam performance.

        **Selected Model:** Linear Regression

        **Test-set Performance:**

        - MAE: 3.0669
        - MSE: 14.4693
        - RMSE: 3.8039
        - R² Score: 0.5625

        The model was selected after comparison with:

        - Decision Tree Regressor
        - Random Forest Regressor
        """
    )

st.divider()

st.caption(
    "Machine Learning Internship Project"
)