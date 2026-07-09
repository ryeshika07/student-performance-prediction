import streamlit as st
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "student_performance_cleaned.csv"
)


# =========================================================
# TRAIN MODEL
# =========================================================

@st.cache_resource
def train_model():

    # Load cleaned dataset
    df = pd.read_csv(DATA_PATH)

    # Define features
    X = df.drop(
        columns=[
            "Student_ID",
            "Final_Exam_Score",
            "Pass_Fail"
        ]
    )

    # Define target
    y = df["Final_Exam_Score"]

    # Numerical features
    numerical_features = [
        "Study_Hours_per_Week",
        "Attendance_Rate",
        "Past_Exam_Scores"
    ]

    # Categorical features
    categorical_features = [
        "Gender",
        "Parental_Education_Level",
        "Internet_Access_at_Home",
        "Extracurricular_Activities"
    ]

    # Numerical preprocessing
    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # Categorical preprocessing
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    # Combined preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                numerical_features
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    # Complete ML pipeline
    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                LinearRegression()
            )
        ]
    )

    # Train model
    model.fit(X, y)

    return model


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

try:
    model = train_model()

except FileNotFoundError:

    st.error(
        "Dataset file not found. Expected location: "
        "data/student_performance_cleaned.csv"
    )

    st.stop()

except Exception as error:

    st.error(
        "Unable to train the prediction model."
    )

    st.exception(error)

    st.stop()


# =========================================================
# APPLICATION HEADER
# =========================================================

st.title(
    "🎓 Student Performance Prediction System"
)

st.markdown(
    """
    Predict a student's expected final exam score
    using academic and personal factors.
    """
)

st.divider()


# =========================================================
# INPUT FORM
# =========================================================

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    # -------------------------
    # LEFT COLUMN
    # -------------------------

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

    # -------------------------
    # RIGHT COLUMN
    # -------------------------

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


# =========================================================
# PREDICTION
# =========================================================

if submitted:

    # Create input dataframe
    input_data = pd.DataFrame({
        "Gender": [
            gender
        ],
        "Study_Hours_per_Week": [
            study_hours
        ],
        "Attendance_Rate": [
            attendance
        ],
        "Past_Exam_Scores": [
            past_scores
        ],
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

    try:

        # Generate prediction
        raw_prediction = model.predict(
            input_data
        )[0]

        # Keep displayed score within 0-100
        predicted_score = max(
            0.0,
            min(
                100.0,
                float(raw_prediction)
            )
        )

        # -------------------------
        # PERFORMANCE CATEGORY
        # -------------------------

        if predicted_score >= 85:

            performance = "Excellent"

        elif predicted_score >= 70:

            performance = "Good"

        elif predicted_score >= 50:

            performance = "Average"

        else:

            performance = "Needs Support"

        # -------------------------
        # DISPLAY RESULT
        # -------------------------

        st.divider()

        st.subheader(
            "Prediction Result"
        )

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.metric(
                label="Predicted Final Exam Score",
                value=(
                    f"{predicted_score:.2f} / 100"
                )
            )

        with result_col2:

            st.metric(
                label="Performance Level",
                value=performance
            )

        # Progress bar
        st.progress(
            int(
                round(predicted_score)
            )
        )

        # -------------------------
        # PERFORMANCE MESSAGE
        # -------------------------

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

    except Exception as error:

        st.error(
            "An error occurred while generating "
            "the prediction."
        )

        st.exception(error)


# =========================================================
# PROJECT INFORMATION
# =========================================================

st.divider()

with st.expander(
    "About this project"
):

    st.markdown(
        """
        This machine learning application predicts
        student final exam performance using academic
        and personal factors.

        **Selected Model:** Linear Regression

        **Baseline Test-set Performance:**

        - MAE: 3.0669
        - MSE: 14.4693
        - RMSE: 3.8039
        - R² Score: 0.5625

        The model was selected after comparison with:

        - Decision Tree Regressor
        - Random Forest Regressor

        The deployed application reconstructs and trains
        the preprocessing and Linear Regression pipeline
        from the cleaned project dataset when the app
        starts. The trained model is cached for subsequent
        predictions.
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Machine Learning Internship Project"
)