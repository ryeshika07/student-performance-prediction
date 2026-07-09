from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)


# -----------------------------------
# Project paths
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "student_performance_cleaned.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "student_performance_model.pkl"
)


# -----------------------------------
# Load dataset
# -----------------------------------

print("Loading dataset from:")
print(DATA_PATH)

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# -----------------------------------
# Define features and target
# -----------------------------------

X = df.drop(
    columns=[
        "Student_ID",
        "Final_Exam_Score",
        "Pass_Fail"
    ]
)

y = df["Final_Exam_Score"]


# -----------------------------------
# Train-test split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------------
# Feature groups
# -----------------------------------

numerical_features = [
    "Study_Hours_per_Week",
    "Attendance_Rate",
    "Past_Exam_Scores"
]

categorical_features = [
    "Gender",
    "Parental_Education_Level",
    "Internet_Access_at_Home",
    "Extracurricular_Activities"
]


# -----------------------------------
# Numerical pipeline
# -----------------------------------

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


# -----------------------------------
# Categorical pipeline
# -----------------------------------

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


# -----------------------------------
# Combined preprocessor
# -----------------------------------

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


# -----------------------------------
# Complete ML pipeline
# -----------------------------------

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


# -----------------------------------
# Train model
# -----------------------------------

print("\nTraining fresh model...")

model.fit(
    X_train,
    y_train
)

print("Model trained successfully.")


# -----------------------------------
# Evaluate model
# -----------------------------------

y_pred = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = mse ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)

print("\nModel Performance")
print("-" * 30)
print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")


# -----------------------------------
# Save fresh model
# -----------------------------------

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_PATH
)

print("\nFresh model saved successfully.")
print("Model path:")
print(MODEL_PATH)


# -----------------------------------
# Immediate reload test
# -----------------------------------

print("\nTesting saved model reload...")

loaded_model = joblib.load(
    MODEL_PATH
)

print("Model reloaded successfully.")


# -----------------------------------
# Prediction test
# -----------------------------------

test_input = X_test.head(1)

test_prediction = loaded_model.predict(
    test_input
)

print(
    "Test prediction:",
    test_prediction[0]
)

print("\nMODEL LOAD OK")