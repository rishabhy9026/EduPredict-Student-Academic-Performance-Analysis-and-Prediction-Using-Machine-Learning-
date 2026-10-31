"""EduPredict: Streamlit app that predicts a student's expected exam score.

Run with:  streamlit run app.py
It loads the trained pipeline saved by the notebook (models/best_model.pkl).
The pipeline already contains the imputer, scaler/encoder and the model, so the
same preprocessing used in training is applied to the values entered here.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "models" / "best_model.pkl"
META_PATH = APP_DIR / "models" / "model_metadata.json"

# Allowed input limits (what is physically/logically valid)
VALID_LIMITS = {
    "Hours_Studied": (0, 80),        # hours per week
    "Attendance": (0, 100),          # percent
    "Sleep_Hours": (0, 12),          # hours per night
    "Previous_Scores": (0, 100),
    "Tutoring_Sessions": (0, 15),    # per month
    "Physical_Activity": (0, 10),    # hours per week
}


@st.cache_resource
def load_model_and_metadata():
    """Load the saved best model and its metadata once."""
    model = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        metadata = json.load(f)
    return model, metadata


def validate_inputs(values, trained_ranges):
    """Return (errors, warnings). Errors block the prediction; warnings do not."""
    errors, warnings = [], []
    for column, (low, high) in VALID_LIMITS.items():
        if not low <= values[column] <= high:
            errors.append(f"{column.replace('_', ' ')} must be between {low} and {high}.")
    for column, (low, high) in trained_ranges.items():
        if VALID_LIMITS[column][0] <= values[column] <= VALID_LIMITS[column][1] and not low <= values[column] <= high:
            warnings.append(f"{column.replace('_', ' ')} = {values[column]} is outside the range the model was trained on "
                            f"({low} to {high}), so the prediction is less reliable.")
    return errors, warnings


def predict_score(model, values, feature_order):
    """Build a one-row table in the training column order and predict (limited to 0-100)."""
    row = pd.DataFrame([values])[feature_order]
    prediction = float(model.predict(row)[0])
    return max(0.0, min(100.0, prediction))


def main():
    st.set_page_config(page_title="EduPredict", page_icon="🎓", layout="centered")
    st.title("🎓 EduPredict")
    st.caption("Student exam score prediction using machine learning")

    try:
        model, meta = load_model_and_metadata()
    except Exception as error:
        st.error(f"Could not load the trained model. Run the notebook first to create models/best_model.pkl. ({error})")
        return

    options = meta["categorical_options"]
    ranges = meta["numeric_ranges"]
    levels = ["Low", "Medium", "High"]

    with st.sidebar:
        st.header("About")
        st.write(f"**Model:** {meta['best_model']}")
        st.write(f"**Test R²:** {meta['test_metrics']['R2']}")
        st.write(f"**Typical error (MAE):** {meta['test_metrics']['MAE']} marks")
        st.caption("Trained on the Kaggle 'Student Performance Factors' dataset. "
                   "The result is an estimate based on patterns in that data, not a guarantee.")

    st.subheader("Enter student details")
    with st.form("prediction_form"):
        st.markdown("**Study habits and academics**")
        col1, col2 = st.columns(2)
        with col1:
            hours = st.number_input("Hours studied per week", 0, 80, 20, step=1)
            sleep = st.number_input("Sleep hours per night", 0, 12, 7, step=1)
            tutoring = st.number_input("Tutoring sessions per month", 0, 15, 1, step=1)
        with col2:
            attendance = st.number_input("Attendance (%)", 0, 100, 80, step=1)
            previous = st.number_input("Previous exam score", 0, 100, 75, step=1)
            physical = st.number_input("Physical activity (hours per week)", 0, 10, 3, step=1)

        st.markdown("**Personal and environment factors**")
        col3, col4 = st.columns(2)
        with col3:
            motivation = st.selectbox("Motivation level", levels, index=1)
            involvement = st.selectbox("Parental involvement", levels, index=1)
            resources = st.selectbox("Access to resources", levels, index=1)
            income = st.selectbox("Family income", levels, index=1)
            teacher = st.selectbox("Teacher quality", levels, index=1)
            peer = st.selectbox("Peer influence", options["Peer_Influence"], index=options["Peer_Influence"].index("Neutral"))
        with col4:
            parent_edu = st.selectbox("Parental education level", options["Parental_Education_Level"],
                                      index=options["Parental_Education_Level"].index("College"))
            distance = st.selectbox("Distance from home", options["Distance_from_Home"],
                                    index=options["Distance_from_Home"].index("Near"))
            school = st.selectbox("School type", options["School_Type"])
            gender = st.selectbox("Gender", options["Gender"])
            extracurricular = st.selectbox("Extracurricular activities", options["Extracurricular_Activities"],
                                           index=options["Extracurricular_Activities"].index("Yes"))
            internet = st.selectbox("Internet access", options["Internet_Access"],
                                    index=options["Internet_Access"].index("Yes"))
            disability = st.selectbox("Learning disabilities", options["Learning_Disabilities"])

        submitted = st.form_submit_button("Predict exam score")

    if submitted:
        values = {
            "Hours_Studied": hours, "Attendance": attendance, "Parental_Involvement": involvement,
            "Access_to_Resources": resources, "Extracurricular_Activities": extracurricular,
            "Sleep_Hours": sleep, "Previous_Scores": previous, "Motivation_Level": motivation,
            "Internet_Access": internet, "Tutoring_Sessions": tutoring, "Family_Income": income,
            "Teacher_Quality": teacher, "School_Type": school, "Peer_Influence": peer,
            "Physical_Activity": physical, "Learning_Disabilities": disability,
            "Parental_Education_Level": parent_edu, "Distance_from_Home": distance, "Gender": gender,
        }
        errors, warnings = validate_inputs(values, ranges)
        if errors:
            for message in errors:
                st.error(message)
            return
        for message in warnings:
            st.warning(message)

        score = predict_score(model, values, meta["feature_order"])
        st.divider()
        st.subheader("Predicted exam score")
        st.metric(label="Expected score (out of 100)", value=f"{score:.1f}")
        st.progress(int(round(score)))
        st.caption(f"On the test data this model's average error was about {meta['test_metrics']['MAE']} marks "
                   f"(RMSE {meta['test_metrics']['RMSE']}). A small number of students score far above what their "
                   "inputs suggest, so treat this as an estimate.")


if __name__ == "__main__":
    main()
