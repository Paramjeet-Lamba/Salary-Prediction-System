import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Salary Predictor", page_icon="💼", layout="centered")

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "Dataset" / "Salary_Data.csv"
MIN_COUNT = 5  # rare job-title grouping threshold, same as the notebook
MIN_WORKING_AGE = 18  # assumed earliest age someone could start accumulating experience


# ------------------------------------------------------------------
# Data loading, cleaning, feature engineering, model training
# (cached so it only runs once per session, not on every interaction)
# ------------------------------------------------------------------
@st.cache_resource
def load_and_train():
    df = pd.read_csv(DATA_PATH)

    # --- Cleaning ---
    df = df.dropna(subset=["Salary", "Years of Experience"])
    df = df.drop_duplicates()
    for col in ["Gender", "Education Level", "Job Title"]:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    def standardize_education(val):
        v = str(val).strip().lower()
        if "phd" in v or "doctor" in v:
            return "PhD"
        if "master" in v:
            return "Master's"
        if "bachelor" in v:
            return "Bachelor's"
        if "high school" in v:
            return "High School"
        return str(val).strip()

    df["Education Level"] = df["Education Level"].apply(standardize_education)

    title_counts = df["Job Title"].value_counts()
    rare_titles = title_counts[title_counts < MIN_COUNT].index
    df["Job Title Cleaned"] = df["Job Title"].apply(
        lambda t: "Other" if t in rare_titles else t
    )

    # --- Feature engineering ---
    def experience_bucket(years):
        if years < 3:
            return "Entry"
        elif years < 8:
            return "Mid"
        else:
            return "Senior"

    df["experience_level"] = df["Years of Experience"].apply(experience_bucket)
    df["senior_advanced_degree"] = (
        (df["experience_level"] == "Senior")
        & (df["Education Level"].isin(["Master's", "PhD"]))
    ).astype(int)

    # --- Build the full engineered feature matrix ---
    base_cat = pd.get_dummies(
        df[["Gender", "Education Level", "Job Title Cleaned"]], drop_first=True
    )
    exp_dummies = pd.get_dummies(
        df["experience_level"], prefix="experience_level", drop_first=True
    )
    X = pd.concat(
        [df[["Age", "Years of Experience"]], base_cat, exp_dummies,
         df[["senior_advanced_degree"]]],
        axis=1,
    )
    y = df["Salary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "R2": round(r2_score(y_test, preds), 4),
        "MAE": round(mean_absolute_error(y_test, preds), 2),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, preds)), 2),
    }

    meta = {
        "genders": sorted(df["Gender"].unique().tolist()),
        "educations": sorted(df["Education Level"].unique().tolist()),
        "job_titles": sorted(df["Job Title Cleaned"].unique().tolist()),
        "feature_columns": X.columns.tolist(),
        "age_min": int(df["Age"].min()),
        "age_max": int(df["Age"].max()),
        "exp_max": float(df["Years of Experience"].max()),
    }
    return model, metrics, meta


def build_input_row(age, gender, education, job_title, experience, meta):
    """Recreate a single-row feature vector matching the training columns exactly."""
    experience_level = "Entry" if experience < 3 else ("Mid" if experience < 8 else "Senior")
    senior_advanced_degree = int(
        experience_level == "Senior" and education in ["Master's", "PhD"]
    )

    row = {col: 0 for col in meta["feature_columns"]}
    row["Age"] = age
    row["Years of Experience"] = experience
    row["senior_advanced_degree"] = senior_advanced_degree

    gender_col = f"Gender_{gender}"
    if gender_col in row:
        row[gender_col] = 1

    edu_col = f"Education Level_{education}"
    if edu_col in row:
        row[edu_col] = 1

    title_col = f"Job Title Cleaned_{job_title}"
    if title_col in row:
        row[title_col] = 1

    exp_col = f"experience_level_{experience_level}"
    if exp_col in row:
        row[exp_col] = 1

    return pd.DataFrame([row])[meta["feature_columns"]], experience_level


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title("💼 Salary Predictor")
st.caption(
    "Linear Regression model trained on HR salary data, with engineered "
    "experience-level and seniority-education interaction features."
)

model, metrics, meta = load_and_train()

with st.sidebar:
    st.header("Model performance")
    st.metric("R²", metrics["R2"])
    st.metric("MAE", f"₹{metrics['MAE']:,.0f}")
    st.metric("RMSE", f"₹{metrics['RMSE']:,.0f}")
    st.caption("Evaluated on a held-out 20% test split.")

st.subheader("Enter candidate details")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", meta["age_min"], meta["age_max"], 30)
    gender = st.selectbox("Gender", meta["genders"])
    education = st.selectbox("Education Level", meta["educations"])
with col2:
    # Experience can never exceed the years available since MIN_WORKING_AGE,
    # so cap the slider dynamically based on the selected age. This prevents
    # physically impossible combinations (e.g. Age=25, Experience=25) from
    # ever being sent to the model, which would otherwise extrapolate wildly
    # and return a meaningless prediction.
    max_possible_exp = float(min(meta["exp_max"], max(0.0, age - MIN_WORKING_AGE)))
    default_exp = min(5.0, max_possible_exp)

    if max_possible_exp <= 0:
        st.slider("Years of Experience", 0.0, 0.1, 0.0, step=0.5, disabled=True)
        experience = 0.0
        st.caption(f"At age {age}, no work experience is possible yet.")
    else:
        experience = st.slider(
            "Years of Experience", 0.0, max_possible_exp, default_exp, step=0.5
        )
    job_title = st.selectbox("Job Title", meta["job_titles"])

if st.button("Predict Salary", type="primary"):
    X_input, exp_level = build_input_row(age, gender, education, job_title, experience, meta)
    prediction = model.predict(X_input)[0]

    st.success(f"### Estimated Salary: ₹{prediction:,.0f}")
    st.caption(
        f"Engineered experience bucket for this candidate: **{exp_level}**"
    )

st.divider()
st.caption(
    "Note: predictions are estimates from a linear model trained on a small "
    "sample dataset and should support, not replace, human compensation judgment."
))
