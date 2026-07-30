# 💼 Salary Prediction using Age, Education, Job Title & Experience

**HR Analytics Project** :- https://salary-prediction-system-pl.streamlit.app

A Linear Regression model that predicts candidate salary from Age, Gender, Education Level, Job Title, and Years of Experience — built to support consistent, bias-aware compensation decisions during hiring, with a strong focus on feature engineering.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![pandas](https://img.shields.io/badge/pandas-data%20wrangling-150458)
![scikit--learn](https://img.shields.io/badge/scikit--learn-LinearRegression-F7931E)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📌 Problem Statement

An HR analytics team wants to estimate a fair salary for a candidate based on their age, education level, job title, and years of experience.

## 🎯 Business Objective

Predict salary from candidate attributes to support consistent, bias-aware compensation decisions during hiring.

## 🗂️ Dataset

| | |
|---|---|
| **File** | `Dataset/Salary_Data.csv` |
| **Rows (after cleaning)** | 19,958 (from 20,160 raw rows) |
| **Columns** | `Age`, `Gender`, `Education Level`, `Job Title`, `Years of Experience`, `Salary` |
| **Age range** | 19 – 53 |
| **Years of Experience range** | 0 – 30 |
| **Salary range** | ₹21,800 – ₹309,900 |

## 🔄 Project Workflow

```
Data Collection → Data Cleaning → EDA → Feature Engineering → Model Building → Evaluation → GitHub Docs
```

## 🧹 Data Cleaning

- Dropped **100 fully-blank rows** (missing across every column).
- Removed **201 duplicate rows**.
- Verified `Education Level` categories were already consistent (`Bachelor's`, `Master's`, `PhD` — no free-text variants to standardize).
- Checked `Job Title` cardinality: **90 unique titles**, all occurring **5+ times**, so no rare-title grouping into `Other` was needed for this dataset.
- Net result: **20,160 → 19,958 rows** (202 rows removed, ~1.0% of the raw data).

## 📊 Exploratory Data Analysis

| Chart | Preview |
|---|---|
| Distribution of Salary | `Images/01_salary_distribution.png` |
| Avg. Salary by Education Level & Gender | `Images/02_salary_by_education_gender.png` |
| Years of Experience vs Salary | `Images/03_experience_vs_salary.png` |
| Top 10 Highest-Paying Job Titles | `Images/04_top_paying_job_titles.png` |

<p align="center">
  <img src="Images/01_salary_distribution.png" width="45%" />
  <img src="Images/03_experience_vs_salary.png" width="45%" />
</p>

**Key findings:**
- Years of Experience is very strongly correlated with Salary (r ≈ 0.98), and Age closely follows (r ≈ 0.96).
- Average salary rises clearly with education: Bachelor's ≈ ₹84.7K → Master's ≈ ₹106.9K → PhD ≈ ₹134.8K.
- Director/VP/Chief-level titles (e.g. Director of Software Development, VP of Data & Analytics, CISO) dominate the top-paying roles, averaging ₹225K–₹239K.
- The average salary gap between genders in this dataset is small (Male ≈ ₹100.2K vs Female ≈ ₹99.8K).

## 🛠️ Feature Engineering *(Additional Requirement)*

Two new features were engineered on top of the baseline feature set:

1. **`experience_level`** — buckets `Years of Experience` into `Entry` (<3 yrs), `Mid` (3–7 yrs), `Senior` (8+ yrs), one-hot encoded.
2. **`senior_advanced_degree`** — a binary interaction flag = 1 when a candidate is **both** `Senior` **and** holds a Master's/PhD, capturing a compounding salary premium that neither raw feature represents alone.

`Gender`, `Education Level`, and `Job Title` are one-hot encoded identically for both models so the comparison isolates the effect of the two engineered features.

> **Note:** An `age_experience_ratio` feature (Age ÷ Years of Experience) was also tested but did not improve performance once `Job Title` was already one-hot encoded, so it was excluded — a realistic reminder that not every engineered feature adds value once strong categorical signal already exists in the model.

## 🤖 Model Building

| | |
|---|---|
| **Features (X)** | Age, Years of Experience, one-hot Gender / Education Level / Job Title (+ engineered features for Model 2) |
| **Target (y)** | Salary |
| **Split** | 80/20 train/test, `random_state = 42` (identical for both models) |
| **Model 1** | Baseline Linear Regression |
| **Model 2** | Linear Regression + `experience_level` + `senior_advanced_degree` |

## 📈 Results

| Model | R² | MAE | RMSE |
|---|---|---|---|
| Baseline (no engineered features) | 0.9875 | 4,649.34 | 6,316.05 |
| **Feature-Engineered** | **0.9899** | **4,129.01** | **5,666.43** |

**Improvement from feature engineering:**
- R² ↑ **+0.0024**
- MAE ↓ **520.33**
- RMSE ↓ **649.61**

The feature-engineered model outperforms the baseline on **all three metrics** on the same held-out test set.

<p align="center">
  <img src="Images/06_r2_comparison.png" width="45%" />
  <img src="Images/07_predicted_vs_actual.png" width="45%" />
</p>

Inspecting model coefficients, `experience_level_Senior` (+₹10,058) carries the largest coefficient magnitude among the engineered terms, followed by `senior_advanced_degree` (+₹8,034) and `experience_level_Mid` (+₹5,527) — the coarse career-stage bucket contributes more than the interaction term, though both help capture non-linear salary progression that raw `Years of Experience` alone misses. Job-title dummies for senior leadership roles (Director/VP/Chief-level) remain the single largest coefficients overall, reflecting how strongly title drives pay in this dataset.

## 📁 Project Structure

```
AIML-Project-RollNo-2302221530074/
├── Dataset/
│   └── Salary_Data.csv
├── Notebook/
│   └── Salary_Prediction.ipynb
├── Images/
│   └── (7 EDA / evaluation charts)
└── README.md
│   
└──  app.py
│   
└── requirement.txt
```

## 🧰 Tech Stack

`pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn` (LinearRegression)

## 📄 License

This project is released under the [MIT License](LICENSE).
