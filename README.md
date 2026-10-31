# 🎓 EduPredict: Student Academic Performance Analysis and Prediction Using Machine Learning

<p align="center">
  <b>An End-to-End Predictive Analytics & Machine Learning Project</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Regression-orange" />
  <img src="https://img.shields.io/badge/Framework-Streamlit-red?logo=streamlit" />
  <img src="https://img.shields.io/badge/Status-Completed-success" />
</p>

---

## 📌 Project Overview

**EduPredict** is a Machine Learning-based Predictive Analytics project designed to analyse and predict students' academic performance using various academic, personal, and environmental factors.

The project explores how factors such as study hours, attendance, previous academic scores, sleep duration, tutoring sessions, motivation, parental involvement, and access to learning resources are associated with examination performance.

It follows a complete Data Science workflow, including:

- Data Collection and Understanding
- Data Preprocessing and Cleaning
- Exploratory Data Analysis (EDA)
- Data Visualization
- Statistical Analysis
- Feature Engineering
- Machine Learning Model Development
- Model Evaluation and Comparison
- Interactive Dashboard Development
- Student Performance Prediction

The project implements four regression algorithms and compares their performance to identify the most suitable model for predicting examination scores.

An interactive application is developed using **Streamlit**, allowing users to explore student performance patterns and generate score predictions.

## 🎯 Project Objectives

The project investigates the following objectives:

**1. Study Hours and Exam Score**

Analyse the relationship between the number of hours students study per week and their examination scores.

**2. Attendance and Exam Score**

Investigate how classroom attendance is associated with students' academic performance.

**3. Previous Academic Performance**

Examine the relationship between students' previous examination scores and their current examination performance.

**4. Sleep and Tutoring**

Study how sleep duration and tutoring sessions are associated with examination scores.

**5. Overall Performance Factors**

Analyse the influence of motivation level, parental involvement, and access to educational resources.

**6. Predictive Modelling**

Develop four regression models, compare their performance, select the best-performing model, and integrate it into an interactive application.

---

## 📂 Dataset Description

The project uses the publicly available **Student Performance Factors** dataset from Kaggle.

**Dataset Source:**  
https://www.kaggle.com/datasets/lainguyn123/student-performance-factors

### Dataset Information

| Attribute | Details |
|---|---|
| Total Records | 6,607 |
| Total Columns | 20 |
| Numerical Input Features | 6 |
| Categorical Input Features | 13 |
| Target Variable | Exam_Score |
| Records After Cleaning | 6,606 |

### Features Used

**Numerical Features**
- Hours_Studied
- Attendance
- Sleep_Hours
- Previous_Scores
- Tutoring_Sessions
- Physical_Activity

**Categorical Features**
- Parental_Involvement
- Access_to_Resources
- Extracurricular_Activities
- Motivation_Level
- Internet_Access
- Family_Income
- Teacher_Quality
- School_Type
- Peer_Influence
- Learning_Disabilities
- Parental_Education_Level
- Distance_from_Home
- Gender

**Target Variable:** `Exam_Score`

The target variable represents the final examination score that the machine learning models aim to predict.

---

## 🛠️ Technologies and Libraries

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation and cleaning |
| NumPy | Numerical computations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| SciPy | Statistical testing and correlation |
| Scikit-learn | Machine learning and preprocessing |
| Joblib | Model saving and loading |
| Streamlit | Interactive dashboard and prediction interface |
| Jupyter Notebook | Data analysis and experimentation |
| Spyder | Python development environment |

---

## ⚙️ Project Workflow

### 1. Data Loading and Initial Inspection

The dataset is loaded using Pandas.

The following functions are used to understand the dataset:

- `head()` – Displays the first five records.
- `tail()` – Displays the last five records.
- `shape` – Shows the number of rows and columns.
- `info()` – Displays column names and data types.
- `describe()` – Generates statistical summaries.

### 2. Data Preprocessing and Cleaning

Data preprocessing is performed to improve data quality before analysis and modelling.

The following operations are performed:

- Identification and treatment of missing values.
- Duplicate record detection.
- Checking inconsistent categorical values.
- Detection of invalid numerical values.
- Outlier detection using the IQR method.
- Removal of one invalid examination score of 101.

**Missing Value Treatment**

| Column | Missing Values | Treatment |
|---|---:|---|
| Teacher_Quality | 78 | Mode Imputation |
| Parental_Education_Level | 90 | Mode Imputation |
| Distance_from_Home | 67 | Mode Imputation |

No duplicate records were found.

After removing the invalid examination score, the cleaned dataset contains **6,606 records**.

### 3. Exploratory Data Analysis (EDA)

EDA is performed to understand the distribution, relationships, and patterns within the dataset.

The analysis includes:

- Univariate Analysis
- Bivariate Analysis
- Multivariate Analysis
- Correlation Analysis
- Distribution Analysis
- Categorical Feature Analysis
- Outlier Detection

Various graphs and statistical visualizations are generated to understand the relationship between student-related factors and examination scores.

### 4. Feature Engineering

Three additional category features are created to make the analysis easier:

- `Study_Category`
- `Attendance_Category`
- `Sleep_Category`

These features help compare examination performance across different student groups.

The original numerical features are retained for model training.

### 5. Data Preprocessing for Machine Learning

The dataset is divided into training and testing sets.

| Dataset | Records | Percentage |
|---|---:|---:|
| Training Set | 5,284 | 80% |
| Testing Set | 1,322 | 20% |
| Total | 6,606 | 100% |

The following preprocessing techniques are used:

- One-Hot Encoding for categorical variables.
- StandardScaler for numerical features used by Linear Regression and KNN.
- SimpleImputer for handling missing values.
- Scikit-learn Pipelines for preprocessing and model training.

The dataset is split using `random_state=42`.

Preprocessing steps are fitted on training data to prevent data leakage.

---

## 📊 Project Visualizations and Graph Gallery

All visualizations are generated using Python, Pandas, Matplotlib, Seaborn, and other statistical libraries.

The project contains **20 graphs and visualizations** covering data exploration, objective-based analysis, feature engineering, model evaluation, and the final dashboard.

### 1. Exploratory Data Analysis (EDA)

**Figure 1: Box Plots of Numerical Features**

Used to identify potential outliers in numerical columns.

<img width="1917" height="998" alt="Screenshot 2026-10-31 220347" src="https://github.com/user-attachments/assets/c0f0e20a-dd9d-4cd5-b00c-cc4668dc8239" />


**Figure 2: Distribution of Examination Scores**

Shows the distribution and frequency of student examination scores.

<img width="1917" height="1000" alt="Screenshot 2026-10-31 220003" src="https://github.com/user-attachments/assets/f6df5162-0f6d-4802-b9ee-1ec0bfbb57f6" />


**Figure 3: Histograms of Numerical Features**

Displays the distribution of numerical variables such as study hours, attendance, sleep hours, and previous scores.

<img width="1917" height="987" alt="Screenshot 2026-10-31 220058" src="https://github.com/user-attachments/assets/ac1691e4-0954-4280-919c-096a3ca8a002" />


**Figure 4: Categorical Feature Distribution**

Shows the number of students belonging to different categorical groups.

<img width="1917" height="993" alt="Screenshot 2026-10-31 220247" src="https://github.com/user-attachments/assets/7bb69285-77a6-4f87-adee-5afca037dc22" />


**Figure 5: Correlation Heatmap**

Represents the correlation between numerical variables and helps identify important relationships.

<img width="1916" height="998" alt="Screenshot 2026-10-31 220133" src="https://github.com/user-attachments/assets/384240e5-d2c6-43a5-a718-752475734c34" />


**Figure 6: Categorical Score Gap Analysis**

Compares the difference in average examination scores between the highest and lowest categories of each categorical feature.

<img width="1910" height="992" alt="Screenshot 2026-10-31 220157" src="https://github.com/user-attachments/assets/ba6602d9-a7ba-450d-91d0-0cbd628deaf0" />


**Figure 7: Multivariate Pair Plot**

Visualizes relationships between numerical features, with observations coloured according to motivation level.

<img width="1910" height="992" alt="Screenshot 2026-10-31 220157" src="https://github.com/user-attachments/assets/bebe0759-89ec-4dfe-811f-e67dbdb9263f" />


### 2. Objective-Based Data Analysis

**Figure 8: Study Hours vs Examination Score**

Analyses the relationship between weekly study hours and examination performance.

![Study Hours](outputs/graphs/08_obj1_hours_vs_score.png)

**Figure 9: Attendance vs Examination Score**

Examines how attendance percentage is associated with examination scores.

![Attendance Analysis](outputs/graphs/09_obj2_attendance_vs_score.png)

**Figure 10: Previous Scores Analysis**

Investigates the relationship between previous academic performance and current examination scores.

![Previous Scores](outputs/graphs/10_obj3_previous_scores.png)

**Figure 11: Sleep Hours and Tutoring Analysis**

Compares examination performance across different sleep durations and tutoring sessions.

![Sleep and Tutoring](outputs/graphs/11_obj4_sleep_tutoring.png)

**Figure 12: Motivation, Parental Involvement and Resources**

Analyses how motivation, parental involvement, and educational resources are associated with student performance.

![Categorical Factors](outputs/graphs/12_obj5_categorical_factors.png)

### 3. Feature Engineering Visualizations

**Figure 13: Engineered Feature Analysis**

Displays the average examination scores across engineered student categories.

![Engineered Features](outputs/graphs/13_engineered_features.png)

**Figure 14: Study Category and Attendance Category Heatmap**

Shows the relationship between study categories, attendance categories, and average examination scores.

![Study Attendance Heatmap](outputs/graphs/14_study_attendance_heatmap.png)

### 4. Machine Learning Model Evaluation

**Figure 15: Model Performance Comparison**

Compares the performance of the four regression models using R², MAE, MSE, and RMSE.

![Model Comparison](outputs/graphs/15_model_comparison.png)

**Figure 16: Actual vs Predicted Examination Scores**

Compares actual examination scores with predictions generated by the four regression models.

![Actual vs Predicted](outputs/graphs/16_actual_vs_predicted_all_models.png)

**Figure 17: Residual Plot of the Best Model**

Examines the prediction errors of the selected Linear Regression model.

![Residual Plot](outputs/graphs/17_residual_plot.png)

**Figure 18: Residual Distribution**

Displays the distribution of prediction errors to understand model performance.

![Residual Distribution](outputs/graphs/18_residual_distribution.png)

**Figure 19: Feature Importance Analysis**

Compares Random Forest feature importance and Linear Regression coefficients to understand the contribution of different features.

![Feature Importance](outputs/graphs/19_feature_importance.png)

### 5. EduPredict Dashboard

**Figure 20: Final Summary Dashboard**

Presents the overall model performance, important factors, and key relationships identified during the project.

![EduPredict Dashboard](outputs/graphs/20_dashboard.png)

---

## 🤖 Machine Learning Models

Four regression algorithms are implemented and evaluated.

### 1. Linear Regression

Linear Regression is used to estimate the relationship between input variables and the target examination score.

It identifies linear patterns in the dataset and generates continuous numerical predictions.

### 2. Decision Tree Regressor

Decision Tree Regressor uses decision-based splitting to divide the dataset into smaller groups and predict examination scores.

### 3. Random Forest Regressor

Random Forest combines multiple decision trees and uses their predictions to generate a final output.

It helps capture complex relationships between different student performance factors.

### 4. K-Nearest Neighbors (KNN) Regressor

KNN predicts examination scores by identifying the nearest training examples and using their target values to estimate the result.

---

## 📈 Model Performance and Comparison

All four models are evaluated using the same test dataset.

The following metrics are used:

- **R² Score:** Measures how much variation in the target variable is explained by the model.
- **MAE:** Measures the average absolute prediction error.
- **MSE:** Measures the average squared prediction error.
- **RMSE:** Measures the square root of the average squared prediction error.

### Model Comparison Results

| Model | R² Score | MAE | MSE | RMSE |
|---|---:|---:|---:|---:|
| Linear Regression | **0.8250** | **0.4160** | **2.3142** | **1.5213** |
| Random Forest | 0.7283 | 1.0234 | 3.5928 | 1.8955 |
| KNN | 0.6482 | 1.3469 | 4.6518 | 2.1568 |
| Decision Tree | 0.6023 | 1.4468 | 5.2596 | 2.2934 |

### 🏆 Best Performing Model: Linear Regression

Based on the test-set evaluation, Linear Regression achieved the best performance among the four implemented models.

**Performance Metrics:**

- R² Score: 0.8250
- MAE: 0.4160
- MSE: 2.3142
- RMSE: 1.5213
- 5-Fold Cross-Validation R² Mean: 0.7166

The R² score is a regression metric and should not be interpreted as classification accuracy.

---

## 🔍 Key Findings

Based on the analysis of 6,606 cleaned student records, the following observations were identified:

1. Attendance has the strongest positive correlation with examination scores (0.582).
2. Study hours also show a positive relationship with examination performance (0.447).
3. Previous scores show a comparatively weaker positive relationship (0.174).
4. Tutoring sessions have a small positive association with examination scores (0.154).
5. Sleep duration shows no statistically significant relationship in this dataset (-0.016, p = 0.19).
6. Higher parental involvement and better access to educational resources are associated with higher average examination scores.

These findings represent associations within the dataset and do not establish cause-and-effect relationships.

---

## 📁 Project Folder Structure

The project is organized into separate folders for datasets, machine learning models, notebooks, visualizations, and reports.

```text
EduPredict/
│
├── app.py
│
├── EduPredict_Dashboard.py
│
├── requirements.txt
│
├── README.md
│
├── dataset/
│   └── StudentPerformanceFactors.csv
│
├── models/
│   ├── best_model.pkl
│   └── model_metadata.json
│
├── notebooks/
│   ├── EduPredict_Analysis.ipynb
│   └── EduPredict_Analysis.py
│
├── outputs/
│   │
│   ├── graphs/
│   │   ├── 01_boxplots_outliers.png
│   │   ├── 02_exam_score_distribution.png
│   │   ├── 03_numeric_histograms.png
│   │   ├── 04_categorical_countplots.png
│   │   ├── 05_correlation_heatmap.png
│   │   ├── 06_categorical_score_gap.png
│   │   ├── 07_pairplot.png
│   │   ├── 08_obj1_hours_vs_score.png
│   │   ├── 09_obj2_attendance_vs_score.png
│   │   ├── 10_obj3_previous_scores.png
│   │   ├── 11_obj4_sleep_tutoring.png
│   │   ├── 12_obj5_categorical_factors.png
│   │   ├── 13_engineered_features.png
│   │   ├── 14_study_attendance_heatmap.png
│   │   ├── 15_model_comparison.png
│   │   ├── 16_actual_vs_predicted_all_models.png
│   │   ├── 17_residual_plot.png
│   │   ├── 18_residual_distribution.png
│   │   ├── 19_feature_importance.png
│   │   └── 20_dashboard.png
│   │
│   ├── cross_validation_r2.csv
│   ├── model_comparison.csv
│   └── tolerance_accuracy.csv
│
└── reports/
    ├── EduPredict_Project_Report.pdf
    ├── EduPredict_Viva_Preparation.md
    └── EduPredict_Viva_Preparation.pdf
```

---

## 💻 Installation and Setup

### Prerequisites

Before running the project, make sure the following are installed:

- Python 3.10 or later
- pip
- Git (optional)

### Step 1: Clone the Repository

```bash
git clone <your-github-repository-url>
```

### Step 2: Navigate to the Project Directory

```bash
cd EduPredict
```

### Step 3: Create a Virtual Environment

For Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

For macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 4: Install Required Libraries

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## ▶️ Running the Project

There are three ways to execute the project.

### Option 1: Run the Complete Interactive Dashboard

```bash
python -m streamlit run EduPredict_Dashboard.py
```

The dashboard trains the four regression models on its first run and provides interactive sections for exploring the data, analysing objectives, comparing models, and making predictions.

### Option 2: Run the Prediction Application

```bash
streamlit run app.py
```

This application loads the saved machine learning pipeline from:

`models/best_model.pkl`

It also reads model information from:

`models/model_metadata.json`

### Option 3: Run the Jupyter Notebook

```bash
jupyter notebook notebooks/EduPredict_Analysis.ipynb
```

Run the notebook cells in sequence to perform data preprocessing, exploratory analysis, model training, and evaluation.

The Python analysis script can also be executed using Spyder or another Python IDE.

---

## 🖥️ Application Features

The EduPredict Streamlit dashboard provides the following sections:

### 1. Overview
- Dataset summary
- Important performance metrics
- Examination score distribution
- Key relationships between features

### 2. Data Explorer
- Interactive dataset exploration
- Feature-based filtering
- Data visualization

### 3. Objectives
- Study hours analysis
- Attendance analysis
- Previous scores analysis
- Sleep and tutoring analysis
- Categorical factor analysis

### 4. Model Performance
- Comparison of four regression models
- Evaluation metrics
- Cross-validation results
- Residual analysis
- Feature importance

### 5. Predict
- Enter student-related information
- Generate an estimated examination score
- View the prediction through the interactive application

---

## ⚠️ Project Limitations

- The analysis is based on one publicly available dataset.
- The dataset may be simulated or heavily processed; this cannot be confirmed from its source.
- Some potentially important factors may not be included in the dataset.
- Sleep duration does not represent sleep quality.
- Correlation does not establish causation.
- Model predictions are estimates and should not be considered guaranteed examination results.

---

## 🚀 Future Scope

The project can be extended through the following improvements:

- Integration of real institutional student records.
- Addition of subject-wise examination and assignment scores.
- Further hyperparameter tuning and model optimization.
- Development of prediction ranges alongside point estimates.
- Fairness evaluation across different student groups.
- Online deployment for students and educational institutions.
- Additional visual analytics and personalized academic insights.

---

## 📚 References

1. [Student Performance Factors Dataset – Kaggle](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)
2. [Scikit-learn Documentation](https://scikit-learn.org/stable/user_guide.html)
3. [Pandas Documentation](https://pandas.pydata.org/docs/)
4. [NumPy Documentation](https://numpy.org/doc/)
5. [Matplotlib Documentation](https://matplotlib.org/stable/)
6. [Seaborn Documentation](https://seaborn.pydata.org/)
7. [Streamlit Documentation](https://docs.streamlit.io/)

---

## 👨‍💻 Author

**Rishabh Yadav**

B.Tech – Computer Science and Engineering  
Lovely Professional University

**Project:** EduPredict – Student Academic Performance Analysis and Prediction Using Machine Learning

**Course:** Predictive Analytics (INT234)

---

<p align="center">
  <b>EduPredict – Turning Student Data into Meaningful Academic Insights.</b>
</p>
