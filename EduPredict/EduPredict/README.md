# EduPredict: Student Academic Performance Analysis and Prediction Using Machine Learning

A complete Data Science and Machine Learning project for the **Predictive Analytics** subject. It analyses the factors behind students' exam scores and predicts the expected exam score with a regression model, available through a Streamlit app.

**Dataset:** [Student Performance Factors (Kaggle)](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors), 6,607 students, 19 input features, target `Exam_Score`. The real CSV is used; no data was generated.

## Project structure

```
EduPredict/
├── dataset/StudentPerformanceFactors.csv
├── notebooks/EduPredict_Analysis.ipynb     # Jupyter / Colab version
│   └── EduPredict_Analysis.py              # Spyder version (run cell by cell)
├── models/
│   ├── best_model.pkl                       # saved best pipeline (preprocessing + model)
│   └── model_metadata.json                  # model name, metrics, feature order, category options
├── outputs/
│   ├── graphs/                              # 20 saved graphs (incl. 20_dashboard.png)
│   ├── model_comparison.csv                 # R2, MAE, MSE, RMSE for all four models
│   └── cross_validation_r2.csv
├── EduPredict_Dashboard.py                  # professional Streamlit dashboard (5 tabs, charts, prediction)
├── app.py                                   # simple prediction-only Streamlit app
├── requirements.txt
├── README.md
└── reports/
    ├── EduPredict_Project_Report.pdf        # 22-page academic report
    ├── EduPredict_Viva_Preparation.pdf      # 42 viva Q&A + 4-minute presentation script
    └── EduPredict_Viva_Preparation.md
```

## How to run

**Locally**
```bash
pip install -r requirements.txt
jupyter notebook notebooks/EduPredict_Analysis.ipynb   # run all cells (re-creates graphs, tables and the model)
streamlit run EduPredict_Dashboard.py     # full dashboard (or: streamlit run app.py for the simple predictor)
```

**Google Colab:** upload `notebooks/EduPredict_Analysis.ipynb` and `dataset/StudentPerformanceFactors.csv` to the same Colab folder and run all cells. The notebook finds the CSV automatically.

> If `app.py` shows a model-loading error (for example, because your scikit-learn version differs from the one used to save the model), simply re-run the notebook; it saves a fresh `models/best_model.pkl` for your environment.

## Workflow

1. **Preprocessing:** 3 categorical columns had missing values (78, 90, 67), imputed with the most frequent value inside the pipeline; 0 duplicates; categories consistent; 1 impossible `Exam_Score` (101) removed; IQR outliers were valid and retained.
2. **EDA:** histograms, count plots, box plots, scatter plots, correlation heatmap, pair plot, each with interpretation.
3. **Five objectives:** study hours, attendance, previous scores, sleep and tutoring, categorical factors (motivation, parental involvement, resources).
4. **Feature engineering:** `Study_Category`, `Attendance_Category`, `Sleep_Category` (for analysis; original numeric columns used for modelling).
5. **Models:** Linear Regression, Decision Tree, Random Forest, KNN in scikit-learn pipelines; same 80/20 split, `random_state=42`; imputation, one-hot encoding and scaling fitted on training data only.
6. **Evaluation:** R², MAE, MSE, RMSE, residual analysis, feature importance, cross-validation.

## Results (test set, 1,322 students)

| Model | R² | MAE | MSE | RMSE |
|---|---|---|---|---|
| **Linear Regression (best)** | **0.8250** | **0.4160** | **2.3142** | **1.5213** |
| Random Forest | 0.7283 | 1.0234 | 3.5928 | 1.8955 |
| KNN | 0.6482 | 1.3469 | 4.6518 | 2.1568 |
| Decision Tree | 0.6023 | 1.4468 | 5.2596 | 2.2934 |

* The best model is chosen automatically by the code (highest test R²). The ranking is the same under 5-fold cross-validation.
* **Strongest factors:** attendance (r = 0.58) and weekly study hours (r = 0.45). Sleep shows no association (r = -0.02).
* 98.6 percent of students are predicted within 1 mark; 49 students (0.7 percent) have scores no feature explains, which limits R². Cross-validated R² is about 0.72, a more cautious estimate than the single test split.

## Limitations

Findings are associations, not causes. The dataset is unusually regular and may be simulated or heavily processed, so results may not transfer to real classrooms.

## Rubric compliance

| # | Requirement | Where |
|---|---|---|
| 1 | Kaggle dataset | `dataset/`, notebook Phase 1 |
| 2 | Cleaning: missing values, duplicates, inconsistent data | Notebook Phase 2 |
| 3 | EDA | Notebook Phase 3 |
| 4 | At least five objectives | Notebook Phase 3 (Objectives 1 to 5) |
| 5 | Visualisation | `outputs/graphs/` (20 graphs) |
| 6 | Feature engineering | Notebook Phase 4 |
| 7 | At least four ML models | Notebook Phase 5 |
| 8 | Best-fit model | Notebook Phase 6, `models/best_model.pkl` |
| 9 | Performance metrics | Notebook Phase 6, `outputs/model_comparison.csv` |
| 10 | Findings, conclusions, recommendations | Notebook Phase 8, report |
