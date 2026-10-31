"""
EduPredict Dashboard
====================
Student Academic Performance Analysis and Prediction (Predictive Analytics project).

Run from a terminal / Anaconda Prompt in the folder that contains this file:
    python -m streamlit run EduPredict_Dashboard.py

What it needs
-------------
* StudentPerformanceFactors.csv  (kept next to this file, or inside a "dataset" folder)
* pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, joblib, streamlit

What it shows (5 tabs)
----------------------
1. Overview            KPI cards, score distribution, key insights
2. Data Explorer       filters + interactive charts
3. Objectives          the five project objectives with charts and statistics
4. Model Performance   accuracy, error, cross-validation, residuals, feature importance
5. Predict             enter student details -> predicted score + what-if charts
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

# --------------------------------------------------------------------------------------
# Page setup and styling
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="EduPredict Dashboard", page_icon="🎓", layout="wide")
st.markdown(
    """
    <style>
    .main-title {font-size: 2.2rem; font-weight: 800; color: #2a9d8f; margin-bottom: 0;}
    .sub-title {color: #8a94a6; margin-top: 0; margin-bottom: 1rem;}
    div[data-testid="stMetric"] {background: rgba(42,157,143,0.10); border-left: 5px solid #2a9d8f;
                                 padding: 12px 16px; border-radius: 8px;}
    </style>
    """,
    unsafe_allow_html=True,
)
sns.set_theme(style="whitegrid")
PALETTE = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]
MODEL_COLORS = {"Linear Regression": "#2a9d8f", "Decision Tree": "#e9c46a", "Random Forest": "#264653", "KNN": "#e76f51"}

TARGET = "Exam_Score"
NUMERIC = ["Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Scores", "Tutoring_Sessions", "Physical_Activity"]
LEVELS = ["Low", "Medium", "High"]
RANDOM_STATE = 42

# Valid limits for user input (what is logically possible)
VALID_LIMITS = {"Hours_Studied": (0, 80), "Attendance": (0, 100), "Sleep_Hours": (0, 12),
                "Previous_Scores": (0, 100), "Tutoring_Sessions": (0, 15), "Physical_Activity": (0, 10)}


# --------------------------------------------------------------------------------------
# Data loading and model training (runs once, then cached)
# --------------------------------------------------------------------------------------
def find_dataset():
    here = Path(__file__).resolve().parent
    for path in [here / "StudentPerformanceFactors.csv", here / "dataset" / "StudentPerformanceFactors.csv",
                 here.parent / "dataset" / "StudentPerformanceFactors.csv"]:
        if path.exists():
            return path
    return None


def make_pipeline(model, scale, numeric_cols, categorical_cols):
    """Preprocessing + model in one pipeline (fitted on training data only)."""
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    prep = ColumnTransformer([
        ("num", Pipeline(numeric_steps), numeric_cols),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols)])
    return Pipeline([("prep", prep), ("model", model)])


@st.cache_resource(show_spinner="Training the four models (only the first time)...")
def build_bundle():
    path = find_dataset()
    if path is None:
        return None
    raw = pd.read_csv(path)
    clean = raw[raw[TARGET] <= 100].copy()                 # one impossible score (101) removed
    X, y = clean.drop(columns=TARGET), clean[TARGET]
    categorical = [c for c in X.columns if c not in NUMERIC]

    eda = clean.copy()                                     # copy used only for charts: fill gaps with the mode
    for col in categorical:
        eda[col] = eda[col].fillna(eda[col].mode()[0])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # Settings for Decision Tree and KNN were chosen by 5-fold cross-validation in the analysis notebook
    specs = {
        "Linear Regression": (LinearRegression(), True),
        "Decision Tree": (DecisionTreeRegressor(max_depth=15, min_samples_leaf=20, random_state=RANDOM_STATE), False),
        "Random Forest": (RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1), False),
        "KNN": (KNeighborsRegressor(n_neighbors=20), True),
    }
    models, predictions, metric_rows, cv_rows = {}, {}, [], []
    for name, (estimator, scale) in specs.items():
        pipe = make_pipeline(estimator, scale, NUMERIC, categorical)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        error = (y_test - pred).abs()
        mse = mean_squared_error(y_test, pred)
        metric_rows.append({"Model": name, "R2": r2_score(y_test, pred), "Accuracy_%": r2_score(y_test, pred) * 100,
                            "MAE": mean_absolute_error(y_test, pred), "MSE": mse, "RMSE": np.sqrt(mse),
                            "Within_1_mark_%": (error <= 1).mean() * 100, "Within_2_marks_%": (error <= 2).mean() * 100,
                            "Within_5_marks_%": (error <= 5).mean() * 100,
                            "Train_R2": r2_score(y_train, pipe.predict(X_train))})
        scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2")
        cv_rows.append({"Model": name, "CV_R2": scores.mean(), "CV_std": scores.std()})
        models[name], predictions[name] = pipe, pred

    metrics = pd.DataFrame(metric_rows).set_index("Model")
    cv = pd.DataFrame(cv_rows).set_index("Model")
    best_name = metrics.sort_values(["R2", "RMSE"], ascending=[False, True]).index[0]   # chosen from test results
    return {"raw": raw, "clean": clean, "eda": eda, "X": X, "categorical": categorical, "X_train": X_train,
            "X_test": X_test, "y_train": y_train, "y_test": y_test, "models": models, "predictions": predictions,
            "metrics": metrics, "cv": cv, "best_name": best_name, "data_path": str(path)}


def load_predictor(bundle):
    """Use the saved best_model.pkl from the notebook if it can be loaded; otherwise the model trained here."""
    here = Path(__file__).resolve().parent
    for candidate in [here / "models" / "best_model.pkl", here.parent / "models" / "best_model.pkl"]:
        if candidate.exists():
            try:
                return joblib.load(candidate), f"saved model file ({candidate.name})"
            except Exception:
                pass
    return bundle["models"][bundle["best_name"]], "model trained in this app"


# --------------------------------------------------------------------------------------
# Small chart helpers
# --------------------------------------------------------------------------------------
def show(fig):
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def grouped_bar(ax, table, columns, labels, title, ylabel, fmt="%.1f"):
    """Bars for every model and every column in `columns`."""
    x = np.arange(len(table.index))
    width = 0.8 / len(columns)
    for i, (col, lab) in enumerate(zip(columns, labels)):
        bars = ax.bar(x + i * width - 0.4 + width / 2, table[col], width, label=lab, color=PALETTE[(i + 1) % 5])
        ax.bar_label(bars, fmt=fmt, fontsize=7, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(table.index, rotation=12)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Model")
    ax.legend(fontsize=8)


# --------------------------------------------------------------------------------------
# Header + data
# --------------------------------------------------------------------------------------
st.markdown('<p class="main-title">🎓 EduPredict Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Student academic performance analysis and exam score prediction using machine learning</p>',
            unsafe_allow_html=True)

bundle = build_bundle()
if bundle is None:
    st.error("StudentPerformanceFactors.csv was not found. Put it next to this file (or inside a 'dataset' folder) and reload.")
    st.stop()

eda, metrics, cv, best = bundle["eda"], bundle["metrics"], bundle["cv"], bundle["best_name"]
best_row = metrics.loc[best]

# --------------------------------------------------------------------------------------
# Sidebar: filters for the Data Explorer and Objectives tabs
# --------------------------------------------------------------------------------------
st.sidebar.header("🔎 Filters")
st.sidebar.caption("Filters apply to the Data Explorer and Objectives tabs.")
gender_f = st.sidebar.multiselect("Gender", sorted(eda["Gender"].unique()), default=sorted(eda["Gender"].unique()))
school_f = st.sidebar.multiselect("School type", sorted(eda["School_Type"].unique()), default=sorted(eda["School_Type"].unique()))
motivation_f = st.sidebar.multiselect("Motivation level", LEVELS, default=LEVELS)
att_low, att_high = st.sidebar.slider("Attendance range (%)", 60, 100, (60, 100))
fdf = eda[eda["Gender"].isin(gender_f) & eda["School_Type"].isin(school_f) & eda["Motivation_Level"].isin(motivation_f)
          & eda["Attendance"].between(att_low, att_high)]
st.sidebar.metric("Students after filtering", f"{len(fdf):,}")
st.sidebar.divider()
st.sidebar.caption(f"Dataset: Kaggle 'Student Performance Factors'. {len(bundle['clean']):,} students after cleaning.")

tab_overview, tab_explore, tab_obj, tab_model, tab_predict = st.tabs(
    ["🏠 Overview", "📊 Data Explorer", "🎯 Objectives", "🤖 Model Performance", "🔮 Predict"])

# ======================================================================================
# TAB 1: OVERVIEW
# ======================================================================================
with tab_overview:
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Students", f"{len(eda):,}")
    k2.metric("Average score", f"{eda[TARGET].mean():.1f}")
    k3.metric("Best model", best)
    k4.metric("Accuracy (R²)", f"{best_row['R2'] * 100:.1f}%")
    k5.metric("Average error (MAE)", f"{best_row['MAE']:.2f} marks")
    k6.metric("RMSE", f"{best_row['RMSE']:.2f} marks")

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(eda[TARGET], bins=range(55, 102, 2), kde=True, color=PALETTE[1], ax=ax)
        ax.axvline(eda[TARGET].mean(), color="red", linestyle="--", label=f"Mean = {eda[TARGET].mean():.1f}")
        ax.set_title("Distribution of Exam Scores")
        ax.set_xlabel("Exam score")
        ax.set_ylabel("Number of students")
        ax.legend()
        show(fig)
    with right:
        corr = eda[NUMERIC + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.barh(corr.index, corr.values, color=[PALETTE[4] if v < 0 else PALETTE[1] for v in corr.values])
        ax.bar_label(bars, fmt="%.2f", padding=3)
        ax.set_xlim(corr.min() - 0.1, corr.max() + 0.12)
        ax.set_title("Correlation of Each Numeric Factor with Exam Score")
        ax.set_xlabel("Pearson correlation")
        ax.set_ylabel("Factor")
        show(fig)

    st.subheader("Key insights")
    corr_sorted = corr.sort_values(ascending=False)
    st.markdown(
        f"- **{corr_sorted.index[0]}** has the strongest relationship with exam score (r = {corr_sorted.iloc[0]:.2f}), "
        f"followed by **{corr_sorted.index[1]}** (r = {corr_sorted.iloc[1]:.2f}).\n"
        f"- **Sleep hours** show almost no relationship with exam score (r = {corr['Sleep_Hours']:.2f}).\n"
        f"- The best model is **{best}** with R² = {best_row['R2']:.3f}; "
        f"**{best_row['Within_1_mark_%']:.1f}%** of test predictions are within 1 mark of the real score.\n"
        "- These are **associations, not proven causes**: for example, motivated students may both attend more and score more.")

# ======================================================================================
# TAB 2: DATA EXPLORER
# ======================================================================================
with tab_explore:
    if len(fdf) < 30:
        st.warning("Too few students match the filters. Widen the filters in the sidebar.")
    else:
        st.caption(f"Showing {len(fdf):,} students (filters from the sidebar).")
        c1, c2 = st.columns(2)
        with c1:
            factor = st.selectbox("Compare average score by", [c for c in bundle["categorical"]], index=0)
            order = LEVELS if set(fdf[factor].unique()) <= set(LEVELS) else None
            means = fdf.groupby(factor)[TARGET].agg(["mean", "count"])
            if order:
                means = means.reindex([o for o in order if o in means.index])
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(means.index.astype(str), means["mean"], color=PALETTE[1])
            ax.bar_label(bars, fmt="%.2f", padding=2)
            ax.set_ylim(fdf[TARGET].mean() - 5, fdf[TARGET].mean() + 5)
            ax.set_title(f"Average Exam Score by {factor}")
            ax.set_xlabel(factor)
            ax.set_ylabel("Average exam score")
            show(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=fdf, x=factor, y=TARGET, order=order, color=PALETTE[2], ax=ax)
            ax.set_title(f"Score Spread by {factor}")
            ax.set_xlabel(factor)
            ax.set_ylabel("Exam score")
            show(fig)

        c3, c4 = st.columns(2)
        with c3:
            x_col = st.selectbox("Scatter plot: numeric factor", NUMERIC, index=1)
            hue_col = st.selectbox("Colour by", ["Motivation_Level", "Parental_Involvement", "Access_to_Resources", "Gender", "School_Type"])
            sample = fdf.sample(min(1500, len(fdf)), random_state=RANDOM_STATE)
            fig, ax = plt.subplots(figsize=(6, 4.2))
            sns.scatterplot(data=sample, x=x_col, y=TARGET, hue=hue_col, alpha=0.5, s=20, ax=ax)
            ax.set_title(f"{x_col} vs Exam Score (sample of {len(sample):,})")
            ax.set_xlabel(x_col)
            ax.set_ylabel("Exam score")
            show(fig)
        with c4:
            st.write("")
            st.write("")
            fig, ax = plt.subplots(figsize=(6, 4.6))
            sns.heatmap(fdf[NUMERIC + [TARGET]].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, cbar_kws={"label": "Correlation"})
            ax.set_title("Correlation Heatmap")
            show(fig)

        with st.expander("Show data table and download"):
            st.dataframe(fdf.head(200))
            st.download_button("Download filtered data (CSV)", fdf.to_csv(index=False).encode("utf-8"),
                               file_name="filtered_students.csv", mime="text/csv")

# ======================================================================================
# TAB 3: OBJECTIVES
# ======================================================================================
with tab_obj:
    if len(fdf) < 30:
        st.warning("Too few students match the filters. Widen the filters in the sidebar.")
    else:
        objective = st.radio("Choose an objective", ["1. Study hours", "2. Attendance", "3. Previous scores",
                                                      "4. Sleep and tutoring", "5. Motivation, parental involvement, resources"], horizontal=True)

        def regression_chart(column, title, xlabel):
            r, p = stats.pearsonr(fdf[column], fdf[TARGET])
            slope, intercept = np.polyfit(fdf[column], fdf[TARGET], 1)
            a, b, c = st.columns(3)
            a.metric("Correlation (r)", f"{r:.3f}")
            b.metric("Marks per extra unit", f"{slope:.3f}")
            c.metric("p-value", "< 0.001" if p < 0.001 else f"{p:.3f}")
            fig, ax = plt.subplots(figsize=(8, 4.5))
            sns.regplot(data=fdf, x=column, y=TARGET, scatter_kws={"alpha": 0.2, "s": 10}, line_kws={"color": "red", "label": f"Trend (r = {r:.2f})"}, ax=ax)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Exam score")
            ax.legend()
            show(fig)

        if objective.startswith("1"):
            st.markdown("**Objective 1:** relationship between weekly study hours and exam score.")
            regression_chart("Hours_Studied", "Study Hours vs Exam Score", "Hours studied per week")
        elif objective.startswith("2"):
            st.markdown("**Objective 2:** how attendance is associated with exam score.")
            regression_chart("Attendance", "Attendance vs Exam Score", "Attendance (%)")
            avg = fdf.groupby("Attendance")[TARGET].mean()
            fig, ax = plt.subplots(figsize=(8, 3.8))
            ax.plot(avg.index, avg.values, marker="o", color=PALETTE[1], label="Average score")
            ax.set_title("Average Exam Score at Each Attendance Level")
            ax.set_xlabel("Attendance (%)")
            ax.set_ylabel("Average exam score")
            ax.legend()
            show(fig)
        elif objective.startswith("3"):
            st.markdown("**Objective 3:** relationship between previous scores and exam score.")
            regression_chart("Previous_Scores", "Previous Scores vs Exam Score", "Previous scores")
        elif objective.startswith("4"):
            st.markdown("**Objective 4:** how sleep hours and tutoring sessions are associated with exam score.")
            r_sleep = stats.pearsonr(fdf["Sleep_Hours"], fdf[TARGET])[0]
            r_tut = stats.pearsonr(fdf["Tutoring_Sessions"], fdf[TARGET])[0]
            a, b = st.columns(2)
            a.metric("Sleep vs score (r)", f"{r_sleep:.3f}")
            b.metric("Tutoring vs score (r)", f"{r_tut:.3f}")
            c1, c2 = st.columns(2)
            with c1:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(data=fdf, x="Sleep_Hours", y=TARGET, color=PALETTE[2], errorbar=None, ax=ax)
                ax.set_ylim(60, 75)
                ax.set_title("Average Score by Sleep Hours")
                ax.set_xlabel("Sleep hours per night")
                ax.set_ylabel("Average exam score")
                show(fig)
            with c2:
                grouped = fdf["Tutoring_Sessions"].clip(upper=5)
                labels = grouped.map(lambda v: "5+" if v >= 5 else str(int(v)))
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.boxplot(x=labels, y=fdf[TARGET], order=["0", "1", "2", "3", "4", "5+"], color=PALETTE[1], ax=ax)
                ax.set_title("Exam Score by Tutoring Sessions")
                ax.set_xlabel("Tutoring sessions per month (5+ = five or more)")
                ax.set_ylabel("Exam score")
                show(fig)
        else:
            st.markdown("**Objective 5:** motivation level, parental involvement and access to resources versus exam score.")
            factors = ["Motivation_Level", "Parental_Involvement", "Access_to_Resources"]
            table = pd.DataFrame({f: fdf.groupby(f)[TARGET].mean().reindex(LEVELS) for f in factors}).T.round(2)
            table["High minus Low"] = (table["High"] - table["Low"]).round(2)
            st.dataframe(table)
            fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
            for ax, f in zip(axes, factors):
                sns.boxplot(data=fdf, x=f, y=TARGET, order=LEVELS, palette="viridis", ax=ax)
                ax.set_title(f"Exam Score by {f}")
                ax.set_xlabel(f)
                ax.set_ylabel("Exam score")
            show(fig)
        st.info("Remember: a relationship in the data does not prove that one factor causes the other.")

# ======================================================================================
# TAB 4: MODEL PERFORMANCE
# ======================================================================================
with tab_model:
    with st.expander("How is 'accuracy' measured for a regression model?"):
        st.markdown(
            "Exam score is a number, so a model cannot be simply 'right or wrong'. We measure accuracy in two ways:\n"
            "- **R² (shown as %)**: the share of the variation in exam scores that the model explains (100% is perfect).\n"
            "- **Within-tolerance accuracy**: the % of students whose predicted score is within 1, 2 or 5 marks of the real score.\n\n"
            "**MAE / MSE / RMSE** measure the size of the error (lower is better).")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Best model", best)
    m2.metric("R² accuracy", f"{best_row['R2'] * 100:.1f}%")
    m3.metric("MAE", f"{best_row['MAE']:.3f}")
    m4.metric("RMSE", f"{best_row['RMSE']:.3f}")
    m5.metric("Within 1 mark", f"{best_row['Within_1_mark_%']:.1f}%")

    st.subheader("Model comparison table (test set)")
    display_cols = ["R2", "MAE", "MSE", "RMSE", "Within_1_mark_%", "Within_2_marks_%", "Within_5_marks_%", "Train_R2"]
    st.dataframe(metrics[display_cols].round(4))
    st.download_button("Download model comparison (CSV)", metrics[display_cols].round(4).to_csv().encode("utf-8"),
                       file_name="model_comparison.csv", mime="text/csv")

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(metrics.index, metrics["Accuracy_%"], color=[MODEL_COLORS[m] for m in metrics.index])
        ax.bar_label(bars, fmt="%.1f%%", padding=2)
        ax.set_ylim(0, 100)
        ax.set_title("Model Accuracy (R² as %): higher is better")
        ax.set_xlabel("Model")
        ax.set_ylabel("R² (%)")
        ax.tick_params(axis="x", rotation=12)
        show(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        grouped_bar(ax, metrics, ["MAE", "RMSE"], ["MAE", "RMSE"], "Prediction Error in Marks: lower is better", "Error (marks)", "%.2f")
        show(fig)

    c3, c4 = st.columns(2)
    with c3:
        fig, ax = plt.subplots(figsize=(6, 4))
        grouped_bar(ax, metrics, ["Within_1_mark_%", "Within_2_marks_%", "Within_5_marks_%"], ["within 1 mark", "within 2 marks", "within 5 marks"],
                    "Share of Predictions Close to the Real Score", "Students (%)", "%.0f")
        ax.set_ylim(0, 110)
        show(fig)
    with c4:
        compare = pd.DataFrame({"Test R2": metrics["R2"], "Cross-validated R2": cv["CV_R2"]})
        fig, ax = plt.subplots(figsize=(6, 4))
        grouped_bar(ax, compare, ["Test R2", "Cross-validated R2"], ["Test R²", "5-fold CV R² (training data)"],
                    "Test R² vs Cross-Validated R²", "R²", "%.2f")
        show(fig)
    st.caption("The model ranking is the same on the test set and under cross-validation, so the choice of the best model is not an accident of one split.")

    st.subheader("Predictions in detail")
    chosen = st.selectbox("Select a model", list(metrics.index), index=list(metrics.index).index(best))
    y_test, pred = bundle["y_test"], bundle["predictions"][chosen]
    residuals = y_test - pred
    d1, d2, d3 = st.columns(3)
    with d1:
        fig, ax = plt.subplots(figsize=(5, 4.2))
        ax.scatter(y_test, pred, alpha=0.35, s=12, color=MODEL_COLORS[chosen], label="Students")
        lims = [y_test.min() - 1, y_test.max() + 1]
        ax.plot(lims, lims, "r--", label="Perfect prediction")
        ax.set_title(f"{chosen}: Actual vs Predicted")
        ax.set_xlabel("Actual score")
        ax.set_ylabel("Predicted score")
        ax.legend()
        show(fig)
    with d2:
        fig, ax = plt.subplots(figsize=(5, 4.2))
        ax.scatter(pred, residuals, alpha=0.35, s=12, color=MODEL_COLORS[chosen], label="Students")
        ax.axhline(0, color="red", linestyle="--", label="Zero error")
        ax.set_title("Residual Plot")
        ax.set_xlabel("Predicted score")
        ax.set_ylabel("Residual (actual - predicted)")
        ax.legend()
        show(fig)
    with d3:
        fig, ax = plt.subplots(figsize=(5, 4.2))
        cut = max(3.0, float(np.percentile(residuals.abs(), 98)))      # focus on the main body of errors
        shown = residuals[residuals.abs() <= cut]
        sns.histplot(shown, bins=40, kde=True, color=MODEL_COLORS[chosen], ax=ax)
        ax.axvline(0, color="red", linestyle="--", label="Zero error")
        ax.set_title(f"Residual Distribution ({len(residuals) - len(shown)} larger errors not shown)", fontsize=10)
        ax.set_xlabel("Residual (marks)")
        ax.set_ylabel("Number of students")
        ax.legend()
        show(fig)

    st.subheader("Which factors matter most?")
    def feature_names(pipe):
        return [n.replace("num__", "").replace("cat__", "") for n in pipe.named_steps["prep"].get_feature_names_out()]

    f1, f2 = st.columns(2)
    with f1:
        forest = bundle["models"]["Random Forest"]
        imp = pd.Series(forest.named_steps["model"].feature_importances_, index=feature_names(forest)).sort_values().tail(12)
        fig, ax = plt.subplots(figsize=(6, 4.6))
        ax.barh(imp.index, imp.values, color=PALETTE[0])
        ax.set_title("Random Forest: Top 12 Feature Importances")
        ax.set_xlabel("Importance (factor names on the left)")
        show(fig)
    with f2:
        linear = bundle["models"]["Linear Regression"]
        coef = pd.Series(linear.named_steps["model"].coef_, index=feature_names(linear))
        top = coef.reindex(coef.abs().sort_values().tail(12).index)
        fig, ax = plt.subplots(figsize=(6, 4.6))
        ax.barh(top.index, top.values, color=[PALETTE[4] if v < 0 else PALETTE[1] for v in top.values])
        ax.set_title("Linear Regression: 12 Largest Coefficients")
        ax.set_xlabel("Coefficient (marks)")
        ax.set_ylabel("Feature")
        show(fig)

    all_pred = bundle["models"][best].predict(bundle["X"])
    big = ((bundle["clean"][TARGET] - all_pred).abs() > 10)
    st.warning(f"Error analysis: {int(big.sum())} students ({big.mean() * 100:.2f}%) have a prediction error above 10 marks. "
               "Their scores are not explained by any recorded factor, which is why R² is below 100% even though most predictions are within 1 mark.")

# ======================================================================================
# TAB 5: PREDICT
# ======================================================================================
with tab_predict:
    predictor, source = load_predictor(bundle)
    st.caption(f"Prediction uses the {source}.")
    options = {c: sorted(eda[c].unique().tolist()) for c in bundle["categorical"]}
    trained_ranges = {c: (int(eda[c].min()), int(eda[c].max())) for c in NUMERIC}

    with st.form("predict_form"):
        st.markdown("**Study habits and academics**")
        a, b, c = st.columns(3)
        hours = a.number_input("Hours studied per week", 0, 80, 20)
        attendance = b.number_input("Attendance (%)", 0, 100, 80)
        previous = c.number_input("Previous exam score", 0, 100, 75)
        sleep = a.number_input("Sleep hours per night", 0, 12, 7)
        tutoring = b.number_input("Tutoring sessions per month", 0, 15, 1)
        physical = c.number_input("Physical activity (hours per week)", 0, 10, 3)

        st.markdown("**Personal and environment factors**")
        d, e, f = st.columns(3)
        motivation = d.selectbox("Motivation level", LEVELS, index=1)
        involvement = e.selectbox("Parental involvement", LEVELS, index=1)
        resources = f.selectbox("Access to resources", LEVELS, index=1)
        income = d.selectbox("Family income", LEVELS, index=1)
        teacher = e.selectbox("Teacher quality", LEVELS, index=1)
        peer = f.selectbox("Peer influence", options["Peer_Influence"], index=options["Peer_Influence"].index("Neutral"))
        parent_edu = d.selectbox("Parental education level", options["Parental_Education_Level"], index=options["Parental_Education_Level"].index("College"))
        distance = e.selectbox("Distance from home", options["Distance_from_Home"], index=options["Distance_from_Home"].index("Near"))
        school = f.selectbox("School type", options["School_Type"])
        gender = d.selectbox("Gender", options["Gender"])
        extra = e.selectbox("Extracurricular activities", options["Extracurricular_Activities"], index=options["Extracurricular_Activities"].index("Yes"))
        internet = f.selectbox("Internet access", options["Internet_Access"], index=options["Internet_Access"].index("Yes"))
        disability = d.selectbox("Learning disabilities", options["Learning_Disabilities"])
        submitted = st.form_submit_button("Predict exam score")

    if submitted:
        values = {"Hours_Studied": hours, "Attendance": attendance, "Parental_Involvement": involvement,
                  "Access_to_Resources": resources, "Extracurricular_Activities": extra, "Sleep_Hours": sleep,
                  "Previous_Scores": previous, "Motivation_Level": motivation, "Internet_Access": internet,
                  "Tutoring_Sessions": tutoring, "Family_Income": income, "Teacher_Quality": teacher, "School_Type": school,
                  "Peer_Influence": peer, "Physical_Activity": physical, "Learning_Disabilities": disability,
                  "Parental_Education_Level": parent_edu, "Distance_from_Home": distance, "Gender": gender}

        # Input validation
        problems = [f"{k.replace('_', ' ')} must be between {lo} and {hi}." for k, (lo, hi) in VALID_LIMITS.items() if not lo <= values[k] <= hi]
        if problems:
            for message in problems:
                st.error(message)
        else:
            for k, (lo, hi) in trained_ranges.items():
                if not lo <= values[k] <= hi:
                    st.warning(f"{k.replace('_', ' ')} = {values[k]} is outside the range the model was trained on ({lo} to {hi}), so the prediction is less reliable.")

            columns_order = list(bundle["X"].columns)
            row = pd.DataFrame([values])[columns_order]
            score = float(np.clip(predictor.predict(row)[0], 0, 100))
            average = eda[TARGET].mean()

            st.divider()
            r1, r2, r3 = st.columns(3)
            r1.metric("Predicted exam score", f"{score:.1f} / 100")
            r2.metric("Class average", f"{average:.1f}", delta=f"{score - average:+.1f} vs average")
            percentile = (eda[TARGET] < score).mean() * 100
            r3.metric("Better than", f"{percentile:.0f}% of students")
            st.progress(int(round(score)))

            g1, g2 = st.columns(2)
            with g1:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(eda[TARGET], bins=range(55, 102, 2), color="#b0bec5", ax=ax, label="All students")
                ax.axvline(score, color="red", linewidth=2.5, label=f"Predicted: {score:.1f}")
                ax.set_title("Where This Student Falls in the Score Distribution")
                ax.set_xlabel("Exam score")
                ax.set_ylabel("Number of students")
                ax.legend()
                show(fig)
            with g2:
                all_scores = {name: float(np.clip(m.predict(row)[0], 0, 100)) for name, m in bundle["models"].items()}
                fig, ax = plt.subplots(figsize=(6, 4))
                bars = ax.bar(all_scores.keys(), all_scores.values(), color=[MODEL_COLORS[n] for n in all_scores])
                ax.bar_label(bars, fmt="%.1f", padding=2)
                ax.set_ylim(min(all_scores.values()) - 5, max(all_scores.values()) + 5)
                ax.set_title("Prediction from Each of the Four Models")
                ax.set_xlabel("Model")
                ax.set_ylabel("Predicted score")
                ax.tick_params(axis="x", rotation=12)
                show(fig)

            st.subheader("What-if analysis: how would the score change?")
            w1, w2 = st.columns(2)
            with w1:
                grid = np.arange(60, 101, 2)
                scenario = pd.concat([row] * len(grid), ignore_index=True)
                scenario["Attendance"] = grid
                fig, ax = plt.subplots(figsize=(6, 3.8))
                ax.plot(grid, np.clip(predictor.predict(scenario), 0, 100), marker="o", color=PALETTE[1], label="Predicted score")
                ax.scatter([attendance], [score], color="red", zorder=5, label="Your input")
                ax.set_title("If Attendance Changes")
                ax.set_xlabel("Attendance (%)")
                ax.set_ylabel("Predicted score")
                ax.legend()
                show(fig)
            with w2:
                grid = np.arange(5, 41, 2)
                scenario = pd.concat([row] * len(grid), ignore_index=True)
                scenario["Hours_Studied"] = grid
                fig, ax = plt.subplots(figsize=(6, 3.8))
                ax.plot(grid, np.clip(predictor.predict(scenario), 0, 100), marker="o", color=PALETTE[3], label="Predicted score")
                ax.scatter([hours], [score], color="red", zorder=5, label="Your input")
                ax.set_title("If Weekly Study Hours Change")
                ax.set_xlabel("Hours studied per week")
                ax.set_ylabel("Predicted score")
                ax.legend()
                show(fig)
            st.caption(f"On the test data this model's typical error was about {best_row['MAE']:.2f} marks (RMSE {best_row['RMSE']:.2f}). "
                       "A few students score far above what their inputs suggest, so treat the result as an estimate, not a guarantee.")
