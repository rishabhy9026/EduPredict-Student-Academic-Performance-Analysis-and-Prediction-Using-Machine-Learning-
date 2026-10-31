# EduPredict: Viva Preparation

## Part A: Questions and answers

### PROJECT INTRODUCTION

**1. What is your project about?**

EduPredict analyses what factors affect students' exam scores and predicts a student's expected score using machine learning. I used the Kaggle Student Performance Factors dataset (6,607 students) and four regression models. *Hinglish:* Yeh project batata hai ki kaun se factors (attendance, study hours, etc.) marks se jude hain, aur ek model banata hai jo marks predict karta hai.

**2. Why is this a regression problem and not classification?**

The target Exam_Score is a continuous number (55 to 100), not a category like Pass/Fail. Predicting a number is regression. *Hinglish:* Marks ek number hai, isliye regression.

**3. What questions does your project answer?**

Does study time help? Does attendance matter? How important are previous scores? Do sleep and tutoring matter? Which categorical factors (motivation, parental involvement, resources) relate to marks? And which model predicts best?

### DATASET

**4. Where did you get the dataset and how big is it?**

From Kaggle: Student Performance Factors (StudentPerformanceFactors.csv). It has 6,607 rows and 20 columns. I used the real file and did not create any data.

**5. What are the target and the features?**

Target: Exam_Score. Features: 19 columns such as Hours_Studied, Attendance, Previous_Scores, Sleep_Hours, Tutoring_Sessions, Motivation_Level, Parental_Involvement, etc.

**6. How many numerical and categorical features are there?**

6 numerical (Hours_Studied, Attendance, Sleep_Hours, Previous_Scores, Tutoring_Sessions, Physical_Activity) and 13 categorical (for example Motivation_Level, Gender, School_Type).

### DATA PREPROCESSING, MISSING VALUES, DUPLICATES

**7. Why do we preprocess data?**

Models need clean, numeric, consistent input. Missing values, wrong values and text categories would cause errors or poor results. *Hinglish:* Kachra data se kachra result aata hai (garbage in, garbage out).

**8. Which columns had missing values and how did you handle them?**

Teacher_Quality (78), Parental_Education_Level (90) and Distance_from_Home (67), about 1 percent each. All are categorical, so I filled them with the most frequent value (mode) using SimpleImputer inside the pipeline.

**9. Why mode and not mean? Why not delete the rows?**

Mean works only for numbers; these are categories, so the most frequent category is the natural choice. Deleting 229 rows would waste data for a very small problem.

**10. Did you find duplicates?**

No, there were 0 duplicate rows, so nothing was removed. I did not add any artificially.

**11. Did you find invalid or inconsistent data?**

Category labels were consistent. One record had Exam_Score = 101, which is impossible on a 100-mark scale, so I removed that one row. *Hinglish:* 100 se upar marks possible nahi, isliye woh data-entry error tha.

**12. How did you detect outliers?**

With box plots and the IQR rule: values below Q1 - 1.5 x IQR or above Q3 + 1.5 x IQR are flagged. Hours_Studied (43), Tutoring_Sessions (430) and Exam_Score (104) had flagged values.

**13. Why did you keep the outliers?**

They were valid, realistic values (for example 4+ tutoring sessions or a score of 90). IQR only says a value is unusual, not wrong. I removed only the impossible 101.

### EDA

**14. What is EDA and why do we do it?**

Exploratory Data Analysis means studying the data with statistics and graphs before modelling to understand patterns, relationships and problems. *Hinglish:* Model banane se pehle data ko samajhna.

**15. What does the exam score distribution look like?**

Bell-shaped, centred near 67.2 with standard deviation 3.9; most students score 65 to 69, with a few high scores up to 100.

**16. Which factor is most related to exam score?**

Attendance (correlation 0.58), then Hours_Studied (0.45), Previous_Scores (0.17) and Tutoring_Sessions (0.15). Sleep is about -0.02, essentially zero.

**17. Does correlation mean causation?**

No. Correlation only shows that two things move together. High attendance may go with high marks because of engagement or motivation, not only attendance itself. *Hinglish:* Saath-saath badhna matlab ek doosre ka karan hona zaroori nahi.

**18. Does sleep affect performance in your data?**

No meaningful association: the average score is about 67 for every sleep duration and r = -0.02 (p = 0.19). This is only about recorded hours, not sleep quality, and does not mean sleep is unimportant for health.

### FEATURE ENGINEERING

**19. What features did you engineer and why?**

Study_Category, Attendance_Category and Sleep_Category, made by grouping numbers into bands with pandas cut. They make results easy to read (low / medium / high) and allowed a study x attendance heatmap. No Exam_Score information was used, so there is no leakage.

**20. Why did you not use these new features in the models?**

They only repeat information already in the numerical columns, and grouping numbers into bands loses detail. So I kept the original numerical columns for training.

### REGRESSION AND THE FOUR ALGORITHMS

**21. What is regression?**

A supervised learning method that predicts a continuous number from input features.

**22. How does Linear Regression work?**

It fits a weighted sum: score = b0 + b1 x Hours + b2 x Attendance + ... The weights are chosen to minimise the squared errors. *Hinglish:* Yeh ek seedhi line (formula) fit karta hai jo data ke sabse kareeb ho.

**23. How does a Decision Tree Regressor work?**

It asks yes/no questions (for example Attendance above 85?) to split students into groups; the prediction is the average score of the final group. A very deep tree memorises the data (overfits).

**24. How does Random Forest work and why is it better than one tree?**

It builds 200 trees on random samples of rows and features and averages their predictions. Averaging cancels individual trees' mistakes, giving more stable predictions. *Hinglish:* Bahut saare trees ka average.

**25. How does KNN regression work?**

To predict a student, it finds the K (here 20) most similar students in the training data and averages their scores. It needs scaled features because it uses distances.

### SPLIT, SCALING, ENCODING, PIPELINES

**26. Why do we split into training and testing sets?**

To check the model on data it has never seen. If we tested on the training data, we would not know whether it can predict new students. I used 80 percent training and 20 percent testing.

**27. What does random_state = 42 do?**

It fixes the random shuffle so the split and results are the same every time we run the code (reproducibility). 42 is just a conventional number.

**28. Why did you use the same split for all models?**

So the comparison is fair: every model is tested on exactly the same students.

**29. What is feature scaling and why did you use StandardScaler?**

It puts numeric features on a similar scale (mean 0, standard deviation 1). KNN uses distance, so Attendance (60-100) would otherwise dominate Tutoring (0-8). I scaled for KNN and Linear Regression; trees do not need it.

**30. What is One-Hot Encoding and why use it?**

It converts a category into 0/1 columns (for example Motivation_Level_High = 1 or 0), because models need numbers and plain numbering like Low=0, Medium=1 can create a false order for unordered categories.

**31. What is a pipeline and what is data leakage?**

A pipeline chains preprocessing and model so everything is learned from the training data only. Data leakage is when test-set information influences training (for example filling missing values using the whole dataset), which gives over-optimistic scores. The pipeline prevents that.

### EVALUATION METRICS

**32. What is R-squared?**

The share of variation in exam scores that the model explains. 1 is perfect, 0 is as bad as always predicting the average. My best model: 0.825, meaning it explains about 82.5 percent of the variation.

**33. What is MAE?**

Mean Absolute Error: the average size of the error in marks, ignoring direction. Linear Regression had MAE = 0.42, so on average it is off by under half a mark.

**34. What is MSE?**

Mean Squared Error: the average of squared errors. Squaring penalises big mistakes heavily. Its unit is marks squared. Linear Regression: 2.31.

**35. What is RMSE and how is it different from MAE?**

RMSE is the square root of MSE, so it is in marks. It is always at least as large as MAE and grows when there are a few big errors. Linear Regression: RMSE 1.52 versus MAE 0.42; the big gap tells us there are a few large errors.

**36. Which model is the best and why?**

Linear Regression: highest R² (0.825), lowest MAE, MSE and RMSE on the test set. Random Forest 0.728, KNN 0.648, Decision Tree 0.602. The relationships in this data are mostly straight-line and additive, which suits Linear Regression. I did not assume it in advance; the code picks the best test R².

### OVERFITTING, VALIDATION, LIMITATIONS, APP

**37. What is overfitting? Did any model overfit?**

Overfitting is when a model memorises training data and performs worse on new data. Random Forest shows mild overfitting (train R² 0.83, test 0.73).

**38. Test R² is higher than train R² for Linear Regression. Is that wrong?**

It is not a bug. The training set contained 44 students with unexplained very high scores, the test set only 5. Excluding those 5, test R² would be 0.985. The cross-validated R² (0.717) is a more cautious estimate.

**39. What is cross-validation and why did you use it?**

The training data is split into 5 parts; the model is trained on 4 and evaluated on 1, repeated 5 times. It gives a steadier estimate than a single split. The ranking of models stayed the same: Linear Regression first.

**40. What are the limitations of your project?**

Associations are not causes; the dataset looks unusually regular (maybe simulated), so results may not hold in real classrooms; 49 students' scores are unexplained by any feature; one exam score only; light tuning.

**41. How does your Streamlit app make predictions?**

app.py loads the saved pipeline (models/best_model.pkl) with joblib, validates the inputs, builds a one-row table in the training column order and calls model.predict. The pipeline applies the same imputation, scaling and encoding used in training. There are no hardcoded predictions.

**42. How can the project be improved?**

Use real institutional data and more features, try tuned gradient-boosting models, give prediction ranges, check fairness across groups, and deploy the app online.

## Part B: Presentation script (about 4 minutes)

**Slide 1: Introduction (30 sec).** Good morning. My project is EduPredict, a system that studies the factors behind students' exam performance and predicts the expected exam score using machine learning. *(Hinglish if needed: Yeh project batata hai ki marks kin cheezon se jude hain.)*

**Slide 2: Dataset (30 sec).** I used the Kaggle Student Performance Factors dataset: 6,607 students, 19 input factors such as study hours, attendance, previous scores, sleep, tutoring, motivation and parental involvement, and the target is Exam Score.

**Slide 3: Cleaning (30 sec).** Three categorical columns had about 1 percent missing values, which I filled with the most frequent value inside a pipeline. There were no duplicates. One record had a score of 101, which is impossible, so I removed it. The remaining outliers were valid values, so I kept them.

**Slide 4: EDA and objectives (60 sec).** I explored the data with histograms, count plots, box plots, scatter plots, a correlation heatmap and a pair plot, and studied five objectives. Attendance has the strongest relationship with exam score, with a correlation of 0.58. Study hours come second at 0.45. Previous scores are weak at 0.17. Tutoring has a small positive link, and sleep shows no relationship. Parental involvement and access to resources are linked to about two marks of difference between high and low levels. These are associations, not proof of cause.

**Slide 5: Feature engineering (15 sec).** I created Study, Attendance and Sleep categories to make the patterns easier to read, but kept the original numeric columns for the models.

**Slide 6: Models (45 sec).** I trained four regression models in scikit-learn pipelines with the same 80/20 split and random state 42: Linear Regression, Decision Tree, Random Forest and KNN. Preprocessing used imputation, one-hot encoding and scaling, fitted on training data only.

**Slide 7: Results (60 sec).** I evaluated them with R-squared, MAE, MSE and RMSE. Linear Regression was the best: R-squared 0.825, an average error of 0.42 marks and RMSE of 1.52. Random Forest was second with 0.73, then KNN with 0.65 and Decision Tree with 0.60. Cross-validation gave the same order. About 99 percent of students are predicted within one mark; a small group of 49 students have scores no feature explains, which limits R-squared.

**Slide 8: App (20 sec).** I built a Streamlit app called EduPredict. It takes the inputs, validates them, loads the saved best model and shows the predicted score. It uses the real trained model.

**Slide 9: Conclusion (30 sec).** Attendance and study hours are the most important factors, and Linear Regression is the best-fit model for this dataset. Limitations are that findings are associations and the dataset looks unusually regular. In future, I would use real institutional data and more features. Thank you.

