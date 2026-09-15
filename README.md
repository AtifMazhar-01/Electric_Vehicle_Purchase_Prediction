# Electric Vehicle Purchase Prediction using Logistic Regression

A machine learning classification project that predicts whether a customer is likely to purchase an Electric Vehicle (EV) based on demographic, financial, commuting, vehicle, charging infrastructure, environmental, subsidy, and range-anxiety related factors.

This project uses Logistic Regression with categorical feature encoding and numerical feature standardization, followed by comprehensive classification evaluation and Kaggle submission generation.

---

## Project Overview

The objective of this project is to build a binary classification model that predicts the target variable:

`Will_Buy_EV`

The model predicts whether a customer:

* `Yes` → Will buy an EV
* `No` → Will not buy an EV

The notebook follows a complete machine learning workflow:

```text
Dataset
   │
   ▼
Data Understanding
   │
   ▼
Data Quality Checks
   │
   ▼
Exploratory Analysis
   │
   ▼
Feature / Target Separation
   │
   ▼
Train-Test Split
   │
   ▼
Categorical Encoding
   │
   ▼
Numerical Feature Scaling
   │
   ▼
Logistic Regression
   │
   ▼
Model Evaluation
   │
   ├── Accuracy
   ├── Precision
   ├── Recall
   ├── F1 Score
   ├── ROC-AUC
   ├── Confusion Matrix
   └── ROC Curve
   │
   ▼
Final Model Training
   │
   ▼
Prediction on Test Data
   │
   ▼
Submission_logistic.csv
```

---

## Dataset

The project uses the Kaggle Playground Series — Season 6 Episode 9 dataset.

The training dataset contains:

* 668,665 records
* 15 columns
* Numerical and categorical features
* One binary target variable

### Features

| Feature                       | Description / Type                |
| ----------------------------- | --------------------------------- |
| `id`                          | Unique record identifier          |
| `Age`                         | Customer age                      |
| `Annual_Income_USD`           | Annual income                     |
| `Daily_Commute_km`            | Daily commuting distance          |
| `Number_of_Cars_Owned`        | Number of cars owned              |
| `Charging_Stations_Near_Home` | Charging stations near home       |
| `Charging_Stations_Near_Work` | Charging stations near workplace  |
| `Environmental_Concern_Level` | Environmental concern score       |
| `Gender`                      | Customer gender                   |
| `City_Type`                   | Urban, Suburban, or Rural         |
| `Current_Car_Type`            | Existing vehicle type             |
| `Home_Charging_Possible`      | Whether home charging is possible |
| `Subsidy_Available`           | Whether EV subsidy is available   |
| `Range_Anxiety_Level`         | Customer range anxiety            |
| `Will_Buy_EV`                 | Target variable                   |

Target:

```text
Will_Buy_EV
```

---

## 1. Data Loading

The notebook loads three files:

```python
train = pd.read_csv("/kaggle/input/datasets/atifmazhar/playground/train.csv")
test = pd.read_csv("/kaggle/input/datasets/atifmazhar/playground/test.csv")
sample_submission = pd.read_csv("/kaggle/input/datasets/atifmazhar/playground/sample_submission.csv")
```

The training data is used for model development and evaluation, while the test data is used for generating the final predictions.

---

## 2. Data Understanding

The notebook examines the dataset using:

```python
train.head()
train.info()
train.describe()
train.isnull().sum()
train.duplicated().sum()
train.dtypes
train.columns.tolist()
```

### Dataset Characteristics

The training dataset contains:

```text
Rows    : 668,665
Columns : 15
```

Data types include:

* Integer features
* Floating-point features
* Categorical/object features

There are no missing values requiring imputation, and the notebook checks for duplicate records before modeling.

---

## 3. Target Variable Analysis

The distribution of the target variable is examined using:

```python
train["Will_Buy_EV"].value_counts()
```

Target distribution:

| Class |   Count |
| ----- | ------: |
| No    | 551,886 |
| Yes   | 116,779 |

This shows that the dataset is imbalanced, with substantially more customers who do not purchase an EV than customers who do.

The target is therefore converted into numerical labels:

```python
y = y.map({
    "Yes": 1,
    "No": 0
})
```

Mapping:

```text
Yes → 1
No  → 0
```

---

## 4. Exploratory Data Analysis

The notebook visualizes the target distribution using a count plot:

```python
sns.countplot(
    data=train,
    x="Will_Buy_EV"
)
```

Categorical features are also identified automatically:

```python
cat_features = train.select_dtypes(
    include="object"
).columns
```

The categorical variables analyzed include:

* `Gender`
* `City_Type`
* `Current_Car_Type`
* `Home_Charging_Possible`
* `Subsidy_Available`
* `Range_Anxiety_Level`
* `Will_Buy_EV`

Their individual distributions are inspected using value counts and count plots.

---

## 5. Feature and Target Separation

The identifier and target are removed from the feature matrix:

```python
X = train.drop(
    ["id", "Will_Buy_EV"],
    axis=1
)

y = train["Will_Buy_EV"]
```

The `id` column is excluded because it is an identifier rather than a meaningful predictive feature.

---

## 6. Train-Test Split

The dataset is divided into training and validation sets using:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Configuration:

| Parameter      | Value   |
| -------------- | ------- |
| Test size      | 20%     |
| Training size  | 80%     |
| Random state   | 42      |
| Stratification | Enabled |

Stratification is used to preserve the target-class distribution between the training and validation sets.

---

## 7. Categorical Feature Encoding

Categorical features are converted into numerical features using one-hot encoding:

```python
X_train = pd.get_dummies(
    X_train,
    columns=cat_features,
    dtype=int
)
```

The same transformation is applied to:

* Training data
* Validation data
* Final test data

This converts categorical values into machine-learning-compatible binary indicator features.

---

## 8. Numerical Feature Standardization

Numerical features are standardized using `StandardScaler`:

```python
scaler = StandardScaler()

X_train[num_features] = scaler.fit_transform(
    X_train[num_features]
)

X_test[num_features] = scaler.transform(
    X_test[num_features]
)

X_final_test[num_features] = scaler.transform(
    X_final_test[num_features]
)
```

The scaler is fitted only on the training data and then applied to the validation and final test data.

This ensures that numerical features are placed on a comparable scale before training Logistic Regression.

---

## 9. Logistic Regression Model

The primary classification algorithm used is Logistic Regression:

```python
lr = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr.fit(X_train, y_train)
```

The trained model generates both class predictions and probability estimates:

```python
y_pred = lr.predict(X_test)

y_prob = lr.predict_proba(X_test)[:, 1]
```

The probability predictions are later used for ROC-AUC and ROC curve analysis.

---

# Model Evaluation

The model is evaluated using multiple classification metrics rather than relying only on accuracy.

The notebook calculates:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Classification Report
* Confusion Matrix
* ROC Curve

---

## 10. Performance Results

The Logistic Regression model achieved the following validation results:

| Metric    |    Score |
| --------- | -------: |
| Accuracy  | 0.894917 |
| Precision | 0.716702 |
| Recall    | 0.658674 |
| F1 Score  | 0.686464 |
| ROC-AUC   | 0.937958 |

### Interpretation

The model achieves approximately:

```text
89.49% Accuracy
93.80% ROC-AUC
```

The ROC-AUC score indicates strong separation between customers who are likely to purchase an EV and those who are not.

However, because the target classes are imbalanced, accuracy alone does not provide the complete picture. Precision, recall, and F1 score are therefore also considered.

---

## 11. Classification Report

The notebook generates the complete classification report:

```text
              precision    recall  f1-score   support

           0       0.93      0.94      0.94    110377
           1       0.72      0.66      0.69     23356

    accuracy                           0.89    133733
   macro avg       0.82      0.80      0.81    133733
weighted avg       0.89      0.89      0.89    133733
```

The positive class (`1` / `Yes`) has:

```text
Precision : 0.72
Recall    : 0.66
F1 Score  : 0.69
```

This provides a more meaningful view of the model's ability to identify potential EV buyers.

---

## 12. Confusion Matrix

The notebook calculates the confusion matrix:

```python
cm = confusion_matrix(
    y_test,
    y_pred
)
```

and visualizes it using a heatmap.

The confusion matrix helps identify:

* True Negatives
* False Positives
* False Negatives
* True Positives

This is particularly useful for understanding the errors made when identifying potential EV buyers.

---

## 13. ROC Curve

The notebook calculates the ROC curve using predicted probabilities:

```python
fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)
```

The ROC curve compares:

```text
True Positive Rate
        vs.
False Positive Rate
```

The resulting curve is evaluated using the ROC-AUC score:

```text
ROC-AUC = 0.937958
```

A high ROC-AUC indicates that the model has strong discriminatory ability across different classification thresholds.

---

# Final Model Training

After evaluating the model on the validation set, the notebook trains a final Logistic Regression model using the complete training dataset.

The complete feature matrix is created by removing:

```text
id
Will_Buy_EV
```

Categorical features are one-hot encoded and numerical features are standardized.

The target is again converted into numerical labels:

```python
y_full = y_full.map({
    "Yes": 1,
    "No": 0
})
```

The final model is then trained:

```python
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_full, y_full)
```

---

## Final Test Predictions

The final model generates probability predictions for the Kaggle test dataset:

```python
test_prob = model.predict_proba(
    X_final_test
)[:, 1]
```

The project uses the probability of the positive class (`Yes`) as the final prediction.

---

## Submission File

The predictions are combined with the original test IDs:

```python
submission = pd.DataFrame(
    {
        "id": X_test_id,
        "Will_Buy_EV": test_prob
    }
)
```

The final submission file is saved as:

```text
Submission_logistic.csv
```

using:

```python
submission.to_csv(
    "Submission_logistic.csv",
    index=False
)
```

This produces the final file for Kaggle submission.

---

# Key Learning Outcomes

This project demonstrates a complete binary classification workflow, including:

* Loading and inspecting large tabular datasets
* Understanding numerical and categorical features
* Checking missing values and duplicates
* Analyzing target-class distribution
* Performing exploratory visualization
* Separating features and target variables
* Handling categorical variables using one-hot encoding
* Standardizing numerical features
* Performing stratified train-test splitting
* Building a Logistic Regression classifier
* Generating class probabilities
* Evaluating classification performance
* Understanding precision, recall, and F1 score
* Using confusion matrices
* Evaluating models using ROC-AUC
* Plotting ROC curves
* Retraining the final model on the complete training dataset
* Generating Kaggle-ready predictions

---

# Tech Stack

| Category          | Technologies                             |
| ----------------- | ---------------------------------------- |
| Language          | Python                                   |
| Data Manipulation | Pandas, NumPy                            |
| Visualization     | Matplotlib, Seaborn                      |
| Machine Learning  | Scikit-learn                             |
| Algorithm         | Logistic Regression                      |
| Preprocessing     | One-Hot Encoding, StandardScaler         |
| Evaluation        | Accuracy, Precision, Recall, F1, ROC-AUC |
| Environment       | Kaggle Notebook                          |
| Dataset           | Kaggle Playground Series — S6E9          |

---

# Project Structure

```text
logistic-regression/
│
├── logistic-regression.ipynb
├── Submission_logistic.csv
└── README.md
```

---

# Machine Learning Workflow

```text
Raw Customer Data
       │
       ▼
Data Inspection
       │
       ├── Missing Values
       ├── Duplicates
       ├── Data Types
       └── Statistical Summary
       │
       ▼
Exploratory Data Analysis
       │
       ▼
Feature / Target Separation
       │
       ▼
Train-Test Split
       │
       ▼
One-Hot Encoding
       │
       ▼
StandardScaler
       │
       ▼
Logistic Regression
       │
       ▼
Predictions + Probabilities
       │
       ▼
Model Evaluation
       │
       ├── Accuracy
       ├── Precision
       ├── Recall
       ├── F1
       ├── ROC-AUC
       ├── Confusion Matrix
       └── ROC Curve
       │
       ▼
Final Model
       │
       ▼
Test Probability Predictions
       │
       ▼
Submission_logistic.csv
```

---

# Results Summary

The Logistic Regression model provides a strong baseline for predicting EV purchase behavior:

```text
Accuracy  : 89.49%
Precision : 71.67%
Recall    : 65.87%
F1 Score  : 68.65%
ROC-AUC   : 93.80%
```

The model demonstrates strong overall classification performance and particularly strong ranking/discrimination capability, as reflected by its ROC-AUC score.

The project also highlights why multiple evaluation metrics are important when working with an imbalanced binary classification problem.

---

# Future Improvements

Potential extensions to this project include:

* Hyperparameter tuning for Logistic Regression
* Threshold optimization to improve recall for potential EV buyers
* Comparing Logistic Regression with tree-based classifiers
* Feature importance and coefficient analysis
* Cross-validation
* Advanced handling of class imbalance
* Model calibration
* Experimenting with ensemble learning methods
* Comparing multiple models using ROC-AUC and F1 score

---

# Conclusion

This project demonstrates an end-to-end machine learning pipeline for predicting Electric Vehicle purchase behavior.

The workflow progresses from raw customer data through data understanding, exploratory analysis, preprocessing, feature engineering, Logistic Regression training, comprehensive evaluation, and final Kaggle submission generation.

With an accuracy of approximately 89.49% and ROC-AUC of approximately 93.80%, Logistic Regression provides a strong baseline for this EV purchase prediction problem.

The project also demonstrates the importance of evaluating classification models using precision, recall, F1 score, confusion matrices, and ROC-AUC rather than relying solely on accuracy.


# Author

Atif Mazhar

Computer Engineering | Machine Learning | Data Science