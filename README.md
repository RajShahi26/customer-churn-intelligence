
# 📊 Customer Churn Intelligence

### Explainable Customer Churn Prediction & Retention Decision Support System

An end-to-end machine learning system that predicts customer churn probability, identifies high-risk customers, explains individual predictions using SHAP, and generates rule-based retention recommendations.

---

## 🚀 Project Overview

Customer churn is a major business problem where identifying customers likely to leave can help organizations prioritize retention efforts.

This project develops an **explainable churn prediction system** that goes beyond binary classification.

The system performs:

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Multiple machine learning model comparison
- XGBoost hyperparameter tuning
- Cross-validation
- Probability threshold optimization
- SHAP-based model explainability
- Customer-level risk segmentation
- Rule-based retention recommendations
- Interactive Streamlit deployment

---

## 🎯 Objectives

The main objectives of this project are:

1. Predict the probability that a customer will churn.
2. Identify customers requiring retention attention.
3. Explain why an individual customer is predicted to churn.
4. Convert model insights into actionable retention recommendations.
5. Deploy the complete workflow through an interactive dashboard.

---

## 📊 Dataset

The project uses the **IBM Telco Customer Churn dataset**.

The dataset contains customer demographic information, account information, service subscriptions, billing information, and churn status.

### Dataset Statistics

- Total customers: **7,043**
- Features: **20**
- Target variable: `Churn`
- Churn = Yes: **1,869**
- Churn = No: **5,174**

The target distribution is approximately:

- No Churn: **73.5%**
- Churn: **26.5%**

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

### Missing / Invalid Values

`TotalCharges` was stored as an object column and contained 11 blank values.

These records corresponded to customers with zero tenure. Therefore, their `TotalCharges` values were imputed as **0** based on the billing context.

### Feature Handling

- `customerID` was retained for customer-level identification but excluded from model training.
- Numerical features were standardized using `StandardScaler`.
- Categorical features were encoded using `OneHotEncoder`.
- `handle_unknown="ignore"` was used for categorical preprocessing.

### Data Split

The data was divided into:

- Training set: **80%**
- Test set: **20%**

A stratified split was used to preserve the churn distribution.

---

## 🔎 Exploratory Data Analysis

EDA was performed using business-driven questions.

Important observations included:

### Contract Type

Month-to-month customers showed substantially higher churn rates than customers with one-year or two-year contracts.

### Tenure

Customers with shorter tenure showed substantially higher churn rates.

### Monthly Charges

Customers who churned had higher average monthly charges than customers who remained.

### Payment Method

Electronic check customers represented an important churn-related segment.

### Additional Services

Customers without Online Security and Technical Support showed higher churn tendencies.

These observations represent associations in the dataset and should not be interpreted as causal effects.

---

## 🤖 Machine Learning Models

Multiple classification algorithms were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- Support Vector Machine
- Artificial Neural Network

### Baseline Model Comparison

| Model | Accuracy | Churn Precision | Churn Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8055 | 0.6572 | 0.5588 | 0.6040 | 0.8421 |
| Decision Tree | 0.7211 | 0.4755 | 0.4920 | 0.4836 | 0.6477 |
| Random Forest | 0.7835 | 0.6186 | 0.4813 | 0.5414 | 0.8206 |
| XGBoost | 0.8055 | 0.6701 | 0.5267 | 0.5898 | 0.8438 |
| SVM | 0.7913 | 0.6410 | 0.4840 | 0.5518 | 0.7902 |
| ANN | 0.7984 | 0.6471 | 0.5294 | 0.5824 | 0.8406 |

---

## ⚙️ XGBoost Optimization

XGBoost was selected for the final system because it provided strong discrimination performance while also supporting efficient tree-based SHAP explanations.

Hyperparameters were optimized using **GridSearchCV with 5-fold Stratified Cross-Validation**.

The test set was kept separate during model selection and threshold optimization.

The final pipeline contains:

```text
Preprocessing
     ↓
One-Hot Encoding + Standard Scaling
     ↓
XGBoost
