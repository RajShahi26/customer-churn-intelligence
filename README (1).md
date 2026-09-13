# 📊 Customer Churn Intelligence

### Explainable Customer Churn Prediction & Retention Decision Support System

An end-to-end machine learning system that predicts customer churn probability, identifies high-risk customers, explains individual predictions using SHAP, and generates rule-based retention recommendations.

🚀 **Live Demo:** [Customer Churn Intelligence](https://raj-churn-intelligence.streamlit.app)

---

## 📌 Project Overview

Customer churn is a major business problem where identifying customers likely to leave can help organizations prioritize retention efforts.

This project develops an **explainable customer churn prediction and decision-support system** that goes beyond simple binary classification.

The system follows the workflow:

```text
Customer Data
     ↓
Data Cleaning & Preprocessing
     ↓
Exploratory Data Analysis
     ↓
Multiple ML Model Comparison
     ↓
XGBoost Hyperparameter Tuning
     ↓
Threshold Optimization
     ↓
Customer Churn Probability
     ↓
Risk Segmentation
     ↓
SHAP Explainability
     ↓
Retention Recommendations
     ↓
Interactive Streamlit Dashboard
```

The main objective is to connect **machine learning predictions with explainability and business-oriented decision support**.

---

## 🎯 Objectives

The main objectives of this project are:

1. Predict the probability that a customer will churn.
2. Identify customers requiring retention attention.
3. Explain why an individual customer is predicted to churn.
4. Segment customers according to their predicted churn risk.
5. Generate rule-based retention recommendations.
6. Deploy the complete workflow through an interactive dashboard.

---

## 📊 Dataset

The project uses the **IBM Telco Customer Churn dataset**.

The dataset contains customer demographic information, account information, service subscriptions, billing information, and churn status.

### Dataset Statistics

| Attribute | Value |
|---|---:|
| Total Customers | 7,043 |
| Input Features | 20 |
| Target Variable | `Churn` |
| Churn = Yes | 1,869 |
| Churn = No | 5,174 |
| Churn Rate | 26.5% |
| No Churn Rate | 73.5% |

The dataset contains both numerical and categorical variables such as:

- Tenure
- Monthly Charges
- Total Charges
- Contract Type
- Internet Service
- Online Security
- Technical Support
- Payment Method
- Paperless Billing
- Demographic attributes

---

## 🧹 Data Preprocessing

### Missing / Invalid Values

`TotalCharges` was originally stored as an object column and contained **11 blank values**.

These records corresponded to customers with zero tenure. Based on the billing context, these values were converted to numeric and imputed as:

```text
TotalCharges = 0
```

### Feature Handling

- `customerID` was retained for customer-level identification but excluded from model training.
- Numerical features were standardized using `StandardScaler`.
- Categorical features were encoded using `OneHotEncoder`.
- `handle_unknown="ignore"` was used to safely process unseen categorical values.

### Train-Test Split

The dataset was divided using a stratified split:

```text
Training Set: 80%
Test Set:     20%
```

Resulting datasets:

```text
Training samples: 5,634
Test samples:     1,409
```

Stratification was used to preserve the churn distribution in both datasets.

---

## 🔎 Exploratory Data Analysis

EDA was performed using business-driven questions rather than generating plots without a specific objective.

### Contract Type

Month-to-month customers showed substantially higher churn rates than customers with one-year or two-year contracts.

### Tenure

Customers with shorter tenure showed substantially higher churn rates, while customers with longer tenure generally exhibited lower churn rates.

### Monthly Charges

Customers who churned had higher average monthly charges than customers who remained.

### Payment Method

Electronic check customers represented an important churn-related segment.

### Additional Services

Customers without Online Security and Technical Support showed higher churn tendencies.

### Important Note

These observations represent **associations in the dataset** and should not be interpreted as causal relationships.

---

## 🤖 Machine Learning Models

Multiple classification algorithms were evaluated to establish a baseline:

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

The baseline comparison showed that **Logistic Regression and XGBoost provided strong overall performance**, while XGBoost was selected for further optimization.

---

## ⚙️ XGBoost Optimization

XGBoost was selected for the final system because it provided strong discrimination performance and is well suited to structured tabular data.

Hyperparameters were optimized using:

- `GridSearchCV`
- 5-fold Stratified Cross-Validation
- ROC-AUC as the optimization metric

The test set was kept completely separate during model selection.

### Final Pipeline

```text
Raw Customer Data
        ↓
ColumnTransformer
   ├── Numerical → StandardScaler
   └── Categorical → OneHotEncoder
        ↓
XGBoost Classifier
        ↓
Churn Probability
```

The final model was implemented as a single scikit-learn `Pipeline`, ensuring that preprocessing and prediction remain consistent during deployment.

---

## 🎚️ Probability Threshold Optimization

The default classification threshold of `0.50` is not necessarily the best operating point for a churn-retention problem.

For retention campaigns, missing a customer who is likely to churn can be more costly than contacting some customers who ultimately would not churn.

Therefore, the classification threshold was optimized using **out-of-fold predictions on the training data**.

The threshold producing the highest out-of-fold F1 score was:

```text
Optimal Threshold = 0.36
```

This threshold was then applied to the untouched test set.

### Threshold Comparison

| Metric | Threshold 0.50 | Threshold 0.36 |
|---|---:|---:|
| Accuracy | 0.8055 | 0.7857 |
| Churn Precision | 0.6736 | 0.5776 |
| Churn Recall | 0.5187 | **0.7166** |
| Churn F1 | 0.5861 | **0.6396** |
| ROC-AUC | 0.8479 | 0.8479 |

The threshold change increased churn recall from approximately **51.9% to 71.7%**, while reducing precision.

This represents a deliberate **business operating-point decision** rather than a universal improvement in every metric.

---

## 🏆 Final Model Performance

The final XGBoost model was evaluated on the held-out test set.

### Final Test Metrics

| Metric | Result |
|---|---:|
| Accuracy | **78.57%** |
| Churn Precision | **57.76%** |
| Churn Recall | **71.66%** |
| Churn F1 | **63.96%** |
| ROC-AUC | **0.8479** |
| Classification Threshold | **0.36** |

### Interpretation

The model achieved a **ROC-AUC of 0.8479**, indicating strong ability to distinguish between customers who churn and those who remain.

The threshold optimization prioritizes **churn recall**, allowing the retention team to identify a larger proportion of potentially churning customers.

---

## 🔍 SHAP Explainability

Model performance alone is not sufficient for a business-facing prediction system.

The project uses **SHAP (SHapley Additive exPlanations)** to understand model behavior.

SHAP is used at two levels:

### Global Explainability

The most influential features identified by the model included:

1. Contract — Month-to-month
2. Tenure
3. Online Security — No
4. Monthly Charges
5. Technical Support — No
6. Internet Service — Fiber optic
7. Payment Method — Electronic check
8. Contract — Two year
9. Total Charges
10. Paperless Billing — No

### Local Explainability

For an individual customer, SHAP values show which features push the model prediction toward higher or lower churn probability.

For example, a high-risk customer may have a combination of:

- Very short tenure
- Month-to-month contract
- High monthly charges
- No Online Security
- No Technical Support

The system displays these contributors directly in the Streamlit application.

### Important Note

SHAP explains **model behavior and associations**. It does not prove that a feature is a causal reason for customer churn.

---

## 👥 Customer Risk Segmentation

Customers are assigned risk categories based on predicted churn probability.

| Risk Level | Probability |
|---|---:|
| 🟢 Low | `< 0.30` |
| 🟡 Medium | `0.30 – < 0.50` |
| 🟠 High | `0.50 – < 0.70` |
| 🔴 Critical | `≥ 0.70` |

The deployed dashboard contains predictions for the **1,409 held-out test customers**.

### Test-Set Risk Distribution

| Risk Level | Customers | Percentage |
|---|---:|---:|
| Low | 860 | 61.0% |
| Medium | 261 | 18.5% |
| High | 185 | 13.1% |
| Critical | 103 | 7.3% |
| **High + Critical** | **288** | **20.4%** |

This allows a business team to prioritize customers instead of treating every customer equally.

---

## 💡 Retention Recommendation Engine

The project includes a rule-based recommendation layer that converts customer attributes and model explanations into potential retention actions.

Examples include:

### Contract Recommendation

For high-risk month-to-month customers:

> Consider offering a suitable long-term contract incentive.

### Early-Tenure Recommendation

For customers with very short tenure:

> Prioritize the customer for an early-tenure retention campaign.

### Online Security

For customers without Online Security:

> Consider offering an Online Security package or trial.

### Technical Support

For customers without Technical Support:

> Consider offering a Technical Support package or trial.

### Payment Method

For electronic-check customers:

> Consider encouraging migration to an automatic payment method.

### Pricing / Bundle Review

For customers with relatively high monthly charges:

> Review pricing and service bundle options and consider a targeted retention offer.

### Important Note

These recommendations are **business-rule heuristics** based on customer attributes and model outputs.

They are not causal treatment recommendations and should be validated through controlled business experiments such as A/B testing.

---

## 🖥️ Streamlit Application

The final system is deployed using **Streamlit Community Cloud**.

### Application Features

The dashboard provides three main sections:

### 1. 📊 Dashboard

Provides an overview of:

- Number of test customers
- High + Critical risk customers
- Critical-risk customers
- Predicted churn customers
- Risk distribution
- Model performance
- Operating threshold

### 2. 👤 Customer Analysis

Users can search for an individual customer and view:

- Customer profile
- Churn probability
- Predicted churn status
- Risk level
- SHAP contributors
- Retention recommendations

### 3. 🚨 High-Risk Customers

Provides a prioritized table of high-risk customers with:

- Customer ID
- Churn probability
- Risk level
- Predicted churn status
- Downloadable CSV output

---

## 🚀 Live Demo

### Try the deployed application

👉 **[Open Customer Churn Intelligence](https://raj-churn-intelligence.streamlit.app)**

The application is publicly deployed using Streamlit Community Cloud.

---

## 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │   IBM Telco Dataset  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Data Cleaning & EDA  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Feature Preprocessing│
                    │ StandardScaler       │
                    │ OneHotEncoder        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Tuned XGBoost Model │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴───────────┐
                    ▼                      ▼
          ┌──────────────────┐   ┌──────────────────┐
          │ Churn Probability│   │  SHAP Explainer  │
          └────────┬─────────┘   └────────┬─────────┘
                   │                      │
                   ▼                      ▼
          ┌──────────────────┐   ┌──────────────────┐
          │ Risk Segmentation│   │ Customer-level   │
          │ Low/Medium/High/ │   │ Explanation      │
          │ Critical         │   └────────┬─────────┘
          └────────┬─────────┘            │
                   │                      │
                   └──────────┬───────────┘
                              ▼
                    ┌──────────────────────┐
                    │ Retention Rules      │
                    │ & Recommendations    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Streamlit Dashboard  │
                    └──────────────────────┘
```

---

## 📁 Project Structure

```text
customer-churn-intelligence/
│
├── app.py
│
├── final_xgb_pipeline.pkl
├── churn_threshold.pkl
├── shap_explainer.pkl
│
├── customer_risk_predictions.csv
│
├── requirements.txt
│
└── README.md
```

### File Description

| File | Purpose |
|---|---|
| `app.py` | Streamlit application |
| `final_xgb_pipeline.pkl` | Trained XGBoost preprocessing + model pipeline |
| `churn_threshold.pkl` | Optimized classification threshold |
| `shap_explainer.pkl` | Saved SHAP TreeExplainer |
| `customer_risk_predictions.csv` | Customer risk predictions |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

---

## 🛠️ Technologies Used

### Programming

- Python

### Data Analysis

- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- XGBoost

### Deep Learning

- TensorFlow / Keras
- Artificial Neural Network used during baseline model comparison

### Explainable AI

- SHAP

### Visualization

- Matplotlib
- Seaborn

### Deployment

- Streamlit
- Streamlit Community Cloud

### Development Environment

- Google Colab
- Jupyter Notebook
- GitHub

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/RajShahi26/customer-churn-intelligence.git
```

Move into the project directory:

```bash
cd customer-churn-intelligence
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will then be available locally through the Streamlit URL shown in the terminal.

---

## 🔐 Reproducibility

The deployed application uses saved model artifacts:

```text
final_xgb_pipeline.pkl
churn_threshold.pkl
shap_explainer.pkl
```

The preprocessing and model are stored together in the XGBoost pipeline to ensure that the same transformations are applied during inference.

The deployment environment uses pinned dependency versions to maintain compatibility with the serialized model artifacts.

---

## ⚠️ Limitations

This project has several limitations:

1. The dataset represents a specific telecom customer population and may not generalize to other industries.
2. Model predictions represent probabilities and are not certain outcomes.
3. SHAP explanations describe model behavior rather than causality.
4. Retention recommendations are heuristic and should not be interpreted as proven interventions.
5. The current application does not include real-time customer transaction or CRM data.
6. The deployed risk dashboard represents the held-out test-set predictions rather than live production customers.
7. The classification threshold of `0.36` reflects the selected F1-oriented operating point and may need to change according to actual business costs.

---

## 🔮 Future Improvements

Potential improvements include:

- Cost-sensitive threshold optimization based on actual retention campaign costs.
- Probability calibration for more reliable churn probabilities.
- Hyperparameter optimization using Bayesian optimization or Optuna.
- Model monitoring and drift detection.
- Integration with CRM systems.
- Real-time customer scoring.
- A/B testing of retention recommendations.
- Uplift modeling to identify customers who are most likely to respond positively to an intervention.
- Automated retraining pipelines.
- More advanced recommendation strategies using causal inference.

---

## 📚 Key Learning Outcomes

Through this project, the following concepts were applied:

- Data cleaning
- Exploratory Data Analysis
- Feature preprocessing
- Categorical encoding
- Feature scaling
- Stratified train-test splitting
- Classification algorithms
- Cross-validation
- Hyperparameter tuning
- XGBoost
- ROC-AUC
- Precision, Recall and F1-score
- Classification threshold optimization
- Out-of-fold predictions
- SHAP explainability
- Customer risk scoring
- Business rule engines
- Model persistence
- Streamlit deployment
- GitHub-based deployment

---

## 💼 Project Value

The project demonstrates an end-to-end **Machine Learning → Explainability → Decision Support → Deployment** workflow.

Instead of stopping at:

```text
"Will this customer churn?"
```

the system attempts to answer:

```text
"How likely is the customer to churn?"
             ↓
"How risky is the customer?"
             ↓
"Why does the model consider the customer risky?"
             ↓
"What retention action could be considered?"
```

This makes the project more representative of a practical **Data Science decision-support system** rather than a standalone machine learning model.

---

## 👨‍💻 Author

**Raj Shahi**

M.Tech — Farm Machinery  
Indian Institute of Technology Kharagpur

### Areas of Interest

- Data Science
- Machine Learning
- Deep Learning
- Explainable AI
- Computer Vision
- AI Applications

---

## ⭐ Project Links

- 🚀 **Live Application:** [Customer Churn Intelligence](https://raj-churn-intelligence.streamlit.app)
- 💻 **GitHub Repository:** [RajShahi26/customer-churn-intelligence](https://github.com/RajShahi26/customer-churn-intelligence)

---

## 📄 License

This project is intended for educational, portfolio, and demonstration purposes.
