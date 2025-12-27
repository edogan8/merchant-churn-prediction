# Merchant Churn Prediction & Retention System 🛡️

This project aims to predict which marketplace sellers (merchants) are likely to stop selling on the platform (churn). Using Machine Learning (XGBoost), it identifies at-risk sellers early, allowing the business to take proactive retention actions.

## 📊 Business Impact (ROI Analysis)
* **Problem:** Acquiring a new seller is 5x more expensive than retaining an existing one.
* **Solution:** An Early Warning System that predicts churn with **91% Recall**.
* **Financial Impact:** In the test simulation, the model identified **173 at-risk sellers**. Assuming a 50% success rate in retention efforts, this model saves an estimated **865,000 TL** in annual revenue.

## 🛠️ Project Architecture
1.  **Data Preprocessing:**
    * Handled missing values using Median/Mode imputation.
    * Addressed class imbalance (17% Churn) using **SMOTE** (Synthetic Minority Over-sampling Technique).
2.  **Model Selection:**
    * Compared Logistic Regression, Random Forest, and XGBoost.
    * **Champion Model:** XGBoost (Selected for its high Recall score).
3.  **Key Insights:**
    * **Tenure:** New sellers are the most likely to churn (Onboarding is critical).
    * **Complaints:** Sellers with open complaints are highly correlated with churn.

## 📈 Model Performance
| Model | Accuracy | F1 Score | Recall (Key Metric) |
|-------|----------|----------|---------------------|
| Logistic Regression | 80% | 0.58 | 82% |
| Random Forest | 95% | 0.86 | 83% |
| **XGBoost (Selected)** | **97%** | **0.91** | **91%** |

## 💻 Tech Stack
* Python 3.13
* XGBoost, Scikit-Learn
* Pandas, NumPy
* Imbalanced-learn (SMOTE)
* Joblib (Model Serialization)

---
*Created by Muhammed Emir Doğan*