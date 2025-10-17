# DA5401 Assignment 6: Handling Missing Data via Imputation & Classification

**Author:** Basavaraj A Naduvinamani  
**Roll No:** DA25C005  

---

## Project Overview

This project focuses on handling missing values in the **UCI Credit Card Default Clients Dataset** and evaluating the impact of various imputation strategies on downstream classification tasks. The main goal is to demonstrate how different imputation methods affect predictive performance and robustness in logistic regression modeling.

---

## Dataset

- **Source:** [UCI Credit Card Default Clients Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)  
- **Samples:** 30,000  
- **Features:** 24 numerical/categorical variables (excluding target)  
- **Target:** `default.payment.next.month` (binary classification)  

**Note:** Artificial missingness (~7%) was introduced in the `AGE` and `BILL_AMT1` columns to simulate a **Missing At Random (MAR)** scenario.

---

## Imputation Strategies

| Dataset | Imputation Strategy | Key Notes |
|---------|-------------------|-----------|
| A       | Median Imputation  | Baseline; robust to outliers and skewed distributions |
| B       | Linear Regression  | Predicts missing `AGE` using other numeric features; assumes MAR |
| C       | Non-Linear Regression (KNN) | Predicts missing `AGE` using KNN; captures potential non-linear relationships |
| D       | Listwise Deletion  | Removes all rows with missing values; smaller dataset, no imputation bias |

---

## Modeling & Evaluation

- **Classifier:** Logistic Regression  
- **Data Split:** 70% training / 30% testing (stratified)  
- **Standardization:** All numeric features scaled using `StandardScaler`  
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score  

**F1-Score Comparison Across Datasets (Logistic Regression)**

| Dataset | Class 0 F1 | Class 1 F1 | Macro Avg F1 | Weighted Avg F1 |
|---------|------------|------------|--------------|----------------|
| A       | 0.887      | 0.351      | 0.619        | 0.768          |
| B       | 0.887      | 0.353      | 0.620        | 0.7687         |
| C       | 0.887      | 0.354      | 0.621        | 0.7691         |
| D       | 0.888      | 0.360      | 0.624        | 0.7718         |

---

## Key Observations

- **Minority Class Performance:** All models exhibit low recall for the minority class (~0.24), highlighting the persistent challenge of class imbalance.  
- **Median vs. Regression Imputation:** Linear and non-linear regression imputation offer minimal improvement over simple median imputation in this dataset.  
- **Listwise Deletion:** Slightly higher overall accuracy (0.8103) and weighted F1-score (0.7718) but reduced dataset size may limit generalization.  
- **Recommendation:** Median imputation is sufficient for this dataset, balancing simplicity, robustness, and predictive performance.

---

## Visualizations

- Distribution of `AGE` before and after each imputation method.  
- F1-score comparison across datasets (Logistic Regression).  

---



```bash
pip install -r requirements.txt
