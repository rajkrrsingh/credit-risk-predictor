# Credit Risk Predictor

**Author:** —
Raj Kumar Singh
---

## Executive summary

This project builds a **binary classification** model to predict whether a loan applicant will **default** on their loan. Using the Kaggle Credit Risk dataset (~32.5k applications, 12 features), we perform exploratory data analysis, cleaning, feature engineering, and train a baseline Logistic Regression model. The goal is to support data-driven lending: identify high-risk borrowers before approval while keeping credit accessible to creditworthy applicants.

---

## Rationale

Loan defaults cost financial institutions billions annually and can contribute to broader economic instability. Accurate risk prediction helps lenders:

- **Reduce losses** by flagging high-risk applicants before approval.
- **Make informed decisions** on terms (amount, rate, or additional checks) instead of a flat yes/no.
- **Keep access fair** for creditworthy applicants who might otherwise be denied.

---

## Research question

**Can we predict whether a loan applicant will default using their financial history, demographics, and loan characteristics?**  
We treat this as a binary classification problem: target variable is `loan_status` (default vs. no default), with the aim of finding the best model to identify applicants with the highest credit risk.

---

## Data sources

- **[Kaggle Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset/data)**  
  - ~32,581 rows × 12 columns.  
  - Features include: `person_age`, `person_income`, `person_home_ownership`, `person_emp_length`, `loan_intent`, `loan_grade`, `loan_amnt`, `loan_int_rate`, `loan_status` (target), `loan_percent_income`, `cb_person_default_on_file`, `cb_person_cred_hist_length`.  
  - Place the CSV as `data/credit_risk_dataset.csv` in the project root to run the notebook.

---

## Methodology

- **EDA & cleaning:** Structure, types, missing values, summary statistics; handling missing data (e.g. `loan_int_rate`, `person_emp_length`).
- **Outliers:** Identification and handling where appropriate.
- **Feature engineering:** Derived features (e.g. income-to-loan ratio, high debt burden flag); scaling/normalization for linear models.
- **Categorical encoding:** One-hot and label encoding for tree-based and linear models.
- **Baseline model:** Logistic Regression with scaled numeric and derived features, evaluated with accuracy, precision, recall, F1, and ROC-AUC.
- **Planned:** Additional classifiers (Decision Tree, KNN, SVM), cross-validation, grid search for hyperparameters, and comparison of evaluation metrics.

---

## Results

- **Baseline:** Logistic Regression (class-weighted, scaled numeric + derived features) provides a test-set benchmark; ROC-AUC ~0.84 on the prepared data.
- **Key drivers of risk (from EDA and presentation):** Debt burden (loan vs. income), interest rate, loan grade (A–F), prior default on file, loan intent, and home ownership.
- The notebook includes visualizations (distributions, correlations, ROC curve) and a reproducible pipeline from raw data to baseline predictions.

---

## Next steps

- Encode categoricals and include them in the baseline.
- Train and compare Decision Tree, KNN, and SVM with the same evaluation metrics.
- Apply begging and boosting to check if it help improve the performance.
- Use cross-validation and grid search to select and tune the best model.
- Document the chosen model and deployment considerations (e.g. threshold for “high risk”).

---

## Outline of project

- [**Credit Risk Predictor — EDA & baseline model**](credit-risk-predictor.ipynb)
- [**Presentation draft (technical & non-technical)**](docs/presentation-draft.md)

---

## Setup

- **Python:** 3.x with dependencies from `requirements.txt`.
- **Data:** Download the [Kaggle Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset/data) and save as `data/credit_risk_dataset.csv`.
- **Run:** Open `credit-risk-predictor.ipynb` in Jupyter and run all cells.

```bash
pip install -r requirements.txt
```

---

## Contact and further information

For questions or collaboration, open an issue or reach out via the repository.
