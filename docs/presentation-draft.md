# Credit Risk Predictor — Presentation Draft

**One-pager for technical and non-technical audiences**

---

## For non-technical audiences (executive / business)

### What we did
We analyzed a dataset of **over 32,000 loan applications** to predict whether an applicant is likely to **default** on their loan. The goal is to help lenders make better decisions: reduce losses from bad loans while still lending to people who can repay.

### Why it matters
- Loan defaults cost the industry **billions** every year.
- A simple, data-driven model can flag **high-risk applicants** before approval.
- Lenders can then offer different terms (e.g. lower amounts, higher rates) or extra checks instead of a flat “yes/no.”

### What we found (in plain language)
- **Debt burden** (loan size vs. income) and **interest rate** are strong signals: higher burden and higher rates go with more defaults.
- **Loan grade** (A–F) lines up with risk: worse grades have higher default rates.
- **Prior default on file** (from credit bureau) is a strong predictor: people who defaulted before are more likely to default again.
- **Purpose of loan** (e.g. medical, education, debt consolidation) and **home ownership** (rent vs. own vs. mortgage) also relate to default rates.

### What we built
- We cleaned the data (fixed missing values and obvious errors), added a few derived metrics (e.g. income-to-loan ratio, high debt burden flag), and built a **baseline model** that predicts default from applicant and loan features.
- This baseline is a **starting point**. Next steps are to try other models (e.g. decision trees, more advanced methods) and tune them for the best trade-off between catching defaults and not over-rejecting good applicants.

### Takeaway
We have a **working baseline** that uses applicant and loan data to predict default risk. The analysis and visualizations in the notebook support both **explaining** the data and **improving** the model in later phases.

---

## For technical audiences (data science / engineering)

### Objective
**Binary classification:** predict `loan_status` (default = 1, no default = 0) from applicant and loan features to minimize credit risk while supporting fair, data-driven lending.

### Data
- **Source:** Kaggle Credit Risk Dataset (~32.5k rows).
- **Target:** `loan_status` (0/1).
- **Features:** Demographics (age, income, home ownership, employment length), loan (intent, grade, amount, interest rate, percent of income), and credit bureau (prior default flag, credit history length).

### EDA and cleaning
- **Missing:** `loan_int_rate` missing for a subset; imputed with median by `loan_grade`.
- **Outliers / errors:** `person_emp_length` capped at 40 years (e.g. 123 → data error).
- **Feature engineering:**  
  - `income_to_loan_ratio`, `debt_burden_high` (loan_percent_income > 0.4),  
  - `cb_person_default_on_file_int`, `loan_grade_ord` (A=0 … F=5).
- **Relationships:** Default rate by loan grade, prior default, intent, home ownership; correlation of numeric features with target (e.g. `loan_percent_income`, `loan_int_rate`).

### Visualizations
- Target balance; distributions of interest rate, percent income, income.
- Correlation heatmap (numeric features + target).
- Default rate by loan grade, prior default, loan intent, home ownership.

### Baseline model
- **Model:** Logistic Regression (class_weight="balanced"), StandardScaler on numeric + derived features.
- **Split:** 75% train / 25% test, stratified on `loan_status`.
- **Metrics (test):** Accuracy, Precision, Recall, F1, ROC-AUC; confusion matrix; ROC curve.
- **Purpose:** Reproducible benchmark before adding categorical encoding, other classifiers (Decision Tree, KNN, SVM), cross-validation, and hyperparameter tuning.

### Next steps (from project draft)
- Encode categoricals (one-hot / label encoding).
- Compare classifiers: Logistic Regression, Decision Tree, KNN, SVM.
- Cross-validation and grid search for hyperparameters.
- Final model selection by Accuracy, Precision, Recall, F1, AUC-ROC and business constraints (e.g. minimizing false negatives for default).

### Reproducibility
- Notebook: `credit-risk-predictor.ipynb` (setup, EDA, cleaning, feature engineering, plots, baseline training and evaluation).
- Data: `data/credit_risk_dataset.csv`.
- Random seed: 42 for splits and model.

---
