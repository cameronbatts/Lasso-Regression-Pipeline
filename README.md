# LASSO Regularization & Variable Selection via Cross-Validation

**Tools:** Python · scikit-learn · LASSO Regression · GridSearchCV · StandardScaler · Linear Regression · NumPy · Joblib

---

## Business Problem

High-dimensional datasets often contain large numbers of irrelevant variables that can reduce model interpretability and increase overfitting risk. This project explores how LASSO regularization can improve regression model performance by automatically selecting the most relevant predictors while shrinking unnecessary coefficients toward zero.

The objective was to compare LASSO regression against traditional linear regression in a controlled high-dimensional environment to evaluate feature selection effectiveness and model simplification.

---

## Project Approach

- Simulated a high-dimensional regression dataset using `make_regression`
- Generated:
  - 1,000 observations
  - 100 predictors
  - 10 truly informative variables
- Built a machine learning pipeline using:
  - `StandardScaler`
  - `Lasso`
- Tuned the regularization parameter (`alpha`) using GridSearchCV
- Compared:
  - standard linear regression
  - LASSO regularization
- Evaluated how effectively LASSO isolated the truly informative variables
- Serialized the trained regression model using Joblib

---

## Analytical Focus Areas

- Regularization
- Feature selection
- High-dimensional regression
- Cross-validation
- Model simplification
- Overfitting reduction
- Pipeline workflows
- Coefficient shrinkage
- Model persistence

---

## Key Results

### LASSO Variable Selection Performance

- True informative predictors in dataset: **10**
- Non-zero coefficients selected by LASSO: **10**
- Non-zero coefficients from standard linear regression: **100**

The LASSO model successfully identified the informative variables while eliminating irrelevant predictors, demonstrating the effectiveness of regularization and automated feature selection.

---

## Repository Structure

```text
data/        -> generated or processed datasets
Images/      -> regression and coefficient visualizations
models/      -> serialized trained models
notebooks/   -> Jupyter notebook workflows
reports/     -> exported reports and outputs
```

---

## Files

| File | Description |
|------|-------------|
| `notebooks/lasso_regression_pipeline.ipynb` | Full LASSO regression and cross-validation workflow |
| `models/linear_regression.joblib` | Serialized trained regression model saved using Joblib |
| `reports/` | Exported reports and project outputs |

---

# Author

Cameron Batts

GitHub: https://github.com/Cameron-Batts

Portfolio: https://cameron-batts.github.io

