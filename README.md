# LASSO Regularization & Variable Selection

**Tools:** Python · scikit-learn · LASSO · StandardScaler · GridSearchCV

---

## Problem

With 100 predictors in a dataset where only 10 are truly relevant, standard regression models overfit by including noise variables, reducing reliability and interpretability.

## Approach

Built a LASSO regression pipeline in Python using scikit-learn with `StandardScaler` and `GridSearchCV` to tune the regularization parameter across 50 candidate values via cross-validation.

## Impact

The model precisely identified all 10 relevant predictors and set the remaining 90 coefficients to zero, demonstrating how regularization improves model parsimony and supports more interpretable, production-ready predictions.

---

## Files

| File | Description |
|------|-------------|
| `portfolio_python_module2.ipynb` | Jupyter notebook with full analysis |
| `portfolio_python_module2.html` | Rendered HTML version |

---

*Part of my data & analytics portfolio — [cameronbatts.github.io](https://cameronbatts.github.io)*
