# ANN vs Classical Machine Learning: A Controlled Comparison

## Problem Statement

Across earlier projects (UFC fight data, DateFruit, powerplant), model comparisons were always entangled with the quirks of real-world data — missing values, class imbalance, leakage risk, and in some cases a genuine low ceiling on how predictable the target was at all. Those projects answered "how well can any model do on *this specific, messy dataset*," but never cleanly isolated "how do different model families actually behave relative to each other, all else being equal."

This project addresses that gap directly: **compare an Artificial Neural Network against three classical machine learning algorithms i.e Logistic/Linear Regression, Random Forest, and Gradient Boosting, on fully controlled, synthetic data**, so that any performance difference observed is attributable to the models themselves, not to hidden data problems.

## Objectives

1. Generate synthetic datasets with known, controllable properties (sample size, feature count, informative vs. noise features, class separability) using `sklearn.datasets.make_regression` and `sklearn.datasets.make_classification`.  
2. Build and train one ANN per task, following the same pipeline discipline established in prior projects (train/test split before any fitting, proper tensor conversion, best-epoch checkpointing).  
3. Train the same three classical models — Linear/Logistic Regression, Random Forest, Gradient Boosting — on identical data splits for a fair comparison.  
4. Evaluate all four models per task using consistent test-set metrics, and analyse *why* any performance gaps appear, rather than just reporting numbers.  
5. Repeat the comparison across three task types — regression, binary classification, and multiclass classification — to check whether any model's relative ranking holds across task types or flips.

## Scope

| Task | Dataset Generator | Notebook |
| :---- | :---- | :---- |
| Regression | `make_regression` — 100,000 samples, 20 features, 12 informative | `ANN_ML_Regression_formatted.ipynb` |
| Binary Classification | `make_classification` — 100,000 samples, 20 features, 12 informative, 2 classes | `ANN_ML_Classification_Binary.ipynb` |
| Multiclass Classification | `make_classification` — 100,000 samples, 20 features, 15 informative, 4 classes | `ANN_ML_Classification_Multiclass.ipynb` |

## Evaluation

- **Regression:** test-set R²  
- **Classification (binary & multiclass):** test-set accuracy, precision, recall, F1

Kept deliberately simple, no train/validation loss curves for the classical models, single test-set metrics only, so the comparison stays readable and the focus stays on cross-model differences rather than per-model diagnostics.

## Expected Outcome

A clear, data-isolated answer to which model family performs best under which conditions (linear vs. non-linear relationships, binary vs. multiclass boundaries), plus a documented contrast against this repo's earlier real-data projects — showing how much of a model's real-world performance ceiling comes from the model itself versus the data it's given.

See `ANN_vs_ML_Regression_Report.md` and `ANN_vs_ML_Classification_Report.md` for full results and findings.  
