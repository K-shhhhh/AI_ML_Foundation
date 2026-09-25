# ANN vs Classical ML — Regression Comparison Report

## Dataset

Synthetic data via `sklearn.datasets.make_regression`, chosen specifically so model comparisons aren't confounded by real-world data quirks. Any performance gap here is attributable to the models, not the data.

- 100,000 samples, 20 features, 12 informative (8 pure noise), `noise=45`, `random_state=42` (fully reproducible)

## Results

| Model | Test R² |
| :---- | :---- |
| Linear Regression | 0.9357 |
| Random Forest (depth=8, 201 trees) | 0.6609 |
| Gradient Boosting (depth=3, 201 trees) | 0.9206 |
| ANN (64→128→256→1, 50 epochs) | *0.93* |

## Findings

**1\. Random Forest is the clear underperformer here — and there's a specific, explainable reason, not just "worse model."** `make_regression` builds its target as a **linear combination** of the informative features by construction. Linear Regression is, unsurprisingly, extremely well-suited to exactly that structure — it *is* a linear combination, so the model class matches the data's true generating process almost perfectly. Random Forest, by contrast, builds decision trees using **axis-aligned splits** (each split asks "is feature X above or below some threshold"), which is a fundamentally clumsy way to approximate a smooth linear relationship — a tree needs many splits to piece together what a straight line represents in one step. At `max_depth=8`, the forest simply doesn't have enough splitting depth to closely approximate a 12-feature linear function, hence the noticeably lower R² (0.66 vs 0.94) on both train and test.

**2\. Gradient Boosting closes most of that gap despite also being tree-based** — because boosting corrects errors iteratively, stage by stage, effectively building a much more refined approximation than a single-depth forest can, even with individually shallow trees (`max_depth=3` here). This is a genuinely useful, non-obvious finding: "tree-based" alone doesn't predict performance — *how* the trees are combined (averaged in parallel vs. sequentially error-corrected) matters enormously when the true relationship is linear.

**3\. This is the mirror image of the UFC classification finding \-** There, tree ensembles (Random Forest, Gradient Boosting) leaned harder into majority-class bias than a plain ANN did, on real, messy, non-linear categorical data. Here, on clean synthetic data with a genuinely linear underlying structure, the *simplest* model (Linear Regression) wins outright, and the more flexible tree ensemble (Random Forest) is worst. Together, these two results demonstrate a broader point worth stating explicitly: **no single model family is universally best — which one wins depends entirely on how well the model's own structural assumptions match the true shape of the data's relationships.**  
