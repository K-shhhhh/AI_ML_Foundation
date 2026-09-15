# ANN vs Classical ML : Classification Comparison Report

## Dataset

Synthetic data via `sklearn.datasets.make_classification`, same reasoning as the regression comparison — isolates model differences from real-world data quirks (the UFC builds already covered that territory). Two separate cases, same underlying generator:

- **Binary:** 100,000 samples, 20 features, 12 informative, 3 redundant, 2 classes, `class_sep=1.0`  
- **Multiclass:** 100,000 samples, 20 features, 15 informative, 3 redundant, 4 classes, 1 cluster per class, `class_sep=1.0`

Both fully reproducible via `random_state=42`. Random Forest and Gradient Boosting run with `n_estimators=101` (reduced from 201 for runtime).

## Results

### Binary Classification

| Model | Accuracy | Precision | Recall | F1 |
| :---- | :---- | :---- | :---- | :---- |
| Logistic Regression | 0.80 | 0.80 | 0.82 | 0.81 |
| Random Forest (depth=8) | 0.91 | 0.92 | 0.90 | 0.91 |
| Gradient Boosting (depth=3) | 0.89 | 0.90 | 0.88 | 0.89 |
| **ANN (64→32→1)** | **0.99** | **0.99** | **0.99** | **0.99** |

### Multiclass Classification (4 classes)

| Model | Accuracy | Precision (weighted) | Recall (weighted) | F1 (weighted) |
| :---- | :---- | :---- | :---- | :---- |
| Logistic Regression | 0.85 | 0.85 | 0.85 | 0.85 |
| Random Forest (depth=8) | 0.92 | 0.92 | 0.92 | 0.92 |
| Gradient Boosting (depth=3) | 0.94 | 0.94 | 0.94 | 0.94 |
| **ANN (64→32→4)** | **0.99** | **0.99** | **0.99** | **0.99** |

## Findings

**1\. This is the mirror image of the regression comparison — here, Logistic Regression is the weakest model, not the strongest.** In the regression report, Linear Regression won outright because `make_regression`'s target is built as a genuinely linear combination of features. `make_classification` works differently — it places class clusters at the vertices of a hypercube in the informative-feature space, which produces decision boundaries that are not purely linear once redundant features and moderate class overlap (`class_sep=1.0`) are introduced. Logistic Regression's core assumption — a single linear decision boundary — is a worse structural match for this problem than it was for the regression target, which is exactly why it trails every other model here. Same underlying lesson as before, opposite direction: model choice should follow the true shape of the relationship, not a fixed ranking.

**2\. Random Forest and Gradient Boosting swap places between the two tasks — a real rank flip, not noise.** Random Forest edges out Gradient Boosting in binary (0.91 vs 0.89), but Gradient Boosting pulls ahead in multiclass (0.94 vs 0.92). This directly answers one of the questions this report set out to check, a model's relative ranking is *not* fixed even between two variants of the same underlying data generator. Plausible explanation: boosting's sequential error-correction has more opportunity to pay off as the problem gets harder (more classes, more boundaries to get right), while Random Forest's parallel-averaging approach doesn't scale its advantage the same way.

**3\. The ANN dominates both tasks decisively, worth being appropriately skeptical of exactly why, not just reporting the number.** 0.99 across accuracy/precision/recall/F1, in both binary and multiclass, is a very strong result — and also a good moment to ask what's driving it rather than taking it as pure evidence of ANN superiority. Contributing factors, most likely combined:

- **Non-linear decision boundaries favor a non-linear model** — same reasoning as point 1, just more pronounced for a flexible neural net than for tree ensembles.  
- **Ample clean data relative to problem complexity** — 100,000 samples for a 20-feature problem with only a handful of classes is a lot of signal for a network with this little capacity (`64→32→output`, only two hidden layers) to work with. This is the kind of setting neural nets are expected to do well in.  
- **Caveat worth stating plainly:** this report only tracks test-set metrics (by design, kept simple), there's no train-vs-test comparison here to independently confirm the ANN isn't simply memorizing extremely well rather than generalizing meaningfully better. Given the dataset size (100k, 80/20 split) genuine memorization is less likely than it was on the UFC builds' much smaller datasets, but it's a real gap in this report's evidence, not something the numbers alone rule out.

**4\. Contrast against the real, messy datasets already in the repo, this is the most useful comparison in the whole report.** On DateFruit (real data, \~900 rows, 7 classes), Logistic Regression was the *best* performer (96.1%), beating Random Forest and Naive Bayes by a wide margin, the opposite of what happened here. On the UFC classification build (real data, class imbalance, low ceiling), every model, including the ANN, hovered within \~2 points of a majority-class baseline, nowhere near the 0.99 seen here. The takeaway isn't "ANN is the best model" or "trees beat logistic regression", it's that **clean, ample, synthetic data lets every model (especially the more flexible ones) perform close to its theoretical ceiling, while real data's noise, small sample sizes, and genuine unpredictability compress the gap between models and cap how well any of them can do.** The synthetic comparison isolates model behavior; the real datasets show what actually happens once that clean-data assumption is stripped away. Both halves of this project (comparison notebooks \+ the earlier real-data builds) are more informative together than either alone.  
