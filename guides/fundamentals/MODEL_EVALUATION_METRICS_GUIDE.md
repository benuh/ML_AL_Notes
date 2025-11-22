# 📊 Model Evaluation Metrics Guide

**"Not everything that counts can be counted, and not everything that can be counted counts."** - Albert Einstein

Choosing the right evaluation metric is crucial. This guide helps you select and interpret metrics for your ML models.

---

## 📋 Table of Contents

- [Why Metrics Matter](#why-metrics-matter)
- [Statistical Foundations and Theory](#statistical-foundations-and-theory)
  - [Metrics as Statistical Estimators](#metrics-as-statistical-estimators)
  - [Confidence Intervals for Classification Metrics](#confidence-intervals-for-classification-metrics)
  - [Hypothesis Testing for Comparing Models](#hypothesis-testing-for-comparing-models)
  - [Decision-Theoretic Foundations](#decision-theoretic-foundations)
  - [Theoretical Properties and Bounds](#theoretical-properties-and-bounds)
- [Classification Metrics](#classification-metrics)
- [Regression Metrics](#regression-metrics)
- [Ranking Metrics](#ranking-metrics)
- [Clustering Metrics](#clustering-metrics)
- [Custom Metrics](#custom-metrics)
- [Metric Selection Guide](#metric-selection-guide)

---

## Why Metrics Matter

### Same Model, Different Metrics

```python
# Fraud detection model
y_true = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # 10% fraud
y_pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Predicts "no fraud" always

accuracy = accuracy_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

print(f"Accuracy: {accuracy:.0%}")  # 90% - Looks great!
print(f"Recall: {recall:.0%}")      # 0% - Catches zero fraud!

# Model is useless but accuracy looks good!
```

**Key lesson:** **Choose metrics that align with business goals.**

---

## Statistical Foundations and Theory

### Metrics as Statistical Estimators

**Every evaluation metric is an estimator of expected performance on the true data distribution.**

**Formal Definition:**
Let P(X, Y) be the true data distribution. The population risk is:
```
R(f) = E_(X,Y)~P [L(f(X), Y)]
```
where L is the loss function and f is our classifier/regressor.

Given a test set D_test = {(x₁, y₁), ..., (xₙ, yₙ)} drawn i.i.d. from P, the empirical risk is:
```
R̂(f) = (1/n) Σᵢ L(f(xᵢ), yᵢ)
```

**Key Properties:**

1. **Unbiasedness:** E[R̂(f)] = R(f)
2. **Consistency:** R̂(f) →^p R(f) as n → ∞ (by Law of Large Numbers)
3. **Variance:** Var(R̂(f)) = σ²/n where σ² = Var(L(f(X), Y))

**Theorem 1 (Central Limit Theorem for Metrics):**
For large n, the sampling distribution of any metric M̂ (with finite variance σ²) is approximately normal:
```
√n(M̂ - M) →^d N(0, σ²)

M̂ ~ N(M, σ²/n) approximately
```

This enables confidence intervals and hypothesis tests.

---

### Confidence Intervals for Classification Metrics

#### Binomial-Based Metrics (Accuracy, Precision, Recall)

These metrics are proportions with **binomial variance**.

**Theorem 2 (Exact Binomial CI):**
For a metric M = k/n (k successes in n trials), the Clopper-Pearson exact CI is:
```
Lower bound: Beta⁻¹(α/2; k, n-k+1)
Upper bound: Beta⁻¹(1-α/2; k+1, n-k)
```
where Beta⁻¹ is the inverse beta CDF.

**Practical Approximation (Wald CI):**
For large n, using normal approximation:
```
M̂ ± z_{α/2} · √(M̂(1-M̂)/n)
```
where z_{α/2} = 1.96 for 95% confidence.

**Implementation:**
```python
import numpy as np
from scipy import stats

def binomial_confidence_interval(metric_value, n_samples, confidence=0.95):
    """
    Wilson score interval (more accurate than Wald for small n or extreme p)
    """
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denominator = 1 + z**2 / n_samples
    center = (metric_value + z**2 / (2 * n_samples)) / denominator
    margin = z * np.sqrt(metric_value * (1 - metric_value) / n_samples + z**2 / (4 * n_samples**2))
    margin /= denominator

    return (center - margin, center + margin)

# Example: Accuracy = 0.85 on n=100 samples
accuracy = 0.85
n = 100
ci_lower, ci_upper = binomial_confidence_interval(accuracy, n)
print(f"95% CI for accuracy: [{ci_lower:.3f}, {ci_upper:.3f}]")
# Output: [0.770, 0.910]
```

**Variance Formula:**
```
Var(Accuracy) = Accuracy · (1 - Accuracy) / n

SE(Accuracy) = √(Accuracy · (1 - Accuracy) / n)
```

**For Precision and Recall:**
```
SE(Precision) = √(Precision · (1 - Precision) / n_predicted_positive)

SE(Recall) = √(Recall · (1 - Recall) / n_actual_positive)
```

**Important:** The effective sample size differs:
- Accuracy: n = total samples
- Precision: n = TP + FP (predicted positives)
- Recall: n = TP + FN (actual positives)

#### ROC-AUC Confidence Intervals: DeLong's Method

**Theorem 3 (AUC as Mann-Whitney U Statistic):**
The ROC-AUC has an elegant interpretation:
```
AUC = P(score(X⁺) > score(X⁻))
```
where X⁺ ~ P(X | Y=1) and X⁻ ~ P(X | Y=0).

**Equivalently:**
```
AUC = (1/(n₊·n₋)) Σᵢ∈pos Σⱼ∈neg 𝟙[score(xᵢ) > score(xⱼ)]
```
where n₊, n₋ are counts of positive and negative examples.

**DeLong's Variance Formula (1988):**

Let S₁₀ = average score assigned to negatives by the classifier ranking each positive higher, and S₀₁ similarly:
```
S₁₀ = (1/n₊) Σᵢ∈pos [number of negatives ranked below xᵢ] / n₋

S₀₁ = (1/n₋) Σⱼ∈neg [number of positives ranked above xⱼ] / n₊
```

Then:
```
Var(AUC) = (1/n₊)·Var(S₁₀) + (1/n₋)·Var(S₀₁)
```

**Implementation:**
```python
from sklearn.metrics import roc_auc_score
import numpy as np
from scipy import stats

def auc_confidence_interval_delong(y_true, y_scores, confidence=0.95):
    """
    Compute AUC confidence interval using DeLong's method

    Reference: DeLong et al. (1988) "Comparing the areas under two or more
    correlated receiver operating characteristic curves"
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # Compute AUC
    auc = roc_auc_score(y_true, y_scores)

    # Get indices
    pos_indices = np.where(y_true == 1)[0]
    neg_indices = np.where(y_true == 0)[0]

    n_pos = len(pos_indices)
    n_neg = len(neg_indices)

    # Compute structural components
    pos_scores = y_scores[pos_indices]
    neg_scores = y_scores[neg_indices]

    # For each positive, count negatives ranked below it
    V10 = np.zeros(n_pos)
    for i, pos_score in enumerate(pos_scores):
        V10[i] = np.mean(pos_score > neg_scores)

    # For each negative, count positives ranked above it
    V01 = np.zeros(n_neg)
    for j, neg_score in enumerate(neg_scores):
        V01[j] = np.mean(pos_scores > neg_score)

    # Compute variance
    var_auc = np.var(V10) / n_pos + np.var(V01) / n_neg
    se_auc = np.sqrt(var_auc)

    # Confidence interval
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    ci_lower = auc - z * se_auc
    ci_upper = auc + z * se_auc

    return auc, ci_lower, ci_upper, se_auc

# Example
y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.7, 0.3, 0.9])

auc, ci_lower, ci_upper, se = auc_confidence_interval_delong(y_true, y_scores)
print(f"AUC: {auc:.3f} ± {se:.3f}")
print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
```

**Hanley-McNeil Approximation (1982):**

Simpler but less accurate alternative:
```
SE(AUC) ≈ √[(AUC·(1-AUC) + (n₊-1)(Q₁-AUC²) + (n₋-1)(Q₂-AUC²)) / (n₊·n₋)]
```
where:
- Q₁ = AUC / (2 - AUC)
- Q₂ = 2·AUC² / (1 + AUC)

#### Bootstrap Confidence Intervals for Any Metric

For complex metrics without analytical variance formulas:

```python
from sklearn.utils import resample

def bootstrap_metric_ci(y_true, y_pred_or_scores, metric_func,
                        n_bootstrap=1000, confidence=0.95):
    """
    Bootstrap confidence interval for any metric

    Args:
        y_true: True labels
        y_pred_or_scores: Predictions or scores
        metric_func: Function that computes the metric
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level

    Returns:
        metric_value, ci_lower, ci_upper
    """
    n = len(y_true)
    bootstrap_metrics = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = resample(range(n), n_samples=n, replace=True)
        y_true_boot = [y_true[i] for i in indices]
        y_pred_boot = [y_pred_or_scores[i] for i in indices]

        # Compute metric on bootstrap sample
        metric_boot = metric_func(y_true_boot, y_pred_boot)
        bootstrap_metrics.append(metric_boot)

    # Original metric
    metric_value = metric_func(y_true, y_pred_or_scores)

    # Percentile confidence interval
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_metrics, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_metrics, 100 * (1 - alpha / 2))

    return metric_value, ci_lower, ci_upper

# Example: Bootstrap CI for F1 score
from sklearn.metrics import f1_score

f1, ci_lower, ci_upper = bootstrap_metric_ci(
    y_true, y_pred,
    lambda y_t, y_p: f1_score(y_t, y_p),
    n_bootstrap=2000
)
print(f"F1: {f1:.3f}, 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
```

---

### Hypothesis Testing for Comparing Models

#### McNemar's Test (Binary Classification)

**Use case:** Compare two classifiers on the same test set.

**Null Hypothesis:** Both classifiers have equal error rates.

**Theorem 4 (McNemar's Test Statistic):**
Construct a 2×2 contingency table of classifier agreements/disagreements:
```
                Classifier B Correct    Classifier B Wrong
Classifier A
Correct              a                      b
Classifier A
Wrong                c                      d
```

Under H₀, the test statistic follows a chi-squared distribution:
```
χ² = (b - c)² / (b + c) ~ χ²(1)
```

**More accurate (with continuity correction):**
```
χ² = (|b - c| - 1)² / (b + c)
```

**Implementation:**
```python
from statsmodels.stats.contingency_tables import mcnemar

def compare_classifiers_mcnemar(y_true, y_pred_A, y_pred_B):
    """
    McNemar's test for paired binary classifiers

    Returns: p-value (reject H₀ if p < 0.05)
    """
    # Create contingency table
    correct_A = (y_true == y_pred_A)
    correct_B = (y_true == y_pred_B)

    # Count disagreements
    a = np.sum(correct_A & correct_B)      # Both correct
    b = np.sum(correct_A & ~correct_B)     # A correct, B wrong
    c = np.sum(~correct_A & correct_B)     # A wrong, B correct
    d = np.sum(~correct_A & ~correct_B)    # Both wrong

    contingency_table = np.array([[a, b], [c, d]])

    # Perform McNemar's test
    result = mcnemar(contingency_table, exact=True)  # exact=True for small samples

    print(f"McNemar's Test Results:")
    print(f"Both correct: {a}, A only: {b}, B only: {c}, Both wrong: {d}")
    print(f"Test statistic: {result.statistic:.3f}")
    print(f"p-value: {result.pvalue:.4f}")

    if result.pvalue < 0.05:
        print("Conclusion: Classifiers have significantly different error rates")
    else:
        print("Conclusion: No significant difference between classifiers")

    return result.pvalue

# Example
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
y_pred_A = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 1])
y_pred_B = np.array([1, 0, 0, 1, 0, 1, 0, 0, 1, 0])

p_value = compare_classifiers_mcnemar(y_true, y_pred_A, y_pred_B)
```

**When to use:**
- ✅ Same test set for both classifiers
- ✅ Binary classification
- ✅ Paired comparison (same examples)

**When NOT to use:**
- ❌ Different test sets
- ❌ Multi-class (use Cochran's Q test instead)

#### Paired t-Test for Metrics (Regression or AUC)

For continuous metrics (RMSE, AUC, etc.), use paired t-test on per-fold cross-validation results:

```python
from scipy.stats import ttest_rel

def compare_models_cv(model_A, model_B, X, y, cv=5, metric_func=None):
    """
    Compare two models using cross-validated paired t-test

    Args:
        model_A, model_B: Sklearn-compatible models
        X, y: Features and labels
        cv: Number of folds
        metric_func: Metric function (default: accuracy for classification)

    Returns:
        p_value, mean_diff, scores_A, scores_B
    """
    from sklearn.model_selection import cross_val_score

    if metric_func is None:
        # Default metric
        metric_func = 'accuracy'

    # Get cross-validated scores
    scores_A = cross_val_score(model_A, X, y, cv=cv, scoring=metric_func)
    scores_B = cross_val_score(model_B, X, y, cv=cv, scoring=metric_func)

    # Paired t-test
    t_stat, p_value = ttest_rel(scores_A, scores_B)
    mean_diff = np.mean(scores_A - scores_B)

    print(f"Model A: {np.mean(scores_A):.4f} ± {np.std(scores_A):.4f}")
    print(f"Model B: {np.mean(scores_B):.4f} ± {np.std(scores_B):.4f}")
    print(f"Mean difference: {mean_diff:.4f}")
    print(f"t-statistic: {t_stat:.3f}")
    print(f"p-value: {p_value:.4f}")

    if p_value < 0.05:
        better_model = "A" if mean_diff > 0 else "B"
        print(f"Conclusion: Model {better_model} is significantly better (p < 0.05)")
    else:
        print("Conclusion: No significant difference between models")

    return p_value, mean_diff, scores_A, scores_B
```

**Important Caveat (Nadeau & Bengio 2003):**
Standard t-test **underestimates variance** due to fold overlap. Use corrected variance:
```
Var_corrected = Var_apparent · (1 + n_test / n_train)
```
See [Cross-Validation Guide](./CROSS_VALIDATION_GUIDE.md) for details.

#### Permutation Test (Non-Parametric Alternative)

When assumptions of parametric tests don't hold:

```python
def permutation_test(y_true, y_pred_A, y_pred_B, metric_func, n_permutations=10000):
    """
    Non-parametric permutation test for comparing classifiers

    H₀: Both classifiers perform equally (metric difference = 0)
    """
    # Observed difference
    metric_A = metric_func(y_true, y_pred_A)
    metric_B = metric_func(y_true, y_pred_B)
    observed_diff = metric_A - metric_B

    # Permutation distribution under H₀
    perm_diffs = []
    for _ in range(n_permutations):
        # Randomly swap predictions for each example
        swap_mask = np.random.rand(len(y_true)) > 0.5
        y_pred_A_perm = np.where(swap_mask, y_pred_B, y_pred_A)
        y_pred_B_perm = np.where(swap_mask, y_pred_A, y_pred_B)

        metric_A_perm = metric_func(y_true, y_pred_A_perm)
        metric_B_perm = metric_func(y_true, y_pred_B_perm)
        perm_diffs.append(metric_A_perm - metric_B_perm)

    # Two-tailed p-value
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))

    print(f"Observed difference: {observed_diff:.4f}")
    print(f"p-value: {p_value:.4f}")

    return p_value, observed_diff, perm_diffs

# Example
from sklearn.metrics import accuracy_score

p_val, obs_diff, perm_dist = permutation_test(
    y_true, y_pred_A, y_pred_B,
    metric_func=accuracy_score,
    n_permutations=10000
)
```

---

### Decision-Theoretic Foundations

#### Bayes Optimal Classifier and Loss Functions

**Theorem 5 (Bayes Optimal Decision Rule):**
Given a loss function L(ŷ, y), the Bayes optimal predictor minimizes expected loss:
```
f*(x) = argmin_ŷ E_Y|X [L(ŷ, Y) | X = x]
       = argmin_ŷ Σ_y L(ŷ, y) · P(Y = y | X = x)
```

**For binary classification (Y ∈ {0, 1}):**

**0-1 Loss:** L(ŷ, y) = 𝟙[ŷ ≠ y]
```
f*(x) = argmin_ŷ [ŷ·P(Y=0|x) + (1-ŷ)·P(Y=1|x)]
      = 𝟙[P(Y=1|x) > 0.5]
```
**Optimal decision:** Predict class with highest posterior probability.
**Associated metric:** Accuracy (misclassification rate).

**Log Loss:** L(p̂, y) = -[y log p̂ + (1-y) log(1-p̂)]
```
f*(x) = P(Y=1|x)  (the true posterior probability)
```
**Optimal decision:** Output calibrated probabilities.
**Associated metric:** Cross-entropy / log loss.

**Cost-Sensitive Loss:** L(ŷ, y) = C_FP·ŷ·(1-y) + C_FN·(1-ŷ)·y
```
f*(x) = 𝟙[P(Y=1|x) > C_FP / (C_FP + C_FN)]
```
**Optimal decision:** Threshold adjusted by cost ratio.
**Associated metric:** Expected cost.

**Example:**
```python
# Medical diagnosis: FN (miss disease) costs 100x more than FP (false alarm)
C_FN = 100
C_FP = 1
optimal_threshold = C_FP / (C_FP + C_FN)  # = 0.0099

# Predict positive if P(disease | symptoms) > 0.0099 (very conservative)
y_pred_optimal = (y_proba > optimal_threshold).astype(int)
```

#### ROC Analysis and Operating Points

**Theorem 6 (ROC Curve is Concave):**
The ROC curve of a proper scoring classifier (one that outputs monotonic transformations of P(Y=1|X)) is concave.

**Proof:**
Any point on the ROC curve corresponds to a threshold t:
```
(FPR(t), TPR(t)) = (P(ŝ(X) > t | Y=0), P(ŝ(X) > t | Y=1))
```

For any two thresholds t₁ < t₂ and λ ∈ [0, 1], we can construct a randomized threshold that achieves:
```
(FPR(λ), TPR(λ)) = λ·(FPR(t₁), TPR(t₁)) + (1-λ)·(FPR(t₂), TPR(t₂))
```
by randomly choosing threshold t₁ with probability λ and t₂ otherwise.

Since the deterministic curve lies on or above all such convex combinations, it is concave. ∎

**Consequence:** Any classifier with a non-concave ROC curve can be improved by randomization.

**Theorem 7 (AUC as Ranking Quality):**
```
AUC = ∫₀¹ TPR(FPR) d(FPR)
    = P(score(X⁺) > score(X⁻))
    = (1/(2·n₊·n₋)) · Σᵢ∈pos Σⱼ∈neg [𝟙(sᵢ > sⱼ) + 0.5·𝟙(sᵢ = sⱼ)]
```

**Interpretation:** AUC is the probability that a randomly chosen positive example is ranked higher than a randomly chosen negative example.

**Perfect ranking:** AUC = 1.0 (all positives ranked above all negatives)
**Random ranking:** AUC = 0.5 (expected value for random scores)
**Inverted ranking:** AUC = 0.0 (flip predictions to get AUC = 1.0)

#### Precision-Recall Analysis

**Theorem 8 (Precision-Recall Relationship):**
Unlike ROC space, Precision-Recall curves are **not** invariant to class imbalance.

Given:
- π = P(Y=1) (prevalence of positive class)
- Precision = TP / (TP + FP)
- Recall = TPR = TP / (TP + FN)
- FPR = FP / (FP + TN)

We can derive:
```
Precision = (Recall · π) / (Recall · π + FPR · (1 - π))
```

**Key insight:** As π → 0 (rare positive class), FPR dominates:
```
Precision ≈ (Recall · π) / (FPR · (1 - π)) → 0 as π → 0
```

Even with low FPR = 0.01, if π = 0.001 (0.1% positive):
```
Precision = (1.0 · 0.001) / (1.0 · 0.001 + 0.01 · 0.999) ≈ 0.09 (9%)
```

**This explains why ROC-AUC can be misleadingly high on imbalanced data while precision is low.**

**Theorem 9 (Optimal F₁ Threshold):**
For a probabilistic classifier with scores s(x), the threshold t* that maximizes F₁ score satisfies:
```
t* = argmax_t [2·Precision(t)·Recall(t) / (Precision(t) + Recall(t))]
```

In general, this must be found numerically. However, for perfectly calibrated classifiers:
```
t* ≈ (TP + FN) / n  (the true prevalence in the test set)
```

---

### Theoretical Properties and Bounds

#### Sample Complexity for Metric Estimation

**Theorem 10 (Hoeffding's Bound for Empirical Error):**
Let R̂(f) be the empirical error rate on n samples. Then with probability at least 1 - δ:
```
|R̂(f) - R(f)| ≤ √(ln(2/δ) / (2n))
```

**Practical implication:** To estimate error within ε with 95% confidence:
```
n ≥ ln(2/0.05) / (2ε²) ≈ 1.84 / ε²
```

**Examples:**
- ε = 0.01 (1% accuracy): n ≥ 18,400 samples
- ε = 0.05 (5% accuracy): n ≥ 736 samples

**For AUC specifically (Cortes & Mohri 2004):**
```
|AUC - AUC_true| ≤ O(√(1/(n₊·n₋)))
```

**Implication:** Need many examples from **both** classes for accurate AUC estimation. If minority class has only 10 examples, standard error is ≈ 1/√(10·n₋) ≈ 0.1 even with large n₋.

#### Metric Calibration and Proper Scoring Rules

**Definition (Proper Scoring Rule):**
A scoring rule S(p̂, y) is **proper** if:
```
E_Y [S(p, Y)] ≤ E_Y [S(p̂, Y)]  for all p̂, when p = P(Y=1)
```

I.e., reporting the true probability minimizes expected loss.

**Theorem 11 (Log Loss is Strictly Proper):**
Log loss is minimized when p̂ = P(Y=1|x):
```
L(p̂) = E_Y [-Y log p̂ - (1-Y) log(1-p̂)]
     = -P(Y=1) log p̂ - P(Y=0) log(1-p̂)

dL/dp̂ = -P(Y=1)/p̂ + P(Y=0)/(1-p̂) = 0
⟹ p̂ = P(Y=1)  ✓
```

**Other proper scoring rules:**
- **Brier score:** (y - p̂)²
- **Spherical score:** y·p̂ / √(p̂² + (1-p̂)²)

**Improper scoring rules:**
- Accuracy (not continuous in probabilities)
- AUC (only depends on ranking, not calibration)

#### Regression Metrics: Optimality and Robustness

**Theorem 12 (Loss Functions and Optimal Predictors):**

| Loss Function | Optimal Predictor | Robustness to Outliers |
|---------------|-------------------|------------------------|
| L₂ (MSE)      | E[Y \| X]        | Low (quadratic penalty) |
| L₁ (MAE)      | Median[Y \| X]   | High (linear penalty) |
| Huber(δ)      | Hybrid           | Medium (adaptive) |
| Quantile(τ)   | Q_τ[Y \| X]      | High |

**Proof for L₂:**
```
argmin_c E[(Y - c)²] = argmin_c E[Y² - 2cY + c²]

d/dc = -2E[Y] + 2c = 0  ⟹  c = E[Y]  ✓
```

**Proof for L₁:**
The median minimizes MAE because it balances mass on both sides. Formally, for any c < median:
```
E[|Y - c|] = E[|Y - c| | Y < c]·P(Y < c) + E[|Y - c| | Y ≥ c]·P(Y ≥ c)
```
Increasing c toward the median decreases the larger (Y ≥ c) term more than it increases the smaller (Y < c) term, until c = median where P(Y < c) = P(Y ≥ c) = 0.5. ∎

---

**Key Takeaways from Statistical Foundations:**

1. **Every metric is an estimator** with sampling variance ∝ 1/n
2. **Always report confidence intervals**, especially with small test sets
3. **Use appropriate statistical tests** for comparing models (McNemar's for classification, paired t-test for CV)
4. **Align metrics with loss functions** that match your decision-theoretic goals
5. **ROC-AUC measures ranking**, not calibration; use log loss for calibrated probabilities
6. **Sample complexity matters**: Accurate metric estimation requires adequate test set size

---

## Classification Metrics

### Binary Classification

#### Confusion Matrix

**Foundation of all classification metrics:**

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, y_pred)

# Visualize
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()

"""
                 Predicted
                 No   Yes
Actual  No      TN    FP
        Yes     FN    TP
"""
```

**Definitions:**
- **TP (True Positive):** Correctly predicted positive
- **TN (True Negative):** Correctly predicted negative
- **FP (False Positive):** Incorrectly predicted positive (Type I error)
- **FN (False Negative):** Incorrectly predicted negative (Type II error)

#### Accuracy

**Formula:** (TP + TN) / (TP + TN + FP + FN)

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.2%}")
```

**When to use:**
- ✅ Balanced classes
- ✅ All errors equally costly

**When NOT to use:**
- ❌ Imbalanced classes
- ❌ Different error costs
- ❌ Rare events (fraud, disease)

**Example where accuracy misleads:**
```python
# 99% healthy, 1% diseased
y_true = [0] * 99 + [1]
y_pred = [0] * 100  # Predict "healthy" always

accuracy = accuracy_score(y_true, y_pred)  # 99%!
# But we miss all diseases!
```

#### Precision

**Formula:**
```
Precision = TP / (TP + FP) = P(y_true=1 | ŷ=1)
```

**Interpretation:** Of all positive predictions, how many are correct? This is the conditional probability that a sample is truly positive given that the model predicted positive.

```python
from sklearn.metrics import precision_score

precision = precision_score(y_true, y_pred)
print(f"Precision: {precision:.2%}")
```

**When to use:**
- Cost of False Positives is high
- Examples:
  - Spam detection (don't want real emails in spam)
  - Medical screening (don't want unnecessary treatment)

**Example:**
```python
# Email spam filter
# Precision = 95% means: Of 100 emails marked as spam, 95 are actually spam
# 5 legitimate emails wrongly marked as spam (bad!)
```

#### Recall (Sensitivity, True Positive Rate)

**Formula:**
```
Recall = TP / (TP + FN) = P(ŷ=1 | y_true=1)
```

**Interpretation:** Of all actual positives, how many did we catch? This is the conditional probability that the model predicts positive given that the sample is truly positive.

**Also known as:** Sensitivity, True Positive Rate (TPR), Hit Rate, Power (in hypothesis testing)

```python
from sklearn.metrics import recall_score

recall = recall_score(y_true, y_pred)
print(f"Recall: {recall:.2%}")
```

**When to use:**
- Cost of False Negatives is high
- Examples:
  - Disease detection (can't miss diseases)
  - Fraud detection (can't miss fraud)
  - Security threats

**Example:**
```python
# Cancer screening
# Recall = 98% means: Of 100 cancer cases, we detected 98
# We missed 2 cases (very bad!)
```

#### F1-Score

**Formula:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
   = 2TP / (2TP + FP + FN)
   = Harmonic Mean(Precision, Recall)
```

**Harmonic mean of precision and recall**

**Derivation:** For two values a and b, harmonic mean = 2/(1/a + 1/b) = 2ab/(a+b)

```python
from sklearn.metrics import f1_score

f1 = f1_score(y_true, y_pred)
print(f"F1-Score: {f1:.2%}")
```

**When to use:**
- Need balance between precision and recall
- Imbalanced classes
- No clear preference for precision vs recall

**Interpretation:**
- F1 = 1.0: Perfect
- F1 = 0.0: Worst
- F1 close to 0: Either precision or recall is very low

**Why harmonic mean (not arithmetic mean)?**
Harmonic mean penalizes extreme values more heavily:
- Arithmetic mean of 100% and 0% = 50% (misleading!)
- Harmonic mean of 100% and 0% = 0% (correctly shows model is useless)

Example with precision=100%, recall=10%:
- Arithmetic mean: (100 + 10)/2 = 55% (sounds okay)
- Harmonic mean (F1): 2×(100×10)/(100+10) = 18% (correctly reflects poor performance)

The harmonic mean ensures both precision AND recall must be high for a good F1 score.

#### F-Beta Score

**Generalization of F1:** Allows weighting precision vs recall

**Formula:**
```
F_β = (1 + β²) × (Precision × Recall) / (β² × Precision + Recall)
    = (1 + β²) × TP / ((1 + β²) × TP + β² × FN + FP)
```

**Mathematical interpretation:** β controls the weight of recall vs precision:
- β² is the ratio of weight given to recall vs precision
- β=1 → equal weight (F1 score)
- β=2 → recall weighted 4× more than precision (β²=4)
- β=0.5 → precision weighted 4× more than recall (1/β²=4)

```python
from sklearn.metrics import fbeta_score

# β = 0.5: Favor precision (2x more weight)
f_half = fbeta_score(y_true, y_pred, beta=0.5)

# β = 2: Favor recall (2x more weight)
f_two = fbeta_score(y_true, y_pred, beta=2.0)

print(f"F0.5: {f_half:.2%} (precision-focused)")
print(f"F2.0: {f_two:.2%} (recall-focused)")
```

**β values:**
- β < 1: Precision more important
- β = 1: F1-score (balanced)
- β > 1: Recall more important

#### ROC-AUC (Receiver Operating Characteristic - Area Under Curve)

**Measures ability to distinguish between classes**

```python
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Need probability scores (not binary predictions)
y_scores = model.predict_proba(X_test)[:, 1]

# Calculate AUC
auc = roc_auc_score(y_true, y_scores)
print(f"ROC-AUC: {auc:.3f}")

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Model (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

**Interpretation:**
- AUC = 1.0: Perfect classifier (can find threshold with TPR=1, FPR=0)
- AUC = 0.5: Random guessing (ROC curve is diagonal line)
- AUC < 0.5: Worse than random (predictions inverted - flip them!)

**Mathematical meaning:** AUC = P(score(positive sample) > score(negative sample))
- Probability that a randomly chosen positive example ranks higher than a randomly chosen negative example
- Equivalent to Mann-Whitney U statistic / Wilcoxon rank-sum test
- Can be computed as: AUC = (Σ_i Σ_j 1[s_i > s_j]) / (n_pos × n_neg) where i∈positives, j∈negatives

**When to use:**
- ✅ Need threshold-independent metric
- ✅ Comparing multiple models
- ✅ Balanced or moderately imbalanced data

**When NOT to use:**
- ❌ Highly imbalanced data (use PR-AUC instead)
- ❌ Care about performance at specific threshold

**Why ROC-AUC fails on imbalanced data:**
ROC curve uses False Positive Rate (FPR) = FP/(FP+TN). When negative class is very large (e.g., 99% of data):
- Even with many false positives, FPR stays low because TN is huge
- ROC-AUC can look great (0.95+) while model is actually terrible for minority class
- PR curve uses Precision = TP/(TP+FP), which directly shows poor performance

**Concrete example:**
- Dataset: 990 negatives, 10 positives (1% positive class)
- Model: Predicts positive for 100 samples, correctly catching 9/10 positives
- TPR (Recall) = 9/10 = 90% (looks good)
- FPR = 91/990 = 9% (looks good → ROC-AUC will be high)
- Precision = 9/100 = 9% (terrible! 91% of positive predictions are wrong)
- PR-AUC will correctly show this model is poor

**Rule of thumb:** If positive class < 10% of data, use PR-AUC instead of ROC-AUC.

#### Precision-Recall AUC

**Better than ROC-AUC for imbalanced data**

```python
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

y_scores = model.predict_proba(X_test)[:, 1]

# Calculate PR curve
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
pr_auc = auc(recall, precision)

print(f"PR-AUC: {pr_auc:.3f}")

# Plot
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'Model (AUC = {pr_auc:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()
```

**When to use:**
- ✅ Highly imbalanced data
- ✅ Positive class is rare but important

#### Log Loss (Cross-Entropy Loss)

**Measures quality of probability predictions**

**Formula:**
```
Log Loss = -(1/n) Σ [y_i log(p_i) + (1-y_i) log(1-p_i)]
```
where:
- y_i ∈ {0,1} is the true label
- p_i is the predicted probability of class 1
- n is the number of samples

```python
from sklearn.metrics import log_loss

y_proba = model.predict_proba(X_test)
logloss = log_loss(y_true, y_proba)
print(f"Log Loss: {logloss:.3f}")
```

**Interpretation:**
- Lower is better (range: [0, ∞))
- Log Loss = 0: Perfect probabilistic predictions
- Log Loss → ∞: Confident wrong predictions (e.g., p=0.99 when y=0)
- Penalizes confident wrong predictions exponentially

**Example penalties:**
- Predict p=0.9 when y=1: Loss = -log(0.9) = 0.105
- Predict p=0.9 when y=0: Loss = -log(0.1) = 2.303 (much worse!)
- Predict p=0.01 when y=1: Loss = -log(0.01) = 4.605 (very bad!)

**When to use:**
- ✅ Need well-calibrated probabilities
- ✅ Predictions used for decision-making
- ✅ Kaggle competitions (common metric)

### Multi-Class Classification

#### Macro vs Micro vs Weighted Averaging

```python
from sklearn.metrics import precision_recall_fscore_support

# Macro: Average of per-class metrics (treats all classes equally)
precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
    y_true, y_pred, average='macro'
)

# Micro: Aggregate TP, FP, FN across all classes
precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
    y_true, y_pred, average='micro'
)

# Weighted: Weighted average by class frequency
precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
    y_true, y_pred, average='weighted'
)

print(f"Macro F1: {f1_macro:.3f}")
print(f"Micro F1: {f1_micro:.3f}")
print(f"Weighted F1: {f1_weighted:.3f}")
```

**When to use:**
- **Macro:** All classes equally important (rare classes matter). Best when you care about performance on all classes regardless of size.
- **Micro:** Larger classes more important. Equivalent to accuracy for multi-class. Good when classes are imbalanced but you care about overall samples.
- **Weighted:** Balance by actual class distribution. Good default for imbalanced multi-class problems.

**Important note:** For imbalanced multi-class, macro-averaged metrics can be misleading if you have very small classes. Consider using weighted average or examining per-class metrics.

#### Classification Report

**Comprehensive overview:**

```python
from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1', 'Class 2']))
```

Output:
```
              precision    recall  f1-score   support

     Class 0       0.85      0.90      0.87       100
     Class 1       0.78      0.75      0.76        80
     Class 2       0.92      0.88      0.90       120

    accuracy                           0.85       300
   macro avg       0.85      0.84      0.84       300
weighted avg       0.86      0.85      0.85       300
```

---

## Regression Metrics

### Mean Absolute Error (MAE)

**Formula:**
```
MAE = (1/n) Σ|y_i - ŷ_i| = (1/n) Σ|e_i|
```
where e_i = y_i - ŷ_i is the residual/error for sample i

**Properties:**
- **L1 norm** of the error vector
- **Median-optimal:** Minimized when ŷ = median(y)
- **Robust to outliers:** Linear penalty for all errors
- **Not differentiable at 0:** Can be problematic for gradient-based optimization

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: {mae:.2f}")
```

**Interpretation:**
- Average absolute difference between predictions and actual values
- Same units as target variable
- Easy to interpret

**When to use:**
- ✅ Want interpretable metric
- ✅ Outliers should not dominate
- ✅ All errors equally important

**Example:**
```python
# House price prediction
mae = 25000  # Predictions off by $25,000 on average
```

### Mean Squared Error (MSE)

**Formula:**
```
MSE = (1/n) Σ(y_i - ŷ_i)² = (1/n) Σe_i²
```

**Properties:**
- **L2 norm squared** of the error vector (divided by n)
- **Mean-optimal:** Minimized when ŷ = mean(y)
- **Sensitive to outliers:** Quadratic penalty amplifies large errors
- **Differentiable everywhere:** Preferred for gradient-based optimization
- **Relation to variance:** MSE = Bias² + Variance (bias-variance decomposition)

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_true, y_pred)
print(f"MSE: {mse:.2f}")
```

**Properties:**
- Penalizes large errors more (squared)
- Not in same units as target
- Always positive

**When to use:**
- ✅ Large errors are especially bad
- ✅ Optimizing for typical ML objective

**When NOT to use:**
- ❌ Want interpretable metric (use MAE or RMSE)

### Root Mean Squared Error (RMSE)

**Formula:**
```
RMSE = √MSE = √((1/n) Σ(y_i - ŷ_i)²)
```

**Comparison with MAE:**
- For errors [1, 1, 1, 1]: MAE = 1, RMSE = 1 (same)
- For errors [0, 0, 0, 4]: MAE = 1, RMSE = 2 (RMSE penalizes large error more)
- **Rule of thumb:** RMSE ≥ MAE always, with equality only if all errors are identical
- **RMSE/MAE ratio:** Indicates error distribution
  - Ratio ≈ 1: Errors are uniform
  - Ratio >> 1: Errors have high variance / many outliers

```python
from sklearn.metrics import mean_squared_error

rmse = mean_squared_error(y_true, y_pred, squared=False)
# or
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print(f"RMSE: {rmse:.2f}")
```

**Properties:**
- Same units as target variable
- Penalizes large errors
- More interpretable than MSE

**When to use:**
- ✅ Same as MSE but want interpretability
- ✅ Common standard metric

### Mean Absolute Percentage Error (MAPE)

**Formula:** (100/n) × Σ|y_true - y_pred| / |y_true|

```python
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(y_true, y_pred)
print(f"MAPE: {mape:.2f}%")
```

**Interpretation:**
- Percentage error (easy to understand)
- Scale-independent

**When to use:**
- ✅ Want scale-independent metric
- ✅ Comparing across different scales

**When NOT to use:**
- ❌ Target has zeros (division by zero!)
- ❌ Target has very small values (unstable)

**Critical problems with MAPE:**
1. **Asymmetric penalty**: MAPE penalizes over-predictions more than under-predictions
   - Predicting 90 when true=100: Error = |100-90|/100 = 10%
   - Predicting 110 when true=100: Error = |100-110|/100 = 10%
   - BUT if predicting 50 when true=100: Error = 50%
   - While predicting 150 when true=100: Error = 50%
   - However, predicting 50 when true=50: Error = 0% vs predicting 100 when true=50: Error = 100%!

2. **Not actually scale-independent**: Large percentage errors on small values dominate
   - If predicting [1000, 10]: errors [100, 1] give MAPE = (10% + 10%)/2 = 10%
   - But errors [50, 5] give MAPE = (5% + 50%)/2 = 27.5% (worse, even though absolute errors are better!)

3. **Undefined for zero values**: Cannot divide by zero

**Alternatives to MAPE:**
- **sMAPE (Symmetric MAPE)**: (100/n) × Σ|y_true - y_pred| / ((|y_true| + |y_pred|)/2)
- **MASE (Mean Absolute Scaled Error)**: Scales by naive forecast error, handles zeros
- **RMSE / MAE**: If you don't actually need scale-independence

### R² Score (Coefficient of Determination)

**Formula:** 1 - (SS_res / SS_tot)

Where:
- SS_res = Σ(y_true - y_pred)² (residual sum of squares)
- SS_tot = Σ(y_true - ȳ)² (total sum of squares, variance)

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_true, y_pred)
print(f"R²: {r2:.3f}")
```

**Interpretation:**
- R² = 1.0: Perfect predictions
- R² = 0.0: Model as good as predicting mean
- R² < 0: Model worse than predicting mean

**Example:**
```python
r2 = 0.85  # Model explains 85% of variance
```

**When to use:**
- ✅ Standard regression metric
- ✅ Want to know variance explained

**When NOT to use:**
- ❌ Target is not normally distributed
- ❌ Non-linear relationships

**Important clarifications:**
1. **R² can be negative**: This happens when your model performs worse than simply predicting the mean. It means SS_res > SS_tot (your predictions are worse than a horizontal line at the mean).

2. **R² can be misleading**:
   - High R² doesn't mean good predictions if target has high variance
   - Low R² doesn't mean bad predictions if target has low variance
   - R² increases as you add features (even random ones!) → use adjusted R²

3. **Adjusted R²**: Penalizes adding useless features
   ```python
   def adjusted_r2(r2, n_samples, n_features):
       return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
   ```

4. **R² ≠ correlation²** (except for simple linear regression with one feature)

Example of misleading R²:
- Predicting daily temperature: R² = 0.60 (actually good! Temperature varies a lot)
- Predicting room temperature: R² = 0.60 (terrible! Room temp should be stable)

### Huber Loss

**Combines MSE and MAE (robust to outliers)**

```python
from sklearn.metrics import mean_squared_error

def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    is_small_error = np.abs(error) <= delta
    squared_loss = 0.5 * error**2
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    return np.mean(np.where(is_small_error, squared_loss, linear_loss))

loss = huber_loss(y_true, y_pred, delta=1.0)
```

**When to use:**
- ✅ Data has outliers
- ✅ Want to penalize large errors less than MSE

---

## Ranking Metrics

### Mean Average Precision (MAP)

**Used in information retrieval, recommendation systems**

```python
def average_precision(y_true, y_scores):
    """Calculate average precision for single query"""
    # Sort by scores descending
    order = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[order]

    # Calculate precision at each relevant position
    precisions = []
    num_hits = 0

    for i, label in enumerate(y_true_sorted):
        if label == 1:
            num_hits += 1
            precisions.append(num_hits / (i + 1))

    if not precisions:
        return 0.0

    return np.mean(precisions)

# Mean over all queries
def mean_average_precision(y_true_list, y_scores_list):
    return np.mean([average_precision(y_t, y_s)
                    for y_t, y_s in zip(y_true_list, y_scores_list)])
```

### Normalized Discounted Cumulative Gain (NDCG)

**Accounts for position of relevant items**

```python
from sklearn.metrics import ndcg_score

# Relevance scores (0-5 scale, e.g., movie ratings)
y_true = [[3, 2, 3, 0, 1, 2]]
y_scores = [[0.9, 0.8, 0.7, 0.6, 0.5, 0.4]]

ndcg = ndcg_score(y_true, y_scores, k=5)  # Top 5
print(f"NDCG@5: {ndcg:.3f}")
```

**Interpretation:**
- NDCG = 1.0: Perfect ranking
- Higher is better
- Penalizes relevant items ranked low

---

## Clustering Metrics

### Silhouette Score

**Measures how similar points are to their own cluster vs other clusters**

```python
from sklearn.metrics import silhouette_score

labels = kmeans.labels_
score = silhouette_score(X, labels)
print(f"Silhouette Score: {score:.3f}")
```

**Interpretation:**
- Score ranges from -1 to 1
- 1: Clusters well separated
- 0: Overlapping clusters
- -1: Wrong clusters

### Davies-Bouldin Index

**Lower is better (measures cluster separation)**

```python
from sklearn.metrics import davies_bouldin_score

score = davies_bouldin_score(X, labels)
print(f"Davies-Bouldin Index: {score:.3f}")
```

### Calinski-Harabasz Index

**Higher is better (ratio of between-cluster to within-cluster variance)**

```python
from sklearn.metrics import calinski_harabasz_score

score = calinski_harabasz_score(X, labels)
print(f"Calinski-Harabasz Index: {score:.1f}")
```

---

## Custom Metrics

### Business-Specific Metrics

```python
def business_metric(y_true, y_pred):
    """
    Custom metric for credit card fraud detection

    Costs:
    - False Positive (block legitimate transaction): $10 customer service
    - False Negative (miss fraud): $500 average fraud loss
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    fp_cost = fp * 10    # $10 per false alarm
    fn_cost = fn * 500   # $500 per missed fraud

    total_cost = fp_cost + fn_cost
    return total_cost

# Use in cross-validation
from sklearn.metrics import make_scorer

custom_scorer = make_scorer(business_metric, greater_is_better=False)
scores = cross_val_score(model, X, y, scoring=custom_scorer, cv=5)
```

### Asymmetric Error Costs

```python
def asymmetric_mse(y_true, y_pred, overpredict_penalty=1.0, underpredict_penalty=2.0):
    """
    MSE with different costs for over vs under prediction

    Example: Inventory forecasting
    - Overpredict: Holding costs
    - Underpredict: Lost sales (more expensive)
    """
    errors = y_true - y_pred
    overpredict = errors < 0
    underpredict = errors > 0

    cost = (
        (errors[overpredict] ** 2).sum() * overpredict_penalty +
        (errors[underpredict] ** 2).sum() * underpredict_penalty
    )

    return cost / len(y_true)
```

---

## Metric Selection Guide

### Decision Tree

```
What type of problem?
│
├─ Classification
│  │
│  ├─ Binary
│  │  ├─ Balanced classes? → Accuracy, F1
│  │  ├─ Imbalanced?
│  │  │  ├─ False Positives expensive? → Precision
│  │  │  ├─ False Negatives expensive? → Recall
│  │  │  └─ Both matter? → F1, PR-AUC
│  │  ├─ Need probabilities? → Log Loss, Brier Score
│  │  └─ Threshold-independent? → ROC-AUC (balanced), PR-AUC (imbalanced)
│  │
│  └─ Multi-class
│     ├─ All classes equally important? → Macro F1
│     ├─ Larger classes more important? → Weighted F1
│     └─ Need probabilities? → Log Loss
│
├─ Regression
│  ├─ Want interpretable metric? → MAE, RMSE
│  ├─ Large errors very bad? → MSE, RMSE
│  ├─ Robust to outliers? → MAE, Huber Loss
│  ├─ Scale-independent? → MAPE, R²
│  └─ Variance explained? → R²
│
├─ Ranking
│  ├─ Position matters? → NDCG
│  ├─ Just order? → MAP
│  └─ Binary relevance? → Precision@K, Recall@K
│
└─ Clustering
   ├─ Don't have ground truth? → Silhouette, Davies-Bouldin
   └─ Have ground truth? → Adjusted Rand Index, NMI
```

### Common Use Cases

| Use Case | Primary Metric | Secondary Metrics |
|----------|----------------|-------------------|
| Spam Detection | Precision | Recall, F1 |
| Fraud Detection | Recall | Precision@90%Recall, PR-AUC |
| Medical Diagnosis | Recall | Specificity, F1 |
| Credit Scoring | ROC-AUC | Log Loss, F1 |
| Recommendation | NDCG@K, MAP | Precision@K, Recall@K |
| House Price Prediction | RMSE | MAE, R² |
| Demand Forecasting | MAPE | RMSE, MAE |
| Customer Churn | PR-AUC | Recall, Precision |

---

## Complete Example

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *

# Generate imbalanced data (10% positive)
X, y = make_classification(
    n_samples=1000,
    weights=[0.9, 0.1],
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("=== CLASSIFICATION METRICS ===\n")

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print()

# Basic metrics
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.3f}")
print()

# Probability-based metrics
print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.3f}")
print(f"Log Loss:  {log_loss(y_test, y_proba):.3f}")
print()

# Detailed report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Custom threshold analysis
print("\n=== THRESHOLD ANALYSIS ===\n")
for threshold in [0.3, 0.5, 0.7]:
    y_pred_thresh = (y_proba >= threshold).astype(int)
    precision = precision_score(y_test, y_pred_thresh)
    recall = recall_score(y_test, y_pred_thresh)
    f1 = f1_score(y_test, y_pred_thresh)

    print(f"Threshold {threshold:.1f}:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1-Score:  {f1:.3f}")
    print()
```

---

## Key Takeaways

**Metric selection rules:**
1. **Align with business goals** (what matters most?)
2. **Consider class balance** (imbalanced → F1, PR-AUC)
3. **Account for error costs** (FP vs FN costs)
4. **Think about deployment** (probabilities vs binary)

**Common mistakes:**
- Using accuracy on imbalanced data
- Not considering error costs
- Optimizing wrong metric
- Forgetting to check multiple metrics

**Best practices:**
- Report multiple metrics
- Always show confusion matrix for classification
- Use cross-validation for robust estimates
- Document why you chose specific metrics

**Remember:** The best metric depends on your problem, not on what's common!

---

**Related guides:**
- [Model Selection Guide](./MODEL_SELECTION_GUIDE.md) - Choose the right algorithm
- [Hyperparameter Tuning](./HYPERPARAMETER_TUNING_GUIDE.md) - Optimize for the right metric
- [Common ML Mistakes](../../resources/COMMON_ML_MISTAKES.md) - Avoid metric pitfalls

---

*Last Updated: October 30, 2025*
