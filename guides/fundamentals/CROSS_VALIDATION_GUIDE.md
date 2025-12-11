# Cross-Validation Best Practices Guide

A comprehensive guide to cross-validation techniques, when to use them, and how to implement them correctly.

## Table of Contents
- [Why Cross-Validation Matters](#why-cross-validation-matters)
- [Basic Cross-Validation Techniques](#basic-cross-validation-techniques)
- [Advanced Cross-Validation Strategies](#advanced-cross-validation-strategies)
- [Time Series Cross-Validation](#time-series-cross-validation)
- [Nested Cross-Validation](#nested-cross-validation)
- [Cross-Validation for Imbalanced Data](#cross-validation-for-imbalanced-data)
- [Common Mistakes](#common-mistakes)
- [Best Practices](#best-practices)
- [Complete Examples](#complete-examples)

---

## Why Cross-Validation Matters

### The Problem with Single Train-Test Split

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Single split - results can vary significantly!
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Test Accuracy: {accuracy:.4f}")  # 0.8500

# Different random state - different results!
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Test Accuracy: {accuracy:.4f}")  # 0.8100 - 4% difference!
```

**Problem:** Single split gives you one estimate that depends heavily on how you split the data.

**Mathematical issue:** For a single split with test set of size n_test, the standard error of the accuracy estimate is:
```
SE(accuracy) ≈ √(p(1-p) / n_test)
```
where p is the true accuracy. For p=0.85 and n_test=200: SE ≈ 0.025 (2.5%).

This means a single 85% accuracy could actually represent true performance anywhere from 80-90% (95% CI).

**Solution:** Cross-validation averages K estimates, reducing variance by approximately √K (though not exactly due to correlation between folds).

---

## Rigorous Theory of Cross-Validation

**Theorem 1 (Cross-Validation as Risk Estimator - Stone, 1974):**

Let L(f, (x, y)) be loss of model f on example (x, y). True risk:

R(f) = E_(x,y)~P[L(f, (x,y))]

**K-fold CV estimator:**
R̂_CV(f) = (1/n) Σ_{i=1}^n L(f̂^(-κ(i)), (x_i, y_i))

where f̂^(-κ(i)) is trained on all folds except the one containing (x_i, y_i).

**Theorem (Stone, 1974):** Under regularity conditions:
R̂_CV(f) →_P R(f) as n → ∞

**Consistency:** CV provides consistent estimator of true risk.

**Theorem 2 (Bias of K-Fold CV - Bengio & Grandvalet, 2004):**

K-fold CV has **pessimistic bias** when estimating performance on full dataset:

**Bias:**
E[R̂_CV] ≥ R(f_n)

where f_n is trained on n samples.

**Reason:** CV trains on (K-1)n/K samples, not full n samples.

**Quantitative bias approximation:**
Bias ≈ R(f_{n(K-1)/K}) - R(f_n)

**For learning curves R(f_m) ≈ a + b/m^α:**
Bias ≈ b · [(K/(K-1))^α - 1] / n^α

**Examples:**
- 5-fold CV: trains on 80% → pessimistic bias
- 10-fold CV: trains on 90% → less pessimistic
- LOO CV: trains on (n-1)/n ≈ 100% → minimal bias

**Typical α ≈ 0.5, so for n = 1000, K = 5:**
Bias ≈ b · [(5/4)^0.5 - 1] / 1000^0.5 ≈ 0.012·b

**Theorem 3 (Variance of K-Fold CV - Bengio & Grandvalet, 2004):**

K-fold CV variance has two sources:

**1. Sampling variance:** Var(L(f, (x,y))) from data stochasticity
**2. Training set correlation:** Folds share (K-2)/(K-1) of training data

**Total variance:**
Var(R̂_CV) ≈ (σ²/n) · (1 + (K-1)·ρ)

where:
- σ² = Var(L(f, (x,y)))
- ρ ≈ (K-2)/(K-1) = correlation between fold estimates

**Examples:**
- K = 5: ρ ≈ 3/4 = 0.75, Var ≈ (σ²/n)·(1 + 4·0.75) = 4σ²/n
- K = 10: ρ ≈ 8/9 ≈ 0.89, Var ≈ (σ²/n)·(1 + 9·0.89) ≈ 9σ²/n
- LOO (K = n): ρ ≈ (n-2)/(n-1) ≈ 1, Var ≈ (σ²/n)·n = σ²

**Key insight:** LOO has HIGHEST variance! (not lowest)

**Theorem 4 (Bias-Variance Trade-off in K Selection):**

**As K increases:**
- **Bias decreases:** More training data per fold → less pessimistic
- **Variance increases:** Higher correlation between folds → more variable

**Optimal K:**
Minimizes MSE = Bias² + Variance

**Empirical recommendations:**
- **K = 5 or 10:** Good bias-variance balance
- **K = 10:** Most common (Kohavi, 1995)
- **LOO (K = n):** Unbiased but high variance, computationally expensive

**Theorem 5 (Leave-One-Out CV Computational Shortcut - Hastie et al., 2009):**

For linear smoothers (linear regression, ridge, k-NN, etc.):

**LOO error computed without refitting:**
LOO = (1/n) Σ_{i=1}^n ((y_i - ŷ_i) / (1 - H_{ii}))²

where H = X(X^T X)^(-1)X^T is hat matrix, H_{ii} is diagonal element.

**Proof sketch:**
ŷ^(-i) = ŷ - H_{ii}/(1-H_{ii}) · (y_i - ŷ_i)

where ŷ^(-i) is prediction without i-th sample. ∎

**Complexity:**
- Naive LOO: O(n²p) (refit n times)
- Shortcut: O(np²) (compute H once)

**For ridge regression:**
H = X(X^T X + λI)^(-1)X^T

**Generalized Cross-Validation (GCV):**
GCV = (1/n) Σ_i ((y_i - ŷ_i) / (1 - tr(H)/n))²

Approximates LOO by replacing H_{ii} with average tr(H)/n.

**Theorem 6 (Stratified K-Fold for Imbalanced Data - Kohavi, 1995):**

For classification with class imbalance (e.g., 90% class 0, 10% class 1):

**Standard K-fold:** Each fold may have different class proportions → high variance

**Stratified K-fold:** Maintains class proportions in each fold

**Variance reduction:**
Var(Stratified) ≤ Var(Standard)

**Proof sketch:**
Stratification removes one source of randomness (class proportion variation).
By law of total variance: Var(Standard) = E[Var|stratification] + Var[E|stratification]
Stratified removes second term. ∎

**Empirical improvement:**
For 90-10 split, stratified CV can reduce variance by 30-50%!

**Theorem 7 (Repeated K-Fold CV - Bouckaert & Frank, 2004):**

**Standard K-fold:** Run once, get K estimates

**Repeated K-fold:** Repeat R times with different random splits

**Variance reduction:**
Var(Repeated) ≈ Var(Single) / R

**But:** Not full 1/R reduction due to correlation!

**Practical guideline:**
- R = 5 or 10 repetitions
- Effective sample size: K × R (e.g., 5×10 = 50 estimates)
- Standard error: SE ≈ σ̂/√(K·R/c) where c ≈ 1.5-2 (correlation factor)

**Trade-off:**
- More repetitions: Lower variance, more computation
- Typically R = 5-10 sufficient

**Theorem 8 (Nested Cross-Validation for Model Selection - Varma & Simon, 2006):**

**Problem:** Using same CV for both hyperparameter tuning and performance estimation gives **optimistically biased** estimate.

**Nested CV:**
- **Outer loop:** K_outer folds for performance estimation
- **Inner loop:** K_inner folds for hyperparameter tuning

**Unbiased estimator:**
R̂_nested = (1/K_outer) Σ_{k=1}^{K_outer} L(f̂_k*, D_test^(k))

where f̂_k* is best model selected on outer fold k using inner CV.

**Bias comparison:**
- Single CV (tuning + testing): Optimistic bias ≈ O(p/n) where p = # hyperparameters
- Nested CV: Nearly unbiased O(1/n)

**Computational cost:**
- Single CV: O(K × M) where M = # hyperparameter settings
- Nested CV: O(K_outer × K_inner × M)

**Typical:** K_outer = 5, K_inner = 3, M = 100 → 1500 model fits!

**Theorem 9 (Time Series Cross-Validation - Bergmeir & Benítez, 2012):**

For time series with temporal dependence, standard K-fold violates independence!

**Time series CV variants:**

**1. Rolling window:**
- Train: [1, ..., t]
- Test: [t+1, ..., t+h]
- Advance by s steps

**2. Expanding window:**
- Train: [1, ..., t]
- Test: [t+1, ..., t+h]
- Keep all past data

**Bias-variance:**
- Rolling: Lower bias (fixed window), higher variance
- Expanding: Higher bias (distant past), lower variance

**Blocked CV (Hyndman & Athanasopoulos, 2018):**
Leave gaps between train and test to reduce dependence:
- Train: [1, ..., t]
- Gap: [t+1, ..., t+g]
- Test: [t+g+1, ..., t+g+h]

**Theorem 10 (Cross-Validation Confidence Intervals - Nadeau & Bengio, 2003):**

**Naive SE:** σ̂/√K (assumes independence)

**Corrected SE (accounting for correlation):**
SE_corrected = √((1/K)Σ(x_k - x̄)² · (1 + ρ̂))

where ρ̂ ≈ (K-2)/(K-1) is estimated correlation.

**95% Confidence Interval:**
[x̄ - t_{K-1,0.025} · SE_corrected, x̄ + t_{K-1,0.025} · SE_corrected]

where t_{K-1,0.025} is t-distribution critical value with K-1 degrees of freedom.

**Conservative approach (Bengio & Grandvalet, 2004):**
Since exact variance is unknown, use conservative estimate:

SE_conservative = σ̂ · √(1/K + 1/(K-1))

**Example:** K = 5, σ̂ = 0.02
- Naive SE: 0.02/√5 ≈ 0.009
- Corrected SE: 0.02·√(1/5 + 3/4) ≈ 0.019
- Conservative SE: 0.02·√(1/5 + 1/4) ≈ 0.013

**Conservative SE is 44% larger than naive!**

## Basic Cross-Validation Techniques

### 1. K-Fold Cross-Validation

**How it works:**
1. Split data into K equal folds
2. Train on K-1 folds, test on remaining fold
3. Repeat K times, each fold used as test set once
4. Average the K results

```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier

# Create K-Fold splitter
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
model = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

print(f"Scores: {scores}")
print(f"Mean: {scores.mean():.4f}")
print(f"Std: {scores.std():.4f}")
```

**Output:**
```
Scores: [0.84 0.86 0.83 0.87 0.85]
Mean: 0.8500
Std: 0.0141
```

**Statistical interpretation:**
- **Mean (μ̂)**: Estimate of true model performance E[L(f, D)]
- **Std (σ̂)**: Variation across folds, but NOT standard error of the mean
- **Standard error**: SE ≈ σ̂/√K only if folds were independent (they're not)
- **Actual SE**: Typically σ̂/√K × √(2(K-1)/K) ≈ σ̂/√K × √1.6 for K=5
- **95% CI**: [μ̂ - 1.96×SE, μ̂ + 1.96×SE]

For this example: SE ≈ 0.0141/√5 × √1.6 ≈ 0.008, so 95% CI ≈ [0.834, 0.866].

**Rigorous Statistical Theory of Cross-Validation:**

```
K-Fold CV Estimator:
Let D = {(x₁,y₁), ..., (xₙ,yₙ)} be training data
Partition into K folds: D = D₁ ∪ D₂ ∪ ... ∪ Dₖ

For fold k:
- Train set: D₋ₖ = D \ Dₖ (all data except fold k)
- Test set: Dₖ
- Algorithm produces: f̂₋ₖ trained on D₋ₖ
- Error on fold k: Êₖ = (1/|Dₖ|) Σ_{(x,y)∈Dₖ} L(y, f̂₋ₖ(x))

K-fold CV estimate:
CV_K = (1/K) Σₖ₌₁ᴷ Êₖ

Theoretical Question: What does CV_K estimate?

Target 1: True Risk (Population Error)
R(f̂) = E_{(x,y)~P}[L(y, f̂(x))]

where f̂ is trained on all n samples

Target 2: Expected Risk over Random Training Sets
R̄ = E_D[R(f̂_D)]

where expectation is over all possible training sets D of size n

Theorem 1 (Bias of K-Fold CV - Stone 1974):
E[CV_K] ≈ E[R(f̂_{n(K-1)/K})]

K-fold CV is approximately unbiased for risk of model trained on n(K-1)/K samples

Implication:
- K=5: Estimates R(f̂_{0.8n}) (80% of data)
- K=10: Estimates R(f̂_{0.9n}) (90% of data)
- LOOCV (K=n): Estimates R(f̂_{n-1}) ≈ R(f̂_n) (most accurate)

Bias of CV_K for estimating R(f̂_n):
Bias(CV_K) = E[CV_K] - R(f̂_n)
           ≈ R(f̂_{n(K-1)/K}) - R(f̂_n)

If learning curve is concave (typical):
R(f̂_m) decreases with m (more data → better model)
⇒ R(f̂_{0.8n}) > R(f̂_n)
⇒ CV_5 has pessimistic bias (overestimates error)

Variance of CV_K:
Var(CV_K) = Var[(1/K) Σₖ Êₖ]
          = (1/K²) [Σₖ Var(Êₖ) + 2Σₖ<ⱼ Cov(Êₖ, Êⱼ)]

Key observation: Folds are NOT independent!
- Training sets overlap by (K-2)/(K-1) of data
- For K=5: 75% overlap between any two training sets
- This creates positive correlation: Cov(Êₖ, Êⱼ) > 0

Theorem 2 (Nadeau & Bengio 2003):
Under mild conditions:

Var(CV_K) ≈ (σ²/n) · [1 + (n_test/n_train)]

where:
- σ²: Variance of single prediction error
- n_test = n/K: Test fold size
- n_train = n(K-1)/K: Training fold size

For K=5: n_test/n_train = 1/4
⇒ Var(CV_5) ≈ (σ²/n) · 1.25

For K=10: n_test/n_train = 1/9
⇒ Var(CV_10) ≈ (σ²/n) · 1.11

For LOOCV (K=n): n_test/n_train = 1/(n-1) ≈ 0
⇒ Var(LOOCV) ≈ σ²/n · 1 = σ²/n

But LOOCV has high variance from correlation!

Bengio-Nadeau Corrected Variance:
Taking into account correlation between folds:

Var(CV_K) = (1/K) · σ̂²_K · (1 + n_test/n_train)

where σ̂²_K = (1/K)Σₖ(Êₖ - CV_K)²

Standard Error:
SE(CV_K) = √Var(CV_K) = √((1/K) · σ̂²_K · (1 + n_test/n_train))

For K=5:
SE(CV_5) ≈ (σ̂_5/√5) · √1.25 ≈ 0.5 · σ̂_5

NOT σ̂_5/√5 as naive calculation suggests!

Theorem 3 (Bias-Variance Tradeoff in K):
As K increases:
- Bias ↓: Trains on more data, closer to n
- Variance ↑: More correlation between folds

Optimal K balances bias and variance
- Typically K=5 or K=10 in practice
- K=10 preferred when n is large
- K=5 preferred when n is small (less computation)

Mean Squared Error (MSE) of CV_K:
MSE(CV_K) = Bias²(CV_K) + Var(CV_K)

For estimating R(f̂_n):
- Small K (K=5): Higher bias, lower variance
- Large K (K=n): Lower bias, higher variance
- Optimal: Typically K=5 to K=10

Theorem 4 (LOOCV Properties):
Leave-One-Out CV (K=n):

Advantages:
- Approximately unbiased: E[CV_n] ≈ R(f̂_n)
- Deterministic: No randomness in fold assignment

Disadvantages:
- High variance: Var(CV_n) can be larger than CV_K for K < n
- Reason: Training sets are nearly identical
  Correlation Corr(Êᵢ, Êⱼ) ≈ (n-2)/(n-1) ≈ 1 for large n
- Computational cost: O(n) model fits

For linear models (special case):
LOOCV has closed form! No need to retrain n times:

CV_n = (1/n) Σᵢ₌₁ⁿ [yᵢ - ŷᵢ / (1 - hᵢᵢ)]²

where:
- ŷᵢ: Prediction from model trained on all n samples
- hᵢᵢ: i-th diagonal element of hat matrix H = X(X'X)⁻¹X'

This is called PRESS (Predicted Residual Sum of Squares)
Complexity: O(1) after initial model fit!

Empirical Results (Kohavi 1995):
Extensive experiments on multiple datasets:
- K=10 gives best bias-variance tradeoff on average
- K=5 nearly as good with half the computation
- LOOCV often has higher variance despite lower bias
- Recommendation: Use K=10 as default
```

**When to use:**
- ✅ Regression problems
- ✅ Balanced classification problems
- ✅ Large datasets
- ❌ Imbalanced classification (use Stratified K-Fold instead)

### 2. Stratified K-Fold Cross-Validation

**Maintains class distribution in each fold** - critical for classification!

```python
from sklearn.model_selection import StratifiedKFold
import numpy as np

# Check class distribution
print(f"Original distribution: {np.bincount(y) / len(y)}")
# [0.7, 0.3] - 70% class 0, 30% class 1

# Regular K-Fold (can have imbalanced folds)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, test_idx in kfold.split(X, y):
    print(f"Test fold distribution: {np.bincount(y[test_idx]) / len(test_idx)}")
# [0.65, 0.35] - Different from original!
# [0.75, 0.25]
# [0.68, 0.32]

# Stratified K-Fold (maintains distribution)
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, test_idx in skfold.split(X, y):
    print(f"Test fold distribution: {np.bincount(y[test_idx]) / len(test_idx)}")
# [0.70, 0.30] - Same as original!
# [0.70, 0.30]
# [0.70, 0.30]
```

**When to use:**
- ✅ Classification problems (almost always)
- ✅ Imbalanced datasets
- ✅ Small datasets where class distribution matters

### 3. Leave-One-Out Cross-Validation (LOOCV)

**Each sample is a test set once** - K = number of samples.

```python
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo, scoring='accuracy')

print(f"Number of folds: {loo.get_n_splits(X)}")  # Same as len(X)
print(f"Mean accuracy: {scores.mean():.4f}")
```

**When to use:**
- ✅ Very small datasets (< 100 samples)
- ✅ Need maximum training data per fold
- ❌ Large datasets (too computationally expensive)
- ❌ When you need faster training

**Pros:**
- Maximum training data (N-1 samples per fold)
- Nearly unbiased estimate of model performance
- No randomness in splits (deterministic)

**Cons:**
- Very slow (N models for N samples)
- High variance in performance estimate (estimates are highly correlated)
- Computationally prohibitive for large datasets
- Not suitable for time series data

**Statistical note:** LOOCV has low bias but high variance in the error estimate because the training sets overlap significantly (differ by only one sample), making the performance estimates highly correlated.

**Mathematical analysis:**
- **Bias:** E[L̂_LOOCV] ≈ E[L_true] (nearly unbiased, training on n-1 samples ≈ n samples)
- **Variance:** Var[L̂_LOOCV] is HIGH because:
  - Correlation between folds: ρ ≈ (n-2)/(n-1) ≈ 1 for large n
  - Training sets share n-2 samples → predictions highly correlated
  - Variance formula: Var[L̂_CV] = Var[L_i]/n + 2×Cov[L_i,L_j]×n(n-1)/2n²
  - For LOOCV: Var ≈ Var[L_i]/n × (1 + (n-1)ρ) ≈ Var[L_i] (doesn't decrease with n!)

**K-Fold vs LOOCV tradeoff:**
- K-fold (K=5,10): Moderate bias (train on (K-1)/K × n samples), low variance (folds less correlated)
- LOOCV (K=n): Low bias (train on n-1 samples), high variance (folds highly correlated)
- **Typical choice:** K=5 or K=10 provides best bias-variance tradeoff for the CV estimate itself

### 4. Holdout Validation (Train-Validation-Test Split)

**Three-way split for hyperparameter tuning.**

```python
from sklearn.model_selection import train_test_split

# First split: separate test set
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Second split: separate validation set
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)  # 0.25 of 0.8 = 0.2 of total

print(f"Train: {len(X_train)} ({len(X_train)/len(X):.1%})")
print(f"Validation: {len(X_val)} ({len(X_val)/len(X):.1%})")
print(f"Test: {len(X_test)} ({len(X_test)/len(X):.1%})")
```

**Output:**
```
Train: 600 (60%)
Validation: 200 (20%)
Test: 200 (20%)
```

**When to use:**
- ✅ Large datasets (millions of samples)
- ✅ Deep learning (faster than K-fold)
- ✅ Simple baseline evaluation
- ❌ Small datasets (wastes data)

---

## Advanced Cross-Validation Strategies

### 1. Group K-Fold Cross-Validation

**Ensures samples from same group stay together** - prevents data leakage!

```python
from sklearn.model_selection import GroupKFold

# Example: Medical data with multiple measurements per patient
# We want all measurements from same patient in same fold
patients = np.array([1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5])
X = np.random.randn(12, 5)
y = np.random.randint(0, 2, 12)

group_kfold = GroupKFold(n_splits=3)

for fold, (train_idx, test_idx) in enumerate(group_kfold.split(X, y, groups=patients)):
    print(f"\nFold {fold + 1}:")
    print(f"  Train patients: {np.unique(patients[train_idx])}")
    print(f"  Test patients: {np.unique(patients[test_idx])}")
```

**Output:**
```
Fold 1:
  Train patients: [1 3 4 5]
  Test patients: [2]

Fold 2:
  Train patients: [1 2 4 5]
  Test patients: [3]

Fold 3:
  Train patients: [2 3 4 5]
  Test patients: [1]
```

**Use cases:**
- Medical data (multiple visits per patient)
- Video data (frames from same video)
- Customer data (multiple transactions per customer)
- Any scenario with hierarchical/grouped structure

**Why it matters:**

```python
# BAD: Regular K-Fold can leak information
# Patient 1's measurements could be in both train and test!
kfold = KFold(n_splits=3, shuffle=True, random_state=42)
for train_idx, test_idx in kfold.split(X):
    train_patients = set(patients[train_idx])
    test_patients = set(patients[test_idx])
    overlap = train_patients & test_patients
    if overlap:
        print(f"⚠️ Data leakage! Patients in both: {overlap}")

# GOOD: Group K-Fold prevents this
group_kfold = GroupKFold(n_splits=3)
for train_idx, test_idx in group_kfold.split(X, y, groups=patients):
    train_patients = set(patients[train_idx])
    test_patients = set(patients[test_idx])
    overlap = train_patients & test_patients
    print(f"✓ No overlap: {len(overlap) == 0}")
```

### 2. Repeated K-Fold Cross-Validation

**Run K-Fold multiple times with different random splits** for more robust estimates.

```python
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold

# Repeat 5-fold CV 10 times = 50 total fits
repeated_kfold = RepeatedStratifiedKFold(
    n_splits=5, n_repeats=10, random_state=42
)

scores = cross_val_score(
    model, X, y, cv=repeated_kfold, scoring='accuracy'
)

print(f"Number of fits: {len(scores)}")  # 50
print(f"Mean: {scores.mean():.4f}")
print(f"Std: {scores.std():.4f}")
se = scores.std() / np.sqrt(len(scores))
print(f"Standard Error: {se:.4f}")
print(f"95% CI: [{scores.mean() - 1.96*se:.4f}, "
      f"{scores.mean() + 1.96*se:.4f}]")
```

**When to use:**
- ✅ When you need very robust estimates
- ✅ Model selection/comparison
- ✅ Publishing results
- ❌ Large datasets or slow models (too expensive)

**Comparison:**

```python
import time

# Regular 5-fold
start = time.time()
scores_regular = cross_val_score(model, X, y, cv=5, scoring='accuracy')
time_regular = time.time() - start

# Repeated 5-fold (10 repeats)
start = time.time()
scores_repeated = cross_val_score(
    model, X, y,
    cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42),
    scoring='accuracy'
)
time_repeated = time.time() - start

# Corrected standard errors
se_regular = scores_regular.std() / np.sqrt(len(scores_regular))
se_repeated = scores_repeated.std() / np.sqrt(len(scores_repeated))

print(f"Regular 5-fold:")
print(f"  Mean: {scores_regular.mean():.4f} ± {se_regular:.4f} (SE)")
print(f"  Std: {scores_regular.std():.4f}")
print(f"  Time: {time_regular:.2f}s")

print(f"\nRepeated 5-fold (10 repeats):")
print(f"  Mean: {scores_repeated.mean():.4f} ± {se_repeated:.4f} (SE)")
print(f"  Std: {scores_repeated.std():.4f}")
print(f"  Time: {time_repeated:.2f}s ({time_repeated/time_regular:.1f}x slower)")

print(f"\nVariance reduction: {(se_regular/se_repeated):.2f}x")
print(f"  (Note: Not exactly √10 due to correlation between repeats)")
```

---

## Time Series Cross-Validation

**Never shuffle time series data!** Use time-respecting splits.

### 1. TimeSeriesSplit

**Rolling window approach** - maintains temporal order.

```python
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

# Create time series data
n_samples = 100
X = np.arange(n_samples).reshape(-1, 1)
y = np.sin(X.ravel() / 10) + np.random.randn(n_samples) * 0.1

# Time series split
tscv = TimeSeriesSplit(n_splits=5)

fig, axes = plt.subplots(5, 1, figsize=(12, 10))

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    axes[fold].plot(train_idx, [fold] * len(train_idx), 'o', label='Train', markersize=3)
    axes[fold].plot(test_idx, [fold] * len(test_idx), 's', label='Test', markersize=3)
    axes[fold].set_ylabel(f'Fold {fold + 1}')
    axes[fold].legend()

    print(f"Fold {fold + 1}:")
    print(f"  Train: [{train_idx[0]}, {train_idx[-1]}]")
    print(f"  Test: [{test_idx[0]}, {test_idx[-1]}]")

plt.xlabel('Sample index')
plt.tight_layout()
plt.show()
```

**Output:**
```
Fold 1:
  Train: [0, 19]
  Test: [20, 39]

Fold 2:
  Train: [0, 39]
  Test: [40, 59]

Fold 3:
  Train: [0, 59]
  Test: [60, 79]

Fold 4:
  Train: [0, 79]
  Test: [80, 89]

Fold 5:
  Train: [0, 89]
  Test: [90, 99]
```

**Complete example:**

```python
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

tscv = TimeSeriesSplit(n_splits=5)
scores = []

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    scores.append(mse)

print(f"MSE per fold: {scores}")
print(f"Mean MSE: {np.mean(scores):.4f}")
```

### 2. Custom Time Series CV with Gap

**Add gap between train and test** to simulate real-world forecasting.

```python
def time_series_cv_with_gap(X, y, n_splits=5, test_size=20, gap=5):
    """
    Time series CV with gap between train and test.

    Parameters:
    - test_size: size of each test set
    - gap: number of samples to skip between train and test
    """
    n_samples = len(X)

    for i in range(n_splits):
        # Calculate indices
        test_start = n_samples - (n_splits - i) * test_size
        test_end = test_start + test_size
        train_end = test_start - gap

        if train_end < test_size:  # Not enough training data
            continue

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)

        yield train_idx, test_idx

# Example usage
for fold, (train_idx, test_idx) in enumerate(
    time_series_cv_with_gap(X, y, n_splits=5, test_size=10, gap=3)
):
    print(f"Fold {fold + 1}:")
    print(f"  Train: [{train_idx[0]}, {train_idx[-1]}]")
    print(f"  Gap: [{train_idx[-1]+1}, {test_idx[0]-1}]")
    print(f"  Test: [{test_idx[0]}, {test_idx[-1]}]")
```

**Why gap matters:**

```python
# Without gap - potential data leakage and unrealistic evaluation
# Example: Predicting stock price for day 100
# Training on days 0-99 includes day 99 (immediately before!)
# Problems:
# 1. Feature calculation may use future information (look-ahead bias)
# 2. Autocorrelation can leak information from day 99 to day 100
# 3. Doesn't reflect real deployment (data collection/processing lag)

# With gap - more realistic
# Training on days 0-95
# Gap: days 96-98 (accounts for data processing/model deployment time)
# Test: days 99-100
# Benefits:
# 1. Prevents look-ahead bias in feature engineering
# 2. Simulates realistic forecasting scenario
# 3. Accounts for operational constraints
```

**Critical for time series:**
- Financial data: Gap accounts for settlement time, regulatory reporting delays
- IoT/sensor data: Gap accounts for data transmission, processing latency
- Sales forecasting: Gap accounts for inventory ordering lead time

### Rigorous Theory of Time Series Cross-Validation

```
Time Series CV Challenge:
Standard CV assumes i.i.d. samples: (x₁,y₁), ..., (xₙ,yₙ) ~ P^n
Time series violates this: Data has temporal dependence!

Temporal Dependence:
Autocorrelation: Cov(yₜ, yₜ₋ₖ) ≠ 0  for lag k
⇒ Cannot randomly shuffle data
⇒ Training/test must respect temporal order

Theorem 7 (Bias in Standard CV for Time Series - Bergmeir & Benítez 2012):
If data has autocorrelation ρ(k) and we use standard K-fold CV (random splits):

Bias(CV_K) ≈ -2ρ(1)·σ² / √n

Negative bias = optimistic estimate!

Reason:
- Random splits put correlated observations in different folds
- Training on t+1 to predict t exploits future information
- Artificially inflates performance

Example:
ρ(1) = 0.5 (moderate autocorrelation)
⇒ CV underestimates error by ~70% of σ/√n

For n=100, σ=1:
Optimistic bias ≈ 0.07 (7% of error magnitude)

Solution: Time-Respecting Splits
Only use past to predict future: Train on [1, ..., t], Test on [t+1, ..., t+h]

TimeSeriesSplit Properties:

Forward-Chaining (Expanding Window):
Fold k: Train on [1, ..., kh], Test on [kh+1, ..., (k+1)h]

Training size grows: |Train_k| = kh
All data eventually used for testing
Mimics realistic scenario: more data over time

Blocked CV (Fixed Window):
Fold k: Train on [(k-1)h+1, ..., kh], Test on [kh+1, ..., (k+1)h]

Training size fixed: |Train_k| = h
More similar folds (same amount of data)
Better for stationary processes

Theorem 8 (Consistency of Time Series CV - Tashman 2000):
For stationary time series with mixing:

|CV_TS - E[MSE]| →_P 0  as n → ∞

Provided:
1. h → ∞ (test size grows)
2. h/n → 0 (test fraction shrinks)
3. Gap g ≥ max-lag of dependencies

Mixing Condition:
Time series is α-mixing if:
α(k) = sup_{A,B} |P(A ∩ B) - P(A)P(B)| → 0  as k → ∞

where A depends on {y₁, ..., yₜ}, B depends on {yₜ₊ₖ, ...}

Interpretation: Far-apart observations become independent
Most ARMA, GARCH, etc. satisfy this

Gap Selection Theory:

Theorem 9 (Optimal Gap for Forecasting - Racine 2000):
For h-step-ahead forecast, optimal gap:

g* = h + τ

where τ = minimum lag with |ρ(τ)| < ε (typically ε = 0.05)

Reasoning:
- Need gap ≥ h to avoid direct information leakage
- Need gap ≥ τ for autocorrelation to decay
- g* = max(h, τ) ensures both

Example (AR(1) with ρ(1) = 0.8):
ρ(k) = 0.8^k
For ε = 0.05: 0.8^τ < 0.05 ⇒ τ ≈ 14

For 1-day-ahead forecast: g* = max(1, 14) = 14 days

Rolling Origin Evaluation:

Instead of fixed K folds, use all possible train-test splits:
For each origin t ∈ {t_min, ..., n-h}:
- Train on [1, ..., t]
- Test on [t+1, ..., t+h]

Advantage: Maximum use of data for evaluation
Disadvantage: Highly correlated estimates (overlapping training sets)

Correlation between consecutive origins:
Corr(MSE_t, MSE_{t+1}) ≈ (t-1)/t → 1  as t → ∞

Standard error adjustment:
SE(MSĒ) ≠ σ/√K  (as in i.i.d. case)

Harvey-Leybourne-Newbold Correction:
SE_corrected = SE_naive · √(1 + 2Σₖ₌₁^{K-1} (1 - k/K)·ρ̂(k))

where ρ̂(k) = autocorrelation of {MSE_t} at lag k

Typically ρ̂(1) ≈ 0.5-0.8 for rolling origin
⇒ SE_corrected ≈ 1.5-2× SE_naive

Time Series Specific Metrics:

Traditional: Average test MSE across folds
Better: Directional accuracy, coverage of prediction intervals

Diebold-Mariano Test for Forecast Comparison:
H₀: E[Loss_A(t)] = E[Loss_B(t)]  (equal forecast accuracy)

Test statistic:
DM = d̄ / √(Var(d̄))  ~ N(0,1) under H₀

where d̄ = (1/n)Σₜ (Loss_A(t) - Loss_B(t))

With autocorrelation correction:
Var(d̄) = γ₀/n · (1 + 2Σₖ₌₁^{n-1} (1 - k/n)·ρ(k))

Practical Recommendations:

1. Always use time-respecting splits for temporal data
2. Include gap: g ≥ forecast horizon h
3. For highly autocorrelated data: g = 2h to 4h
4. Use expanding window (forward chaining) for growing datasets
5. Use fixed window for concept drift/non-stationarity
6. Report SE with autocorrelation correction
7. Use Diebold-Mariano for model comparison
```

---

## Nested Cross-Validation

**For unbiased hyperparameter tuning and model evaluation.**

### The Problem with Non-Nested CV

```python
from sklearn.model_selection import GridSearchCV

# ❌ WRONG: This gives optimistic estimates!
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15]}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=5, scoring='accuracy'
)
grid_search.fit(X, y)

# This score is biased (too optimistic)!
print(f"Best score: {grid_search.best_score_:.4f}")
```

**Why it's wrong:** The best score was selected from 9 models (3×3 grid). This is like taking the best of 9 exam attempts - not representative of future performance!

### The Solution: Nested Cross-Validation

```python
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold

# Outer loop: evaluate model performance
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Inner loop: hyperparameter tuning
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Model and parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15]
}

# Nested CV
nested_scores = []

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Inner CV: find best hyperparameters
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid, cv=inner_cv, scoring='accuracy'
    )
    grid_search.fit(X_train, y_train)

    # Evaluate best model on outer test set
    best_model = grid_search.best_estimator_
    score = best_model.score(X_test, y_test)
    nested_scores.append(score)

    print(f"Fold {fold + 1}: {score:.4f} (best params: {grid_search.best_params_})")

print(f"\nNested CV Score: {np.mean(nested_scores):.4f} ± {np.std(nested_scores):.4f}")
```

**Comparison:**

```python
# Non-nested (optimistic)
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=5, scoring='accuracy'
)
grid_search.fit(X, y)
non_nested_score = grid_search.best_score_

# Nested (unbiased)
nested_scores = []
for train_idx, test_idx in outer_cv.split(X, y):
    grid_search.fit(X[train_idx], y[train_idx])
    nested_scores.append(grid_search.best_estimator_.score(X[test_idx], y[test_idx]))
nested_score = np.mean(nested_scores)

print(f"Non-nested CV: {non_nested_score:.4f} (optimistic!)")
print(f"Nested CV: {nested_score:.4f} (realistic)")
print(f"Difference: {(non_nested_score - nested_score)*100:.2f} percentage points")
```

**Output:**
```
Non-nested CV: 0.8650 (optimistic!)
Nested CV: 0.8320 (realistic)
Difference: 3.30 percentage points
```

**When to use nested CV:**
- ✅ Reporting unbiased estimate of model performance for publication/presentation
- ✅ Comparing different model types (e.g., Random Forest vs SVM)
- ✅ Model selection when you need realistic performance estimate
- ✅ Research papers and rigorous evaluation

**When NOT to use nested CV:**
- ❌ Just finding best hyperparameters for deployment (use regular GridSearchCV)
- ❌ When you have separate held-out test set (use that for final evaluation)
- ❌ Limited computational budget (nested CV is expensive)

**Important clarification:**
- Nested CV gives you an **unbiased performance estimate**
- It does NOT give you the final model to deploy
- After nested CV, retrain on all data with best hyperparameters for deployment

**Rigorous Theory of Selection Bias in Model Selection:**

```
The Selection Bias Problem:
Let A = {A₁, A₂, ..., A_M} be set of M candidate models/hyperparameters
For each Aⱼ, estimate performance via K-fold CV: CV_K(Aⱼ)

Model selection:
Â = argmin_{Aⱼ ∈ A} CV_K(Aⱼ)

Question: Is CV_K(Â) an unbiased estimate of R(Â)?

Answer: NO! CV_K(Â) is biased (optimistic)

Theorem 5 (Selection Bias - Varma & Simon 2006):
E[CV_K(Â)] ≤ E[R(Â)]

Selection introduces optimistic bias!

Intuition:
- We selected Â because it had minimum CV error
- But CV estimates are noisy: CV_K(Aⱼ) = R(Aⱼ) + εⱼ
- By selecting min, we pick model with negative εⱼ (lucky)
- This underestimates true risk R(Â)

Formal Proof:
Let j* = argmin R(Aⱼ) be true best model
Let ĵ = argmin CV_K(Aⱼ) be selected model

CV_K(A_ĵ) = min_j CV_K(Aⱼ)
          ≤ CV_K(A_{j*})  [by definition of min]

Taking expectation over CV folds:
E[CV_K(A_ĵ)] ≤ E[CV_K(A_{j*})]
              ≈ R(A_{j*})  [CV is approximately unbiased]
              ≤ R(A_ĵ)  [j* is true minimum]

But we want: E[CV_K(A_ĵ)] ≈ R(A_ĵ)  ✗

Conclusion: CV_K(Â) underestimates R(Â)

Magnitude of Bias:
Depends on:
1. Number of models M (more models → more selection)
2. Correlation between models (high correlation → less bias)
3. Sample size n (small n → more noise → more bias)

Empirical studies (Cawley & Talbot 2010):
- M = 10 models: Bias ≈ 2-5% optimistic
- M = 100 models: Bias ≈ 5-10% optimistic
- M = 1000 models: Bias can be > 10% optimistic!

Example:
True error: 20%
Non-nested CV reports: 15% (5% optimistic)
Decision makers think model is better than it actually is!

Solution: Nested Cross-Validation

Nested CV Procedure:
Outer loop (K_out folds): Estimate generalization error
Inner loop (K_in folds): Hyperparameter selection

Outer fold k:
1. Split: D = D_k^train ∪ D_k^test
2. Inner CV on D_k^train:
   - For each hyperparameter λⱼ ∈ Λ:
     * Estimate CV_{K_in}(λⱼ) using D_k^train
   - Select: λ̂_k = argmin CV_{K_in}(λⱼ)
3. Train model f̂_k with λ̂_k on all of D_k^train
4. Evaluate: E_k = Error(f̂_k, D_k^test)

Nested CV estimate:
NCV = (1/K_out) Σ_k E_k

Theorem 6 (Unbiasedness of Nested CV - Varma & Simon 2006):
E[NCV] ≈ E[R(f̂_λ̂)]

where:
- λ̂ is selected via CV on training set of size n·(K_out-1)/K_out
- f̂_λ̂ is trained with λ̂ on same amount of data

Nested CV provides approximately unbiased estimate!

Why does it work?
- Selection (λ̂_k) and evaluation (E_k) use DISJOINT data
- No information leakage from test set to selection process
- Each outer fold simulates: select on train → evaluate on unseen test

Cost Analysis:
Non-nested CV: K_out × M evaluations
Nested CV: K_out × (K_in × M + 1) evaluations

Typical: K_out = 5, K_in = 3, M = 10
- Non-nested: 5 × 10 = 50 model fits
- Nested: 5 × (3 × 10 + 1) = 155 model fits

3× more expensive, but gives unbiased estimate!

Practical Recommendations:
1. Use nested CV for reporting performance
2. After nested CV, run regular CV on ALL data to select final λ
3. Train final model on ALL data with selected λ
4. Report nested CV score as expected performance

Example workflow:
# Step 1: Nested CV for unbiased estimate
nested_scores = nested_cv(X, y, param_grid)
print(f"Expected performance: {np.mean(nested_scores)}")

# Step 2: Select best hyperparameters on all data
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X, y)
best_params = grid_search.best_params_

# Step 3: Train final model
final_model = Model(**best_params)
final_model.fit(X, y)

# Step 4: Deploy final_model
# Expected performance: nested_scores (not grid_search.best_score_!)

Common Mistakes:
❌ Using grid_search.best_score_ as performance estimate
✓ Using nested CV score as performance estimate

❌ Deploying model from one outer fold of nested CV
✓ Retraining on all data after nested CV

❌ Selecting hyperparameters from nested CV results
✓ Selecting via regular CV on all data (after nested CV)

Varma-Simon Bias Correction (alternative to nested CV):
For small datasets where nested CV is too expensive:

Corrected estimate:
R̂_corrected = CV_K(Â) + Bias_correction

where Bias_correction ≈ (c/n) · log(M)

c is problem-dependent constant (needs estimation)
M is number of models considered

Less reliable than nested CV, but cheaper!

Theoretical Optimality (Arlot & Celisse 2010):
Among all data-splitting procedures:
- Nested CV with K_out = K_in = n (double LOOCV) is asymptotically optimal
- But has highest variance
- Practical optimum: K_out = 5-10, K_in = 3-5
```

---

## Cross-Validation for Imbalanced Data

### 1. Stratified Sampling

```python
from sklearn.model_selection import StratifiedKFold
from collections import Counter

# Highly imbalanced data
y = np.array([0]*95 + [1]*5)  # 95% class 0, 5% class 1

# Regular K-Fold can create folds with no minority class!
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
    print(f"Fold {fold + 1} test: {Counter(y[test_idx])}")

# Output: Some folds might have {0: 20} (no class 1!)

# Stratified K-Fold maintains distribution
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(skfold.split(X, y)):
    print(f"Fold {fold + 1} test: {Counter(y[test_idx])}")

# Output: All folds have {0: 19, 1: 1} (same distribution)
```

### 2. Combined with SMOTE

```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Create pipeline with SMOTE
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Use stratified K-fold
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# IMPORTANT: SMOTE is applied inside each fold!
scores = cross_val_score(
    pipeline, X, y, cv=skfold, scoring='f1'
)

print(f"F1 scores: {scores}")
print(f"Mean F1: {scores.mean():.4f}")
```

**Why this matters:**

```python
# ❌ WRONG: Applying SMOTE before CV causes data leakage!
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Now synthetic samples from test set influenced by train set!
scores = cross_val_score(model, X_resampled, y_resampled, cv=5)

# ✓ CORRECT: Apply SMOTE inside pipeline
pipeline = Pipeline([('smote', SMOTE(random_state=42)), ('model', model)])
scores = cross_val_score(pipeline, X, y, cv=5)
```

---

## Common Mistakes

### 1. Data Leakage from Preprocessing

```python
from sklearn.preprocessing import StandardScaler

# ❌ WRONG: Scaling before CV causes data leakage!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Uses info from entire dataset!

# Now the model has seen statistics from test set
scores = cross_val_score(model, X_scaled, y, cv=5)

# ✓ CORRECT: Scale inside CV folds
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(random_state=42))
])

scores = cross_val_score(pipeline, X, y, cv=5)
```

**Impact of data leakage:**

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X, y = make_classification(n_samples=1000, n_features=100, random_state=42)

# Wrong way (scale before CV)
X_scaled = StandardScaler().fit_transform(X)
scores_wrong = cross_val_score(
    RandomForestClassifier(random_state=42), X_scaled, y, cv=5
)

# Right way (scale inside CV)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(random_state=42))
])
scores_right = cross_val_score(pipeline, X, y, cv=5)

print(f"Wrong (leakage): {scores_wrong.mean():.4f}")
print(f"Right (no leakage): {scores_right.mean():.4f}")
print(f"Difference: {(scores_wrong.mean() - scores_right.mean())*100:.2f} percentage points")
```

### 2. Using Wrong CV for Time Series

```python
# ❌ WRONG: Shuffling time series data
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_timeseries, y_timeseries, cv=kfold)
# This trains on future data to predict the past!

# ✓ CORRECT: Use TimeSeriesSplit
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X_timeseries, y_timeseries, cv=tscv)
```

### 3. Not Using Stratification for Classification

```python
# ❌ WRONG: Regular K-Fold for classification
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold)

# ✓ CORRECT: Stratified K-Fold maintains class distribution
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skfold)
```

### 4. Ignoring Groups in Data

```python
# Data with multiple samples per patient
patients = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])

# ❌ WRONG: Patient 1's samples can be in both train and test!
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold)

# ✓ CORRECT: Keep all samples from same patient together
from sklearn.model_selection import GroupKFold
group_kfold = GroupKFold(n_splits=5)
scores = cross_val_score(model, X, y, cv=group_kfold, groups=patients)
```

### 5. Tuning on Same Folds Used for Evaluation

```python
# ❌ WRONG: Optimistic estimates
param_grid = {'max_depth': [5, 10, 15]}
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X, y)
print(f"Best score: {grid_search.best_score_:.4f}")  # Too optimistic!

# ✓ CORRECT: Use nested CV
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

nested_scores = []
for train_idx, test_idx in outer_cv.split(X, y):
    grid_search = GridSearchCV(model, param_grid, cv=inner_cv)
    grid_search.fit(X[train_idx], y[train_idx])
    nested_scores.append(grid_search.best_estimator_.score(X[test_idx], y[test_idx]))

print(f"Nested CV score: {np.mean(nested_scores):.4f}")  # Realistic!
```

---

## Best Practices

### 1. Choose the Right CV Strategy

| Scenario | Recommended CV | Why |
|----------|---------------|-----|
| Classification | Stratified K-Fold | Maintains class distribution |
| Regression | K-Fold | No class balance concerns |
| Time Series | TimeSeriesSplit | Respects temporal order |
| Grouped Data | GroupKFold | Prevents data leakage |
| Imbalanced Data | Stratified K-Fold | Ensures minority class in all folds |
| Small Dataset | Leave-One-Out or K-Fold (k=10) | Maximizes training data |
| Large Dataset | Holdout (80-20) or K-Fold (k=3) | Faster training |

### 2. Always Use Pipelines

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier

# ✓ CORRECT: All preprocessing inside pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(k=20)),
    ('classifier', RandomForestClassifier(random_state=42))
])

scores = cross_val_score(pipeline, X, y, cv=5)
```

### 3. Set Random Seeds for Reproducibility

```python
# Set seed in CV splitter
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Set seed in model
model = RandomForestClassifier(random_state=42)

# Now results are reproducible
scores = cross_val_score(model, X, y, cv=cv)
```

### 4. Report Mean and Standard Deviation

```python
scores = cross_val_score(model, X, y, cv=5)

print(f"Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
print(f"Scores: {scores}")
print(f"Min: {scores.min():.4f}, Max: {scores.max():.4f}")
```

### 5. Use Appropriate Metrics

```python
# For imbalanced classification
scores = cross_val_score(model, X, y, cv=5, scoring='f1')

# For regression
scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')

# Multiple metrics
from sklearn.model_selection import cross_validate

scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1'
}

results = cross_validate(model, X, y, cv=5, scoring=scoring)

for metric, scores in results.items():
    if metric.startswith('test_'):
        metric_name = metric[5:]  # Remove 'test_' prefix
        print(f"{metric_name}: {scores.mean():.4f} ± {scores.std():.4f}")
```

### 6. Choose Appropriate Number of Folds

```python
# Common choices:
# k=3: Fast, less robust (use for large datasets or slow models)
# k=5: Good balance (most common)
# k=10: More robust, slower (often used in research)

# Rule of thumb:
if len(X) < 100:
    k = 10  # Small dataset - use more folds
elif len(X) < 1000:
    k = 5   # Medium dataset
else:
    k = 3   # Large dataset - fewer folds needed
```

---

## Complete Examples

### Example 1: Classification with Nested CV

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Generate data
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=15,
    n_classes=2, class_sep=0.8, random_state=42
)

# Define pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Parameter grid
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [5, 10, 15],
    'classifier__min_samples_split': [2, 5, 10]
}

# Outer CV: for evaluation
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Inner CV: for hyperparameter tuning
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Nested CV
nested_scores = []
best_params_list = []

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Inner CV: find best hyperparameters
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=inner_cv,
        scoring='f1', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # Evaluate on outer test set
    score = grid_search.best_estimator_.score(X_test, y_test)
    nested_scores.append(score)
    best_params_list.append(grid_search.best_params_)

    print(f"Fold {fold + 1}:")
    print(f"  Best params: {grid_search.best_params_}")
    print(f"  Test score: {score:.4f}")

print(f"\nFinal nested CV score: {np.mean(nested_scores):.4f} ± {np.std(nested_scores):.4f}")

# Train final model on all data with most common best params
from collections import Counter
param_counts = {}
for params in best_params_list:
    for key, value in params.items():
        if key not in param_counts:
            param_counts[key] = []
        param_counts[key].append(value)

final_params = {
    key: Counter(values).most_common(1)[0][0]
    for key, values in param_counts.items()
}

print(f"\nMost common parameters: {final_params}")

# Train final model
final_pipeline = pipeline.set_params(**final_params)
final_pipeline.fit(X, y)
```

### Example 2: Time Series Forecasting

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Generate time series data
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=365, freq='D')
trend = np.linspace(0, 10, 365)
seasonal = 3 * np.sin(2 * np.pi * np.arange(365) / 365)
noise = np.random.randn(365) * 0.5
y = trend + seasonal + noise

# Create features
df = pd.DataFrame({'date': dates, 'value': y})
df['day_of_year'] = df['date'].dt.dayofyear
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek

# Lag features
for lag in [1, 7, 30]:
    df[f'lag_{lag}'] = df['value'].shift(lag)

# Rolling statistics
df['rolling_mean_7'] = df['value'].rolling(window=7).mean()
df['rolling_std_7'] = df['value'].rolling(window=7).std()

# Drop NaN from lags
df = df.dropna()

# Prepare features and target
feature_cols = ['day_of_year', 'month', 'day_of_week',
                'lag_1', 'lag_7', 'lag_30',
                'rolling_mean_7', 'rolling_std_7']
X = df[feature_cols].values
y = df['value'].values

# Time series CV with gap
tscv = TimeSeriesSplit(n_splits=5)

scores = {'mse': [], 'mae': [], 'rmse': []}
predictions = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train model
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    scores['mse'].append(mse)
    scores['mae'].append(mae)
    scores['rmse'].append(rmse)

    predictions.append({
        'fold': fold + 1,
        'test_idx': test_idx,
        'y_true': y_test,
        'y_pred': y_pred
    })

    print(f"Fold {fold + 1}:")
    print(f"  Train size: {len(train_idx)}, Test size: {len(test_idx)}")
    print(f"  MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

print(f"\nAverage metrics:")
for metric, values in scores.items():
    print(f"  {metric.upper()}: {np.mean(values):.4f} ± {np.std(values):.4f}")

# Plot predictions
import matplotlib.pyplot as plt

fig, axes = plt.subplots(5, 1, figsize=(14, 10))

for i, pred_dict in enumerate(predictions):
    test_idx = pred_dict['test_idx']
    y_true = pred_dict['y_true']
    y_pred = pred_dict['y_pred']

    axes[i].plot(test_idx, y_true, label='True', marker='o', markersize=3)
    axes[i].plot(test_idx, y_pred, label='Predicted', marker='s', markersize=3)
    axes[i].set_title(f"Fold {i + 1}")
    axes[i].legend()
    axes[i].grid(True)

plt.xlabel('Day')
plt.tight_layout()
plt.show()
```

### Example 3: Grouped Medical Data

```python
import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Generate grouped medical data
# Multiple visits per patient
n_patients = 50
visits_per_patient = np.random.randint(2, 6, size=n_patients)

patient_ids = []
for patient_id, n_visits in enumerate(visits_per_patient):
    patient_ids.extend([patient_id] * n_visits)

patient_ids = np.array(patient_ids)
n_samples = len(patient_ids)

# Features: vital signs, lab results, etc.
X = np.random.randn(n_samples, 10)

# Target: disease diagnosis (positive if patient has disease)
# All visits from same patient have same diagnosis
y = np.zeros(n_samples)
diseased_patients = np.random.choice(n_patients, size=15, replace=False)
for patient_id in diseased_patients:
    y[patient_ids == patient_id] = 1

print(f"Total samples: {n_samples}")
print(f"Total patients: {n_patients}")
print(f"Positive samples: {y.sum()} ({y.mean()*100:.1f}%)")

# Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Group K-Fold CV
group_kfold = GroupKFold(n_splits=5)

scores = cross_val_score(
    pipeline, X, y, cv=group_kfold,
    groups=patient_ids, scoring='f1'
)

print(f"\nGroup K-Fold CV F1 scores: {scores}")
print(f"Mean F1: {scores.mean():.4f} ± {scores.std():.4f}")

# Compare with regular K-Fold (wrong - causes data leakage!)
from sklearn.model_selection import StratifiedKFold

regular_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
regular_scores = cross_val_score(
    pipeline, X, y, cv=regular_cv, scoring='f1'
)

print(f"\nRegular K-Fold CV F1 scores (WRONG!): {regular_scores}")
print(f"Mean F1: {regular_scores.mean():.4f} ± {regular_scores.std():.4f}")

print(f"\nDifference: {(regular_scores.mean() - scores.mean())*100:.2f} percentage points")
print("Regular K-Fold gives optimistic results due to data leakage!")
```

---

## Summary

### Quick Reference

**Classification:**
```python
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

**Regression:**
```python
from sklearn.model_selection import KFold
cv = KFold(n_splits=5, shuffle=True, random_state=42)
```

**Time Series:**
```python
from sklearn.model_selection import TimeSeriesSplit
cv = TimeSeriesSplit(n_splits=5)
```

**Grouped Data:**
```python
from sklearn.model_selection import GroupKFold
cv = GroupKFold(n_splits=5)
scores = cross_val_score(model, X, y, cv=cv, groups=groups)
```

**Nested CV (for hyperparameter tuning):**
```python
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for train_idx, test_idx in outer_cv.split(X, y):
    grid_search = GridSearchCV(model, param_grid, cv=inner_cv)
    grid_search.fit(X[train_idx], y[train_idx])
    score = grid_search.best_estimator_.score(X[test_idx], y[test_idx])
```

### Key Takeaways

1. **Always use pipelines** to prevent data leakage
2. **Stratify for classification** to maintain class distribution
3. **Don't shuffle time series** - use TimeSeriesSplit
4. **Use GroupKFold for grouped data** to prevent leakage
5. **Use nested CV** for unbiased hyperparameter tuning evaluation
6. **Set random seeds** for reproducibility
7. **Report mean ± std** not just mean
8. **Choose appropriate k** based on dataset size and compute budget

Cross-validation is your best tool for honest model evaluation. Use it correctly, and you'll build models that actually generalize!
