# 🎯 Model Selection Guide: Choosing the Right Algorithm

**"All models are wrong, but some are useful."** - George Box

This guide helps you choose the right ML algorithm for your problem based on data characteristics, problem type, and constraints.

---

## 📋 Table of Contents

- [Quick Decision Tree](#quick-decision-tree)
- [Problem Type Classification](#problem-type-classification)
- [Statistical Learning Theory and Model Selection](#statistical-learning-theory-and-model-selection)
  - [Foundations of Generalization](#foundations-of-generalization)
  - [VC Dimension and Sample Complexity](#vc-dimension-and-sample-complexity)
  - [Rademacher Complexity](#rademacher-complexity-modern-generalization-theory)
  - [PAC Learning Framework](#pac-learning-framework)
  - [Information Criteria (AIC, BIC, MDL, Adjusted R²)](#information-criteria-for-model-selection)
  - [Structural Risk Minimization](#structural-risk-minimization-srm)
  - [No Free Lunch Theorem](#no-free-lunch-theorem)
  - [Model Complexity and Effective Degrees of Freedom](#model-complexity-and-effective-degrees-of-freedom)
- [Regression Models](#regression-models)
- [Classification Models](#classification-models)
- [Clustering Models](#clustering-models)
- [Model Comparison Framework](#model-comparison-framework)
- [Practical Considerations](#practical-considerations)
- [Algorithm Cheat Sheet](#algorithm-cheat-sheet)

---

## Quick Decision Tree

```
What's your problem?
│
├─ Predict continuous value (price, temperature, etc.)
│  └─ REGRESSION
│     ├─ Linear relationship? → Linear Regression
│     ├─ Need interpretability? → Decision Tree Regressor
│     ├─ Complex patterns? → Random Forest / XGBoost
│     └─ Very large dataset? → SGD Regressor
│
├─ Predict category (spam/not spam, cat/dog/bird)
│  └─ CLASSIFICATION
│     ├─ Linear separable? → Logistic Regression / SVM (linear)
│     ├─ Need interpretability? → Decision Tree
│     ├─ Complex boundaries? → Random Forest / XGBoost
│     ├─ Image/text/sequence? → Neural Networks
│     └─ Very few samples? → Naive Bayes
│
├─ Find groups in data (customer segments)
│  └─ CLUSTERING
│     ├─ Know number of clusters? → K-Means
│     ├─ Density-based groups? → DBSCAN
│     └─ Hierarchical structure? → Agglomerative Clustering
│
└─ Reduce dimensions (visualization, compression)
   └─ DIMENSIONALITY REDUCTION
      ├─ Linear projection? → PCA
      ├─ Non-linear? → t-SNE, UMAP
      └─ Feature selection? → SelectKBest, RFE
```

---

## Problem Type Classification

### Supervised Learning

**You have labeled data (input → output pairs)**

**Regression:**
- Predict house prices
- Forecast sales
- Estimate delivery time
- Predict temperature

**Classification:**
- Binary: Spam detection, fraud detection, churn prediction
- Multi-class: Digit recognition (0-9), species classification
- Multi-label: Tag articles (sports, politics, both)

### Unsupervised Learning

**You have unlabeled data (find patterns)**

**Clustering:**
- Customer segmentation
- Anomaly detection
- Document grouping

**Dimensionality Reduction:**
- Data visualization
- Feature extraction
- Noise reduction

### Semi-Supervised Learning

**You have mostly unlabeled data, few labels**
- Use clustering + few labels
- Self-training algorithms
- Co-training

---

## Statistical Learning Theory and Model Selection

### Foundations of Generalization

**The Central Question:** How do we select a model that generalizes well to unseen data?

**Formal Framework:**
Let:
- D = {(x₁, y₁), ..., (xₙ, yₙ)} be training data drawn i.i.d. from P(X, Y)
- F be a hypothesis class (e.g., linear functions, decision trees of depth ≤ 5)
- f̂ ∈ F be the learned model
- L(y, f(x)) be the loss function

**Population Risk (True Error):**
```
R(f) = E_(X,Y)~P [L(Y, f(X))]
```

**Empirical Risk (Training Error):**
```
R̂(f) = (1/n) Σᵢ L(yᵢ, f(xᵢ))
```

**Generalization Gap:**
```
R(f̂) - R̂(f̂) = ?
```

**Key Insight:** The generalization gap depends on the **complexity** of the hypothesis class F, not just the specific function f̂.

---

### VC Dimension and Sample Complexity

**Definition (VC Dimension):**
The Vapnik-Chervonenkis (VC) dimension of a hypothesis class F is the maximum number of points that can be **shattered** (perfectly classified for all possible labelings) by F.

**Examples:**

1. **Linear classifiers in ℝᵈ:** VC(F) = d + 1
   - Proof: Can shatter d+1 points in general position, but not d+2 (Radon's theorem)

2. **Decision trees of depth h:** VC(F) ≈ O(2^h · log(2^h))

3. **Neural networks with W weights:** VC(F) = O(W log W)

**Theorem 1 (VC Generalization Bound - Vapnik & Chervonenkis 1971):**
For binary classification (L ∈ {0, 1}), with probability at least 1 - δ:
```
R(f̂) ≤ R̂(f̂) + √((d_VC · (log(2n/d_VC) + 1) + log(4/δ)) / n)
```

where d_VC = VC(F).

**Sample Complexity:**
To achieve generalization error ε with confidence 1-δ, we need:
```
n = O((d_VC/ε²) · log(1/ε) · log(1/δ))
```

**Practical Implications:**
- **More complex models** (higher VC dimension) require **more data** to generalize well
- **Simple models** generalize better with **limited data**
- Trade-off between **expressiveness** (low bias) and **generalization** (low variance)

**Example:**
```python
# Linear classifier in 10 dimensions: VC = 11
# To achieve ε = 0.05 error with 95% confidence:
d_VC = 11
epsilon = 0.05
delta = 0.05

n_required = (d_VC / epsilon**2) * np.log(1/epsilon) * np.log(1/delta)
# n ≈ 3,300 samples

# Deep neural network with 10,000 parameters: VC ≈ 100,000
# Would require millions of samples for same guarantee!
```

---

### Rademacher Complexity (Modern Generalization Theory)

**More refined than VC dimension**, Rademacher complexity measures the ability of F to fit random noise.

**Definition:**
```
Rad_n(F) = E_σ [sup_{f∈F} (1/n) |Σᵢ σᵢ f(xᵢ)|]
```
where σᵢ ∈ {-1, +1} are random Rademacher variables (uniform coin flips).

**Interpretation:** Expected maximum correlation between functions in F and random labels.

**Theorem 2 (Rademacher Generalization Bound):**
With probability at least 1 - δ:
```
R(f̂) ≤ R̂(f̂) + 2·Rad_n(F) + √(log(1/δ)/(2n))
```

**Examples:**

1. **Linear functions with bounded norm:**
   If F = {x ↦ w^T x : ||w|| ≤ W, ||x|| ≤ X}, then:
   ```
   Rad_n(F) = O(W·X / √n)
   ```

2. **Neural networks (Bartlett et al. 2017):**
   ```
   Rad_n(F) = O((∏ₗ ||Wₗ||_2) / √n)
   ```
   Depends on **spectral norm** of weight matrices, not number of parameters!

**Modern Deep Learning Implication:**
Networks with millions of parameters can generalize if weight norms are controlled (via regularization or implicit bias of SGD).

---

### PAC Learning Framework

**Probably Approximately Correct (PAC) Learning** (Valiant 1984) formalizes "learnable from finite samples."

**Definition:**
A hypothesis class F is PAC-learnable if there exists an algorithm A such that for any distribution P, any ε > 0, δ > 0:
```
Given n ≥ poly(1/ε, 1/δ, d) samples,
A outputs f̂ such that:
P[R(f̂) ≤ min_{f∈F} R(f) + ε] ≥ 1 - δ
```

**Key Properties:**

1. **Polynomial sample complexity:** n = poly(1/ε, 1/δ, complexity(F))
2. **Distribution-free:** Works for any P
3. **Agnostic:** Allows for F not containing the true function

**Theorem 3 (Fundamental Theorem of PAC Learning):**
A hypothesis class F is PAC-learnable if and only if it has finite VC dimension.

**Sample Complexity for PAC Learning:**
```
n = O((d_VC/ε) · log(1/δ))
```

**Example:**
```python
# Axis-aligned rectangles in ℝ²: VC dimension = 4
# To PAC-learn with ε=0.1, δ=0.05:
d_VC = 4
epsilon = 0.1
delta = 0.05

n_pac = (d_VC / epsilon) * np.log(1/delta)
# n ≈ 120 samples

# Guaranteed to find rectangle with error ≤ 0.1 + optimal_error
```

---

### Information Criteria for Model Selection

#### Akaike Information Criterion (AIC)

**Derivation:**

Start with Kullback-Leibler divergence from true distribution to model:
```
KL(P_true || P_model) = E_P_true [log P_true(y|x) - log P_model(y|x; θ̂)]
                      = const - E_P_true [log P_model(y|x; θ̂)]
```

For maximum likelihood estimator θ̂:
```
E[log P_model(y|x; θ̂)] ≈ (1/n) Σᵢ log P(yᵢ|xᵢ; θ̂) - k/n
```
where k = number of parameters (bias correction via asymptotic analysis).

**AIC Formula:**
```
AIC = -2·log L(θ̂) + 2k
    = -2·(log-likelihood) + 2·(# parameters)
```

**Lower AIC is better** (minimizes estimated KL divergence).

**Properties:**
- **Asymptotically efficient:** Selects the model closest to truth as n → ∞
- **Risk of overfitting:** For small n, may select overly complex models
- **Assumes MLE:** Model must be fit via maximum likelihood

**Example:**
```python
from sklearn.linear_model import LinearRegression, Ridge
import numpy as np

def compute_aic(model, X, y):
    """
    Compute AIC for regression model
    Assumes Gaussian errors
    """
    n = len(y)
    y_pred = model.predict(X)

    # RSS (residual sum of squares)
    rss = np.sum((y - y_pred)**2)

    # Log-likelihood (Gaussian assumption)
    log_likelihood = -n/2 * np.log(2*np.pi) - n/2 * np.log(rss/n) - n/2

    # Number of parameters (coefficients + intercept + variance)
    k = X.shape[1] + 2

    aic = -2 * log_likelihood + 2 * k
    return aic

# Compare models
models = {
    'Linear (all features)': LinearRegression().fit(X_train, y_train),
    'Linear (subset)': LinearRegression().fit(X_train[:, :5], y_train)
}

for name, model in models.items():
    aic = compute_aic(model, X_train if '(all' in name else X_train[:, :5], y_train)
    print(f"{name}: AIC = {aic:.2f}")
```

#### Bayesian Information Criterion (BIC)

**Derivation (Bayesian Model Selection):**

Under Bayesian framework, we want to compute:
```
P(Model_i | Data) ∝ P(Data | Model_i) · P(Model_i)
```

The **marginal likelihood** (evidence) is:
```
P(Data | Model_i) = ∫ P(Data | θ, Model_i) · P(θ | Model_i) dθ
```

Using **Laplace approximation** around MLE θ̂:
```
log P(Data | Model) ≈ log P(Data | θ̂, Model) - (k/2) log n + O(1)
```

**BIC Formula:**
```
BIC = -2·log L(θ̂) + k·log(n)
    = -2·(log-likelihood) + (# parameters)·log(sample size)
```

**Lower BIC is better**.

**Comparison with AIC:**
```
AIC penalty: 2k
BIC penalty: k·log(n)

For n > 8: BIC penalizes complexity more heavily than AIC
```

**Properties:**
- **Consistent:** Selects true model as n → ∞ (if true model is in candidate set)
- **More conservative:** Prefers simpler models than AIC
- **Bayesian interpretation:** Approximates log posterior probability

**Theorem 4 (BIC Consistency):**
If the true model M* is in the candidate set, then:
```
P[BIC selects M* as n → ∞] → 1
```

**Example:**
```python
def compute_bic(model, X, y):
    """
    Compute BIC for regression model
    """
    n = len(y)
    y_pred = model.predict(X)

    # RSS
    rss = np.sum((y - y_pred)**2)

    # Log-likelihood
    log_likelihood = -n/2 * np.log(2*np.pi) - n/2 * np.log(rss/n) - n/2

    # Number of parameters
    k = X.shape[1] + 2

    bic = -2 * log_likelihood + k * np.log(n)
    return bic

# BIC tends to select simpler models than AIC for large n
for name, model in models.items():
    X_model = X_train if '(all' in name else X_train[:, :5]
    aic = compute_aic(model, X_model, y_train)
    bic = compute_bic(model, X_model, y_train)
    print(f"{name}:")
    print(f"  AIC = {aic:.2f}, BIC = {bic:.2f}")
```

#### Minimum Description Length (MDL)

**Information-Theoretic View:**

MDL principle: The best model is the one that **minimizes** the total description length:
```
MDL = L(Model) + L(Data | Model)
```

where:
- L(Model) = bits needed to encode the model
- L(Data | Model) = bits needed to encode data given model

**Connection to BIC:**
BIC is asymptotically equivalent to MDL under certain encoding schemes.

**Practical MDL:**
```
MDL ≈ -log P(Data | θ̂) + (k/2) log n
```

**Interpretation:** Balance between model fit and model complexity.

#### Adjusted R² (Regression Specific)

**Motivation:** Regular R² always increases when adding features, even if they're noise.

**Regular R²:**
```
R² = 1 - (SS_res / SS_tot)
   = 1 - Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²
```

**Adjusted R² Derivation:**

Adjust for degrees of freedom:
```
Adjusted R² = 1 - [(SS_res / (n - k - 1)) / (SS_tot / (n - 1))]
            = 1 - [(1 - R²) · (n - 1) / (n - k - 1)]
```

where:
- n = sample size
- k = number of predictors

**Properties:**
- **Penalizes additional features:** Can decrease when adding irrelevant features
- **Compares models with different # of predictors**
- **Range:** Can be negative (unlike R²)

**Relationship to F-statistic:**
```
F = [(R² / k) / ((1 - R²) / (n - k - 1))]
```

Testing if all coefficients are zero.

**Example:**
```python
from sklearn.metrics import r2_score

def adjusted_r2(r2, n, k):
    """
    Compute adjusted R²

    Args:
        r2: Regular R² score
        n: Number of samples
        k: Number of features

    Returns:
        Adjusted R²
    """
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

# Compare models
r2_full = r2_score(y_test, model_full.predict(X_test))
r2_subset = r2_score(y_test, model_subset.predict(X_test[:, :5]))

n = len(y_test)
adj_r2_full = adjusted_r2(r2_full, n, X_test.shape[1])
adj_r2_subset = adjusted_r2(r2_subset, n, 5)

print(f"Full model: R² = {r2_full:.3f}, Adj R² = {adj_r2_full:.3f}")
print(f"Subset model: R² = {r2_subset:.3f}, Adj R² = {adj_r2_subset:.3f}")

# If R² difference is small but adj_r2_subset > adj_r2_full,
# subset model is better (simpler and nearly as good)
```

---

### Structural Risk Minimization (SRM)

**Framework (Vapnik):**

Instead of minimizing empirical risk alone, minimize **structural risk**:
```
R_struct(f) = R̂(f) + Ω(f)
```

where Ω(f) is a **complexity penalty**.

**Formal SRM:**

Define a nested sequence of hypothesis classes:
```
F₁ ⊂ F₂ ⊂ F₃ ⊂ ... ⊂ Fₖ
```

with increasing complexity (e.g., VC(F₁) < VC(F₂) < ...).

For each class Fᵢ, compute:
```
R_struct^(i) = min_{f∈Fᵢ} R̂(f) + √((VC(Fᵢ) · log(n)) / n)
```

**Select** the class i* that minimizes R_struct^(i).

**Theorem 5 (SRM Generalization Bound):**
With probability at least 1 - δ, the selected model f̂ from F_{i*} satisfies:
```
R(f̂) ≤ R_struct^(i*) + O(√(log(K/δ) / n))
```

where K is the number of hypothesis classes considered.

**Practical Implementation:**

SRM is implemented in:
- **Ridge/Lasso regression:** Regularization parameter λ controls complexity
- **Decision tree pruning:** Tree depth controls complexity
- **Neural network regularization:** Weight decay, dropout

**Example:**
```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
import numpy as np

def structural_risk_minimization_trees(X, y, max_depths, cv=5):
    """
    Select optimal tree depth via SRM
    """
    results = []

    for depth in max_depths:
        model = DecisionTreeRegressor(max_depth=depth, random_state=42)

        # Empirical risk (negative CV score)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
        empirical_risk = -cv_scores.mean()

        # Complexity: Approximate VC dimension for tree
        # VC(tree of depth h) ≈ 2^h
        vc_dim = 2 ** depth

        # Complexity penalty
        n = len(X)
        complexity_penalty = np.sqrt((vc_dim * np.log(n)) / n)

        # Structural risk
        structural_risk = empirical_risk + complexity_penalty

        results.append({
            'depth': depth,
            'empirical_risk': empirical_risk,
            'complexity_penalty': complexity_penalty,
            'structural_risk': structural_risk
        })

    # Select depth with minimum structural risk
    best = min(results, key=lambda x: x['structural_risk'])

    print("Structural Risk Minimization Results:")
    for r in results:
        print(f"Depth {r['depth']}: "
              f"Emp Risk = {r['empirical_risk']:.4f}, "
              f"Complexity = {r['complexity_penalty']:.4f}, "
              f"Structural Risk = {r['structural_risk']:.4f}")

    print(f"\nSelected depth: {best['depth']}")
    return best['depth']

# Example usage
max_depths = [2, 4, 6, 8, 10, 12]
optimal_depth = structural_risk_minimization_trees(X_train, y_train, max_depths)
```

---

### No Free Lunch Theorem

**Statement (Wolpert & Macready 1997):**

Averaged over **all possible target functions**, all learning algorithms have the **same expected performance**.

**Formal Theorem:**

Let A₁, A₂ be two learning algorithms, and f be a target function. The **off-training-set error** is:
```
E(A, f, n) = expected error of A on data not in training set
```

Then:
```
Σ_f E(A₁, f, n) = Σ_f E(A₂, f, n)
```

where the sum is over all possible target functions f.

**Implications:**

1. **No universally best algorithm:** Algorithm performance is **problem-dependent**
2. **Prior knowledge matters:** Good algorithms incorporate **inductive bias** matching the problem domain
3. **Evaluation is crucial:** Must test algorithms on **realistic problems**, not worst-case

**Why Some Algorithms Work Better in Practice:**

Real-world problems have **structure**:
- **Smoothness:** Nearby inputs have similar outputs (favors local methods, neural networks)
- **Sparsity:** Few features are relevant (favors Lasso, tree-based methods)
- **Low intrinsic dimension:** Data lies on low-dimensional manifold (favors PCA, deep learning)

**Practical Takeaway:**
- Always validate on **your specific problem**
- Use **domain knowledge** to guide model selection
- No silver bullet exists

**Example:**
```python
# Neural networks excel on image data (spatial structure)
# But may underperform on tabular data compared to gradient boosting

# Tabular data (no spatial structure):
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor

xgb = XGBRegressor().fit(X_tabular_train, y_train)
nn = MLPRegressor(hidden_layers=(100, 100)).fit(X_tabular_train, y_train)

# XGBoost often wins on tabular
print(f"XGBoost R²: {xgb.score(X_tabular_test, y_test):.3f}")
print(f"Neural Net R²: {nn.score(X_tabular_test, y_test):.3f}")

# Image data (spatial structure):
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

# CNN excels, XGBoost struggles
cnn = build_cnn_model()  # Convolutional neural network
# xgb on flattened images performs poorly
```

---

### Model Complexity and Effective Degrees of Freedom

**Degrees of Freedom (df):**

Measures the **effective number of parameters** after regularization or constraints.

**Definition (Stein's Unbiased Risk Estimate):**
```
df(ŷ) = (1/σ²) · Σᵢ Cov(ŷᵢ, yᵢ)
```

**For linear smoothers** ŷ = Hy (where H is the "hat matrix"):
```
df = tr(H)
```

**Examples:**

1. **Ordinary Least Squares:**
   ```
   H = X(X^T X)^(-1) X^T
   df = tr(H) = p (number of features)
   ```

2. **Ridge Regression:**
   ```
   H = X(X^T X + λI)^(-1) X^T
   df(λ) = Σᵢ σᵢ² / (σᵢ² + λ)
   ```
   where σᵢ are singular values of X.

   **Property:** df(λ) decreases from p (λ=0) to 0 (λ→∞).

3. **Lasso:**
   ```
   df ≈ number of non-zero coefficients
   ```

**AIC with Effective df:**
```
AIC = -2·log L(θ̂) + 2·df
```

**Example:**
```python
from sklearn.linear_model import Ridge
import numpy as np

def effective_degrees_of_freedom_ridge(X, lambda_param):
    """
    Compute effective df for ridge regression
    """
    # SVD of X
    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    # Effective df
    df = np.sum(s**2 / (s**2 + lambda_param))

    return df

# Example: 10 features, different regularization strengths
X_standardized = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)

lambdas = [0.01, 0.1, 1.0, 10.0, 100.0]
for lam in lambdas:
    df = effective_degrees_of_freedom_ridge(X_standardized, lam)
    print(f"λ = {lam:6.2f}: df = {df:.2f} (out of {X_standardized.shape[1]} features)")

# Output shows df decreasing from ~10 to near 0 as λ increases
```

---

**Key Takeaways from Statistical Learning Theory:**

1. **Generalization depends on complexity:** More complex models need more data (VC dimension, Rademacher complexity)

2. **Information criteria trade fit and complexity:**
   - AIC: Asymptotically efficient, can overfit with small n
   - BIC: Consistent, more conservative
   - Use BIC for model selection, AIC for prediction

3. **Structural Risk Minimization:** Balance empirical error and model complexity explicitly

4. **No Free Lunch:** No universally best algorithm—use domain knowledge and empirical evaluation

5. **Effective degrees of freedom:** Captures true model complexity after regularization

6. **PAC Learning:** Formalizes "learnable from finite samples" (finite VC dimension sufficient)

---

## Regression Models

### Linear Regression

**When to use:**
- Linear relationship between features and target
- Need interpretability (coefficients show feature impact)
- Fast training required
- Baseline model

**Pros:**
- Fast to train
- Interpretable
- Works well with small data
- No hyperparameters to tune

**Cons:**
- Assumes linear relationship
- Sensitive to outliers
- Can't model complex patterns
- Assumes features are independent

**Example:**
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

# Interpret coefficients
for feature, coef in zip(feature_names, model.coef_):
    print(f"{feature}: {coef:.2f}")
```

**Best for:**
- Initial baseline
- Problems with clear linear trends
- When you need to explain predictions

### Ridge Regression (L2)

**When to use:**
- Features are correlated (multicollinearity)
- Need regularization to prevent overfitting
- All features are relevant

**Key hyperparameter:** alpha (regularization strength)

```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
```

### Lasso Regression (L1)

**When to use:**
- Feature selection needed (sets coefficients to 0)
- Sparse models preferred
- Many irrelevant features

```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=0.1)
model.fit(X_train, y_train)

# Features with non-zero coefficients are selected
selected_features = np.where(model.coef_ != 0)[0]
```

### Decision Tree Regressor

**When to use:**
- Non-linear relationships
- Need interpretability (can visualize tree)
- Mixed feature types (numerical + categorical)
- No need for feature scaling

**Pros:**
- Easy to interpret
- Handles non-linearity
- No feature scaling needed
- Captures interactions automatically

**Cons:**
- Prone to overfitting
- High variance (unstable)
- Biased to majority class

```python
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(max_depth=5, min_samples_split=20)
model.fit(X_train, y_train)
```

### Random Forest Regressor

**When to use:**
- Don't know which algorithm to use (great default)
- Need good out-of-box performance
- Mixed feature types
- Want feature importance

**Pros:**
- Reduces overfitting vs single tree
- Robust to outliers
- Handles missing values
- Feature importance built-in

**Cons:**
- Less interpretable than linear
- Slower than linear models
- Large memory footprint

**Why Random Forest is such a good default:**

1. **Bias-Variance tradeoff sweet spot:**
   - Single tree: Low bias (can fit complex patterns) but high variance (unstable)
   - Ensemble of trees: Maintains low bias while dramatically reducing variance
   - Formula: Var(average) = Var(tree) / n_trees (if trees were independent)
   - In practice, trees are correlated, but variance still reduces significantly

2. **Built-in regularization through randomness:**
   - Bootstrap sampling (row randomness): Each tree sees different data
   - Feature subsetting (column randomness): Each split considers random subset of features
   - Combined effect: Decorrelates trees, improves ensemble performance
   - Default feature subset: √n_features (classification) or n_features/3 (regression)

3. **Robust to hyperparameter choices:**
   - More trees almost always better (diminishing returns after ~100-500)
   - Max_depth can often be left unlimited without severe overfitting
   - Min_samples_split/leaf provide gentle regularization
   - Hard to make Random Forest perform terribly with default settings

4. **No feature scaling required:**
   - Tree splits are based on thresholds, not distances
   - Scale-invariant: Same tree whether feature is 0-1 or 0-1000
   - Saves preprocessing time and reduces error potential

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    random_state=42
)
model.fit(X_train, y_train)

# Feature importance
importances = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
```

### XGBoost / LightGBM / CatBoost

**When to use:**
- Need state-of-the-art performance
- Kaggle competitions
- Tabular data
- Can tune hyperparameters

**Key differences explained:**

**XGBoost (2014):**
- **Tree growth:** Level-wise (grows all nodes at same depth before moving deeper)
- **Speed:** Moderate (pre-sorting for splits is expensive)
- **Memory:** Higher (stores pre-sorted features)
- **Best for:** General purpose, well-tested, most mature
- **Special features:** Built-in regularization, handles missing values well

**LightGBM (2017):**
- **Tree growth:** Leaf-wise (grows tree by choosing leaf with max delta loss)
- **Speed:** Fastest (histogram-based algorithm, no pre-sorting)
- **Memory:** Lowest (histogram binning)
- **Best for:** Large datasets (>10K rows), many features (>100)
- **Caution:** Leaf-wise growth can overfit on small datasets (<10K rows)
- **Special features:** Categorical feature support, faster training

**CatBoost (2017):**
- **Tree growth:** Symmetric (balanced trees)
- **Speed:** Moderate-fast
- **Memory:** Moderate
- **Best for:** Datasets with many categorical features, need minimal tuning
- **Special features:** Best-in-class categorical encoding (ordered target statistics), less prone to overfitting
- **Advantage:** Often works well with default parameters

**Performance comparison (typical):**
```
Accuracy:    XGBoost ≈ LightGBM ≈ CatBoost (all within 1%)
Speed:       LightGBM > CatBoost > XGBoost (2-10x difference)
Tuning ease: CatBoost > LightGBM > XGBoost
Categorical: CatBoost > LightGBM > XGBoost (manual encoding)
```

**Selection guide:**
- **XGBoost:** Default choice, most documentation/examples, proven track record
- **LightGBM:** Large datasets where speed matters, deep trees (leaf-wise growth)
- **CatBoost:** Many categorical features, want good defaults without tuning

**XGBoost:**
```python
import xgboost as xgb

model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,  # Level-wise: limits depth
    random_state=42
)
model.fit(X_train, y_train)
```

**LightGBM:**
```python
import lightgbm as lgb

model = lgb.LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,  # Leaf-wise: limits leaves (2^max_depth)
    random_state=42
)
model.fit(X_train, y_train)
```

**CatBoost:**
```python
from catboost import CatBoostRegressor

model = CatBoostRegressor(
    iterations=100,
    learning_rate=0.1,
    depth=5,
    random_state=42,
    verbose=False
)
# No need to encode categorical features!
model.fit(X_train, y_train, cat_features=['city', 'category'])
```

### Support Vector Regression (SVR)

**When to use:**
- Small to medium datasets
- High-dimensional data
- Need robust predictions

**Avoid:**
- Very large datasets (slow)
- Need interpretability

```python
from sklearn.svm import SVR

model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
model.fit(X_train, y_train)
```

### Neural Networks (MLP Regressor)

**When to use:**
- Very large datasets
- Complex non-linear patterns
- Images, text, sequences

**Avoid:**
- Small datasets (<1000 samples)
- Need quick training
- Need interpretability

```python
from sklearn.neural_network import MLPRegressor

model = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    max_iter=500,
    random_state=42
)
model.fit(X_train, y_train)
```

---

## Classification Models

### Logistic Regression

**When to use:**
- Binary classification
- Need probability estimates
- Linear decision boundary
- Baseline model

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Get probabilities
probabilities = model.predict_proba(X_test)
```

### Decision Tree Classifier

**When to use:**
- Need interpretability
- Non-linear boundaries
- Mixed feature types

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=5, min_samples_split=20)
model.fit(X_train, y_train)
```

### Random Forest Classifier

**When to use:**
- Default choice for tabular data
- Need robust performance
- Want feature importance

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    class_weight='balanced',  # For imbalanced data
    random_state=42
)
model.fit(X_train, y_train)
```

### Gradient Boosting (XGBoost, LightGBM, CatBoost)

**When to use:**
- Need best performance
- Willing to tune hyperparameters
- Tabular data

```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),  # For imbalance
    random_state=42
)
model.fit(X_train, y_train)
```

### Support Vector Machines (SVM)

**When to use:**
- High-dimensional data (text classification)
- Clear margin of separation
- Small to medium datasets

**Kernels:**
- Linear: For linearly separable data
- RBF: For non-linear boundaries
- Polynomial: For polynomial boundaries

```python
from sklearn.svm import SVC

# Linear kernel
model = SVC(kernel='linear', C=1.0, probability=True)

# RBF kernel
model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)

model.fit(X_train, y_train)
```

### Naive Bayes

**When to use:**
- Text classification
- Very small datasets
- Need fast training
- Features are independent (or when violation doesn't hurt much)

**Variants:**
- **GaussianNB**: Continuous features (assumes normal distribution)
- **MultinomialNB**: Count data (text, word frequencies)
- **BernoulliNB**: Binary features (presence/absence)

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

# For continuous features
model = GaussianNB()

# For text (word counts)
model = MultinomialNB(alpha=1.0)

model.fit(X_train, y_train)
```

**Independence assumption explained:**

**The assumption:** Features are conditionally independent given the class.

Formula: P(x₁, x₂, ..., xₙ | y) = P(x₁|y) × P(x₂|y) × ... × P(xₙ|y)

**Example of violation:**
- Spam detection with features: contains("free"), contains("money"), contains("offer")
- These words are correlated: emails with "free" often have "money" and "offer"
- Naive Bayes assumes: P(free, money | spam) = P(free | spam) × P(money | spam)
- Reality: P(free, money | spam) < P(free | spam) × P(money | spam) (due to correlation)

**Why it often works despite violation:**

1. **Classification only needs correct ranking:**
   - Don't need exact probabilities, just need P(spam | x) > P(not spam | x)
   - Independence violation affects both classes similarly
   - Relative ordering often preserved even if absolute probabilities wrong

2. **Errors can cancel out:**
   - Overestimating some feature correlations, underestimating others
   - Systematic errors may balance across features
   - Final prediction can still be correct

3. **Low variance:**
   - Fewer parameters to estimate (each feature separately)
   - Less prone to overfitting on small datasets
   - Can outperform complex models that overfit

**When independence violation hurts:**
- Strong feature dependencies that affect one class more than another
- Example: In medical diagnosis, symptoms that always occur together should be treated as one feature
- Very redundant features (duplicate columns, word stems + words)

**Best practices:**
- Remove highly correlated features before training
- Feature selection to reduce redundancy
- Despite "naive" assumption, often competitive with sophisticated models on text
- Especially good baseline for text: 10-20 lines of code, trains in seconds

### K-Nearest Neighbors (KNN)

**When to use:**
- Small datasets
- Non-parametric (no assumptions)
- Simple baseline

**Avoid:**
- Large datasets (slow)
- High-dimensional data (curse of dimensionality)

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance',  # Weight by distance
    metric='euclidean'
)
model.fit(X_train, y_train)
```

**Curse of dimensionality explained:**

**Problem:** As dimensions increase, distance-based methods break down.

**Why this happens:**

1. **All points become equidistant:**
   - In high dimensions, distance between nearest and farthest neighbors converges
   - Example with random points in d dimensions:
     - d=2: nearest neighbor distance ≈ 0.1, farthest ≈ 1.0 (10x difference)
     - d=10: nearest ≈ 0.7, farthest ≈ 1.2 (1.7x difference)
     - d=100: nearest ≈ 0.95, farthest ≈ 1.05 (1.1x difference)
   - Result: Cannot distinguish "close" from "far"

2. **Data becomes sparse:**
   - To cover same density, need exponentially more points
   - Example: Cover unit interval [0,1] with 10 points → 10 points needed
   - Cover unit square [0,1]×[0,1] same density → 100 points needed
   - Cover d-dimensional cube → 10^d points needed
   - At d=10, need 10 billion points!

3. **Volume concentrates at boundaries:**
   - Almost all volume of hypersphere is near surface
   - Example: Fraction of volume within 99% of radius:
     - d=2 (circle): 98% of volume
     - d=10: 90.4% of volume
     - d=100: 36.6% of volume
   - Most points are far from center, near surface

**Practical impact on KNN:**
- Neighbors aren't actually "near" in any meaningful sense
- Distance metric loses discriminative power
- Need exponentially more data as dimensions grow
- Performance degrades dramatically above ~10-20 dimensions

**Solutions:**
- Dimensionality reduction (PCA, t-SNE) before KNN
- Feature selection to remove irrelevant features
- Use algorithms designed for high dimensions (tree-based, linear)

### Neural Networks

**When to use:**
- Very large datasets (>100K samples)
- Images, text, sequences
- Complex patterns

**Avoid:**
- Small datasets
- Need interpretability
- Limited compute

```python
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    max_iter=500,
    random_state=42
)
model.fit(X_train, y_train)
```

---

## Clustering Models

### K-Means

**When to use:**
- Know number of clusters
- Spherical clusters
- Large datasets (scales well)

```python
from sklearn.cluster import KMeans

model = KMeans(n_clusters=3, random_state=42)
labels = model.fit_predict(X)

# Evaluate (if you have ground truth)
from sklearn.metrics import silhouette_score
score = silhouette_score(X, labels)
```

### DBSCAN

**When to use:**
- Don't know number of clusters
- Arbitrary cluster shapes
- Want to find outliers

```python
from sklearn.cluster import DBSCAN

model = DBSCAN(eps=0.5, min_samples=5)
labels = model.fit_predict(X)

# -1 indicates outliers
n_outliers = (labels == -1).sum()
```

### Hierarchical Clustering

**When to use:**
- Want dendrogram
- Hierarchical structure
- Small to medium datasets

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Agglomerative clustering
model = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = model.fit_predict(X)

# Create dendrogram
linkage_matrix = linkage(X, method='ward')
dendrogram(linkage_matrix)
```

---

## Statistical Learning Theory Foundations

Understanding the mathematical principles behind model selection helps make principled decisions rather than relying solely on empirical comparisons.

### Bias-Variance Decomposition in Model Selection

**Fundamental Decomposition:**
```
For regression with squared loss, expected prediction error decomposes as:

E[(y - f̂(x))²] = Bias²[f̂(x)] + Var[f̂(x)] + σ²

where:
- Bias²[f̂(x)] = (E[f̂(x)] - f(x))²  (model assumption error)
- Var[f̂(x)] = E[(f̂(x) - E[f̂(x)])²]  (estimation variance)
- σ² = irreducible error (noise)

Model Complexity Trade-off:
┌─────────────────────────────────────┐
│ Simple Models (Linear Regression)  │
│ - High Bias, Low Variance          │
│ - Underfitting risk                │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ Complex Models (Deep NN, KNN k=1)  │
│ - Low Bias, High Variance          │
│ - Overfitting risk                 │
└─────────────────────────────────────┘

Optimal Complexity: Minimizes Bias² + Variance

Mathematical Insight:
- As model complexity ↑: Bias ↓, Variance ↑
- Sweet spot: Trade-off between bias and variance
- Cross-validation estimates this trade-off empirically
```

### PAC Learning Framework

**Probably Approximately Correct (PAC) Learning:**
```
A hypothesis class H is PAC-learnable if:
∃ algorithm A and polynomial function poly(·,·,·,·) such that:

∀ε > 0, δ > 0, distribution D, target concept c:

If m ≥ poly(1/ε, 1/δ, n, size(c)), then with probability ≥ 1-δ:

error(h) ≤ error(h*) + ε

where:
- ε: accuracy parameter (approximation error)
- δ: confidence parameter (probability of failure)
- m: number of training samples
- n: dimension of input space
- h*: best hypothesis in H (empirical risk minimizer)
- error(h): true risk (generalization error)

Key Insight: Need m = O((1/ε²) log(1/δ)) samples for finite H

Practical Implication:
To achieve 95% accuracy (ε=0.05) with 99% confidence (δ=0.01):
m ≈ (1/0.05²) × log(1/0.01) ≈ 400 × 4.6 ≈ 1,840 samples minimum
```

### VC Dimension and Model Capacity

**Vapnik-Chervonenkis (VC) Dimension:**
```
Definition: VC(H) = largest n such that H can shatter n points

Shatter: For any labeling of n points, ∃h ∈ H that perfectly classifies them

Examples:
1. Linear classifiers in ℝ²: VC = 3
   - Can shatter any 3 points (not collinear)
   - Cannot shatter all configurations of 4 points

2. Linear classifiers in ℝᵈ: VC = d + 1

3. Neural network with W weights: VC = O(W log W)

4. Decision trees of depth h: VC = O(h × 2^h)

5. RBF kernel SVM: VC = ∞ (infinite capacity)

Sample Complexity Bound (Fundamental Theorem):
For hypothesis class H with VC dimension d:

m ≥ O((d/ε) log(1/ε) + (1/ε) log(1/δ))

suffices for PAC learning with error ε and confidence 1-δ

Generalization Bound:
With probability ≥ 1-δ:

error_true(h) ≤ error_train(h) + O(√[(d log(m/d) + log(1/δ)) / m])

Trade-off:
- High VC dimension → More expressive model → Better training fit
- High VC dimension → Need more samples → Risk overfitting

Model Selection Implication:
Choose H with VC dimension proportional to available data:
d ≈ m / (c × log m) where c ≈ 10-20 (rule of thumb)
```

### Structural Risk Minimization (SRM)

**Principle:**
```
Instead of minimizing empirical risk alone:

R_emp(h) = (1/n) Σᵢ L(h(xᵢ), yᵢ)

Minimize structural risk:

R_struct(h) = R_emp(h) + Ω(h)
              └─ fit ─┘   └complexity penalty┘

where Ω(h) increases with model complexity

Implementations:
1. Regularization:
   min L(θ) + λ||θ||²  (Ridge)
   min L(θ) + λ||θ||₁  (Lasso)

2. Architecture constraints:
   - Limit tree depth
   - Limit number of parameters
   - Early stopping

3. Bayesian model selection:
   Maximize log P(D|M) - log |H_M|
            └─likelihood┘  └model complexity┘

Connection to Information Criteria:
- AIC: -2 log L + 2k  (k = # parameters)
- BIC: -2 log L + k log n  (n = # samples)
- MDL: -log L + (k/2) log n

Model Selection Rule:
Choose model minimizing information criterion
(Lower is better: better fit with fewer parameters)
```

### No Free Lunch Theorem

**Statement:**
```
Theorem (Wolpert & Macready, 1997):
Averaged over all possible problems, all algorithms perform equally well.

Formally: For any pair of algorithms A₁, A₂:

E_f [error(A₁, f)] = E_f [error(A₂, f)]

where expectation is over all possible target functions f

Implications:
1. No universally best algorithm exists
2. Algorithm performance depends on problem domain
3. Inductive bias must match problem structure

Practical Consequences:
✓ Domain knowledge matters for algorithm selection
✓ Must validate on problem-specific data
✓ "Best practices" are domain-dependent
✗ Don't blindly apply "best" algorithm from benchmarks

Example:
- Linear regression: Best for linear relationships
- K-NN: Best for local similarity patterns
- Neural nets: Best for compositional hierarchical patterns
- Each fails miserably on problems not matching their bias!
```

### Sample Complexity for Different Models

**Theoretical Requirements:**
```
Model                    | VC Dimension | Sample Complexity
-------------------------|--------------|-------------------
Linear (d features)      | d + 1        | O(d/ε²)
Polynomial degree p      | O(d^p)       | O(d^p/ε²)
Decision tree depth h    | O(h·2^h)     | O(h·2^h/ε²)
k-NN (k=1)              | ∞            | O(ε^(-d+2)/d))
RBF kernel SVM          | ∞            | Problem-dependent
Neural net (W weights)   | O(W log W)   | O(W log W/ε²)

where:
- d: input dimension
- ε: target error rate
- W: total number of weights

Practical Sample Size Rules:
1. Linear models: n ≥ 10d (minimum), n ≥ 20d (recommended)

2. Tree-based models: n ≥ 50 × num_leaves

3. Neural networks: n ≥ 10W (minimum), n ≥ 100W (recommended)
   Example: 1000-parameter network needs ≥100,000 samples

4. k-NN: n ≥ exp(d) samples needed (curse of dimensionality!)
   For d=10, need exponentially many samples

Model Selection Based on Sample Size:
n < 100:      Linear/Logistic, Naive Bayes
100 ≤ n < 1K: + Decision Trees, k-NN
1K ≤ n < 10K: + Random Forest, SVM
10K ≤ n < 100K: + Gradient Boosting, shallow NNs
n ≥ 100K:     + Deep Neural Networks
```

### Generalization Bounds

**Rademacher Complexity:**
```
Definition: Measures richness of hypothesis class

R̂_m(H) = E_σ [sup_{h∈H} (1/m) Σᵢ σᵢ h(xᵢ)]

where σᵢ ∈ {-1, +1} are random signs (Rademacher variables)

Generalization Bound:
With probability ≥ 1-δ:

error_true(h) ≤ error_train(h) + 2R̂_m(H) + O(√[log(1/δ)/m])

Properties:
1. R̂_m(H) ≤ √(VC(H)/m)  (connects to VC dimension)
2. For finite H: R̂_m(H) ≤ √(log|H|/m)
3. Decreases with m: more data → tighter bounds

Model Comparison:
Model with smaller Rademacher complexity generalizes better
(for same training error)
```

**Margin-based Bounds (SVM Theory):**
```
For linear classifiers with margin γ:

error_true ≤ error_train + O(R²/(γ²m))

where:
- R: radius of smallest ball containing data
- γ: margin (distance to decision boundary)
- m: number of samples

Key Insight: Large margin → Better generalization
(Even with infinite VC dimension!)

This justifies SVM's maximum margin principle:
max γ subject to correct classification
```

### Occam's Razor Formalized

**Minimum Description Length (MDL):**
```
Best model: Minimizes total description length

MDL(M, D) = L(M) + L(D|M)
            └model─┘  └data given model┘

where:
- L(M): bits to encode model
- L(D|M): bits to encode data using model

Interpretation:
- Simple model + complex patterns: High L(D|M)
- Complex model + simple patterns: High L(M)
- Optimal: Balance between model and data complexity

Connection to Bayesian Model Selection:
MDL ≈ -log P(M|D) = -log P(D|M) - log P(M)
                      └─likelihood┘  └─prior─┘

Practical Application:
Prefer simpler models (fewer parameters) when:
- Limited data
- Similar training performance to complex models
- Need interpretability/generalization
```

### Model Selection Algorithm

**Principled Selection Process:**
```python
import numpy as np
from sklearn.model_selection import learning_curve, validation_curve

def principled_model_selection(models, X, y, cv=5):
    """
    Select model using statistical learning principles

    Considers:
    1. Empirical performance (cross-validation)
    2. Model complexity (# parameters, VC dimension proxy)
    3. Sample efficiency (learning curves)
    4. Stability (variance across folds)
    """
    results = []

    for name, model in models.items():
        # 1. Cross-validation performance
        cv_scores = cross_val_score(model, X, y, cv=cv)
        mean_score = cv_scores.mean()
        std_score = cv_scores.std()

        # 2. Model complexity (parameter count)
        try:
            n_params = len(model.get_params())
        except:
            n_params = np.inf

        # 3. Learning curve (sample efficiency)
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv,
            train_sizes=np.linspace(0.1, 1.0, 10)
        )

        # Convergence: gap between train and validation
        final_gap = train_scores[-1].mean() - val_scores[-1].mean()

        # 4. Structural risk (CV score + complexity penalty)
        n_samples = len(X)
        # BIC-like penalty: penalize complex models
        complexity_penalty = (n_params / n_samples) * np.log(n_samples)
        structural_risk = -mean_score + complexity_penalty

        results.append({
            'model': name,
            'cv_score': mean_score,
            'cv_std': std_score,
            'n_params': n_params,
            'convergence_gap': final_gap,
            'structural_risk': structural_risk,
            'stable': std_score < 0.1  # Low variance across folds
        })

    # Sort by structural risk (lower is better)
    results_df = pd.DataFrame(results).sort_values('structural_risk')

    print("Model Selection Results:")
    print(results_df)

    # Recommend model
    best = results_df.iloc[0]
    print(f"\nRecommended: {best['model']}")
    print(f"CV Score: {best['cv_score']:.4f} ± {best['cv_std']:.4f}")
    print(f"Structural Risk: {best['structural_risk']:.4f}")

    return results_df

# Example usage:
# models = {
#     'Linear': LinearRegression(),
#     'Ridge': Ridge(alpha=1.0),
#     'Tree': DecisionTreeRegressor(max_depth=5),
#     'Forest': RandomForestRegressor(n_estimators=100)
# }
# results = principled_model_selection(models, X_train, y_train)
```

---

## Model Comparison Framework

### Step 1: Start Simple

```python
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(max_depth=5),
    'Random Forest': RandomForestRegressor(n_estimators=100)
}

results = {}

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    results[name] = {
        'mean': scores.mean(),
        'std': scores.std()
    }
    print(f"{name}: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

### Step 2: Select Best Performers

```python
# Continue with top 2-3 models
# Tune hyperparameters for those
```

### Step 3: Evaluate on Test Set

```python
# Train final model
best_model = RandomForestRegressor(n_estimators=100, max_depth=10)
best_model.fit(X_train, y_train)

# Evaluate on test set (ONCE)
test_score = best_model.score(X_test, y_test)
print(f"Test R²: {test_score:.3f}")
```

### Comparison Metrics

**Regression:**
- R² (coefficient of determination)
- MSE / RMSE
- MAE
- MAPE (mean absolute percentage error)

**Classification:**
- Accuracy (be careful with imbalanced data!)
- Precision / Recall / F1
- ROC-AUC
- PR-AUC (for imbalanced data)
- Confusion matrix

```python
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

---

## Practical Considerations

### Data Size

**Small (<1K samples):**
- Simple models (Linear, Logistic)
- K-Nearest Neighbors
- Naive Bayes
- Avoid: Deep learning

**Medium (1K-100K):**
- Random Forest
- Gradient Boosting
- SVM
- Shallow neural networks

**Large (>100K):**
- Gradient Boosting (LightGBM, XGBoost)
- SGD-based linear models
- Deep learning
- Avoid: KNN, full-batch methods

### Feature Count

**Low dimensional (<10 features):**
- Any algorithm works
- Start with Linear/Logistic

**Medium (10-100 features):**
- Tree-based models
- Linear models with regularization
- SVM

**High dimensional (>100 features):**
- Linear models with L1 regularization
- Gradient boosting
- Feature selection first
- Avoid: KNN (curse of dimensionality)

### Interpretability Requirements

**High interpretability needed:**
1. Linear/Logistic Regression (coefficients)
2. Decision Trees (rules)
3. Naive Bayes (probabilities)

**Medium interpretability:**
1. Random Forest (feature importance)
2. Gradient Boosting (SHAP values)

**Low interpretability (black box):**
1. Deep Neural Networks
2. SVM with RBF kernel

### Training Time

**Fast training needed:**
1. Linear/Logistic Regression
2. Naive Bayes
3. Decision Tree (single)

**Medium:**
1. Random Forest
2. SVM (small data)

**Slow:**
1. Gradient Boosting (many iterations)
2. Deep Neural Networks
3. SVM (large data)

### Prediction Speed

**Real-time (<1ms):**
- Linear/Logistic Regression
- Single Decision Tree
- Naive Bayes

**Fast (<10ms):**
- Random Forest (few trees)
- Small neural networks

**Slower:**
- Large ensembles
- Deep neural networks

---

## Algorithm Cheat Sheet

### Regression

| Algorithm | Best For | Pros | Cons | Training Time |
|-----------|----------|------|------|---------------|
| Linear Regression | Baseline, interpretability | Fast, interpretable | Assumes linearity | Very fast |
| Ridge/Lasso | Regularization, feature selection | Prevents overfitting | Still assumes linearity | Very fast |
| Decision Tree | Non-linear, interpretability | Easy to interpret | Overfits easily | Fast |
| Random Forest | Default choice | Robust, accurate | Less interpretable | Medium |
| XGBoost/LightGBM | Best performance | State-of-the-art | Needs tuning | Medium-Slow |
| SVR | Small data, high-dim | Robust | Slow on large data | Medium-Slow |
| Neural Networks | Very large data | Flexible | Needs lots of data | Slow |

### Classification

| Algorithm | Best For | Pros | Cons | Training Time |
|-----------|----------|------|------|---------------|
| Logistic Regression | Baseline, probabilities | Fast, interpretable | Linear boundary | Very fast |
| Decision Tree | Interpretability | Easy to understand | Overfits | Fast |
| Random Forest | Default choice | Robust, accurate | Black box | Medium |
| XGBoost/LightGBM | Best performance | State-of-the-art | Needs tuning | Medium-Slow |
| SVM | High-dim (text) | Effective in high-dim | Slow on large data | Medium-Slow |
| Naive Bayes | Text, small data | Very fast | Independence assumption | Very fast |
| KNN | Simple baseline | No training | Slow prediction | Fast (training) |
| Neural Networks | Images, text | Very flexible | Needs lots of data | Slow |

### Quick Selection Guide

**Start with:**
1. Linear/Logistic Regression (baseline)
2. Random Forest (good default)
3. Gradient Boosting (if RF works well)

**For specific cases:**
- **Text:** Naive Bayes, SVM, Transformers
- **Images:** CNNs (ResNet, EfficientNet)
- **Time series:** ARIMA, LSTM, Prophet
- **Tabular:** XGBoost, LightGBM, CatBoost
- **Small data:** Linear models, Naive Bayes
- **Need interpretability:** Linear, Decision Tree
- **Need speed:** Linear, Naive Bayes

---

## Complete Example

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Load data
X, y = load_data()

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features (for linear models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Try multiple models
models = {
    'Linear Regression': (LinearRegression(), X_train_scaled, X_test_scaled),
    'Ridge': (Ridge(alpha=1.0), X_train_scaled, X_test_scaled),
    'Decision Tree': (DecisionTreeRegressor(max_depth=10), X_train, X_test),
    'Random Forest': (RandomForestRegressor(n_estimators=100), X_train, X_test),
    'XGBoost': (xgb.XGBRegressor(n_estimators=100), X_train, X_test)
}

results = {}

for name, (model, X_tr, X_te) in models.items():
    # Cross-validation
    cv_scores = cross_val_score(model, X_tr, y_train, cv=5, scoring='r2')

    # Train on full training set
    model.fit(X_tr, y_train)

    # Test score
    test_score = model.score(X_te, y_test)

    results[name] = {
        'CV mean': cv_scores.mean(),
        'CV std': cv_scores.std(),
        'Test R²': test_score
    }

    print(f"{name}:")
    print(f"  CV: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    print(f"  Test: {test_score:.3f}\n")

# Select best model
best_model_name = max(results, key=lambda x: results[x]['Test R²'])
print(f"Best model: {best_model_name}")
```

---

## Key Takeaways

**Model selection strategy:**
1. **Start simple** (Linear/Logistic baseline)
2. **Try tree-based** (Random Forest as default)
3. **Optimize best** (Tune hyperparameters)
4. **Evaluate properly** (Cross-validation + test set)

**Rules of thumb:**
- Tabular data → Gradient Boosting
- Images → CNNs
- Text → Transformers (BERT, GPT)
- Time series → ARIMA, LSTM, Prophet
- Small data → Simple models
- Need interpretability → Linear/Tree
- Need speed → Linear/Naive Bayes

**Remember:**
- No single best algorithm for all problems
- Start simple, increase complexity if needed
- More data > better algorithm
- Feature engineering > algorithm choice
- Validate with cross-validation, evaluate once on test set

---

**Related guides:**
- [Hyperparameter Tuning](./HYPERPARAMETER_TUNING_GUIDE.md) - Optimize your chosen model
- [Feature Engineering](./FEATURE_ENGINEERING_GUIDE.md) - Improve performance
- [Common ML Mistakes](../../resources/COMMON_ML_MISTAKES.md) - Avoid pitfalls

---

*Last Updated: October 30, 2025*
