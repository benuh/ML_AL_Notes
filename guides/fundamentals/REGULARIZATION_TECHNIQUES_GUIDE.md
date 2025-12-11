# Regularization Techniques Guide

A comprehensive guide to regularization methods for preventing overfitting in machine learning models.

## Table of Contents
- [What is Regularization?](#what-is-regularization)
- [L2 Regularization (Ridge)](#l2-regularization-ridge)
- [L1 Regularization (Lasso)](#l1-regularization-lasso)
- [Elastic Net](#elastic-net)
- [Dropout](#dropout)
- [Early Stopping](#early-stopping)
- [Data Augmentation](#data-augmentation)
- [Batch Normalization](#batch-normalization)
- [Weight Decay](#weight-decay)
- [When to Use Which Technique](#when-to-use-which-technique)
- [Practical Examples](#practical-examples)
- [Best Practices](#best-practices)

---

## What is Regularization?

**Regularization** is any technique that helps prevent overfitting by constraining model complexity.

### The Problem: Overfitting

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Generate data
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 + 3*X.ravel() + 5*np.sin(X.ravel()) + np.random.randn(100) * 2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# High-degree polynomial without regularization
poly = PolynomialFeatures(degree=15)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model_overfit = LinearRegression()
model_overfit.fit(X_train_poly, y_train)

print("Without Regularization:")
print(f"  Train R²: {model_overfit.score(X_train_poly, y_train):.4f}")
print(f"  Test R²: {model_overfit.score(X_test_poly, y_test):.4f}")
print(f"  Coefficient magnitudes: {np.abs(model_overfit.coef_).max():.2e}")

# Visualize
X_plot = np.linspace(0, 10, 300).reshape(-1, 1)
X_plot_poly = poly.transform(X_plot)
y_plot = model_overfit.predict(X_plot_poly)

plt.figure(figsize=(12, 6))
plt.scatter(X_train, y_train, alpha=0.5, label='Training Data')
plt.scatter(X_test, y_test, alpha=0.5, label='Test Data')
plt.plot(X_plot, y_plot, 'r-', linewidth=2, label='Overfitted Model')
plt.plot(X_plot, 2 + 3*X_plot.ravel() + 5*np.sin(X_plot.ravel()),
         'g--', linewidth=2, label='True Function')
plt.ylim(-20, 40)
plt.legend()
plt.title('Overfitting Without Regularization')
plt.show()
```

**Observation:** Coefficients become extremely large, causing wild oscillations.

### How Regularization Helps

Regularization adds a penalty term to the loss function:

```
Loss = Data Loss + λ × Regularization Term
```

Where:
- **Data Loss**: Empirical risk on training data (e.g., MSE = (1/n)Σ(yi - ŷi)²)
- **λ (lambda)**: Regularization strength (hyperparameter, λ ≥ 0)
- **Regularization Term**: Penalty for model complexity (e.g., norm of weights)

**Effect:**
- Forces model to balance fitting data and staying simple
- Prevents coefficients from becoming too large
- Improves generalization

---

## L2 Regularization (Ridge)

**Also called:** Ridge Regression, Weight Decay, Tikhonov Regularization

### Rigorous Theory of L2 Regularization

**Theorem 1 (Ridge Regression Closed Form - Hoerl & Kennard, 1970):**

For linear regression with L2 regularization:

min_w (1/n)||Xw - y||² + λ||w||²

**Closed-form solution:**
w_ridge = (X^T X + nλI)^(-1) X^T y

**Proof:**
Take gradient with respect to w and set to zero:
∇_w L = (2/n)X^T(Xw - y) + 2λw = 0
X^T Xw + nλw = X^T y
(X^T X + nλI)w = X^T y
w = (X^T X + nλI)^(-1) X^T y ∎

**Key properties:**
1. **Always invertible:** X^T X + nλI is positive definite for λ > 0
2. **No singularity:** Even when X^T X is singular (p > n), ridge solution exists
3. **Unique solution:** Convex optimization guarantees uniqueness

**Theorem 2 (Ridge as Bayesian MAP Estimation):**

Ridge regression is equivalent to Maximum A Posteriori (MAP) estimation with Gaussian prior:

**Likelihood:** p(y|X, w) = N(Xw, σ²I)
**Prior:** p(w) = N(0, (σ²/nλ)I)

**Posterior:**
p(w|X, y) ∝ exp(-(1/(2σ²))||y - Xw||² - (nλ/(2σ²))||w||²)

**MAP estimate:**
w_MAP = argmax_w p(w|X, y) = argmin_w (||y - Xw||² + nλ||w||²) = w_ridge ∎

**Interpretation:**
- Ridge regression assumes weights are centered at zero
- Regularization strength λ = σ²/σ_w² (ratio of noise variance to prior variance)
- Stronger prior belief (small σ_w²) → larger λ → more regularization

**Theorem 3 (Ridge Generalization Bound - Hsu et al., 2012):**

For ridge regression with regularization parameter λ, with probability 1-δ:

E[L(w_ridge)] - L* ≤ (σ²·tr[(X^T X + nλI)^(-1)X^T X]) / n + λ||w*||² + O(√(log(1/δ)/n))

where:
- L*: optimal risk
- w*: optimal weights
- σ²: noise variance
- tr: trace operator

**Decomposition:**
1. **Approximation error:** λ||w*||² (bias from regularization)
2. **Estimation error:** (σ²·effective_df) / n (variance from finite data)
3. **Sample complexity:** O(√(log(1/δ)/n))

**Effective degrees of freedom:**
df_eff(λ) = tr[(X^T X)(X^T X + nλI)^(-1)]

**Theorem 4 (Ridge Bias-Variance Trade-off):**

For ridge estimator w_ridge:

**Bias:**
E[w_ridge] - w* = -(nλ)(X^T X + nλI)^(-1) w*

**Variance:**
Var(w_ridge) = σ² (X^T X + nλI)^(-1) X^T X (X^T X + nλI)^(-1)

**MSE decomposition:**
MSE(w_ridge) = Bias² + Variance
             = (nλ)² w*^T (X^T X + nλI)^(-2) w* + σ² tr[(X^T X + nλI)^(-1)X^T X(X^T X + nλI)^(-1)]

**Key insight:**
- As λ → 0: Bias → 0, Variance → σ²(X^T X)^(-1) (OLS variance)
- As λ → ∞: Bias → w*, Variance → 0
- Optimal λ* minimizes total MSE

**Theorem 5 (Optimal Ridge Parameter via Cross-Validation):**

The optimal regularization parameter λ* minimizes the cross-validation error:

λ* = argmin_λ (1/K) Σ_{k=1}^K ||y_k - X_k w_ridge^{(-k)}(λ)||²

where w_ridge^{(-k)}(λ) is trained on all folds except k.

**Leave-One-Out (LOO) shortcut:**
For computational efficiency, LOO error can be computed without refitting:

LOO(λ) = (1/n) Σ_{i=1}^n ((y_i - x_i^T w_ridge) / (1 - H_ii))²

where H = X(X^T X + nλI)^(-1)X^T is the hat matrix.

**Complexity:** O(n³) for direct computation, O(n²p) with smart updates.

**Theorem 6 (Ridge Shrinkage in Eigenspace):**

Let X^T X = UΣ²U^T be the eigendecomposition.

**Ridge solution in eigenspace:**
w_ridge = U diag(σ_j² / (σ_j² + nλ)) U^T X^T y

where σ_j are singular values of X.

**Shrinkage factor for jth component:**
s_j(λ) = σ_j² / (σ_j² + nλ)

**Properties:**
- Large σ_j (principal directions): s_j ≈ 1 (minimal shrinkage)
- Small σ_j (noise directions): s_j ≈ 0 (maximal shrinkage)

**Interpretation:** Ridge shrinks more in directions with low variance!

**Example:**
- σ_1 = 100: s_1(λ=1) = 10000/10001 ≈ 0.9999 (1% shrinkage)
- σ_100 = 1: s_100(λ=1) = 1/2 = 0.5 (50% shrinkage)

**Theorem 7 (Ridge Sample Complexity):**

To achieve ε-accuracy with probability 1-δ:

n = O((d/ε²) · log(d/δ))

where d is the effective dimensionality: d = tr[(X^T X)(X^T X + nλI)^(-1)]

**For λ → 0 (no regularization):** d = p (full dimension)
**For λ → ∞ (strong regularization):** d → 0

**Key insight:** Ridge reduces effective dimensionality!

**Quantitative example:**
- p = 1000 features
- λ = 1: d_eff ≈ 100 (10× reduction)
- Sample complexity: n = O(100/ε²) instead of O(1000/ε²)
- **10× less data needed for same accuracy!**

### Formula

```
Loss = (1/n)Σ(yi - ŷi)² + λ × Σ(wi²)
     = MSE + λ × ||w||₂²
```

Where:
- **n**: Number of training samples
- **||w||₂²**: L2 norm squared of weight vector (sum of squared weights)
- **λ**: Regularization parameter (higher λ = stronger regularization)

Penalizes sum of squared weights, encouraging small but non-zero coefficients.

### Implementation

```python
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Ridge regression with different alpha values
alphas = [0, 0.01, 0.1, 1, 10, 100]
results = []

for alpha in alphas:
    model = Ridge(alpha=alpha)
    model.fit(X_train_poly, y_train)

    train_score = model.score(X_train_poly, y_train)
    test_score = model.score(X_test_poly, y_test)
    max_coef = np.abs(model.coef_).max()

    results.append({
        'Alpha': alpha,
        'Train R²': train_score,
        'Test R²': test_score,
        'Gap': train_score - test_score,
        'Max |Coef|': max_coef
    })

    print(f"Alpha={alpha}:")
    print(f"  Train R²: {train_score:.4f}, Test R²: {test_score:.4f}")
    print(f"  Max coefficient: {max_coef:.2e}")
```

### Visualizing Ridge Effect

```python
import pandas as pd

df_results = pd.DataFrame(results)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Train vs Test scores
axes[0].plot(df_results['Alpha'], df_results['Train R²'], 'o-',
            label='Train R²', linewidth=2)
axes[0].plot(df_results['Alpha'], df_results['Test R²'], 's-',
            label='Test R²', linewidth=2)
axes[0].set_xscale('log')
axes[0].set_xlabel('Alpha (λ)')
axes[0].set_ylabel('R² Score')
axes[0].set_title('Ridge: Performance vs Regularization')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Gap
axes[1].plot(df_results['Alpha'], df_results['Gap'], 'o-', linewidth=2)
axes[1].set_xscale('log')
axes[1].set_xlabel('Alpha (λ)')
axes[1].set_ylabel('Train - Test Gap')
axes[1].set_title('Overfitting Gap')
axes[1].grid(True, alpha=0.3)

# Plot 3: Coefficient magnitudes
axes[2].plot(df_results['Alpha'], df_results['Max |Coef|'], 'o-', linewidth=2)
axes[2].set_xscale('log')
axes[2].set_yscale('log')
axes[2].set_xlabel('Alpha (λ)')
axes[2].set_ylabel('Max |Coefficient|')
axes[2].set_title('Coefficient Shrinkage')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Coefficient Paths

```python
# Show how coefficients shrink with regularization
alphas_range = np.logspace(-2, 4, 100)
coef_paths = []

for alpha in alphas_range:
    model = Ridge(alpha=alpha)
    model.fit(X_train_poly, y_train)
    coef_paths.append(model.coef_)

coef_paths = np.array(coef_paths)

plt.figure(figsize=(12, 6))
for i in range(coef_paths.shape[1]):
    plt.plot(alphas_range, coef_paths[:, i], linewidth=1)

plt.xscale('log')
plt.xlabel('Alpha (λ)')
plt.ylabel('Coefficient Value')
plt.title('Ridge: Coefficient Paths')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
plt.show()
```

### Rigorous Mathematical Theory of Ridge Regression

**Bayesian Interpretation:**

```
Ridge regression is equivalent to Maximum A Posteriori (MAP) estimation
with Gaussian prior on weights:

Likelihood: p(y|X,w) = N(Xw, σ²I)
Prior: p(w) = N(0, τ²I)

Posterior: p(w|X,y) ∝ p(y|X,w) · p(w)

Taking negative log:
-log p(w|X,y) = (1/2σ²)||y - Xw||² + (1/2τ²)||w||² + const

Ridge objective with λ = σ²/τ²:
argmin_w [(1/2)||y - Xw||² + (λ/2)||w||²]

Interpretation: Ridge assumes weights are normally distributed!
Stronger regularization λ ⇔ tighter prior (smaller τ²)
```

**Closed-Form Solution with SVD Analysis:**

```
Ridge objective:
L(w) = (1/2)||y - Xw||² + (λ/2)||w||²

Setting gradient to zero:
∇_w L = -Xᵀ(y - Xw) + λw = 0
Xᵀy = (XᵀX + λI)w

Closed-form solution:
ŵ_ridge = (XᵀX + λI)⁻¹Xᵀy

Compare to OLS: ŵ_OLS = (XᵀX)⁻¹Xᵀy

SVD Analysis:
Let X = UΣVᵀ be SVD of X (n×d matrix, rank r ≤ min(n,d))
U: n×r orthonormal (left singular vectors)
Σ: r×r diagonal with singular values σᵢ
V: d×r orthonormal (right singular vectors)

Then:
XᵀX = VΣ²Vᵀ
(XᵀX + λI)⁻¹ = V(Σ² + λI)⁻¹Vᵀ

Ridge solution:
ŵ_ridge = V(Σ² + λI)⁻¹ΣUᵀy
        = Σᵢ₌₁ʳ [σᵢ/(σᵢ² + λ)] · (uᵢᵀy) · vᵢ

OLS solution:
ŵ_OLS = Σᵢ₌₁ʳ [1/σᵢ] · (uᵢᵀy) · vᵢ

Shrinkage factor: σᵢ/(σᵢ² + λ) < 1/σᵢ

Effect of λ on singular values:
- Large σᵢ (strong signal): σᵢ²/(σᵢ² + λ) ≈ 1 (minimal shrinkage)
- Small σᵢ (weak signal): σᵢ²/(σᵢ² + λ) ≈ σᵢ²/λ ≈ 0 (strong shrinkage)

Key insight: Ridge shrinks components in weak directions more!

Effective degrees of freedom:
df(λ) = Σᵢ₌₁ʳ σᵢ²/(σᵢ² + λ)

Properties:
- λ = 0: df = r (full rank)
- λ → ∞: df → 0 (shrink to 0)
- Smooth function of λ
```

**Stability and Condition Number:**

```
Condition number of XᵀX:
κ(XᵀX) = σ_max²/σ_min²

For ill-conditioned problems (σ_min ≈ 0):
- OLS unstable: (XᵀX)⁻¹ requires dividing by σ_min² ≈ 0
- Coefficients have huge variance
- Small data changes → large coefficient changes

Ridge fixes this:
κ(XᵀX + λI) = (σ_max² + λ)/(σ_min² + λ)

As λ increases:
- Condition number decreases
- Solution becomes more stable
- Variance of ŵ decreases

Bias-Variance Tradeoff:
E[ŵ_ridge] = (XᵀX + λI)⁻¹XᵀXw* ≠ w* (biased!)
Var[ŵ_ridge] = σ²(XᵀX + λI)⁻¹XᵀX(XᵀX + λI)⁻¹ (lower than OLS!)

MSE decomposition:
MSE(ŵ) = Bias²(ŵ) + Var(ŵ) + σ²

Optimal λ balances bias-variance:
λ_optimal ≈ argmin_λ [Bias²(λ) + Var(λ)]

Typically found via cross-validation
```

**Connections to Other Methods:**

```
Ridge as Constrained Optimization:
Lagrangian formulation equivalent to:

minimize   (1/2)||y - Xw||²
subject to ||w||² ≤ t

where t and λ are in one-to-one correspondence (larger λ ⇔ smaller t)

Ridge as Kernel Ridge Regression:
When n < d (more features than samples), use kernel trick:

ŵ = Xᵀα where α = (XXᵀ + λI)⁻¹y

Prediction: ŷ = Xŵ = Xᵀ(XXᵀ + λI)⁻¹y
This is kernel ridge regression with linear kernel K = XXᵀ

Computational advantage: Invert n×n matrix instead of d×d
When n << d, this is much faster!

Ridge as Tikhonov Regularization:
Generalized form with penalty matrix Γ:

ŵ = argmin_w [||y - Xw||² + ||Γw||²]

Standard Ridge: Γ = √λ · I
Can use other Γ to encode prior knowledge about w
```

### Rigorous Statistical Theory of Ridge Regression

**Theorem 1 (Ridge Regression Generalization Bound - Hsu et al., 2014):**

```
For Ridge regression with parameter λ, with probability ≥ 1-δ over n samples:

L(ŵ_ridge) - L(w*) ≤ O((σ²·tr[(XᵀX + λI)⁻¹XᵀX] / n) + λ||w*||²) + O(√(log(1/δ) / n))

where:
- L(w) = E[(y - Xw)²] (population risk)
- σ² = Var[ε] (noise variance)
- tr[(XᵀX + λI)⁻¹XᵀX] = effective degrees of freedom

Key Insights:
1. Bias term: λ||w*||² (increases with λ)
2. Variance term: σ²·df(λ)/n (decreases with λ)
3. Optimal λ balances bias-variance tradeoff

Sample Complexity:
For ε-optimal solution: n = Ω(df(λ)/ε²)

where df(λ) = Σᵢ σᵢ²/(σᵢ² + λ) ≤ rank(X)
```

**Theorem 2 (Ridge Regression Consistency):**

```
Under mild conditions, as n → ∞:

||ŵ_ridge - w_λ*|| →_P 0

where w_λ* = argmin_w [E[(y - Xw)²] + λ||w||²] (population minimizer)

If additionally λ = λ(n) → 0 and λn → ∞:
||ŵ_ridge - w*|| →_P 0  (consistent for true w*)

Optimal rate: λ ~ n⁻¹/² gives ||ŵ - w*|| = O_P(n⁻¹/²)
```

**Theorem 3 (Ridge MSE Decomposition):**

```
For fixed design matrix X:

MSE(ŵ_ridge) = E[||ŵ_ridge - w*||²]
             = ||E[ŵ_ridge] - w*||² + tr[Cov(ŵ_ridge)]
             = Bias²(λ) + Variance(λ)

Explicit formulas:
Bias²(λ) = ||λ(XᵀX + λI)⁻¹w*||²
         = λ² · wᵀ(XᵀX + λI)⁻²w*

Variance(λ) = σ² · tr[(XᵀX + λI)⁻¹XᵀX(XᵀX + λI)⁻¹]
            = σ² · Σᵢ σᵢ²/(σᵢ² + λ)²

Optimal λ:
λ* = argmin_λ MSE(λ) satisfies:
dMSE/dλ|_λ* = 0

Closed form (if w* and σ² known):
λ_optimal = σ² / ||w*||²

In practice: Use cross-validation to estimate λ_optimal
```

**Theorem 4 (Stability of Ridge Regression - Bousquet & Elisseeff, 2002):**

```
Ridge regression has uniform stability:

sup_{S,z} |L(ŵ_S, z) - L(ŵ_S', z)| ≤ 2M² / (nλ)

where:
- S' differs from S in one sample
- M = bound on ||x||
- ŵ_S = Ridge estimator on S

Generalization bound (from stability):
E[L(ŵ)] - L̂(ŵ) ≤ 2M² / (nλ) + O(√(log(1/δ) / n))

Interpretation:
- Larger λ → more stable → better generalization
- Stability decreases with 1/λ
```

**Theorem 5 (Ridge vs OLS MSE Comparison):**

```
Theorem (Hoerl & Kennard, 1970):
There always exists λ > 0 such that:

MSE(ŵ_ridge(λ)) < MSE(ŵ_OLS)

Proof sketch:
MSE(ŵ_ridge) = Bias² + Variance
MSE(ŵ_OLS) = 0 + Variance_OLS

For small λ > 0:
Bias² = O(λ²)  (quadratic in λ)
Variance_OLS - Variance_ridge = O(λ)  (linear in λ)

⇒ Net reduction = O(λ) - O(λ²) > 0 for small λ > 0 ∎

Practical implication:
Ridge with properly chosen λ ALWAYS beats OLS in MSE!
```

### Properties of Ridge

**Advantages:**
- ✅ Smooth, continuous shrinkage
- ✅ Computationally efficient (closed-form solution)
- ✅ Handles correlated features well
- ✅ Always has a solution (well-posed even when XᵀX singular)
- ✅ Reduces condition number → numerical stability
- ✅ Bayesian interpretation (Gaussian prior)
- ✅ Uniform stability: O(1/(nλ))
- ✅ Always better than OLS in MSE (with proper λ)

**Disadvantages:**
- ❌ Doesn't perform feature selection (keeps all features)
- ❌ Coefficients never exactly zero
- ❌ Biased estimator (E[ŵ] ≠ w*)

**When to use:**
- Many features with small-to-medium effects
- Correlated features (multicollinearity)
- When you want to keep all features
- Ill-conditioned design matrices (numerical stability needed)

---

## L1 Regularization (Lasso)

**Also called:** Lasso Regression (Least Absolute Shrinkage and Selection Operator)

### Rigorous Theory of L1 Regularization

**Theorem 8 (Lasso Sparsity - Tibshirani, 1996):**

For Lasso regression:

min_w (1/(2n))||Xw - y||² + λ||w||₁

**Key property:** Lasso produces **sparse** solutions (many coefficients exactly zero).

**Geometric intuition:**
- Constraint form: min ||Xw - y||² subject to ||w||₁ ≤ t
- L1 ball has corners at coordinate axes
- Optimal solution often lies at corner → sparsity!

**Comparison with Ridge:**
- Ridge constraint: ||w||₂² ≤ t (circular ball, no corners)
- Ridge rarely produces exactly zero coefficients
- Lasso constraint: ||w||₁ ≤ t (diamond shape with corners)

**Theorem 9 (Lasso Soft-Thresholding - Donoho & Johnstone, 1994):**

For orthonormal design (X^T X = I):

**Closed-form solution:**
w_lasso,j = sign(w_OLS,j) · max(|w_OLS,j| - λ, 0)

This is the **soft-thresholding operator:**

S_λ(z) = {
  z - λ,  if z > λ
  0,      if |z| ≤ λ
  z + λ,  if z < -λ
}

**Proof:**
For orthonormal X, objective separates:
L(w) = Σ_j [(w_j - w_OLS,j)² / 2 + λ|w_j|]

Take subdifferential and set to zero:
∂L/∂w_j = w_j - w_OLS,j + λ·sign(w_j) = 0 (if w_j ≠ 0)

Solving: w_j = w_OLS,j - λ·sign(w_j)

For w_j > 0: w_j = w_OLS,j - λ
For w_j < 0: w_j = w_OLS,j + λ
For w_j = 0: |w_OLS,j| ≤ λ ∎

**Key insight:** Lasso shrinks by constant amount λ, Ridge shrinks by proportion!

**Theorem 10 (Lasso Support Recovery - Wainwright, 2009):**

Let S = {j : w*_j ≠ 0} be the true support (non-zero indices).

**Under restricted eigenvalue condition:**
If λ ≥ c · σ·√(log p / n) for some constant c,

then with probability 1-δ:

**1. Sparsistency:** supp(w_lasso) = S (exact support recovery)
**2. ℓ₂ consistency:** ||w_lasso - w*||₂ ≤ O(σ√(|S|log p / n))
**3. ℓ∞ consistency:** ||w_lasso - w*||∞ ≤ O(σ√(log p / n))

where:
- |S| = number of true non-zeros (sparsity level)
- p = total number of features
- n = number of samples
- σ = noise level

**Sample complexity:**
For exact support recovery: n = Ω(|S|·log p)

**Interpretation:**
- Lasso can recover true support with O(s log p) samples
- Logarithmic scaling in p enables high-dimensional learning!
- Compare to OLS: needs n > p (linear scaling)

**Theorem 11 (Lasso as Bayesian MAP with Laplace Prior):**

Lasso is equivalent to MAP estimation with Laplace prior:

**Likelihood:** p(y|X, w) = N(Xw, σ²I)
**Prior:** p(w) = ∏_j (λ/(2σ²)) exp(-λ|w_j|/σ²)

**Posterior:**
p(w|X, y) ∝ exp(-(1/(2σ²))||y - Xw||² - (λ/σ²)||w||₁)

**MAP estimate:**
w_MAP = argmax_w p(w|X, y) = argmin_w (||y - Xw||² + 2λ||w||₁) ∎

**Interpretation:**
- Laplace prior p(w_j) ∝ exp(-λ|w_j|) has peak at zero
- Encourages sparsity (many weights at zero)
- Heavier tails than Gaussian → less penalty for large weights

**Theorem 12 (Lasso Non-Uniqueness and Instability):**

**Non-uniqueness:** If features are highly correlated, Lasso solution may not be unique.

**Example:** If x_i = x_j (identical features), any convex combination satisfies optimality:
w_lasso,i + w_lasso,j = constant, but individual values arbitrary!

**Instability:** Small perturbations in data can change selected features.

**Theorem (Zou & Hastie, 2005):** If X^T X is not full rank, Lasso cannot select more than n features.

**Practical implication:** Lasso can be unstable with high correlation!

**Solution:** Elastic Net (combines L1 + L2)

**Theorem 13 (Lasso Computational Complexity):**

**No closed form!** Unlike Ridge, Lasso requires iterative optimization.

**Common algorithms:**

**1. Coordinate Descent (CD):**
- Update one w_j at a time
- Complexity per iteration: O(np)
- Total: O(np · iterations) ≈ O(np · log(1/ε))
- **Fastest for moderate p**

**2. LARS (Least Angle Regression):**
- Computes entire regularization path
- Complexity: O(min(n², np))
- Exact solution (no approximation)

**3. Proximal Gradient Descent:**
- Iterative: w^(t+1) = S_λη(w^(t) - η∇f(w^(t)))
- Complexity per iteration: O(np)
- Convergence: O(1/ε) iterations for ε-accuracy

**Comparison:**
- Ridge: O(np²) or O(n²p) (matrix inversion, one-shot)
- Lasso: O(np · log(1/ε)) per λ (iterative)
- Lasso is slower but provides feature selection!

**Theorem 14 (Lasso Oracle Inequality - Bickel et al., 2009):**

Under restricted eigenvalue condition, with λ ~ σ√(log p / n):

||w_lasso - w*||²₂ ≤ C · |S| · σ² · (log p / n)

with probability 1-δ.

**Oracle property:**
If we knew true support S, OLS on S achieves:
||w_OLS,S - w*||²₂ = O(|S|·σ² / n)

**Lasso pays only log p factor:**
Lasso rate / Oracle rate = (log p / n) / (1 / n) = log p

**This is nearly optimal!**

**Theorem 15 (Lasso Cross-Validation Consistency - Homrighausen & McDonald, 2013):**

Let λ_CV be the λ selected by K-fold cross-validation.

**Theorem:** Under regularity conditions, as n → ∞:

λ_CV → λ_optimal with probability 1

where λ_optimal minimizes prediction error.

**Practical guideline:**
- Use 5-fold or 10-fold CV
- Search λ on logarithmic grid: [10⁻⁴, 10⁻³, ..., 10²]
- Select λ with minimum CV error
- "One-standard-error rule": Select largest λ within 1 SE of minimum

### Formula

```
Loss = (1/n)Σ(yi - ŷi)² + λ × Σ|wi|
     = MSE + λ × ||w||₁
```

Where:
- **||w||₁**: L1 norm of weight vector (sum of absolute values)
- **λ**: Regularization parameter (higher λ = more sparsity)

Penalizes sum of absolute values of weights. The L1 penalty creates sparsity because its derivative is constant (±1 for non-zero weights), driving small coefficients to exactly zero.

### Implementation

```python
from sklearn.linear_model import Lasso

# Lasso regression with different alpha values
alphas = [0.001, 0.01, 0.1, 1, 10]
results_lasso = []

for alpha in alphas:
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(X_train_poly, y_train)

    train_score = model.score(X_train_poly, y_train)
    test_score = model.score(X_test_poly, y_test)
    n_nonzero = np.sum(model.coef_ != 0)

    results_lasso.append({
        'Alpha': alpha,
        'Train R²': train_score,
        'Test R²': test_score,
        'Non-zero Coefs': n_nonzero,
        'Total Coefs': len(model.coef_)
    })

    print(f"Alpha={alpha}:")
    print(f"  Train R²: {train_score:.4f}, Test R²: {test_score:.4f}")
    print(f"  Non-zero coefficients: {n_nonzero}/{len(model.coef_)}")
```

### Feature Selection with Lasso

```python
# Visualize which features Lasso selects
model_lasso = Lasso(alpha=0.1, max_iter=10000)
model_lasso.fit(X_train_poly, y_train)

# Plot coefficients
plt.figure(figsize=(12, 6))
plt.bar(range(len(model_lasso.coef_)), model_lasso.coef_)
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.title(f'Lasso Coefficients (α=0.1) - {np.sum(model_lasso.coef_ != 0)} non-zero')
plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
plt.grid(True, alpha=0.3, axis='y')
plt.show()

print(f"Features set to zero: {np.sum(model_lasso.coef_ == 0)}/{len(model_lasso.coef_)}")
```

### Coefficient Paths for Lasso

```python
# Show how coefficients go to exactly zero
alphas_range = np.logspace(-3, 1, 100)
coef_paths_lasso = []

for alpha in alphas_range:
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(X_train_poly, y_train)
    coef_paths_lasso.append(model.coef_)

coef_paths_lasso = np.array(coef_paths_lasso)

plt.figure(figsize=(12, 6))
for i in range(coef_paths_lasso.shape[1]):
    plt.plot(alphas_range, coef_paths_lasso[:, i], linewidth=1)

plt.xscale('log')
plt.xlabel('Alpha (λ)')
plt.ylabel('Coefficient Value')
plt.title('Lasso: Coefficient Paths (Note: Some paths hit zero)')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
plt.show()
```

### Rigorous Mathematical Theory of Lasso

**Why L1 Creates Sparsity - Subdifferential Analysis:**

```
Lasso objective:
L(w) = (1/2)||y - Xw||² + λ||w||₁

L1 norm is non-differentiable at wᵢ = 0!
Use subdifferential ∂|wᵢ|:

∂|wᵢ| = { +1      if wᵢ > 0
        { -1      if wᵢ < 0
        { [-1,1]  if wᵢ = 0

Optimality condition (0 ∈ ∂L(ŵ)):
For each coordinate i:
0 ∈ -xᵢᵀ(y - Xŵ) + λ·∂|ŵᵢ|

Case 1: ŵᵢ > 0
-xᵢᵀ(y - Xŵ) + λ = 0
⇒ xᵢᵀ(y - Xŵ) = λ

Case 2: ŵᵢ < 0
-xᵢᵀ(y - Xŵ) - λ = 0
⇒ xᵢᵀ(y - Xŵ) = -λ

Case 3: ŵᵢ = 0
-λ ≤ xᵢᵀ(y - Xŵ) ≤ λ
⇒ |xᵢᵀ(y - Xŵ)| ≤ λ

Key Insight:
If |xᵢᵀ(y - X₋ᵢŵ₋ᵢ)| < λ, then ŵᵢ = 0 exactly!

This is why L1 produces exact zeros:
The flat subdifferential at 0 allows ŵᵢ = 0 to satisfy optimality

Compare to L2:
∂(wᵢ²) = 2wᵢ (smooth, always defined)
For wᵢ = 0: requires gradient = 0 (measure-zero event)
⇒ Ridge never produces exact zeros!
```

**Soft-Thresholding Operator:**

```
For orthogonal X (XᵀX = I), Lasso has closed form!

Coordinate-wise solution:
ŵᵢ = S_λ(ŵᵢ^OLS)

where S_λ is soft-thresholding operator:

S_λ(z) = sign(z)·max(|z| - λ, 0)
       = { z - λ    if z > λ
         { 0        if |z| ≤ λ
         { z + λ    if z < -λ

Geometric interpretation:
- If |ŵᵢ^OLS| ≤ λ: Shrink to exactly 0
- If |ŵᵢ^OLS| > λ: Shrink by λ toward 0

Compare to Ridge (for orthogonal X):
ŵᵢ^Ridge = ŵᵢ^OLS / (1 + λ)  (proportional shrinkage)

Visual comparison:
Ridge: ŵ = cŵ^OLS where c < 1 (scale down)
Lasso: ŵ = S_λ(ŵ^OLS) (shift toward 0, then truncate)
```

**Bayesian Interpretation:**

```
Lasso equivalent to MAP estimation with Laplace prior:

Prior: p(wᵢ) = (1/2b)exp(-|wᵢ|/b)  [Laplace/double exponential]

Taking negative log:
-log p(w|X,y) ∝ (1/2σ²)||y - Xw||² + (1/b)Σ|wᵢ|

Lasso with λ = σ²/b

Laplace prior properties:
- Peaked at 0 (encourages sparsity)
- Heavy tails (allows large weights when needed)
- L1 penalty is MAP with Laplace prior

Gaussian vs Laplace prior:
- Gaussian: p(w) ∝ exp(-w²/2τ²)  → Ridge (L2)
- Laplace: p(w) ∝ exp(-|w|/b)   → Lasso (L1)

Laplace has sharper peak at 0 → more mass on w=0 → sparsity!
```

**Coordinate Descent Algorithm:**

```
Lasso has no closed form for general X
Use coordinate descent (very efficient!):

Initialize: ŵ = 0
Repeat until convergence:
  For j = 1, ..., d:
    # Partial residual (fixing all weights except wⱼ)
    rⱼ = y - Σₖ≠ⱼ Xₖŵₖ

    # Coordinate-wise update
    ŵⱼ^OLS = xⱼᵀrⱼ / ||xⱼ||²

    # Soft-threshold
    ŵⱼ ← S_{λ/||xⱼ||²}(ŵⱼ^OLS)

Convergence: Guaranteed for convex Lasso problem
Complexity: O(nd) per pass through coordinates
Typically converges in 10-100 passes

Why coordinate descent works well:
- Each subproblem has closed-form solution (soft-threshold)
- Exploits sparsity (skip zero coordinates)
- No matrix inversions needed
- Very fast in practice
```

**Statistical Properties:**

```
Theorem (Lasso Consistency - Zhao & Yu, 2006):
Under "irrepresentable condition" on X:

With probability → 1 as n → ∞:
(i)  Lasso correctly identifies support: {i : ŵᵢ ≠ 0} = {i : wᵢ* ≠ 0}
(ii) ||ŵ - w*|| → 0 (consistent)

Irrepresentable condition:
Requires X_inactive not too correlated with X_active

Warning: This is a strong condition!
- Fails when features highly correlated
- Explains Lasso instability with correlation

Degrees of Freedom (Zou et al., 2007):
For Lasso: df(λ) ≈ E[# non-zero coefficients]

Not smooth like Ridge! Jumps when variables enter/leave model

Prediction Error Bound:
Under restricted eigenvalue condition:
||Xŵ - Xw*||² ≤ C·s·log(d)/n

where s = |{i : wᵢ* ≠ 0}| (true sparsity)

Key: Error depends on log(d), not d!
⇒ Lasso works even when d >> n (high-dimensional regime)
```

**Geometric Interpretation:**

```
Constrained formulation:
minimize   (1/2)||y - Xw||²
subject to ||w||₁ ≤ t

L1 ball geometry:
- In 2D: Diamond shape |w₁| + |w₂| ≤ t
- Has corners at axes (w₁=0 or w₂=0)
- Elliptical contours of ||y - Xw||² hit corners
- ⇒ Solution often at corner (sparse!)

L2 ball geometry:
- In 2D: Circle w₁² + w₂² ≤ t²
- No corners, smooth boundary
- Solution rarely at axis (w₁=0 or w₂=0)
- ⇒ No sparsity!

General Lp penalty (p ∈ (0,2]):
||w||_p^p = Σ|wᵢ|^p

p < 1: Even sparser (non-convex!)
p = 1: Lasso (sparsest convex penalty)
p = 2: Ridge (no sparsity)
```

### Properties of Lasso

**Advantages:**
- ✅ Performs automatic feature selection
- ✅ Produces sparse models (many coefficients exactly = 0)
- ✅ Interpretable (fewer features to explain)
- ✅ Good for high-dimensional data (d >> n)
- ✅ Prediction error depends on log(d), not d
- ✅ Bayesian interpretation (Laplace prior)

**Disadvantages:**
- ❌ Unstable with correlated features (violates irrepresentable condition)
- ❌ Arbitrarily selects one feature from correlated group
- ❌ Slower to compute than Ridge (no closed form, needs coordinate descent)
- ❌ May select at most n features when p > n
- ❌ Biased for large coefficients (shrinks everything by λ)

**Why instability with correlated features?**
When features are highly correlated (Pearson correlation |ρ| > 0.9), Lasso exhibits selection instability:
- For perfectly correlated features (ρ = 1), Lasso arbitrarily assigns weight to one feature
- Example: If X₁ and X₂ are identical, solutions can range from (w₁=k, w₂=0) to (w₁=0, w₂=k)
- Small perturbations in data can cause drastic changes in which feature is selected
- Ridge regression distributes weight equally: (w₁=k/2, w₂=k/2) for identical features
- Violates irrepresentable condition when X_inactive correlated with X_active

**When to use:**
- High-dimensional data (d >> n)
- When you believe many features are irrelevant (sparse ground truth)
- When interpretability is important (want few features)
- When you want automatic feature selection
- When features not too correlated (|ρ| < 0.5)

---

## Elastic Net

**Combines L1 and L2 regularization**

### Formula

```
Loss = MSE + λ₁ × Σ|wi| + λ₂ × Σ(wi²)
     = MSE + λ₁ × ||w||₁ + λ₂ × ||w||₂²
```

Or equivalently (sklearn parameterization):
```
Loss = MSE + λ × (α × ||w||₁ + (1-α)/2 × ||w||₂²)
```

Where:
- **α** (l1_ratio): Mixing parameter between L1 and L2 (0 ≤ α ≤ 1)
  - α = 1: Pure Lasso
  - α = 0: Pure Ridge
  - 0 < α < 1: Elastic Net
- **λ** (alpha): Overall regularization strength

### Implementation

```python
from sklearn.linear_model import ElasticNet

# Test different l1_ratio values
l1_ratios = [0, 0.25, 0.5, 0.75, 1.0]  # 0=Ridge, 1=Lasso

results_elastic = []

for l1_ratio in l1_ratios:
    model = ElasticNet(alpha=0.1, l1_ratio=l1_ratio, max_iter=10000)
    model.fit(X_train_poly, y_train)

    train_score = model.score(X_train_poly, y_train)
    test_score = model.score(X_test_poly, y_test)
    n_nonzero = np.sum(model.coef_ != 0)

    results_elastic.append({
        'L1 Ratio': l1_ratio,
        'Train R²': train_score,
        'Test R²': test_score,
        'Non-zero Coefs': n_nonzero
    })

    print(f"L1 Ratio={l1_ratio} ({'Lasso' if l1_ratio==1 else 'Ridge' if l1_ratio==0 else 'Elastic Net'}):")
    print(f"  Train R²: {train_score:.4f}, Test R²: {test_score:.4f}")
    print(f"  Non-zero coefficients: {n_nonzero}/{len(model.coef_)}")
```

### Visualizing Elastic Net Effect

```python
# Compare Ridge, Lasso, and Elastic Net
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

models = [
    ('Ridge (L2)', Ridge(alpha=0.1)),
    ('Elastic Net', ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)),
    ('Lasso (L1)', Lasso(alpha=0.1, max_iter=10000))
]

for ax, (name, model) in zip(axes, models):
    model.fit(X_train_poly, y_train)

    ax.bar(range(len(model.coef_)), model.coef_)
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Coefficient Value')
    ax.set_title(f'{name}\nNon-zero: {np.sum(model.coef_ != 0)}/{len(model.coef_)}')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```

### Rigorous Theory of Elastic Net

**Theorem 6 (Elastic Net Objective and Grouping Effect - Zou & Hastie, 2005):**

```
Elastic Net objective:
L_EN(w) = (1/2n)||y - Xw||² + λ₁||w||₁ + λ₂||w||²

Equivalent naive formulation:
ŵ_EN = argmin_w [(1/2)||y - Xw||² + λ(α||w||₁ + (1-α)||w||²/2)]

where α = λ₁/(λ₁ + λ₂), λ = λ₁ + λ₂

Grouping Effect Property:
For highly correlated features xᵢ, xⱼ (correlation ρ):

|ŵᵢ - ŵⱼ| ≤ C·(1 - ρ)/λ₂

Interpretation:
- When ρ → 1 (perfect correlation): ŵᵢ → ŵⱼ
- Elastic Net assigns similar weights to correlated features
- Lasso (λ₂=0) arbitrarily selects one feature
- Ridge (λ₁=0) averages weights equally

Proof Sketch:
The L2 penalty λ₂||w||² creates strong convexity:
∇²L_EN = XᵀX + λ₂I ≽ λ₂I

For correlated xᵢ ≈ xⱼ:
Perturbing wᵢ ↔ wⱼ increases ||w||² penalty
⇒ Optimal solution keeps wᵢ ≈ wⱼ ∎
```

**Theorem 7 (Elastic Net Oracle Inequalities - Zou & Zhang, 2009):**

```
Under restricted eigenvalue condition on X:

||ŵ_EN - w*||₁ ≤ C₁·s·λ₁
||Xŵ_EN - Xw*||² ≤ C₂·s·λ₁²

where s = |supp(w*)| (true sparsity)

Compared to Lasso:
- Similar oracle bounds
- Better finite-sample performance with correlation
- More stable variable selection

Sample Complexity:
For ε-optimal solution with probability ≥ 1-δ:

n = O((s·log(d))/(ε²)·log(1/δ))

Key: Depends on log(d), not d → works in high dimensions!
```

**Theorem 8 (Strong Oracle Property - Zou & Zhang, 2009):**

```
Under compatibility condition and appropriate λ₁, λ₂:

With probability → 1 as n → ∞:

(i)  Sign consistency: sign(ŵ_EN,i) = sign(w_i*) for all i ∈ supp(w*)
(ii) Support recovery: {i : ŵ_EN,i ≠ 0} = {i : w_i* ≠ 0}
(iii) Asymptotic normality: √n(ŵ_EN - w*) →_d N(0, Σ)

Conditions required:
- λ₁ = O(√(log(d)/n))  (Lasso rate)
- λ₂ = O(1/√n)  (shrinking to 0)
- Compatibility constant bounded away from 0

Weaker than Lasso's irrepresentable condition!
```

**Elastic Net vs Lasso vs Ridge Comparison:**

```
| Property | Ridge | Lasso | Elastic Net |
|----------|-------|-------|-------------|
| Sparsity | No | Yes | Yes |
| Grouping correlated | Yes (equal) | No | Yes (similar) |
| Oracle property | No | Yes* | Yes** |
| Computational cost | O(d³) | O(nd·K) | O(nd·K) |
| # Hyperparameters | 1 | 1 | 2 |

* Requires irrepresentable condition (strong)
** Requires compatibility condition (weaker)
K = number of coordinate descent iterations
```

### Properties of Elastic Net

**Advantages:**
- ✅ Best of both worlds: feature selection + stable with correlated features
- ✅ Works well when features are correlated
- ✅ Can select groups of correlated features (grouping effect)
- ✅ Weaker conditions than Lasso for oracle property
- ✅ Bounded coefficient differences: |ŵᵢ - ŵⱼ| ≤ C(1-ρ)/λ₂

**Disadvantages:**
- ❌ Two hyperparameters to tune (α and λ)
- ❌ More computationally expensive

**When to use:**
- Correlated features + want feature selection
- Not sure whether to use Ridge or Lasso
- High-dimensional data with feature groups
- When Lasso unstable (violates irrepresentable condition)

---

## Dropout

**Randomly deactivates neurons during training** (for neural networks)

### How It Works

During training:
1. For each forward pass, randomly deactivate each neuron with probability p (dropout rate)
2. Scale active neuron outputs by 1/(1-p) to maintain expected value
3. This creates an ensemble effect: each mini-batch trains a different sub-network
4. Prevents co-adaptation: neurons cannot rely on presence of specific other neurons

During inference (testing):
- Use all neurons without dropout (p=0)
- No scaling needed because training already scaled by 1/(1-p)

**Mathematical formulation:**
- Training: h = mask ⊙ activation, where mask ~ Bernoulli(1-p), then scale by 1/(1-p)
- Inference: h = activation (all neurons active)

```python
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

# Generate data
X_nn, y_nn = make_regression(n_samples=1000, n_features=20, noise=10, random_state=42)
X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(
    X_nn, y_nn, test_size=0.2, random_state=42
)

# Scale data
scaler = StandardScaler()
X_train_nn = scaler.fit_transform(X_train_nn)
X_test_nn = scaler.transform(X_test_nn)

# Model WITHOUT dropout
model_no_dropout = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(20,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1)
])

model_no_dropout.compile(optimizer='adam', loss='mse', metrics=['mae'])

history_no_dropout = model_no_dropout.fit(
    X_train_nn, y_train_nn,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    verbose=0
)

# Model WITH dropout
model_dropout = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(20,)),
    keras.layers.Dropout(0.5),  # 50% dropout
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1)
])

model_dropout.compile(optimizer='adam', loss='mse', metrics=['mae'])

history_dropout = model_dropout.fit(
    X_train_nn, y_train_nn,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    verbose=0
)

# Compare training curves
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Without dropout
axes[0].plot(history_no_dropout.history['loss'], label='Train Loss')
axes[0].plot(history_no_dropout.history['val_loss'], label='Val Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Without Dropout')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# With dropout
axes[1].plot(history_dropout.history['loss'], label='Train Loss')
axes[1].plot(history_dropout.history['val_loss'], label='Val Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('With Dropout (p=0.5)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Final test performance
test_loss_no_dropout = model_no_dropout.evaluate(X_test_nn, y_test_nn, verbose=0)[0]
test_loss_dropout = model_dropout.evaluate(X_test_nn, y_test_nn, verbose=0)[0]

print(f"Test Loss without Dropout: {test_loss_no_dropout:.2f}")
print(f"Test Loss with Dropout: {test_loss_dropout:.2f}")
```

### Different Dropout Rates

```python
# Test different dropout rates
dropout_rates = [0, 0.1, 0.3, 0.5, 0.7]
results_dropout = []

for rate in dropout_rates:
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(20,)),
        keras.layers.Dropout(rate),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(rate),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    history = model.fit(
        X_train_nn, y_train_nn,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        verbose=0
    )

    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]

    results_dropout.append({
        'Rate': rate,
        'Train Loss': final_train_loss,
        'Val Loss': final_val_loss,
        'Gap': final_train_loss - final_val_loss
    })

df_dropout = pd.DataFrame(results_dropout)
print(df_dropout.to_string(index=False))
```

### Rigorous Theory of Dropout

**Theorem 9 (Dropout as Ensemble Learning - Srivastava et al., 2014):**

```
Dropout with rate p trains exponentially many sub-networks:

Number of possible sub-networks: 2^n (for n neurons)

Training objective (Monte Carlo approximation):
E_δ~Bernoulli(1-p)[L(f(x; w ⊙ δ), y)]

where δ = dropout mask, ⊙ = element-wise product

Inference approximation:
E[h_dropout] = (1-p)·h  (expectation over masks)

⇒ At test time: Use all weights scaled by (1-p)

Ensemble interpretation:
Dropout ≈ Averaging predictions from 2^n models
- Each model trained on a different architecture
- Weight sharing through common underlying weights
- Prevents co-adaptation of neurons
```

**Theorem 10 (Dropout Regularization Effect - Wager et al., 2013):**

```
For generalized linear models with dropout:

Dropout objective ≈ L2 regularization + data-dependent penalty

Specifically for linear regression:
E_δ[L_dropout] ≈ MSE + (p/(2(1-p)))·Σᵢ||wᵢ||²·||xᵢ||²

Key insight: Dropout adds penalty proportional to:
- Weight magnitude: ||wᵢ||²
- Feature magnitude: ||xᵢ||² (data-dependent!)

Compare to standard L2: penalty only on ||wᵢ||²

Advantage:
- Features with larger variance get stronger regularization
- Adaptive regularization based on data statistics
```

**Theorem 11 (Dropout Convergence - Helmbold & Long, 2017):**

```
For neural networks trained with dropout rate p:

Generalization bound with probability ≥ 1-δ:

L(ŵ) ≤ L̂(ŵ) + O(√(W·log(W)·log(1/p)/(n·(1-p)²)) + √(log(1/δ)/n))

where:
- W = number of weights in network
- n = training samples
- p = dropout rate

Sample Complexity:
For ε-optimal solution:
n = O(W·log(W)·log(1/p)/(ε²·(1-p)²))

Key insights:
1. Effective sample size: n·(1-p)² (reduced by dropout)
2. Need more samples to compensate for dropout
3. Regularization strength increases with p
```

**Theorem 12 (Dropout Training Dynamics):**

```
Expected gradient under dropout:

E_δ[∇L(w; δ)] ≠ ∇E_δ[L(w; δ)]  (biased gradient!)

Variance of dropout gradient:
Var[∇L_dropout] = Var[∇L] + Var_dropout[∇L]

Dropout adds gradient noise:
- Acts as implicit regularization
- Helps escape sharp minima
- Prefers flat minima (better generalization)

Connection to Sharp vs Flat Minima:
Dropout noise: σ² ~ O(p/(1-p))

At sharp minimum (large Hessian eigenvalues):
- High gradient variance
- Unstable training
- SGD pushed away

At flat minimum (small Hessian eigenvalues):
- Low gradient variance
- Stable training
- SGD attracted

⇒ Dropout implicitly finds flat minima!
```

**Theorem 13 (Optimal Dropout Rate - Baldi & Sadowski, 2013):**

```
For single-layer network with d inputs:

Optimal dropout rate p* minimizes generalization error:

p* ≈ 1 - √((n-1)/(n + d - 1))

Asymptotic behavior:
- Small d (few features): p* → 0 (little dropout)
- Large d (many features): p* → 1 - √(n/d)
- n >> d: p* → 0 (enough data, less regularization)
- d >> n: p* → 1 - √(n/d) (underdetermined, strong regularization)

Typical values:
| Setting | n/d | Optimal p* |
|---------|-----|------------|
| Underdetermined | 0.5 | ≈ 0.3 |
| Balanced | 1.0 | ≈ 0.3 |
| Overdetermined | 2.0 | ≈ 0.2 |
| Plenty data | 10.0 | ≈ 0.05 |

Rule of thumb: Start with p=0.5 for hidden layers, p=0.2 for input
```

**Comparison: Dropout vs L2 Regularization:**

```
| Property | Dropout | L2 (Weight Decay) |
|----------|---------|-------------------|
| Regularization type | Multiplicative noise | Penalty on weights |
| Data-dependent | Yes (||x||²) | No |
| Ensemble effect | Yes (2^n models) | No |
| Training time | Slower (2× epochs) | Normal |
| Gradient noise | High | Low |
| Minimum preference | Flat | Any |
| Optimal rate | Data-dependent | Fixed λ |

Empirical observation (Hinton et al.):
Dropout often outperforms L2 for deep networks
Reason: Ensemble effect + preference for flat minima
```

### Properties of Dropout

**Advantages:**
- ✅ Very effective for deep networks
- ✅ Acts as ensemble of 2^n sub-networks
- ✅ Simple to implement
- ✅ No additional parameters to learn
- ✅ Data-dependent regularization: penalty ∝ ||x||²
- ✅ Implicit preference for flat minima
- ✅ Prevents co-adaptation of neurons

**Disadvantages:**
- ❌ Increases training time (need ~2× epochs)
- ❌ Only applicable to neural networks
- ❌ Adds gradient noise: Var ∝ p/(1-p)
- ❌ Effective sample size reduced to n(1-p)²

**When to use:**
- Deep neural networks
- Network is overfitting (large train-validation gap)
- High-dimensional features (d >> n)
- Typical dropout rates:
  - Hidden layers: 0.2-0.5 (commonly 0.3-0.5)
  - Input layer: 0.1-0.2 (more conservative)
  - Recurrent layers: 0.1-0.3 (lower rates for RNNs/LSTMs)
  - Use Theorem 13 for data-driven p*: p* ≈ 1 - √(n/d)

---

## Early Stopping

**Stop training when validation performance stops improving**

### Implementation

```python
from sklearn.neural_network import MLPRegressor

# Without early stopping
model_no_early = MLPRegressor(
    hidden_layer_sizes=(100, 100),
    max_iter=500,
    early_stopping=False,
    random_state=42
)

model_no_early.fit(X_train_nn, y_train_nn)

# With early stopping
model_early = MLPRegressor(
    hidden_layer_sizes=(100, 100),
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=10,  # Stop if no improvement for 10 iterations
    random_state=42
)

model_early.fit(X_train_nn, y_train_nn)

print("Without Early Stopping:")
print(f"  Iterations: {model_no_early.n_iter_}")
print(f"  Train Loss: {model_no_early.loss_:.4f}")

print("\nWith Early Stopping:")
print(f"  Iterations: {model_early.n_iter_} (stopped early)")
print(f"  Train Loss: {model_early.loss_:.4f}")
print(f"  Best Validation Score: {model_early.best_validation_score_:.4f}")

# Plot loss curves
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(model_no_early.loss_curve_)
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Loss')
axes[0].set_title('Without Early Stopping')
axes[0].grid(True, alpha=0.3)

axes[1].plot(model_early.loss_curve_, label='Train')
axes[1].plot(model_early.validation_scores_, label='Validation')
axes[1].axvline(x=len(model_early.loss_curve_), color='r',
               linestyle='--', label='Stopped Here')
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Loss/Score')
axes[1].set_title('With Early Stopping')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Keras Early Stopping

```python
# More sophisticated early stopping with Keras
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',      # Metric to monitor
    patience=10,             # Number of epochs to wait
    restore_best_weights=True,  # Restore weights from best epoch
    verbose=1
)

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(20,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

history = model.fit(
    X_train_nn, y_train_nn,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=0
)

print(f"Training stopped at epoch {len(history.history['loss'])}")

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.axvline(x=len(history.history['loss']) - early_stopping.patience,
           color='r', linestyle='--', label='Best Model')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Early Stopping')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Rigorous Theory of Early Stopping

**Theorem 14 (Early Stopping as Implicit Regularization - Yao et al., 2007):**

```
For gradient descent with step size η on quadratic loss:

w_t = (I - ηH)^t w_0 + Σ_{τ=0}^{t-1} (I - ηH)^τ η∇L̂

where H = Hessian of population loss L

Connection to Ridge Regression:
Let H = UΛU^T (eigendecomposition), then:

w_t → w_ridge as t → ∞

with implicit regularization:
λ(t) ≈ 1/(ηt)

Key insight: Stopping at iteration t* ⇔ Ridge with λ ~ 1/(ηt*)

Early stopping implicitly regularizes!
- More iterations → less regularization (smaller λ)
- Fewer iterations → more regularization (larger λ)

Proof sketch:
lim_{t→∞} Σ_{τ=0}^{t-1} (I - ηH)^τ = H^{-1}  (geometric series)

For finite t:
Σ_{τ=0}^{t-1} (I - ηH)^τ ≈ (H + λ(t)I)^{-1}  where λ(t) ~ 1/(ηt)
```

**Theorem 15 (Optimal Stopping Time - Hardt et al., 2016):**

```
For stochastic gradient descent with step size η:

Generalization bound:
E[L(w_t) - L̂(w_t)] ≤ O(ηt·σ²/n + ||w_0 - w*||²/(ηt))

where σ² = variance of stochastic gradients

Minimizing w.r.t. t:
d/dt [...] = 0 ⇒ ηt* ~ ||w_0 - w*||/σ·√n

Optimal iterations:
t* = O(||w_0 - w*||/(η·σ)·√n)

Sample complexity (for fixed t):
n = O(ηt·σ²/ε²)

Interpretation:
- More data (larger n) → can train longer (larger t*)
- Higher noise (larger σ) → should stop earlier (smaller t*)
- Better initialization (smaller ||w_0 - w*||) → can stop earlier
```

**Theorem 16 (Early Stopping Validation Error - Shalev-Shwartz et al., 2010):**

```
With probability ≥ 1-δ over validation set V of size m:

True stopping point t_true satisfies:
|L(w_{t_val}) - L(w_{t_true})| ≤ O(√(T·log(T/δ)/m))

where:
- t_val = iteration with best validation error
- T = total iterations
- m = validation set size

Sample complexity for validation set:
m = Ω(T·log(T)/ε²)  to get ε-accurate stopping

Key insight: Need O(T) validation samples!
- Training for longer → need larger validation set
- Rule of thumb: m ≥ 0.2n (20% for validation)
```

**Theorem 17 (Patience Parameter Analysis):**

```
Patience p = number of epochs without improvement before stopping

Optimal patience p* balances:
1. Risk of stopping too early (high variance in validation loss)
2. Risk of overfitting by continuing too long

For validation loss with noise variance σ_val²:

Optimal patience:
p* ≈ c·σ_val/ΔL_epoch

where:
- ΔL_epoch = typical improvement per epoch
- c ≈ 1-3 (tuning constant)

Interpretation:
- High noise (σ_val large) → need more patience
- Fast improvement (ΔL_epoch large) → less patience needed

Typical values:
- Clean validation: p = 5-10
- Noisy validation: p = 10-20
- Very noisy: p = 20-50
```

**Early Stopping vs Explicit Regularization:**

```
| Aspect | Early Stopping | L2 Regularization |
|--------|---------------|-------------------|
| Regularization parameter | t (iterations) | λ (penalty) |
| Equivalent λ | ~1/(ηt) | Explicit λ |
| Computational cost | Saves time | Full training |
| Validation set | Required | Optional (can use CV) |
| Hyperparameter tuning | Automatic | Need to tune λ |
| Theoretical guarantee | Yes (implicit) | Yes (explicit) |
| Practical performance | Often similar | Often similar |

Empirical finding (Caruana et al.):
Early stopping ≈ L2 regularization in final performance
But early stopping is:
- Easier to use (no λ to tune)
- Faster (stops before convergence)
- More principled (data-driven stopping)
```

### Properties of Early Stopping

**Advantages:**
- ✅ Simple and effective
- ✅ No hyperparameters to tune (besides patience)
- ✅ Saves training time
- ✅ Often as effective as other regularization
- ✅ Implicit regularization: λ(t) ~ 1/(ηt)
- ✅ Data-driven stopping criterion
- ✅ Equivalent to Ridge with adaptive λ

**Disadvantages:**
- ❌ Requires validation set (m = Ω(T·log(T)/ε²))
- ❌ Can stop too early with noisy validation loss
- ❌ Patience parameter p needs tuning
- ❌ Validation noise affects stopping quality

**When to use:**
- Always! (for iterative training algorithms)
- Neural networks
- Gradient boosting
- Any model trained iteratively
- When you want automatic regularization strength
- Especially effective when combined with explicit regularization

---

## Data Augmentation

**Create additional training samples by applying transformations**

### Image Data Augmentation

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Example augmentation pipeline
datagen = ImageDataGenerator(
    rotation_range=20,          # Rotate ±20 degrees
    width_shift_range=0.2,      # Shift horizontally ±20%
    height_shift_range=0.2,     # Shift vertically ±20%
    horizontal_flip=True,       # Random horizontal flip
    zoom_range=0.2,             # Random zoom ±20%
    fill_mode='nearest'         # Fill pixels with nearest value
)

print("Data augmentation techniques:")
print("- Rotation, flips, shifts, zooms")
print("- Increases effective training set size")
print("- Improves model robustness")
```

### Tabular Data Augmentation

```python
def augment_tabular_data(X, y, noise_level=0.05, n_augmentations=3):
    """
    Augment tabular data by adding noise.
    """
    X_aug = [X]
    y_aug = [y]

    for _ in range(n_augmentations):
        # Add Gaussian noise
        X_noisy = X + np.random.randn(*X.shape) * noise_level * X.std(axis=0)
        X_aug.append(X_noisy)
        y_aug.append(y)

    return np.vstack(X_aug), np.concatenate(y_aug)

# Example usage
X_orig = X_train[:100]
y_orig = y_train[:100]

X_aug, y_aug = augment_tabular_data(X_orig, y_orig, noise_level=0.1, n_augmentations=5)

print(f"Original size: {len(X_orig)}")
print(f"Augmented size: {len(X_aug)}")

# Train on augmented data
model_aug = Ridge(alpha=1.0)
model_aug.fit(X_aug, y_aug)

model_orig = Ridge(alpha=1.0)
model_orig.fit(X_orig, y_orig)

print(f"\nTest score (original): {model_orig.score(X_test, y_test):.4f}")
print(f"Test score (augmented): {model_aug.score(X_test, y_test):.4f}")
```

### Properties of Data Augmentation

**Advantages:**
- ✅ Increases effective training set size
- ✅ Improves robustness
- ✅ Domain-specific (can encode prior knowledge)

**Disadvantages:**
- ❌ Requires domain knowledge
- ❌ Can introduce unrealistic samples
- ❌ Increases training time

**When to use:**
- Small training sets
- Image/audio data (very effective)
- When transformations preserve label

---

## Batch Normalization

**Normalizes layer inputs during training**

### Implementation

```python
# Without Batch Normalization
model_no_bn = keras.Sequential([
    keras.layers.Dense(128, input_shape=(20,)),
    keras.layers.Activation('relu'),
    keras.layers.Dense(128),
    keras.layers.Activation('relu'),
    keras.layers.Dense(1)
])

# With Batch Normalization
model_bn = keras.Sequential([
    keras.layers.Dense(128, input_shape=(20,)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Dense(128),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Dense(1)
])

# Train both models
for name, model in [('Without BN', model_no_bn), ('With BN', model_bn)]:
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(
        X_train_nn, y_train_nn,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        verbose=0
    )

    print(f"{name}:")
    print(f"  Final train loss: {history.history['loss'][-1]:.4f}")
    print(f"  Final val loss: {history.history['val_loss'][-1]:.4f}")
```

### Rigorous Theory of Batch Normalization

**Theorem 18 (Batch Normalization Transform - Ioffe & Szegedy, 2015):**

```
For mini-batch B = {x₁, ..., x_m}:

BN(x) = γ·(x - μ_B)/√(σ_B² + ε) + β

where:
- μ_B = (1/m)Σᵢxᵢ  (batch mean)
- σ_B² = (1/m)Σᵢ(xᵢ - μ_B)²  (batch variance)
- γ, β = learnable scale and shift parameters
- ε = small constant for numerical stability (typically 10⁻⁵)

Key properties:
1. Zero mean, unit variance: E[BN(x)] = β, Var[BN(x)] = γ²
2. Differentiable w.r.t. all inputs and parameters
3. Preserves representational power (γ and β restore any distribution)

Gradients (for backpropagation):
∂L/∂γ = Σᵢ (∂L/∂yᵢ)·x̂ᵢ
∂L/∂β = Σᵢ (∂L/∂yᵢ)
∂L/∂xᵢ = (γ/√(σ_B² + ε))·[∂L/∂yᵢ - (1/m)Σⱼ∂L/∂yⱼ - x̂ᵢ·(1/m)Σⱼ(∂L/∂yⱼ)·x̂ⱼ]
```

**Theorem 19 (BN Reduces Internal Covariate Shift - Santurkar et al., 2018):**

```
Original hypothesis (Ioffe & Szegedy):
BN reduces "internal covariate shift" (changing distributions in layers)

Modern understanding (Santurkar et al.):
BN primarily smooths the loss landscape!

Loss landscape smoothness:
Without BN: ||∇²L|| ~ O(L²)  (Lipschitz constant of gradient)
With BN: ||∇²L|| ~ O(1)  (much smoother)

Theorem: For network with BN layers,

||∇L(w + Δw) - ∇L(w)|| ≤ β·||Δw||

where β = O(1) independent of depth L

Without BN: β = O(L²) (explodes with depth!)

Consequence:
- Gradients are more predictive
- Can use larger learning rates: η ~ O(1) instead of η ~ O(1/L²)
- Faster convergence
- More stable training
```

**Theorem 20 (BN as Regularization - Luo et al., 2019):**

```
Batch Normalization provides implicit regularization:

1. Noise injection effect:
   μ_B, σ_B² are random (depend on mini-batch B)
   ⇒ Adds noise to activations during training
   ⇒ Similar to dropout (but different mechanism)

2. Regularization strength:
   Var[BN(x)] ∝ 1/m  (decreases with batch size)

   Larger batch → less noise → less regularization
   Smaller batch → more noise → more regularization

3. Effective noise variance:
   σ_noise² ≈ σ²/(m-1)  (sampling variance)

4. Generalization bound:
   With probability ≥ 1-δ:

   L(ŵ) - L̂(ŵ) ≤ O(√(W/(n·m)) + √(log(1/δ)/n))

   where m = batch size, W = # weights

   Key: Generalization improves with larger m (less noise)
   But convergence speed decreases with larger m (less stochasticity)
```

**Theorem 21 (BN and Learning Rate - Bjorck et al., 2018):**

```
Batch Normalization enables stable training with large learning rates:

Effective learning rate on normalized layer:
η_eff = η·(γ/√(σ²))

Scale invariance property:
For any c > 0, scaling weights w → cw doesn't change BN output:

BN(cw·x) = γ·((cw·x - μ)/(c·σ)) + β = BN(w·x)

⇒ Gradients automatically rescaled!

Weight norm growth:
||w||² grows with iterations, but doesn't affect activations
BN normalizes away the scale

Consequence:
Can use η = O(1) instead of carefully tuned small values
Typical: η ∈ [0.01, 0.1] with BN vs η ∈ [0.001, 0.01] without BN
```

**Theorem 22 (BN Convergence Analysis - Cai et al., 2019):**

```
For neural network trained with BN and SGD:

Convergence rate (with appropriate η and initialization):
E[||∇L(w_t)||²] ≤ O(1/√t)

Without BN (standard analysis):
E[||∇L(w_t)||²] ≤ O(1/√t) but with worse constants

Key improvements with BN:
1. Constants independent of depth L
2. Less sensitive to initialization
3. Larger effective learning rate possible

Sample complexity:
n = O(d·L²/ε²)  without BN
n = O(d·log(L)/ε²)  with BN

⇒ Exponential improvement in depth dependence!
```

**BN Training vs Inference Dichotomy:**

```
Training mode:
- Use batch statistics: μ_B, σ_B²
- Adds noise (regularization)
- Different output for same input in different batches

Inference mode:
- Use population statistics: μ_pop, σ_pop²
- Estimated via exponential moving average during training:

  μ_pop ← α·μ_pop + (1-α)·μ_B  (typically α = 0.9 or 0.99)
  σ_pop² ← α·σ_pop² + (1-α)·σ_B²

- Deterministic output
- No noise

This dichotomy can cause issues:
- Training-inference discrepancy if batch size varies
- Batch size affects regularization strength
- Need sufficient training batches to estimate μ_pop, σ_pop² accurately

Solutions:
- Use consistent batch sizes
- Use Group Normalization or Layer Normalization (batch-independent)
- Calibrate population statistics on validation set
```

**Comparison: Different Normalization Techniques:**

```
| Method | Normalization | Batch-dependent | Best for |
|--------|---------------|-----------------|----------|
| Batch Norm | Across batch | Yes | Large batches, CNNs |
| Layer Norm | Across features | No | RNNs, Transformers |
| Instance Norm | Per sample | No | Style transfer |
| Group Norm | Feature groups | No | Small batches |

Computational complexity (all O(d) per sample):
- Batch Norm: Need m ≥ 2 for stable variance estimate
- Layer Norm: Works with m = 1 (no batch dependence)
- Group Norm: Intermediate (groups of features)
```

### Properties of Batch Normalization

**Advantages:**
- ✅ Accelerates training (can use larger learning rates)
- ✅ Reduces sensitivity to initialization
- ✅ Acts as regularization (noise injection)
- ✅ Allows higher learning rates (η ~ O(1) instead of O(1/L²))
- ✅ Smooths loss landscape: ||∇²L|| = O(1) instead of O(L²)
- ✅ Scale invariance: BN(cw·x) = BN(w·x)
- ✅ Depth-independent convergence guarantees

**Disadvantages:**
- ❌ Adds complexity (extra parameters γ, β)
- ❌ Small batch sizes problematic (m < 2 unstable)
- ❌ Behavior differs between training and inference
- ❌ Training-inference discrepancy if batch sizes differ
- ❌ Need sufficient batches to estimate population statistics
- ❌ Regularization strength varies with batch size (∝ 1/m)

**When to use:**
- Deep neural networks (L > 3 layers)
- Struggling with convergence
- Want faster training
- Large batch sizes available (m ≥ 16)
- CNNs and feedforward networks
- Avoid for: small batches (use Group/Layer Norm instead), online learning (m=1)

---

## Weight Decay

**Directly shrinks weights during optimization**

### Distinction from L2 Regularization

While often confused, weight decay and L2 regularization differ:

**L2 Regularization (in loss function):**
```
Loss = MSE + (λ/2) × ||w||₂²
Gradient: ∇w = ∇MSE + λw
Update: w ← w - η(∇MSE + λw)
```

**Weight Decay (in optimizer):**
```
Update: w ← w - η∇MSE - ηλw
      = (1 - ηλ)w - η∇MSE
```

For standard SGD, these are equivalent. However, for adaptive optimizers (Adam, RMSprop):
- L2 regularization: Penalty added before adaptive scaling
- Weight decay: Applied after adaptive learning rate computation

**Result:** Weight decay provides more effective regularization for Adam/AdamW.

### Implementation

```python
# Weight decay in optimizer (recommended approach for Adam)
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(20,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1)
])

# Adam with weight decay (also called AdamW)
optimizer = keras.optimizers.Adam(learning_rate=0.001, weight_decay=0.01)

model.compile(optimizer=optimizer, loss='mse')
```

**Typical weight decay values:** 0.0001 to 0.01 (commonly 0.0001 for transformers, 0.01 for CNNs)

---

## When to Use Which Technique

### Decision Tree

```
Start here
│
├─ Linear Model?
│  ├─ YES → Want feature selection?
│  │  ├─ YES → Use Lasso (L1)
│  │  └─ NO → Use Ridge (L2)
│  │
│  └─ NO → Neural Network?
│     ├─ YES → Deep?
│     │  ├─ YES → Use Dropout + Batch Norm + Early Stopping
│     │  └─ NO → Use L2 + Early Stopping
│     │
│     └─ NO → Tree-based?
│        ├─ YES → Limit depth, min_samples_split
│        └─ NO → Use appropriate regularization for algorithm
│
└─ Still overfitting?
   ├─ Get more data
   ├─ Use data augmentation
   └─ Use ensemble methods
```

### Quick Reference Table

| Model Type | Primary Regularization | Secondary | Hyperparameter |
|-----------|----------------------|-----------|----------------|
| Linear Regression | Ridge (L2) | Lasso (L1), Elastic Net | α (alpha) |
| Logistic Regression | L2 | L1 | C (inverse of α) |
| Neural Networks | Dropout | Early Stopping, L2 | dropout_rate, α |
| Decision Trees | max_depth | min_samples_split | max_depth |
| Random Forest | max_depth | max_features | max_depth |
| SVM | C parameter | Kernel choice | C |
| Gradient Boosting | learning_rate | max_depth, subsample | learning_rate |

### Choosing Between L1 and L2

**Use Ridge (L2) when:**
- All features potentially relevant
- Features are correlated
- Want smooth coefficient shrinkage

**Use Lasso (L1) when:**
- Many irrelevant features
- Want automatic feature selection
- Need interpretable model (few features)

**Use Elastic Net when:**
- Unsure about L1 vs L2
- Correlated features + want selection
- High-dimensional data

### Combining Multiple Techniques

```python
# Example: Combining multiple regularization techniques

# 1. L2 regularization through optimizer
optimizer = keras.optimizers.Adam(learning_rate=0.001, weight_decay=0.01)

# 2. Dropout
# 3. Batch Normalization
# 4. Early Stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

model = keras.Sequential([
    # Input layer
    keras.layers.Dense(256, input_shape=(20,)),
    keras.layers.BatchNormalization(),  # Batch norm
    keras.layers.Activation('relu'),
    keras.layers.Dropout(0.3),  # Dropout

    # Hidden layer
    keras.layers.Dense(128),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Dropout(0.3),

    # Output layer
    keras.layers.Dense(1)
])

model.compile(optimizer=optimizer, loss='mse')

history = model.fit(
    X_train_nn, y_train_nn,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=[early_stopping],  # Early stopping
    verbose=0
)

print("Model uses:")
print("- L2 regularization (weight decay)")
print("- Dropout (30%)")
print("- Batch Normalization")
print("- Early Stopping")
```

---

## Practical Examples

### Example 1: Finding Optimal Regularization Strength

```python
from sklearn.model_selection import cross_val_score

# Test range of alpha values
alphas = np.logspace(-4, 2, 50)
train_scores = []
cv_scores = []

for alpha in alphas:
    model = Ridge(alpha=alpha)

    # Training score
    model.fit(X_train_poly, y_train)
    train_scores.append(model.score(X_train_poly, y_train))

    # Cross-validation score
    cv_score = cross_val_score(model, X_train_poly, y_train, cv=5, scoring='r2')
    cv_scores.append(cv_score.mean())

# Find optimal alpha
optimal_idx = np.argmax(cv_scores)
optimal_alpha = alphas[optimal_idx]

# Plot
plt.figure(figsize=(12, 6))
plt.plot(alphas, train_scores, 'o-', label='Train Score', linewidth=2)
plt.plot(alphas, cv_scores, 's-', label='CV Score', linewidth=2)
plt.axvline(x=optimal_alpha, color='r', linestyle='--',
           label=f'Optimal α = {optimal_alpha:.4f}')
plt.xscale('log')
plt.xlabel('Alpha (λ)')
plt.ylabel('R² Score')
plt.title('Finding Optimal Regularization Strength')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Optimal alpha: {optimal_alpha:.4f}")
print(f"CV Score: {cv_scores[optimal_idx]:.4f}")
```

### Example 2: Comparing All Regularization Methods

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor

# Define models
models = {
    'No Regularization': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1, max_iter=10000),
    'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000),
}

results_comparison = []

for name, model in models.items():
    model.fit(X_train_poly, y_train)

    train_score = model.score(X_train_poly, y_train)
    test_score = model.score(X_test_poly, y_test)

    if hasattr(model, 'coef_'):
        n_nonzero = np.sum(np.abs(model.coef_) > 1e-10)
        max_coef = np.abs(model.coef_).max()
    else:
        n_nonzero = n_features
        max_coef = np.nan

    results_comparison.append({
        'Model': name,
        'Train R²': train_score,
        'Test R²': test_score,
        'Gap': train_score - test_score,
        'Non-zero Coefs': n_nonzero,
        'Max |Coef|': max_coef
    })

df_comparison = pd.DataFrame(results_comparison)
print(df_comparison.to_string(index=False))
```

---

## Best Practices

### 1. Always Start with Validation Curve

```python
from sklearn.model_selection import validation_curve

# Find good range for alpha
param_range = np.logspace(-4, 4, 20)

train_scores, val_scores = validation_curve(
    Ridge(), X_train_poly, y_train,
    param_name='alpha',
    param_range=param_range,
    cv=5,
    scoring='r2'
)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(param_range, train_scores.mean(axis=1), 'o-', label='Train')
plt.plot(param_range, val_scores.mean(axis=1), 's-', label='Validation')
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('R² Score')
plt.title('Validation Curve for Ridge')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 2. Use Cross-Validation for Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'alpha': np.logspace(-4, 2, 20)
}

grid_search = GridSearchCV(
    Ridge(), param_grid, cv=5, scoring='r2', n_jobs=-1
)

grid_search.fit(X_train_poly, y_train)

print(f"Best alpha: {grid_search.best_params_['alpha']:.4f}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
print(f"Test score: {grid_search.score(X_test_poly, y_test):.4f}")
```

### 3. Monitor Training vs Validation Loss

```python
# Always plot train vs validation curves
def plot_train_val_curves(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    final_gap = history.history['loss'][-1] - history.history['val_loss'][-1]
    if final_gap > 0.1 * history.history['val_loss'][-1]:
        print("⚠️ Possible overfitting detected!")
    else:
        print("✓ Good generalization")
```

### 4. Combine Multiple Techniques

```python
# Best practice: Use multiple regularization techniques together
# Example for neural networks:
# 1. L2 weight decay in optimizer
# 2. Dropout
# 3. Batch normalization
# 4. Early stopping
# 5. Data augmentation (if applicable)
```

### 5. Check Coefficient Magnitudes

```python
def check_coefficient_health(model):
    """
    Check if coefficients are too large (potential overfitting).
    """
    if hasattr(model, 'coef_'):
        max_coef = np.abs(model.coef_).max()
        mean_coef = np.abs(model.coef_).mean()

        print(f"Max |coefficient|: {max_coef:.2e}")
        print(f"Mean |coefficient|: {mean_coef:.2e}")

        if max_coef > 1000:
            print("⚠️ Very large coefficients - consider increasing regularization")
        else:
            print("✓ Coefficient magnitudes are reasonable")
```

---

## Summary

### Key Takeaways

1. **Regularization prevents overfitting** by constraining model complexity
2. **Ridge (L2)** shrinks all coefficients smoothly
3. **Lasso (L1)** performs feature selection by setting coefficients to zero
4. **Elastic Net** combines benefits of L1 and L2
5. **Dropout** is essential for deep neural networks
6. **Early stopping** is simple but very effective
7. **Always use validation curves** to find optimal regularization strength
8. **Combine multiple techniques** for best results

### Quick Reference

| Problem | Solution |
|---------|----------|
| Overfitting in linear models | Ridge, Lasso, or Elastic Net |
| Too many features | Lasso or Elastic Net |
| Correlated features | Ridge or Elastic Net |
| Neural network overfitting | Dropout + Early Stopping + L2 |
| Deep network training | Batch Normalization |
| Small dataset | Data Augmentation |

### Checklist

- [ ] Split data into train/validation/test
- [ ] Try model without regularization (baseline)
- [ ] Plot validation curves for different regularization strengths
- [ ] Use cross-validation to select hyperparameters
- [ ] Check coefficient magnitudes
- [ ] Monitor train vs validation loss
- [ ] Use early stopping for iterative algorithms
- [ ] Combine multiple regularization techniques
- [ ] Evaluate on held-out test set

Proper regularization is essential for building models that generalize well!
