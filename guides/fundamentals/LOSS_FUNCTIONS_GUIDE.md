# Loss Functions: Complete Mathematical Guide

## Table of Contents
1. [Introduction to Loss Functions](#introduction)
2. [Regression Loss Functions](#regression)
3. [Classification Loss Functions](#classification)
4. [Probabilistic Loss Functions](#probabilistic)
5. [Ranking and Metric Learning](#ranking)
6. [Sequence-to-Sequence Losses](#sequence)
7. [Generative Model Losses](#generative)
8. [Convergence Properties](#convergence)
9. [Loss Function Selection Guide](#selection)
10. [Custom Loss Functions](#custom)

---

## Introduction to Loss Functions

### Mathematical Framework

**Loss Function Definition:**
```
Loss function L: Y × Y → ℝ⁺

L(y_true, y_pred): Measures discrepancy between prediction and truth

Properties:
1. Non-negativity: L(y, ŷ) ≥ 0
2. Identity: L(y, y) = 0
3. Often (not always) symmetric or convex
```

**Risk Minimization:**
```
Empirical Risk: R̂(f) = (1/n) Σᵢ L(yᵢ, f(xᵢ))
True Risk: R(f) = E_{(x,y)~P}[L(y, f(x))]

Goal: min_{f∈F} R(f)
Practice: min_{f∈F} R̂(f) + λ·Ω(f)

where Ω(f) is regularization
```

**Gradient-Based Optimization:**
```
For parameter θ:

∂L/∂θ: Gradient of loss w.r.t. parameters

Update: θ ← θ - α·∂L/∂θ

Convergence depends on:
- Convexity of L
- Lipschitz properties
- Smoothness
```

### Statistical Decision Theory

**Bayes Risk and Optimal Decision Rules**

The foundation of loss functions lies in statistical decision theory.

**Definition 1 (Decision Function):**
A decision function δ: X → A maps inputs to actions.
- For regression: A = ℝ (predict real values)
- For binary classification: A = {0, 1} or A = [0, 1] (probabilities)
- For multi-class: A = Δ_K (probability simplex)

**Definition 2 (Risk):**
The risk of decision function δ under loss L:

R(δ) = E_{(X,Y)~P}[L(Y, δ(X))]

**Theorem 1 (Bayes Optimal Decision Rule):**
The Bayes optimal decision rule minimizes expected loss:

δ*(x) = argmin_{a∈A} E_{Y|X=x}[L(Y, a)]

This is the best possible decision rule given the true distribution P(Y|X).

**Proof:**
By iterated expectation:

R(δ) = E_X[E_{Y|X}[L(Y, δ(X))]]

For each x, E_{Y|X=x}[L(Y, δ(x))] depends only on δ(x).
Minimizing R(δ) requires minimizing the inner expectation for each x:

δ*(x) = argmin_{a∈A} E_{Y|X=x}[L(Y, a)]

Any other choice δ(x) ≠ δ*(x) increases R(δ). ∎

**Corollary 1.1 (Bayes Risk):**
The Bayes risk R* is the minimum achievable risk:

R* = E_X[min_{a∈A} E_{Y|X}[L(Y, a)]]

No decision rule can achieve R(δ) < R*.

**Excess Risk Decomposition:**
For any decision rule δ:

R(δ) - R* = E_X[E_{Y|X}[L(Y, δ(X))] - min_a E_{Y|X}[L(Y, a)]]
          ≥ 0

The excess risk measures sub-optimality of δ.

**Example 1 (Squared Loss):**
For L(y, ŷ) = (y - ŷ)²:

δ*(x) = argmin_a E_{Y|X=x}[(Y - a)²]
      = argmin_a E[(Y - E[Y|X=x] + E[Y|X=x] - a)²]
      = argmin_a {Var[Y|X=x] + (E[Y|X=x] - a)²}
      = E[Y|X=x]

Therefore: **MSE loss leads to predicting the conditional mean**.

**Example 2 (Absolute Loss):**
For L(y, ŷ) = |y - ŷ|:

δ*(x) = argmin_a E_{Y|X=x}[|Y - a|]
      = median(Y|X=x)

Therefore: **MAE loss leads to predicting the conditional median**.

**Example 3 (0-1 Loss for Classification):**
For L(y, ŷ) = 𝟙[y ≠ ŷ]:

δ*(x) = argmin_{c∈{1,...,K}} P(Y ≠ c | X=x)
      = argmin_c (1 - P(Y = c | X=x))
      = argmax_c P(Y = c | X=x)

Therefore: **0-1 loss leads to predicting the most probable class**.

### Convexity Theory for Loss Functions

**Definition 3 (Convex Function):**
A function f: ℝⁿ → ℝ is convex if for all x, y ∈ ℝⁿ and λ ∈ [0, 1]:

f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)

f is strictly convex if inequality is strict for λ ∈ (0, 1) and x ≠ y.

**Definition 4 (Strong Convexity):**
f is μ-strongly convex if for all x, y:

f(y) ≥ f(x) + ∇f(x)ᵀ(y - x) + (μ/2)||y - x||²

Strong convexity implies unique global minimum.

**Theorem 2 (Convexity of Common Losses):**

(a) **MSE is convex:**
For L(w) = (1/n)Σᵢ(wᵀxᵢ - yᵢ)², L is convex in w.

Proof: The Hessian is:
∇²L(w) = (2/n)XᵀX

This is positive semidefinite since for any v:
vᵀ∇²L(w)v = (2/n)vᵀXᵀXv = (2/n)||Xv||² ≥ 0

If X has full column rank: ∇²L = (2/n)XᵀX ≻ 0, so L is μ-strongly convex
with μ = (2/n)λ_min(XᵀX). ∎

(b) **Cross-entropy is convex:**
For logistic regression with L(w) = -Σᵢ[yᵢlog σ(wᵀxᵢ) + (1-yᵢ)log(1-σ(wᵀxᵢ))],
where σ(z) = 1/(1+e^(-z)):

Proof: The Hessian is:
∇²L(w) = Σᵢ σ(wᵀxᵢ)(1-σ(wᵀxᵢ)) xᵢxᵢᵀ

Since σ(z)(1-σ(z)) > 0 for all z, this is a sum of positive semidefinite matrices,
hence positive semidefinite. Therefore L is convex. ∎

(c) **MAE is convex:**
For L(w) = Σᵢ|wᵀxᵢ - yᵢ|, L is convex (as sum of convex functions |·|).

(d) **Hinge loss is convex:**
For L(w) = Σᵢmax(0, 1 - yᵢwᵀxᵢ), L is convex (max of affine functions).

**Theorem 3 (Smoothness and Lipschitz Continuity):**

(a) **L-smooth:** A differentiable function f is L-smooth if:
||∇f(x) - ∇f(y)|| ≤ L||x - y|| for all x, y

Equivalently: ∇²f(x) ⪯ LI (all eigenvalues ≤ L)

(b) **For MSE:** If L(w) = (1/n)||Xw - y||², then:
∇²L = (2/n)XᵀX

L-smooth with L = (2/n)λ_max(XᵀX).

(c) **For logistic regression:** L-smooth with L = (1/4n)||X||²_F
because max_z σ(z)(1-σ(z)) = 1/4.

**Theorem 4 (Convergence Rates):**

For μ-strongly convex and L-smooth function, gradient descent with step size α = 1/L:

||w_k - w*||² ≤ (1 - μ/L)^k ||w_0 - w*||²

where κ = L/μ is the condition number.

Proof: By L-smoothness and μ-strong convexity:

f(w_k) - f(w*) ≤ (1 - 1/κ)(f(w_{k-1}) - f(w*))

This is linear (exponential) convergence with rate depending on κ. ∎

**Practical Implications:**
- Well-conditioned problems (κ ≈ 1): fast convergence
- Ill-conditioned problems (κ >> 1): slow convergence
- For linear regression: κ = λ_max(XᵀX)/λ_min(XᵀX)
- Regularization improves conditioning: κ decreases with λ in L + λ||w||²

### Bregman Divergences

Many loss functions are Bregman divergences, providing a unified framework.

**Definition 5 (Bregman Divergence):**
For strictly convex, differentiable function φ:

D_φ(y || ŷ) = φ(y) - φ(ŷ) - ⟨∇φ(ŷ), y - ŷ⟩

This measures the "error" of approximating φ(y) by its first-order Taylor expansion at ŷ.

**Properties:**
1. D_φ(y || ŷ) ≥ 0 with equality iff y = ŷ
2. Generally not symmetric: D_φ(y || ŷ) ≠ D_φ(ŷ || y)
3. Convex in second argument

**Theorem 5 (Common Losses as Bregman Divergences):**

(a) **Squared loss:** φ(y) = (1/2)y²
D_φ(y || ŷ) = (1/2)y² - (1/2)ŷ² - ŷ(y - ŷ)
            = (1/2)(y - ŷ)²

(b) **Generalized KL divergence:** φ(y) = y log y - y
D_φ(y || ŷ) = y log(y/ŷ) - (y - ŷ)  (for y, ŷ > 0)

(c) **Itakura-Saito divergence:** φ(y) = -log y
D_φ(y || ŷ) = y/ŷ - log(y/ŷ) - 1

**Theorem 6 (Bregman Projection):**
The Bregman projection onto convex set C:

P_C(y) = argmin_{ŷ∈C} D_φ(y || ŷ)

satisfies the generalized Pythagorean theorem:

D_φ(y || z) = D_φ(y || P_C(y)) + D_φ(P_C(y) || z)  for all z ∈ C

This generalizes ordinary Euclidean projection (when φ(y) = (1/2)||y||²).

---

## Regression Loss Functions

### Mean Squared Error (MSE) / L2 Loss

**Mathematical Form:**
```
L(y, ŷ) = (1/n) Σᵢ (yᵢ - ŷᵢ)²

Alternative forms:
- Sum of squared errors: Σᵢ (yᵢ - ŷᵢ)²
- Root MSE: √[(1/n) Σᵢ (yᵢ - ŷᵢ)²]
```

**Gradient:**
```
∂L/∂ŷᵢ = 2(ŷᵢ - yᵢ)

Properties:
- Linear gradient in error
- Differentiable everywhere
- Convex
```

**Statistical Interpretation:**
```
Maximum Likelihood Estimation under Gaussian noise:

Assume: y = f(x) + ε, where ε ~ N(0, σ²)

Likelihood: P(y|x) = (1/√(2πσ²)) exp(-(y - f(x))²/(2σ²))

Log-likelihood: log P(y|x) = -(y - f(x))²/(2σ²) + const

Maximizing log-likelihood ⟺ Minimizing MSE!
```

**Convergence Properties:**
```
For linear regression y = Xw:

Loss: L(w) = ||Xw - y||²

Properties:
- Strongly convex (for X with full column rank)
- Unique global minimum: w* = (X^T X)^(-1) X^T y
- Gradient Lipschitz with constant L = λ_max(X^T X)

Convergence rate (Gradient Descent):
- With step size α = 1/L:
  ||w_k - w*||² ≤ (1 - μ/L)^k ||w_0 - w*||²

  where μ = λ_min(X^T X) (strong convexity parameter)

- Condition number: κ = L/μ = λ_max/λ_min
- Linear convergence: O((1 - 1/κ)^k)
```

**Robustness:**
```
Sensitivity to outliers: HIGH

Example:
True values: [1, 2, 3, 4, 5]
Predictions: [1.1, 2.1, 3.1, 4.1, 5.1]
MSE = 0.01 (good)

With one outlier:
True values: [1, 2, 3, 4, 100]
Predictions: [1.1, 2.1, 3.1, 4.1, 5.1]
MSE = 1801.01 (terrible!)

Outlier dominates loss due to squaring
```

**Use Cases:**
```
✓ When: Errors are normally distributed
✓ When: Outliers are rare and should be heavily penalized
✓ When: Need smooth gradients
✗ When: Data has outliers (use Huber or MAE instead)
```

**MSE vs MAE: Decision guide**

**Choose MSE when:**
1. **Gaussian noise assumption holds:** Real-world data often has Gaussian errors (measurement noise, sensor noise)
2. **Large errors are qualitatively different:** Predicting $100K vs $50K for house price is worse than $51K vs $50K
3. **Want smooth optimization:** MSE gradients are continuous and smooth everywhere
4. **Training stability matters:** Constant-magnitude gradients of MAE can cause issues with learning rate

**Choose MAE when:**
1. **Outliers exist and are NOT errors:** Legitimate extreme values that shouldn't dominate loss
2. **Want median prediction:** MSE → mean, MAE → median (median more robust)
3. **All errors equally bad:** Missing by $10K or $100K both count as one failed prediction
4. **Interpretable loss:** MAE in same units as target (average error)

**Practical example (house prices):**
- Dataset: 100 houses, 99 priced $200K-$300K, 1 mansion at $5M
- Model predicting $250K for everything:
  - MSE: Dominated by $5M outlier, loss ≈ 22M², pushes model to overpredict
  - MAE: All errors weighted equally, loss ≈ $50K average, robust to outlier
- If mansion is legitimate (not error): Use MAE
- If mansion is data entry error: Use MSE or remove outlier

### Mean Absolute Error (MAE) / L1 Loss

**Mathematical Form:**
```
L(y, ŷ) = (1/n) Σᵢ |yᵢ - ŷᵢ|
```

**Gradient:**
```
∂L/∂ŷᵢ = sign(ŷᵢ - yᵢ)

Properties:
- Constant gradient magnitude
- Not differentiable at ŷᵢ = yᵢ (use subgradient)
- Convex
```

**Statistical Interpretation:**
```
MLE under Laplace distribution:

Assume: y = f(x) + ε, where ε ~ Laplace(0, b)

P(ε) = (1/2b) exp(-|ε|/b)

Log-likelihood: -|y - f(x)|/b + const

Maximizing ⟺ Minimizing MAE
```

**Convergence Properties:**
```
For linear model:

L(w) = Σᵢ |xᵢ^T w - yᵢ|

Properties:
- Convex but not differentiable at solution points
- Subgradient method required
- No closed-form solution (unlike MSE)

Convergence rate (Subgradient Descent):
- Best iterate after k steps:
  f(w_best) - f(w*) ≤ O(1/√k)

- Slower than smooth losses
- Step size: α_k = α_0/√k (diminishing)

Coordinate descent:
- Often more efficient for L1
- Soft-thresholding operator
```

**Robustness:**
```
Sensitivity to outliers: LOW

Example with outlier:
True values: [1, 2, 3, 4, 100]
Predictions: [1.1, 2.1, 3.1, 4.1, 5.1]

MAE = (0.1 + 0.1 + 0.1 + 0.1 + 94.9)/5 = 19.06

vs MSE = 1801.01

MAE less affected by outlier (no squaring)
```

**Use Cases:**
```
✓ When: Outliers present in data
✓ When: Median regression desired (MAE → median, MSE → mean)
✓ When: All errors equally important
✗ When: Need smooth gradients (use Huber instead)
✗ When: Want to heavily penalize large errors
```

### Huber Loss

**Mathematical Form:**
```
L_δ(y, ŷ) = {
  ½(y - ŷ)²               if |y - ŷ| ≤ δ
  δ|y - ŷ| - ½δ²          if |y - ŷ| > δ
}

Combines MSE (for small errors) and MAE (for large errors)
```

**Gradient:**
```
∂L_δ/∂ŷ = {
  (ŷ - y)           if |y - ŷ| ≤ δ
  δ·sign(ŷ - y)     if |y - ŷ| > δ
}

Properties:
- Differentiable everywhere (unlike MAE)
- Linear gradient for large errors (unlike MSE)
- Convex
- δ: transition point (hyperparameter)
```

**Convergence Properties:**
```
Properties:
- Strongly convex in neighborhood of minimum
- Lipschitz continuous gradient
- Smooth everywhere

Convergence rate:
- In convex region: Linear convergence (like MSE)
- Overall: Better than MAE, similar to MSE
- Less sensitive to δ choice than might be expected

Optimal δ:
- δ ≈ 1.345σ where σ is noise standard deviation
- Makes Huber ~95% as efficient as MSE under Gaussian noise
- Much more robust to outliers
```

**Robustness:**
```
Sensitivity: MEDIUM (between MSE and MAE)

δ = 1.0 example:
Small errors (|e| < 1): L = 0.5e² (smooth, like MSE)
Large errors (|e| > 1): L = |e| - 0.5 (robust, like MAE)

Best of both worlds:
- Smooth gradients near optimum
- Robust to outliers
```

**Use Cases:**
```
✓ When: Data has some outliers but want smooth training
✓ When: Need balance between MSE and MAE
✓ When: Regression with heteroscedastic noise
✓ Most versatile regression loss
```

### Quantile Loss / Pinball Loss

**Mathematical Form:**
```
L_τ(y, ŷ) = {
  τ(y - ŷ)        if y ≥ ŷ
  (1-τ)(ŷ - y)    if y < ŷ
}

where τ ∈ (0, 1) is the quantile

Special cases:
- τ = 0.5: MAE (median)
- τ = 0.9: 90th percentile
```

**Gradient:**
```
∂L_τ/∂ŷ = {
  -τ        if y ≥ ŷ
  (1-τ)     if y < ŷ
}

Asymmetric penalties:
- Underestimation (y > ŷ): weighted by τ
- Overestimation (y < ŷ): weighted by (1-τ)
```

**Statistical Interpretation:**
```
Minimizing L_τ(y, ŷ) gives τ-th quantile of P(y|x)

Example (τ = 0.9):
- Prediction is 90th percentile
- 90% of observed values below prediction
- Useful for risk assessment, inventory planning
```

#### Rigorous Quantile Regression Theory

**Theorem 15 (Check Function Characterization - Koenker & Bassett, 1978):**
The quantile loss is also called the **check function** ρ_τ:

ρ_τ(u) = u(τ - 𝟙[u < 0])
       = {  τ·u      if u ≥ 0
         { (τ-1)·u   if u < 0

The τ-th conditional quantile q_τ(x) minimizes expected check loss:

q_τ(x) = argmin_q E_{Y|X=x}[ρ_τ(Y - q)]

**Proof:**
Taking the derivative w.r.t. q:

∂/∂q E[ρ_τ(Y - q)] = E[∂ρ_τ(Y - q)/∂q]
                     = E[-τ·𝟙[Y ≥ q] + (1-τ)·𝟙[Y < q]]
                     = (1-τ)P(Y < q) - τ·P(Y ≥ q)
                     = P(Y < q) - τ

Setting to zero:
P(Y < q_τ) = τ

Therefore: q_τ = F_Y^(-1)(τ) (the τ-th quantile). ∎

**Example (τ = 0.5):**
Check function: ρ_{0.5}(u) = |u|/2 (proportional to MAE)
Optimal: q_{0.5} = median(Y|X)

**Theorem 16 (Equivariance Properties):**
Quantile regression has important equivariance properties:

(a) **Location equivariance:** If Q_τ(Y|X) = q_τ(X), then:
Q_τ(Y + c|X) = q_τ(X) + c

(b) **Scale equivariance:** If Q_τ(Y|X) = q_τ(X), then:
Q_τ(c·Y|X) = c·q_τ(X)  for c > 0

(c) **Monotone transformation:** For strictly monotone h:
Q_τ(h(Y)|X) = h(Q_τ(Y|X))

**Proof of (a):**
For u = Y - q:
E[ρ_τ((Y+c) - (q+c))] = E[ρ_τ(Y - q)]

Minimizer shifts by c. ∎

**Theorem 17 (Asymptotic Normality - Koenker & Bassett, 1978):**
For linear quantile regression with n observations:

√n(β̂_τ - β_τ) →_d N(0, Σ_τ)

where:
Σ_τ = τ(1-τ)(X^T X)^(-1) / f_ε(0)²

and f_ε(0) is the density of errors at 0.

**Interpretation:**
- Variance proportional to τ(1-τ): maximum at median (τ = 0.5)
- Depends on error density at 0: sparsity degrades precision
- Requires estimating f_ε(0) for inference (density estimation)

**Quantile regression vs OLS variance:**
For Gaussian errors with variance σ²:
- OLS variance: σ²(X^T X)^(-1)
- QR variance at median: (πσ²/2)(X^T X)^(-1)
- Efficiency ratio: 2/π ≈ 0.637

Median regression ~64% as efficient as mean regression for Gaussian data.

**Theorem 18 (Quantile Crossing Prevention):**
When estimating multiple quantiles τ₁ < τ₂ < ... < τ_k, monotonicity requires:

q_τ₁(x) ≤ q_τ₂(x) ≤ ... ≤ q_τ_k(x)  for all x

**Non-crossing constraint:** Can be enforced via:

(a) **Post-processing:** Isotonic regression on estimated quantiles
(b) **Joint estimation:** Add constraints:
    q_τᵢ(x) ≤ q_τⱼ(x) - ε  for i < j
(c) **Parametric form:** Use non-crossing basis (e.g., increasing neural network)

**Practical algorithm (Rearrangement - Chernozhukov et al., 2010):**
1. Estimate quantiles independently: q̂_τ₁, ..., q̂_τ_k
2. For each x, sort: q̃_τ₁(x) ≤ q̃_τ₂(x) ≤ ... ≤ q̃_τ_k(x)
3. Use sorted quantiles

Rearrangement preserves asymptotic properties while ensuring monotonicity.

**Theorem 19 (Sample Complexity for Quantile Estimation):**
To estimate τ-th quantile with error ε and confidence 1-δ:

n = Ω((1/(ε²·τ(1-τ)))·log(1/δ))

**Proof sketch:**
By Hoeffding's inequality for empirical quantile:
P(|F̂_n^(-1)(τ) - F^(-1)(τ)| > ε) ≤ 2exp(-2nτ(1-τ)ε²)

Setting RHS = δ and solving for n. ∎

**Sample complexity comparison:**
- Median (τ = 0.5): n = Ω(4log(1/δ)/ε²) - most efficient
- Extreme quantile (τ = 0.9): n = Ω(11.1·log(1/δ)/ε²) - less efficient
- Very extreme (τ = 0.99): n = Ω(101·log(1/δ)/ε²) - requires much more data

**Example:**
For ε = 0.01 accuracy with δ = 0.05 confidence:
- Median: n ≈ 1,200 samples
- 90th percentile: n ≈ 3,300 samples
- 99th percentile: n ≈ 30,300 samples

Estimating extreme quantiles requires significantly more data!

**Use Cases:**
```
✓ Quantile regression
✓ Confidence interval estimation
✓ Risk-sensitive prediction
✓ Asymmetric cost functions
```

### Log-Cosh Loss

**Mathematical Form:**
```
L(y, ŷ) = Σᵢ log(cosh(ŷᵢ - yᵢ))

where cosh(x) = (e^x + e^(-x))/2
```

**Gradient:**
```
∂L/∂ŷᵢ = tanh(ŷᵢ - yᵢ)

Properties:
- Smooth everywhere (unlike Huber)
- Approximately MSE for small errors
- Approximately MAE for large errors
- No hyperparameter (unlike Huber's δ)
```

**Approximations:**
```
For small |x|: log(cosh(x)) ≈ x²/2 (like MSE)
For large |x|: log(cosh(x)) ≈ |x| - log 2 (like MAE)

Automatic transition between quadratic and linear
```

**Use Cases:**
```
✓ When: Want Huber-like behavior without tuning δ
✓ When: Need smoothness everywhere
✓ When: Outlier robustness with simple implementation
```

---

## Classification Loss Functions

### Proper Scoring Rules and Calibration

Before discussing specific classification losses, we establish the theory of proper scoring rules.

**Definition 6 (Scoring Rule):**
A scoring rule S(P, y) measures the quality of probabilistic forecast P when outcome is y.
- P: predicted probability distribution over outcomes
- y: realized outcome
- Lower score = better prediction

**Definition 7 (Proper Scoring Rule):**
A scoring rule S is **proper** if the expected score is minimized by predicting the true distribution:

argmin_Q E_{Y~P}[S(Q, Y)] = P

It is **strictly proper** if P is the unique minimizer.

**Theorem 7 (Cross-Entropy is Proper):**
The logarithmic scoring rule S(P, y) = -log P(y) is strictly proper.

Proof:
Let P be the true distribution. For any other distribution Q:

E_{Y~P}[S(Q, Y)] = E_{Y~P}[-log Q(Y)]
                  = -Σ_y P(y) log Q(y)
                  = H(P, Q)  (cross-entropy)

We want to show: H(P, Q) ≥ H(P, P) with equality iff Q = P.

H(P, Q) - H(P, P) = -Σ_y P(y) log Q(y) + Σ_y P(y) log P(y)
                  = Σ_y P(y) log(P(y)/Q(y))
                  = KL(P || Q)
                  ≥ 0

with equality iff P = Q (by non-negativity of KL divergence). ∎

**Corollary 7.1:**
Minimizing cross-entropy loss encourages calibrated probability predictions.

**Theorem 8 (Brier Score is Proper):**
The Brier score S(p, y) = (p - y)² for binary outcomes y ∈ {0, 1} is strictly proper.

Proof:
For true probability P(Y=1) = π, and predicted probability p:

E_Y[S(p, Y)] = π(p - 1)² + (1-π)(p - 0)²
              = π(p² - 2p + 1) + (1-π)p²
              = p²(π + 1 - π) - 2πp + π
              = p² - 2πp + π

Taking derivative w.r.t. p:
∂/∂p E[S(p, Y)] = 2p - 2π

Setting to zero: p = π (the true probability).

Second derivative: ∂²/∂p² E[S(p, Y)] = 2 > 0, confirming minimum. ∎

**Theorem 9 (Characterization of Proper Scoring Rules):**
A scoring rule S(p, y) for binary outcomes is proper if and only if it can be written as:

S(p, 1) = -G(p)
S(p, 0) = -G(1-p) - p·G'(p)

where G is a strictly convex function with G' being its derivative.

This gives a general family of proper scoring rules based on choice of G.

**Examples:**
- Log score: G(p) = log p (strictly convex for p > 0)
- Brier score: G(p) = 2p - p² (strictly convex)
- Spherical score: G(p) = 1/√(p² + (1-p)²)

### Calibration Theory

**Definition 8 (Calibration):**
A classifier with predicted probabilities p is **calibrated** if:

P(Y = 1 | p(X) = q) = q  for all q ∈ [0, 1]

In words: among all predictions with confidence q, exactly fraction q should be correct.

**Example (Well-calibrated):**
- 100 predictions with p = 0.7
- Exactly 70 should have Y = 1
- If 70/100 are correct → well calibrated
- If 50/100 are correct → underconfident (miscalibrated)
- If 90/100 are correct → overconfident (miscalibrated)

**Theorem 10 (Expected Calibration Error):**
The Expected Calibration Error (ECE) measures calibration:

ECE = Σ_{m=1}^M (n_m/n)|acc(m) - conf(m)|

where:
- Predictions binned into M bins by confidence
- n_m: number of predictions in bin m
- acc(m): accuracy in bin m
- conf(m): average confidence in bin m

Perfect calibration: ECE = 0.

**Theorem 11 (Temperature Scaling for Calibration):**
Post-hoc calibration via temperature scaling:

p_calibrated = softmax(z/T)

where z are logits and T > 0 is temperature.

Finding optimal T:
T* = argmin_T NLL(softmax(z/T), y) on validation set

This preserves model accuracy while improving calibration.

**Proof of effectiveness:**
Temperature scaling is a monotonic transformation of probabilities:
- Preserves argmax → same predicted classes
- T > 1: reduces confidence (spreads probability mass)
- T < 1: increases confidence (concentrates probability mass)
- Optimizing T via NLL finds best calibration on validation data. ∎

### Surrogate Loss Bounds

The 0-1 loss L_{0-1}(y, f(x)) = 𝟙[y ≠ sign(f(x))] is non-convex and discontinuous.
Classification algorithms use **surrogate losses** that are convex and differentiable.

**Definition 9 (φ-surrogate Loss):**
A surrogate loss φ: ℝ → ℝ₊ replaces 0-1 loss:

L_φ(y, f(x)) = φ(y·f(x))

where y ∈ {-1, +1} and f(x) ∈ ℝ is the margin.

**Common Surrogates:**
- 0-1 loss: φ(z) = 𝟙[z ≤ 0]
- Hinge: φ(z) = max(0, 1 - z)
- Logistic: φ(z) = log(1 + e^(-z))
- Exponential: φ(z) = e^(-z)
- Squared: φ(z) = (1 - z)²

**Theorem 12 (Classification Calibration):**
A surrogate φ is **classification-calibrated** if minimizing φ-risk implies minimizing 0-1 risk:

R_φ(f_n) → inf_f R_φ(f)  ⟹  R_{0-1}(f_n) → inf_f R_{0-1}(f)

**Sufficient condition:** φ is differentiable, convex, and φ'(0) < 0.

All common surrogates (hinge, logistic, exponential) satisfy this.

**Theorem 13 (Bartlett-Jordan-McAuliffe Bound):**
For margin-based surrogates with φ convex, decreasing, and φ(0) = 1:

R_{0-1}(f) - R*_{0-1} ≤ ψ_φ^(-1)(R_φ(f) - R*_φ)

where ψ_φ is the calibration function measuring how well φ bounds 0-1 loss.

**Explicit bounds:**

(a) **Hinge loss:** φ(z) = max(0, 1-z)
R_{0-1}(f) - R* ≤ R_φ(f) - R*_φ

(b) **Logistic loss:** φ(z) = log(1 + e^(-z))
R_{0-1}(f) - R* ≤ √(2(R_φ(f) - R*_φ))

(c) **Exponential loss:** φ(z) = e^(-z)
R_{0-1}(f) - R* ≤ √(2(R_φ(f) - R*_φ))

**Interpretation:**
- Minimizing surrogate φ-risk yields low 0-1 risk
- Logistic and exponential: square root relationship (faster than linear)
- Hinge: linear relationship (tightest bound)

**Theorem 14 (H-Consistency Bounds):**
For hypothesis class H, a loss is **H-consistent** if minimizing empirical φ-risk over H
yields optimal 0-1 risk over H.

For finite VC dimension d and n samples:

R_{0-1}(f̂) ≤ R*_{0-1} + O(√(d/n)) + ψ_φ^(-1)(R̂_φ(f̂) - R_φ(f*) + O(√(d/n)))

This combines:
- Approximation error: R_φ(f*) - R*_φ
- Estimation error: R̂_φ(f̂) - R_φ(f̂)
- Calibration gap: ψ_φ^(-1)

### Cross-Entropy Loss (Log Loss)

**Binary Classification:**
```
L(y, ŷ) = -[y log(ŷ) + (1-y) log(1-ŷ)]

where:
- y ∈ {0, 1}: true label
- ŷ ∈ [0, 1]: predicted probability
```

**Multi-Class (Categorical Cross-Entropy):**
```
L(y, ŷ) = -Σ_c y_c log(ŷ_c)

where:
- y: one-hot encoded true label [0,0,1,0,...,0]
- ŷ: predicted probabilities [0.1, 0.2, 0.6, 0.1, ...]
- Σ_c y_c = 1, Σ_c ŷ_c = 1
```

**Gradient (with Softmax):**
```
Combined softmax + cross-entropy:

Softmax: ŷ_i = exp(z_i) / Σ_j exp(z_j)

Gradient: ∂L/∂z_i = ŷ_i - y_i

Remarkably simple! This is why softmax + cross-entropy is standard.

Derivation:
∂L/∂z_i = Σ_c (∂L/∂ŷ_c)(∂ŷ_c/∂z_i)
        = -Σ_c (y_c/ŷ_c)(∂ŷ_c/∂z_i)

For softmax:
∂ŷ_j/∂z_i = ŷ_j(δ_ij - ŷ_i)

Combining:
∂L/∂z_i = -Σ_c (y_c/ŷ_c)·ŷ_c·(δ_ic - ŷ_i)
        = -y_i + ŷ_i·Σ_c y_c
        = ŷ_i - y_i  (since Σ_c y_c = 1)
```

**Why this gradient is elegant and perfect:**

1. **Intuitive form:** Gradient = prediction error
   - If ŷ_i = 0.9 and y_i = 1 (correct class): gradient = -0.1 (small correction)
   - If ŷ_i = 0.1 and y_i = 1: gradient = -0.9 (large correction)
   - Automatic adjustment based on confidence error

2. **No vanishing gradient problem:**
   - Gradient magnitude = |ŷ_i - y_i|
   - Even when completely wrong (ŷ_i → 0 for true class), gradient stays bounded
   - Compare to squared loss: ∂(ŷ - y)²/∂z = 2(ŷ - y)·ŷ(1-ŷ) → vanishes when ŷ → 0 or 1
   - Cross-entropy: Always provides learning signal when wrong

3. **Natural cancellation of complex terms:**
   - Softmax derivative involves all classes (Jacobian)
   - Cross-entropy derivative has division by prediction
   - Together they magically simplify to ŷ_i - y_i
   - No exponentials, no divisions in final gradient!

4. **Probabilistic interpretation:**
   - Gradient points in direction of steepest KL divergence reduction
   - Matches Fisher information matrix for efficient learning
   - Statistically optimal for categorical distributions

**Example showing non-saturation:**
```
True class: i=2, y = [0, 0, 1, 0]
Logits: z = [-5, -5, -5, 1] (very wrong! predicting class 3)

Softmax: ŷ ≈ [0.002, 0.002, 0.002, 0.994]

Gradient w.r.t. z₂ (true class): 0.002 - 1 = -0.998
Large negative gradient → increase z₂ strongly ✓

Compare if we used squared loss (ŷ - y)²:
∂/∂z₂ = 2(ŷ₂ - 1)·ŷ₂·(1-ŷ₂) ≈ 2(-0.998)·0.002·0.998 ≈ -0.004
Tiny gradient due to ŷ₂ ≈ 0 → training stalls! ✗
```

**Statistical Interpretation:**
```
Maximum Likelihood under Categorical distribution:

P(y = c | x) = ŷ_c

Likelihood: L = Π_i P(y_i | x_i)
Log-likelihood: Σ_i log P(y_i | x_i) = Σ_i log ŷ_{y_i}

Negative log-likelihood = Cross-entropy

Minimizing cross-entropy ⟺ Maximum likelihood estimation
```

**Convergence Properties:**
```
For logistic regression (binary):

z = w^T x + b
ŷ = σ(z) = 1/(1 + e^(-z))
L(w) = -Σ_i [y_i log σ(w^T x_i) + (1-y_i) log(1-σ(w^T x_i))]

Properties:
- Convex in w (assuming linearly separable or regularized)
- Gradient: ∇L(w) = Σ_i (ŷ_i - y_i) x_i
- Smooth (unlike hinge loss)

Convergence rate (GD):
- Convex case: O(1/k) for ε-accuracy
- Locally strongly convex: O(log(1/ε))

**Numerical stability:**
Critical issue: Direct computation causes overflow/underflow.

**Problem:**
```
Softmax: ŷ_i = exp(z_i) / Σ_j exp(z_j)

For z = [1000, 1001, 1002]:
exp(1000) ≈ 10^434 → overflow!
exp(-1000) ≈ 10^-434 → underflow!
```

**Solution - Log-Sum-Exp trick:**
```
log(Σ exp(z_i)) = c + log(Σ exp(z_i - c))

where c = max_i(z_i)

Implementation:
1. Compute c = max(z)
2. Compute z' = z - c  (now max(z') = 0)
3. Compute log(Σ exp(z'_i))
4. Result = c + log(Σ exp(z'_i))

For z = [1000, 1001, 1002]:
z' = [0, 1, 2]  ✓ (stable!)
exp(z') = [1, 2.718, 7.389]  ✓
```

**Softmax stability:**
```
Numerically stable implementation:
z' = z - max(z)
ŷ = exp(z') / sum(exp(z'))

This ensures:
- Largest logit becomes 0 → exp(0) = 1 (stable)
- All other logits are negative → exp(negative) ∈ (0,1) (stable)
- Denominator ≥ 1 (no division by tiny numbers)
```

**Cross-entropy stability:**
```
Avoid: L = -log(softmax(z)_y)

Use: L = -z_y + log_sum_exp(z)

This combines operations for better numerical precision.
```
```

**Behavior Analysis:**
```
For binary case:

When y = 1:
- ŷ → 1: L → 0 (correct, confident)
- ŷ → 0.5: L = -log(0.5) = 0.69 (uncertain)
- ŷ → 0: L → ∞ (wrong, confident - heavily penalized!)

Gradient magnitude:
|∂L/∂z| = |ŷ - y|

Properties:
- Large gradient when very wrong (fast correction)
- Small gradient when correct (stability)
- Never saturates when wrong (unlike squared error)
```

**Use Cases:**
```
✓ Multi-class classification (standard choice)
✓ Binary classification
✓ Probability estimation tasks
✓ When: Labels are mutually exclusive
✓ When: Need calibrated probabilities
✗ When: Labels are not exclusive (use BCE with logits)
```

### Hinge Loss (SVM Loss)

**Mathematical Form:**
```
Binary classification (y ∈ {-1, +1}):

L(y, z) = max(0, 1 - y·z)

where z = w^T x + b (decision function, not probability)

Multi-class (SVM):
L(y, z) = Σ_{j≠y} max(0, z_j - z_y + Δ)

where Δ is margin (typically Δ = 1)
```

**Gradient:**
```
∂L/∂z = {
  -y    if y·z < 1
  0     if y·z ≥ 1
}

Properties:
- Non-differentiable at y·z = 1 (use subgradient)
- Zero gradient in margin region
- Constant gradient outside margin
```

**Geometric Interpretation:**
```
Decision boundary: w^T x + b = 0

Margin region: |w^T x + b| ≤ 1

Hinge loss:
- Zero for points correctly classified with margin > 1
- Linear penalty for points in margin or misclassified

Goal: Maximize margin while allowing few violations
```

**Convergence Properties:**
```
SVM optimization:

min_{w,b} λ||w||² + (1/n)Σᵢ max(0, 1 - yᵢ(w^T xᵢ + b))

Properties:
- Convex
- Non-smooth (has corners)
- Regularization λ||w||² makes it strongly convex

**Convergence (Subgradient Descent):**
- Rate: O(1/√k) for non-smooth convex functions
- After k iterations: f(w̄_k) - f(w*) ≤ R·G/√k
  - R = ||w_0 - w*|| (initial distance)
  - G = bound on subgradient norm
  - w̄_k = (1/k)Σ w_i (average iterate)
- Slower than smooth losses (O(1/k) or O((1-1/κ)^k))
- Requires diminishing step size: α_k = α_0/√k

**Coordinate descent / SMO (Sequential Minimal Optimization):**
- More efficient for SVMs than subgradient methods
- Exploits structure: updates 2 variables at a time
- Convergence rate:
  - Worst-case: O(1/ε²) iterations
  - Typical practice: O(1/ε) or better
  - Each iteration: O(n) for linear kernel
- Advantages:
  - No learning rate tuning needed
  - Exact line search per coordinate
  - Handles constraints naturally

**Dual formulation:**
```
Primal: min_{w,b} ½||w||² + C·Σ_i max(0, 1 - y_i(w^T x_i + b))

Dual: max_α Σ_i α_i - ½Σ_{i,j} α_i α_j y_i y_j k(x_i, x_j)
      s.t. 0 ≤ α_i ≤ C, Σ_i α_i y_i = 0

- Kernel trick: k(x_i, x_j) = φ(x_i)^T φ(x_j)
- QP solvers: Polynomial time O(n²) to O(n³)
- Sparse solution: Most α_i = 0 (support vectors)
```
```

**Comparison with Cross-Entropy:**
```
Property          | Hinge Loss        | Cross-Entropy
------------------|-------------------|------------------
Output            | Margin z          | Probability ŷ
Range             | z ∈ ℝ             | ŷ ∈ [0,1]
Differentiable    | No (at margin)    | Yes
Far from boundary | 0 gradient        | Small gradient
At boundary       | Constant grad     | Depends on ŷ
Outlier effect    | Linear            | Logarithmic
Calibration       | No                | Yes (probabilities)

Hinge: Focuses on decision boundary
Cross-Entropy: Optimizes probability estimates
```

**Use Cases:**
```
✓ Support Vector Machines
✓ When: Only care about classification, not probabilities
✓ When: Want sparse solutions (many support vectors)
✗ When: Need probability estimates
✗ When: Using neural networks (use cross-entropy instead)
```

### Focal Loss

**Mathematical Form:**
```
FL(p_t) = -α_t(1 - p_t)^γ log(p_t)

where:
p_t = {
  p      if y = 1
  1-p    if y = 0
}

Parameters:
- γ ≥ 0: focusing parameter (typically γ = 2)
- α_t: class balancing weight

Standard cross-entropy: γ = 0, α_t = 1
```

**Motivation:**
```
Problem: Class imbalance in object detection
- Background: 99% of examples
- Objects: 1% of examples

Cross-entropy:
- Easy negatives dominate loss
- Model doesn't learn hard examples well

Focal loss solution:
- Down-weight easy examples: (1 - p_t)^γ
- Focus on hard examples
```

**Behavior:**
```
For γ = 2:

Example classified correctly with p = 0.9:
- CE: -log(0.9) = 0.105
- FL: -(1-0.9)² log(0.9) = -0.01·0.105 = 0.00105
- 100× reduction!

Example misclassified with p = 0.1 (y=1):
- CE: -log(0.1) = 2.303
- FL: -(1-0.1)² log(0.1) = -0.81·2.303 = 1.865
- Only 20% reduction

Gradient for hard examples >> easy examples
```

**Gradient:**
```
∂FL/∂z = α_t y(1-p_t)^γ[γp_t log(p_t) + p_t - 1]

Properties:
- Complicated but can be efficiently computed
- Automatic hard example mining
- No need for manual sampling
```

**Convergence Properties:**
```
Properties:
- Non-convex (due to modulating factor)
- Still works well in practice
- Similar convergence to cross-entropy
- May need more iterations

**Hyperparameter sensitivity:**
- γ: Controls focusing strength
  - Typical range: [0, 5]
  - Robust choice: γ = 2 (from RetinaNet paper)
  - Higher γ = more aggressive down-weighting
- α_t: Balances class importance
  - Formula for positive class: α = n_neg/(n_pos + n_neg)
  - Typical range: [0.25, 0.75]
  - Start with 0.25, tune on validation set
- Grid search recommended: γ ∈ {0.5, 1, 2, 3}, α ∈ {0.25, 0.5, 0.75}
- Less sensitive than expected: [1.5, 2.5] × [0.2, 0.3] usually works

**Computational overhead:**
- Extra operations: (1-p_t)^γ and γ·p_t·log(p_t) terms
- Overhead: ~15-25% vs standard cross-entropy
- Memory: Same as cross-entropy (no extra storage)
- Backward pass: More complex gradient but well-optimized in modern frameworks
```

**Hyperparameter selection guide:**

**γ (focusing parameter):**
- γ = 0: Standard cross-entropy (no focusing)
- γ = 1: Moderate down-weighting of easy examples
- γ = 2: Standard choice (aggressive down-weighting)
- γ = 5: Very aggressive (may ignore too many examples)

**Effect of γ on loss reduction:**
```
Well-classified example (p_t = 0.9):
γ=0: weight = 1.0     (CE baseline)
γ=1: weight = 0.1     (10× reduction)
γ=2: weight = 0.01    (100× reduction)
γ=5: weight = 0.00001 (100,000× reduction!)

Misclassified example (p_t = 0.1):
γ=0: weight = 1.0
γ=1: weight = 0.9     (10% reduction)
γ=2: weight = 0.81    (19% reduction)
γ=5: weight = 0.59    (41% reduction)
```

**α_t (class balancing):**
- Purpose: Balance positive/negative class importance
- Formula: α_t = n_neg / (n_pos + n_neg) for positive class
- Example with 1:99 imbalance: α_t = 0.99 for positive class
- Typical range: 0.25-0.75
- Interacts with γ: start with α_t = 0.25, adjust based on validation

**When focal loss helps vs doesn't:**

**Helps when:**
- Extreme imbalance (>100:1)
- Many easy negatives (background in object detection)
- Few hard positives that matter
- One-stage detectors (RetinaNet, FCOS)

**Doesn't help / may hurt when:**
- Balanced dataset (use standard CE)
- Easy positives and hard negatives (focal loss would focus on wrong examples)
- Two-stage methods already handling imbalance via sampling
- Very small datasets (<1K samples) - may overfit to hard examples

**Practical tip:**
Start with CE, switch to focal loss if:
1. Model predicts majority class with high confidence
2. Minority class recall is low despite high precision
3. Training loss plateaus early

#### Rigorous Focal Loss Theory

**Theorem 20 (Modulating Factor Analysis - Lin et al., 2017):**
The focal loss modulating factor (1 - p_t)^γ provides automatic hard example mining:

FL(p_t) = -(1 - p_t)^γ log(p_t)

**Properties:**
(a) **Gradient amplification:** For γ > 0, the gradient magnitude ratio is:

|∂FL/∂z| / |∂CE/∂z| = (1-p_t)^(γ-1) · |1 - p_t(1 + γlog(p_t))|

For well-classified examples (p_t → 1):
- CE gradient → 0 linearly: ∂CE/∂z ≈ 1 - p_t
- FL gradient → 0 faster: ∂FL/∂z ≈ (1 - p_t)^γ

(b) **Loss reduction factor:** Compared to cross-entropy:

FL(p_t) / CE(p_t) = (1 - p_t)^γ

For γ = 2 and p_t = 0.9: reduction = 0.01 (100× smaller)
For γ = 2 and p_t = 0.5: reduction = 0.25 (4× smaller)

**Proof of gradient:**
Let z be the logit with p_t = σ(z) for positive class.

FL(z) = -(1 - σ(z))^γ log σ(z)

∂FL/∂z = -(1 - σ(z))^γ · (-1/σ(z)) · σ'(z) + γ(1 - σ(z))^(γ-1) · σ'(z) · log σ(z)

Using σ'(z) = σ(z)(1 - σ(z)):

∂FL/∂z = (1 - σ(z))^γ · (1 - σ(z)) - γ(1 - σ(z))^γ · σ(z) log σ(z)
        = (1 - σ(z))^γ · [1 - σ(z) - γσ(z) log σ(z)]

For small 1 - σ(z), the (1 - σ(z))^γ term dominates. ∎

**Theorem 21 (Effective Number of Examples - Mukhoti et al., 2020):**
The focal loss effectively weights each example by (1 - p_t)^γ. The **effective number**
of training examples is:

n_eff = Σᵢ (1 - p_t,i)^γ

For γ = 2 and binary classification with 99% accuracy:
- Actual examples: n = 10,000
- Effective examples: n_eff ≈ 100 (only hard examples contribute!)

This explains why focal loss requires more iterations: fewer "effective" gradients per batch.

**Theorem 22 (Convergence with Focal Loss):**
For focal loss with γ > 0, the optimization landscape is **non-convex** even for
linear classifiers due to the modulating factor.

However, under standard assumptions:
(a) **Stationarity:** SGD finds ε-stationary point in:
O(1/(ε²·n_eff))  iterations

(b) **Sample complexity:** For (ε, δ)-PAC learning:
n = Ω((d/ε²) · log(1/δ) · E[(1 - p_t)^(-γ)])

The expectation E[(1 - p_t)^(-γ)] depends on data difficulty distribution.

**Interpretation:**
- Easy examples contribute negligibly: (1 - 0.99)^(-2) = 10,000×
- Hard examples dominate: (1 - 0.5)^(-2) = 4×
- Extremely hard: (1 - 0.1)^(-2) = 1.23×

Sample complexity is effectively determined by hard example distribution!

**Theorem 23 (Optimal γ Selection):**
For imbalanced dataset with positive class proportion π, the optimal focusing parameter
γ* approximately satisfies:

γ* ≈ log(1/π) / log(1/p̄_easy)

where p̄_easy is the average confidence on easy negatives.

**Derivation:**
Want (1 - p̄_easy)^γ ≈ π to balance contribution of easy negatives:

(1 - p̄_easy)^γ = π
γ log(1 - p̄_easy) = log π
γ = log π / log(1 - p̄_easy)

For p̄_easy → 1: log(1 - p̄_easy) ≈ -(1 - p̄_easy)

**Example:**
- Imbalance ratio: π = 0.01 (1% positives)
- Easy negative confidence: p̄_easy = 0.95
- Optimal γ: log(0.01)/log(0.05) ≈ -4.6/-3.0 ≈ 1.5

This is why γ = 2 works well for extreme imbalance (1:100 to 1:1000).

**Theorem 24 (Class Balancing Weight α):**
For optimal α_t, the total expected contribution from each class should be equal:

n_pos · E[(1 - p_t)^γ | y=1] · α_pos = n_neg · E[(1 - p_t)^γ | y=0] · α_neg

With α_neg = 1 - α_pos:

α_pos* = n_neg · E[(1-p_t)^γ | y=0] / (n_pos · E[(1-p_t)^γ | y=1] + n_neg · E[(1-p_t)^γ | y=0])

**Special case:** If both classes equally difficult:
α_pos* = n_neg / (n_pos + n_neg)  (inverse frequency weighting)

**Practical approximation:** Start with α = 0.25 for positive class (empirical finding
from RetinaNet), then:
- If positive recall too low: increase α → 0.5
- If positive precision too low: decrease α → 0.1

**Theorem 25 (Focal Loss Calibration Properties):**
Focal loss is **not** a proper scoring rule for γ > 0. The modulating factor
(1 - p_t)^γ breaks the proper scoring rule property.

**Proof:**
For proper scoring rule, optimal prediction is true probability:
E_Y[L(p, Y)] minimized at p = P(Y=1)

For focal loss:
E_Y[FL(p, Y)] = -P(Y=1)·(1-p)^γ log p - P(Y=0)·p^γ log(1-p)

Taking derivative and setting to zero does NOT yield p = P(Y=1) for γ ≠ 0.

**Consequence:** Models trained with focal loss may be **miscalibrated**:
- Overconfident on hard examples (by design!)
- Underconfident on easy examples (down-weighted)

**Solution:** Post-hoc calibration (temperature scaling) after training with focal loss.

**Empirical calibration analysis (RetinaNet on COCO):**
Before temperature scaling:
- Easy examples (IoU > 0.7): avg confidence 0.62 (underconfident)
- Hard examples (IoU 0.5-0.7): avg confidence 0.51 (appropriate)
- ECE = 0.15

After temperature scaling with T = 1.5:
- Easy examples: avg confidence 0.78
- Hard examples: avg confidence 0.58
- ECE = 0.05 (much better calibrated!)

**Use Cases:**
```
✓ Object detection (RetinaNet)
✓ Extreme class imbalance
✓ When: Hard examples are important
✓ When: Easy negatives dominate
✗ When: Classes are balanced (use cross-entropy)
✗ When: Need calibrated probabilities (use CE or calibrate post-hoc)
```

### Label Smoothing

**Mathematical Form:**
```
Standard one-hot: y_i = δ_ij (1 if i=j, 0 otherwise)

Label smoothing:
y'_i = (1 - ε)δ_ij + ε/K

where:
- ε: smoothing parameter (typically 0.1)
- K: number of classes

Loss: L = -Σ_i y'_i log(ŷ_i)
```

**Effect:**
```
Instead of [0, 0, 1, 0, 0] (K=5)
Use: [0.02, 0.02, 0.92, 0.02, 0.02] with ε=0.1

Prevents overfitting to training labels
Encourages model not to be overconfident
```

**Gradient:**
```
∂L/∂z_i = ŷ_i - y'_i
        = ŷ_i - [(1-ε)δ_ij + ε/K]

For true class j:
∂L/∂z_j = ŷ_j - (1-ε) - ε/K
        = ŷ_j - 1 + ε(1 - 1/K)

For other classes:
∂L/∂z_i = ŷ_i - ε/K

Model can never drive ŷ_j → 1 (always has small error ε)
```

**Regularization Effect:**
```
Theorem (Müller et al., 2019):
Label smoothing is equivalent to adding KL divergence to uniform:

L_LS = (1-ε)L_CE + ε·KL(u || ŷ)

where u is uniform distribution

Effect:
- Prevents largest logit from becoming much larger than others
- Improves generalization
- Better calibration
```

**Why label smoothing works:**

1. **Prevents overconfidence:**
   - Hard labels encourage model to make predictions arbitrarily confident
   - Softmax with z_correct >> z_wrong → ŷ_correct ≈ 1
   - Model achieves this by increasing logit magnitudes: ||z|| → ∞
   - Label smoothing caps maximum achievable probability: ŷ_max = 1 - ε + ε/K

2. **Implicit regularization on logits:**
   - Without smoothing: logits can grow unbounded
   - With smoothing: bounded logits (can't make perfect prediction)
   - Example with K=10, ε=0.1:
     - Target: [0.01, 0.01, 0.91, ..., 0.01]
     - Model can't achieve this by making z_2 → ∞
     - Forces model to learn reasonable feature magnitudes

3. **Better calibration:**
   - Overconfident models: predicted probability doesn't match true probability
   - Example: Model predicts 99% confidence but is only 90% accurate
   - Label smoothing reduces confidence → better calibrated probabilities
   - Calibration: P(correct | confidence=p) ≈ p

4. **Robustness to label noise:**
   - Real labels may have ~5-10% error rate (human annotation mistakes)
   - Hard labels assume 100% certainty → model fits noise
   - Soft labels admit uncertainty → model less affected by noise

**Optimal ε selection:**
```
Theory suggests: ε ≈ estimated label noise rate

Practical recommendations:
- ε = 0.1: Standard choice for ImageNet (works for most cases)
- ε = 0.05: Conservative, when labels are high quality
- ε = 0.2: Aggressive, when labels are noisy or dataset is small
- ε = 0.3: Very aggressive, rarely used (may hurt)

Rule of thumb: Start with ε = 0.1, increase if overfitting, decrease if underfitting
```

**Use Cases:**
```
✓ Image classification (standard practice now)
✓ When: Want better generalization
✓ When: Training labels may be noisy
✓ When: Model tends to overfit
✓ When: Need calibrated probabilities for decision-making
✗ When: Dataset is very small (<1K samples)
✗ When: Need perfect accuracy on training set (e.g., memorization tasks)
✗ When: Labels are verified perfect (rare in practice)
```

---

## Probabilistic Loss Functions

### f-Divergences: A Unified Framework

Many divergences between probability distributions belong to the family of f-divergences.

**Definition 10 (f-Divergence):**
For convex function f: ℝ₊ → ℝ with f(1) = 0:

D_f(P || Q) = E_Q[f(dP/dQ)] = Σ_x Q(x)·f(P(x)/Q(x))

where dP/dQ is the Radon-Nikodym derivative (density ratio).

**Properties:**
1. Non-negativity: D_f(P || Q) ≥ f(1) = 0 (by Jensen's inequality)
2. Identity: D_f(P || Q) = 0 ⟺ P = Q
3. Convexity: D_f(P || Q) is convex in the pair (P, Q)

**Proof of non-negativity:**
By Jensen's inequality (f is convex):

D_f(P || Q) = E_Q[f(dP/dQ)]
            ≥ f(E_Q[dP/dQ])
            = f(Σ_x Q(x)·(P(x)/Q(x)))
            = f(Σ_x P(x))
            = f(1)
            = 0 ∎

**Theorem 15 (Common Divergences as f-Divergences):**

(a) **KL Divergence:** f(t) = t log t
D_f(P || Q) = KL(P || Q) = Σ_x P(x) log(P(x)/Q(x))

(b) **Reverse KL:** f(t) = -log t
D_f(P || Q) = KL(Q || P) = Σ_x Q(x) log(Q(x)/P(x))

(c) **Total Variation:** f(t) = (1/2)|t - 1|
D_f(P || Q) = (1/2)Σ_x |P(x) - Q(x)| = TV(P, Q)

(d) **Squared Hellinger:** f(t) = (√t - 1)²
D_f(P || Q) = Σ_x Q(x)(√(P(x)/Q(x)) - 1)²
            = Σ_x (√P(x) - √Q(x))²

(e) **Chi-squared:** f(t) = (t - 1)²
D_f(P || Q) = Σ_x Q(x)((P(x)/Q(x)) - 1)²
            = Σ_x (P(x) - Q(x))²/Q(x)
            = χ²(P || Q)

(f) **Jensen-Shannon:** Symmetrized KL
JS(P || Q) = (1/2)KL(P || M) + (1/2)KL(Q || M)
where M = (1/2)(P + Q)

Properties:
- Symmetric: JS(P || Q) = JS(Q || P)
- Bounded: 0 ≤ JS(P || Q) ≤ log 2
- Square root is a metric: √JS is a proper distance

**Theorem 16 (Variational Representation of f-Divergences):**
For any f-divergence:

D_f(P || Q) = sup_{T: X→ℝ} {E_P[T(X)] - E_Q[f*(T(X))]}

where f* is the convex conjugate of f:
f*(y) = sup_t {ty - f(t)}

This variational form is the basis for adversarial training (GANs).

**Example (KL Divergence):**
For f(t) = t log t:
f*(y) = e^(y-1)

Thus:
KL(P || Q) = sup_T {E_P[T] - E_Q[e^(T-1)]}

**Theorem 17 (Data Processing Inequality):**
For any f-divergence and Markov chain X → Y → Z:

D_f(P_X || Q_X) ≥ D_f(P_Y || Q_Y) ≥ D_f(P_Z || Q_Z)

In words: processing data through any channel cannot increase divergence.

Proof: Uses Jensen's inequality and the Markov property. ∎

**Practical implication:**
- Feature extraction reduces divergence between distributions
- Information is lost, never gained, through transformations
- Applies to all f-divergences simultaneously

### Kullback-Leibler (KL) Divergence

**Mathematical Form:**
```
KL(P || Q) = Σ_x P(x) log(P(x)/Q(x))
           = E_P[log P(x)] - E_P[log Q(x)]

Continuous case:
KL(P || Q) = ∫ p(x) log(p(x)/q(x)) dx
```

**Properties:**
```
1. Non-negativity: KL(P || Q) ≥ 0
2. Identity: KL(P || Q) = 0 ⟺ P = Q (almost everywhere)
3. Asymmetry: KL(P || Q) ≠ KL(Q || P) in general
4. Not a metric (no triangle inequality)

Proof of non-negativity (Jensen's inequality):
KL(P || Q) = -E_P[log(Q/P)]
           ≥ -log E_P[Q/P]
           = -log Σ_x P(x)·(Q(x)/P(x))
           = -log Σ_x Q(x)
           = -log 1 = 0
```

**Relationship to Cross-Entropy:**
```
H(P, Q) = -E_P[log Q(x)]         (cross-entropy)
H(P) = -E_P[log P(x)]            (entropy)

KL(P || Q) = H(P, Q) - H(P)

In supervised learning:
- P: true distribution (one-hot)
- Q: predicted distribution
- H(P) = 0 for one-hot labels

Therefore: KL(P || Q) = H(P, Q)

Minimizing KL ⟺ Minimizing cross-entropy
```

**Forward vs Reverse KL:**
```
Forward KL: KL(P || Q)
- Minimizing Q w.r.t. P
- Mean-seeking: Q covers all modes of P
- Used in: Maximum likelihood

Reverse KL: KL(Q || P)
- Mode-seeking: Q focuses on one mode of P
- Used in: Variational inference

Behavior difference:
P has two modes, Q is unimodal
- Forward KL: Q spreads between modes (bad)
- Reverse KL: Q picks one mode (better for VI)
```

**Intuitive explanation of forward vs reverse:**

**Forward KL: KL(P || Q) = E_P[log P - log Q]**

Behavior:
- Expectation over P: Must have high probability under Q wherever P has probability
- If P(x) > 0 but Q(x) ≈ 0: log Q(x) → -∞, KL → ∞ (infinite penalty!)
- Forces Q to cover all of P (Q spreads out to avoid zero probability under P)

Example (mixture of two Gaussians):
```
P = 0.5·N(-3, 1) + 0.5·N(+3, 1)  (two modes at -3 and +3)
Q = N(μ, σ²)                      (single Gaussian)

Forward KL minimization:
- Q must have P(x)>0 → Q(x)>0 everywhere P has mass
- Result: Q = N(0, 4) (wide Gaussian covering both modes)
- Mean μ=0 (average of -3 and +3)
- Large variance to cover both modes
```

**Reverse KL: KL(Q || P) = E_Q[log Q - log P]**

Behavior:
- Expectation over Q: Only cares about where Q has probability
- If Q(x) > 0 but P(x) ≈ 0: log P(x) → -∞, KL → ∞
- Forces Q to only put mass where P has high probability (Q is zero-avoiding under P)
- Q ignores low-probability regions of P

Example (same mixture):
```
P = 0.5·N(-3, 1) + 0.5·N(+3, 1)
Q = N(μ, σ²)

Reverse KL minimization:
- Q can put mass anywhere, but gets penalized if P is low there
- Result: Q = N(-3, 1) OR N(+3, 1) (picks one mode!)
- Which mode depends on initialization
- Tight fit to chosen mode
```

**Practical consequences:**

**Use Forward KL (Maximum Likelihood) when:**
- Have samples from P (data distribution)
- Want Q to explain all data variations
- Can't afford to miss any data modes
- Example: Generative modeling (GAN, diffusion)

**Use Reverse KL (Variational Inference) when:**
- Need to sample from Q and evaluate P
- Want tractable approximate posterior
- Okay with approximating one mode well
- Example: VAE, variational Bayes

**Visual intuition:**
```
P: ⚫  ⚫    (two modes)

Forward KL → Q: ━━━━━ (covers both, wide)
Reverse KL → Q:  ⚫    (picks one, tight)
```

**Mathematical reason:**
```
Forward: ∫ p(x) log[p(x)/q(x)] dx
- Integral weighted by p(x)
- High penalty where p(x) is large but q(x) is small
- q must spread to cover all of p

Reverse: ∫ q(x) log[q(x)/p(x)] dx
- Integral weighted by q(x)
- Only penalty where q(x) is large
- q can ignore regions where p is small
```

**Gradient (for Variational Inference):**
```
For parameterized Q_θ:

∇_θ KL(Q_θ || P) = ∇_θ E_Q[log Q_θ(z) - log P(z)]

Using reparameterization trick:
z = g_θ(ε, x) where ε ~ p(ε)

∇_θ KL = E_ε[∇_θ(log Q_θ(g_θ(ε,x)) - log P(g_θ(ε,x)))]

Enables backpropagation through sampling
```

**Use Cases:**
```
✓ Variational inference (VAE, VIB)
✓ Knowledge distillation
✓ Distribution matching
✓ Policy optimization (RL)
✗ When: Need symmetric divergence (use JS divergence)
```

### Wasserstein Distance (Earth Mover's Distance)

**Mathematical Form:**
```
W_p(P, Q) = (inf_{γ∈Γ(P,Q)} E_{(x,y)~γ}[||x-y||^p])^{1/p}

where Γ(P,Q) is set of joint distributions with marginals P and Q

For p=1: W_1(P, Q) = inf_{γ} E_{(x,y)~γ}[||x-y||]

Interpretation: Minimum cost to transport mass from P to Q
```

**Properties:**
```
1. Metric: Satisfies triangle inequality
2. Weaker topology than KL divergence
3. Doesn't suffer from mode collapse
4. Well-defined even when P and Q don't overlap

Advantage over KL:
- KL(P || Q) = ∞ if supports don't overlap
- W(P, Q) is always finite
```

**Kantorovich-Rubinstein Duality:**
```
W_1(P, Q) = sup_{||f||_L≤1} [E_P[f(x)] - E_Q[f(x)]]

where ||f||_L ≤ 1 means f is 1-Lipschitz

This is computable! (used in WGANs)
```

**Gradient (WGAN):**
```
Discriminator (critic) f must be 1-Lipschitz:

Methods:
1. Weight clipping: clip weights to [-c, c]
2. Gradient penalty: ||∇_x f(x)||₂ ≈ 1

WGAN-GP loss:
L_D = E_Q[f(x)] - E_P[f(x)] + λ·E_x̂[(||∇_x̂ f(x̂)||₂ - 1)²]

where x̂ = αx_real + (1-α)x_fake, α ~ U[0,1]
```

**Convergence Properties:**
```
WGAN theoretical guarantees:

Theorem: Optimizing W_1 distance:
- Generator converges to data distribution
- Even when distributions have disjoint supports

Gradient behavior:
- Non-saturating (unlike original GAN)
- Meaningful loss values (correlate with quality)
- Stable training

Practical convergence:
- Requires many critic iterations per generator step
- GP version more stable than weight clipping
```

**Use Cases:**
```
✓ GANs (WGAN, WGAN-GP)
✓ When: Distributions may not overlap
✓ When: Need meaningful loss values
✓ Optimal transport problems
✗ When: Computational cost is critical (expensive)
```

---

## Ranking and Metric Learning

### Triplet Loss

**Mathematical Form:**
```
L = max(0, d(a, p) - d(a, n) + margin)

where:
- a: anchor sample
- p: positive sample (same class as anchor)
- n: negative sample (different class)
- d(·,·): distance metric (usually L2)
- margin: hyperparameter (typically 0.2-1.0)

Goal: d(a, p) + margin < d(a, n)
```

**Gradient:**
```
When loss > 0:

∂L/∂a = 2(a - p) - 2(a - n) = 2(n - p)
∂L/∂p = 2(p - a)
∂L/∂n = 2(n - a)

Properties:
- Pulls anchor to positive
- Pushes anchor from negative
- Zero gradient when margin satisfied
```

**Mining Strategies:**
```
Triplet selection critical!

1. Hard negative mining:
   n = argmax_n d(a, n) s.t. d(a,p) > d(a,n)
   Most violating negative

2. Semi-hard negative:
   d(a, p) < d(a, n) < d(a, p) + margin
   Violates margin but not order

3. Easy negative:
   d(a, n) > d(a, p) + margin
   Don't contribute to loss (skip)

4. Batch-all mining:
   Use all valid triplets in batch
   Can be many: O(batch_size³)
```

**Convergence Properties:**
```
Challenges:
- Non-smooth (max function)
- Many triplets satisfy margin (zero gradient)
- Requires good mining strategy

Convergence:
- Depends heavily on mining
- Hard mining: faster but unstable
- Semi-hard: slower but more stable

Practical tips:
- Start with easy/random triplets
- Gradually increase difficulty
- Use online mining within batches
```

#### Rigorous Metric Learning Theory

**Theorem 26 (Triplet Loss as Structured Hinge Loss - Schroff et al., 2015):**
Triplet loss is a structured margin-based loss enforcing relative ordering:

L_triplet = max(0, ||f(a) - f(p)||² - ||f(a) - f(n)||² + α)

This is equivalent to hinge loss on the margin:
L = max(0, margin(a,p,n) + α)

where margin(a,p,n) = d(a,p) - d(a,n) is the **relative distance difference**.

**Properties:**
(a) **Non-convex:** Even for linear embeddings f(x) = Wx, loss is non-convex in W
(b) **Lipschitz:** With bounded features ||f(x)|| ≤ R:
|L(f) - L(f')| ≤ 4R·||f - f'||_∞

(c) **Zero gradient region:** When margin > α, gradient is zero (no learning signal)

**Theorem 27 (Generalization Bound for Triplet Loss - Cao et al., 2016):**
For hypothesis class F with Rademacher complexity R_n(F), the generalization error for
triplet loss satisfies with probability ≥ 1-δ:

R(f) - R̂_n(f) ≤ 4R_n(F) + 3√(log(2/δ)/(2n))

where R̂_n is empirical risk over n triplets.

**For linear embeddings** f(x) = Wx with ||W||_F ≤ B and ||x|| ≤ R:
R_n(F) ≤ (2BR√d) / √n

**Sample complexity:** For (ε, δ)-accurate embedding:
n = Ω((B²R²d/ε²)·log(1/δ))

**Example:**
- Embedding dimension: d = 128
- Feature norm bound: R = 10
- Weight bound: B = 1
- Target error: ε = 0.1
- Confidence: δ = 0.05

Required triplets: n ≈ 13,000 · log(20) ≈ 39,000 triplets!

**Theorem 28 (Triplet Mining and Convergence - Wu et al., 2017):**
The choice of triplet mining strategy critically affects convergence:

(a) **Random triplets:** Convergence rate O(1/√T) but requires ~O(N³) triplets
(b) **Hard negatives:** Convergence rate O(1/T) but may cause instability
(c) **Semi-hard negatives:** Convergence O(1/√T) with better stability

**Formal definition (Semi-hard negative):**
n is semi-hard for anchor a and positive p if:
||f(a) - f(p)||² < ||f(a) - f(n)||² < ||f(a) - f(p)||² + α

**Convergence guarantee (semi-hard mining):**
After T iterations with batch size B:
E[margin violation rate] ≤ O(√(log B / T))

**Theorem 29 (Embedding Space Properties):**
After training with triplet loss, the learned embedding f: X → ℝ^d satisfies:

(a) **Triangle inequality:** Not guaranteed! Triplet loss doesn't enforce metric properties.

(b) **Clustering property:** Same-class embeddings form clusters with radius:
r_class ≤ √α  (margin parameter bounds cluster radius)

(c) **Separation:** Different-class cluster centers separated by ≥ 2√α

**Proof of (c):**
For converged model, all triplets satisfy:
||f(a) - f(p)||² + α ≤ ||f(a) - f(n)||²

Taking expectations over class distributions:
E[||f(x_i) - c_i||²] + α ≤ E[||f(x_i) - c_j||²]  for i ≠ j

where c_i, c_j are class centroids.

By triangle inequality:
||c_i - c_j||² ≥ (√E[||f(x_i) - c_j||²] - √E[||f(x_i) - c_i||²])²
              ≥ (√α)² = α

But empirically: ||c_i - c_j|| ≥ 2√α (factor of 2 from typical distributions). ∎

**Theorem 30 (Number of Triplets and Saturation):**
For dataset with N examples and C classes:
- Total valid triplets: O(N³) (combinatorial explosion!)
- After k epochs, fraction satisfying margin: p_sat(k)

**Saturation curve:**
p_sat(k) ≈ 1 - exp(-λk)

where λ depends on data difficulty and model capacity.

**Practical implications:**
- Early training: most triplets violate margin → strong gradients
- Late training: most triplets satisfy margin → sparse gradients
- Solution: Online mining (select hard triplets within each batch)

**Effective batch size for online mining:**
For batch size B, number of valid triplets: O(B³/C)
- B = 32, C = 8: ~2,000 triplets per batch!
- But only ~5-10% violate margin after initial training

**Use Cases:**
```
✓ Face recognition (FaceNet)
✓ Metric learning
✓ Image retrieval
✓ When: Need embedding space with metric properties
✓ When: Many classes (>100)
✗ When: Number of classes is small (use softmax)
✗ When: Cannot form meaningful triplets
```

### Contrastive Loss

**Mathematical Form:**
```
L = (1-y)·½d² + y·½max(0, margin - d)²

where:
- y = 1 if same class, 0 if different
- d = ||f(x₁) - f(x₂)||₂

Alternative (SimCLR style):
L = -log(exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ))

where sim(u,v) = u·v / (||u||·||v||) (cosine similarity)
```

**Gradient:**
```
For positive pair (y=1):
∂L/∂z₁ = (z₁ - z₂)·max(0, margin - d)/d

For negative pair (y=0):
∂L/∂z₁ = (z₁ - z₂)·d

Behavior:
- Positive: push together until margin
- Negative: push apart always (no limit)
```

**InfoNCE (Contrastive Predictive Coding):**
```
L = -E[log(f(x,c_pos) / (f(x,c_pos) + Σ_i f(x,c_neg,i)))]

where f(x,c) = exp(x^T c / τ)

Properties:
- Lower bound on mutual information
- As negatives → ∞: approaches MI
- Temperature τ controls difficulty
```

#### Rigorous Contrastive Learning Theory

**Theorem 31 (InfoNCE as MI Lower Bound - van den Oord et al., 2018):**
The InfoNCE (Noise Contrastive Estimation) loss provides a lower bound on mutual information:

L_InfoNCE = -E[log(exp(f(x,c_+)/τ) / (exp(f(x,c_+)/τ) + Σᵢ exp(f(x,c_-,i)/τ)))]

where:
- c_+ is positive context (similar to x)
- c_-,i are K negative contexts (dissimilar to x)
- τ is temperature
- f(x,c) is similarity score

**Theorem:** InfoNCE lower-bounds mutual information:

I(X; C) ≥ log(K+1) - L_InfoNCE

**Proof:**
Let p_+ = exp(f(x,c_+)/τ) / Z be the probability assigned to positive.

InfoNCE = -E[log p_+] = H_cross(p_true, p_model)

By properties of cross-entropy:
H_cross ≤ H(p_true) + KL(p_true || p_model)

With K negatives uniformly sampled:
H(p_true) = log(K+1)

Therefore:
I(X; C) = H(C) - H(C|X)
        ≥ log(K+1) - L_InfoNCE

**Key insight:** More negatives → tighter bound! ∎

**Theorem 32 (Temperature Scaling Effects - Wang & Isola, 2020):**
Temperature τ controls the concentration of the distribution:

(a) **Small τ (τ → 0):** Hard assignment (winner-take-all)
- Focuses on hardest negatives
- Can be unstable
- Typical: τ = 0.05-0.1

(b) **Large τ (τ → ∞):** Uniform distribution
- Treats all negatives equally
- Slower learning
- Typical: τ = 0.5-1.0

**Optimal τ:** For data with noise level σ, optimal temperature:
τ* ≈ σ/√d

where d is embedding dimension.

**Empirical finding (SimCLR):**
- d = 128: τ = 0.07 works best
- d = 512: τ = 0.1 works best
- d = 2048: τ = 0.2 works best

**Theorem 33 (Contrastive Learning Convergence - Arora et al., 2019):**
For contrastive learning with K negatives, the embedding quality after T steps satisfies:

||f*_θ - f*_optimal||² ≤ O(√(d/(K·T)))

where:
- d is embedding dimension
- K is number of negatives per positive
- T is number of training steps

**Implications:**
(a) **More negatives help:** Convergence ∝ 1/√K
(b) **Diminishing returns:** K = 4096 only √2 better than K = 2048
(c) **Need large batches:** K typically = batch_size - 1

**Example:**
- Target error: ε = 0.01
- Embedding dim: d = 128
- Negatives: K = 1024
- Required steps: T = d/(K·ε²) = 128/(1024·0.0001) ≈ 1,250 steps

With batch size 256: ~5 epochs on 64K examples.

**Theorem 34 (Alignment and Uniformity - Wang & Isola, 2020):**
Good contrastive representations satisfy two properties:

(a) **Alignment:** Positive pairs close together:
L_align = E_{(x,x_+)~p_pos}[||f(x) - f(x_+)||²]

(b) **Uniformity:** Features uniformly distributed on hypersphere:
L_uniform = log E_{x,y~p_data}[exp(-||f(x) - f(y)||²)]

**Optimal embedding:** Minimizes weighted combination:
L = L_align + λ·L_uniform

**Relationship to contrastive loss:**
InfoNCE implicitly optimizes both:
- Numerator → minimizes L_align
- Denominator → maximizes L_uniform (pushes negatives away)

**Proof sketch:**
∂L_InfoNCE/∂f(x) = -f(c_+) + E_neg[f(c_-)]

First term: pulls x toward positive (alignment)
Second term: pushes x from negatives (uniformity) ∎

**Theorem 35 (Representation Collapse Prevention):**
Without sufficient negatives, representations **collapse** to constant:
f(x) = c for all x

**Collapse condition:** When K < d/log d (too few negatives for embedding dimension)

**Proof (informal):**
For d-dimensional embeddings on unit sphere, need K = Ω(d/log d) random negatives
to distinguish between different points.

**Practical safeguards:**
(a) **Large batch sizes:** K = batch_size - 1 ≥ 256
(b) **Momentum encoder:** MoCo queue maintains K = 65,536 negatives
(c) **Explicit uniformity loss:** Add L_uniform regularization
(d) **Batch normalization:** Prevents complete collapse (each feature has variance)

**Empirical collapse indicators:**
- Training loss → 0 but validation performance poor
- Embedding norm ||f(x)|| → 0
- Embedding variance Var[f(x)] → 0
- Cosine similarity matrix rank deficient

**Theorem 36 (Sample Complexity for Contrastive Learning):**
To learn ε-optimal representations with δ confidence:

n = Ω((d²/ε²)·log(1/δ)·log(K))

**Comparison to supervised learning:**
Supervised (K classes): n = O((d/ε²)·K·log(1/δ))
Contrastive: n = O((d²/ε²)·log(K)·log(1/δ))

**Key difference:** Contrastive requires O(d) more samples but is label-free!

**Example:**
- d = 128, K = 1000 classes, ε = 0.1, δ = 0.05
- Supervised: n ≈ 164M samples
- Contrastive: n ≈ 21B samples (128× more!)

But contrastive doesn't need labels → can use unlabeled data at scale!

**Use Cases:**
```
✓ Self-supervised learning (SimCLR, MoCo)
✓ Representation learning
✓ When: Have pairs of similar/dissimilar samples
✓ When: Unsupervised or semi-supervised setting
✓ When: Large unlabeled dataset available
✗ When: Cannot create meaningful positive pairs
✗ When: Small batch size (< 256)
```

---

## Sequence-to-Sequence Losses

### Teacher Forcing with Cross-Entropy

**Mathematical Form:**
```
For sequence y = [y₁, y₂, ..., y_T]:

L = -(1/T) Σ_t log P(y_t | y₁, ..., y_{t-1}, x)

Teacher forcing: Use ground truth y_{t-1} during training

Problems:
- Exposure bias: Model sees gold labels in training
- Test time: Uses own predictions → distribution mismatch
```

**Scheduled Sampling:**
```
Mix gold and predicted tokens during training:

With probability p: use y_{t-1} (gold)
With probability 1-p: use ŷ_{t-1} (predicted)

Schedule: p decreases over training (e.g., p = 1/(1+k/k₀))

Reduces exposure bias but makes training non-differentiable
```

### Connectionist Temporal Classification (CTC)

**Mathematical Form:**
```
For input X of length T, output Y of length U ≤ T:

P(Y | X) = Σ_{π∈Alignments(Y)} Π_t P(π_t | X)

where Alignments(Y) are all valid alignments

Example:
Y = "cat"
Alignments: [c,c,a,t], [c,a,a,t], [c,a,t,t], [-,c,a,t], etc.

CTC loss: L = -log P(Y | X)
```

**Forward-Backward Algorithm:**
```
Efficiently compute sum over exponential alignments:

Forward: α(t,s) = P(labels[1:s] | input[1:t])
Backward: β(t,s) = P(labels[s:] | input[t:])

Combined: P(Y|X) = α(T,|Y|)

Complexity: O(T · U) instead of O(T^U)
```

**Gradient:**
```
∂L/∂y_{t,k} = p_{t,k} - (1/P(Y|X)) Σ_{s:π_s=k} α(t,s)β(t,s)

where p_{t,k} = P(label k at time t)

Computed efficiently with forward-backward
```

#### Rigorous CTC Theory

**Theorem 37 (CTC Forward-Backward Correctness - Graves et al., 2006):**
The forward-backward algorithm correctly computes P(Y|X) in O(T·U) time,
where T = input length, U = output length.

**Forward variable:** α(t, s) = P(π[1:t] matches Y[1:s] | X[1:t])

Recursion:
α(t, s) = [α(t-1, s) + α(t-1, s-1) + α(t-1, s-2)·𝟙[Y_s = blank]] · P(Y_s | X_t)

**Backward variable:** β(t, s) = P(π[t:T] matches Y[s:U] | X[t:T])

Recursion:
β(t, s) = [β(t+1, s) + β(t+1, s+1) + β(t+1, s+2)·𝟙[Y_s = blank]] · P(Y_s | X_{t+1})

**Total probability:**
P(Y | X) = Σ_s α(T, s)·β(T, s) = α(T, U)

**Proof of O(T·U) complexity:**
- State space: T time steps × U label positions = O(T·U) states
- Each state computed in O(1) using 3 previous states
- Total: O(T·U) time and space ∎

**Comparison to naive enumeration:**
- All possible alignments: |Σ|^T where |Σ| = vocabulary size
- For |Σ| = 30, T = 100: 30^100 ≈ 10^147 alignments!
- Forward-backward: T·U = 100·20 = 2,000 operations (10^144× speedup!)

**Theorem 38 (CTC Gradient Computation):**
The gradient of CTC loss w.r.t. logits can be computed efficiently:

∂L_CTC/∂log P(k | X_t) = P(k | X_t) - Σ_{s: Y_s=k} α(t,s)·β(t,s) / P(Y|X)

**Interpretation:**
- First term: Model's prediction P(k | X_t)
- Second term: Expected label at time t given Y
- Gradient = prediction error at each time step!

**Computational complexity:**
- Forward pass: O(T·U)
- Backward pass: O(T·U)
- Gradient: O(T·|Σ|) where |Σ| is vocabulary size
- Total per example: O(T·(U + |Σ|))

**Theorem 39 (CTC Blank Label Necessity):**
The blank label is **necessary** for CTC to represent all possible alignments.

**Proof by counterexample:**
Consider Y = "aa" (repeated label) with T = 2 time steps.
- Without blank: only possible alignment is [a, a]
- Cannot distinguish from Y = "a" with alignment [a, a]!

With blank:
- Y = "aa": alignments include [a, blank], [blank, a], [a, a] (via blank)
- Y = "a": alignments are [a, blank], [blank, a], [a, a] (single a)

Blank allows CTC to:
1. Represent repeated labels: a-blank-a → "aa"
2. Handle variable length: blank tokens for timing
3. Model optionality: blank for silence ∎

**Theorem 40 (CTC Conditional Independence Assumption):**
CTC assumes **conditional independence** of outputs at each time step:

P(π_t | π_{1:t-1}, X) = P(π_t | X_t)

This is a **strong assumption** that limits CTC expressiveness!

**Consequences:**
(a) Cannot model output dependencies: P(Y_i | Y_{i-1}) = const (no language model)
(b) Alignment depends only on acoustics, not on label sequence
(c) Need external language model for good performance

**Comparison to attention:**
- CTC: O(T) independent predictions
- Attention: O(U) autoregressive predictions with full history

**Theorem 41 (CTC Alignment Ambiguity):**
For a given Y, CTC marginalizes over **exponentially many** alignments:

Number of valid alignments ≈ (T choose U)·2^U

**Example:**
- Y = "cat" (U = 3 labels)
- T = 10 time steps
- Valid alignments: (10 choose 3)·2³ = 120·8 = 960 alignments!

Each alignment π has probability:
P(π | X) = Π_{t=1}^T P(π_t | X_t)

CTC loss:
L = -log Σ_π P(π | X)  (sum over all 960 alignments!)

**Theorem 42 (CTC Decoding Complexity):**

(a) **Greedy decoding:** O(T·|Σ|)
- Take argmax at each time step
- Remove blanks and duplicates
- Fast but suboptimal!

(b) **Beam search:** O(T·|Σ|·B·log B)
- Maintain top-B hypotheses
- Merge paths with same label sequence
- Much better accuracy
- B = 100 typical

(c) **Optimal decoding:** O(T·U·|Σ|)
- Find most probable label sequence Y* = argmax_Y P(Y|X)
- Uses prefix search beam search
- Exponentially slow without pruning

**Practical beam search:**
For B = 100, |Σ| = 30, T = 100:
- Operations: 100·30·100·log(100) ≈ 20M
- Greedy: 100·30 = 3K (6000× faster!)
- Accuracy gap: ~5-10% WER improvement with beam search

**Theorem 43 (CTC Sample Complexity):**
For CTC with vocabulary size |Σ| and max length T, sample complexity:

n = Ω((|Σ|·T·log T / ε²)·log(1/δ))

**Interpretation:**
- Linear in vocabulary size |Σ|
- Linear in sequence length T
- Requires log T factor for alignment uncertainty

**Comparison to frame-level classification:**
Frame-level: n = Ω((|Σ|·T/ε²)·log(1/δ))
CTC: Extra log T factor due to marginalization over alignments.

**Example:**
- Vocabulary: |Σ| = 30 (characters)
- Sequence length: T = 100
- Target error: ε = 0.1
- Confidence: δ = 0.05

CTC: n ≈ 30·100·log(100)·100·3 ≈ 4.1M sequences
Frame-level: n ≈ 30·100·100·3 ≈ 900K frames

CTC needs ~4-5× more data due to alignment ambiguity!

**Use Cases:**
```
✓ Speech recognition (DeepSpeech, Wav2Vec)
✓ OCR (handwriting recognition)
✓ When: Input and output lengths differ
✓ When: Alignment is unknown
✓ When: Conditional independence acceptable
✗ When: Need attention mechanism (use seq2seq instead)
✗ When: Strong output dependencies (use autoregressive models)
✗ When: Small dataset (CTC needs more data than frame-level)
```

---

## Generative Model Losses

### Variational Lower Bound (ELBO)

**Mathematical Form:**
```
For VAE with encoder q_φ(z|x) and decoder p_θ(x|z):

ELBO(θ,φ) = E_q[log p_θ(x|z)] - KL(q_φ(z|x) || p(z))

Equivalent forms:
= log p(x) - KL(q_φ(z|x) || p(z|x))
= E_q[log p_θ(x,z) - log q_φ(z|x)]

Loss: L = -ELBO (we minimize negative ELBO)
```

**Derivation:**
```
log p(x) = log∫ p(x,z) dz
         = log∫ p(x,z)·(q(z|x)/q(z|x)) dz
         = log E_q[p(x,z)/q(z|x)]
         ≥ E_q[log p(x,z)/q(z|x)]    (Jensen's inequality)
         = E_q[log p(x|z)] - KL(q(z|x) || p(z))
         = ELBO

Gap: KL(q(z|x) || p(z|x))
Tight when: q(z|x) = p(z|x) (intractable)
```

**Gradient Estimation:**
```
Problem: Can't backprop through z ~ q_φ(z|x)

Reparameterization trick:
z = μ_φ(x) + σ_φ(x) ⊙ ε, where ε ~ N(0,I)

Now:
∇_φ E_q[f(z)] = ∇_φ E_ε[f(μ_φ(x) + σ_φ(x) ⊙ ε)]
              = E_ε[∇_φ f(μ_φ(x) + σ_φ(x) ⊙ ε)]

Low variance, unbiased gradient!
```

**β-VAE:**
```
L = E_q[log p_θ(x|z)] - β·KL(q_φ(z|x) || p(z))

β > 1: Stronger regularization
- More disentangled representations
- Worse reconstruction

β < 1: Weaker regularization
- Better reconstruction
- Less disentangled

Trade-off between reconstruction and regularization
```

**Convergence Properties:**
```
ELBO optimization:

Properties:
- ELBO is lower bound on log p(x)
- Maximizing ELBO ⟹ good generative model
- Non-convex (neural networks)

Convergence:
- Alternating optimization (not simultaneous)
- Can use Adam/SGD
- Often reaches good local optimum

Challenges:
- KL vanishing: KL → 0, model ignores z
- Solutions: KL annealing, free bits
```

**Use Cases:**
```
✓ Variational Autoencoders
✓ Bayesian inference approximation
✓ Latent variable models
✓ Semi-supervised learning
```

### GAN Losses

**Original GAN (Minimax):**
```
L_D = -E_real[log D(x)] - E_fake[log(1 - D(G(z)))]
L_G = -E_fake[log D(G(z))]

Minimax formulation:
min_G max_D V(D,G) = E_real[log D(x)] + E_fake[log(1-D(G(z)))]

Problems:
- Vanishing gradients for generator
- When D is optimal: log(1-D(G(z))) ≈ 0
```

**Non-Saturating GAN:**
```
L_G = -E_fake[log D(G(z))]

Instead of minimizing log(1-D(G(z))), maximize log D(G(z))

Gradient:
- Original: -D'(G(z)) when D(G(z)) small (weak signal)
- Non-saturating: D'(G(z))/D(G(z)) (stronger)

Still minimax, just different G objective
```

**Wasserstein GAN (WGAN):**
```
L_D = -E_real[D(x)] + E_fake[D(G(z))]
L_G = -E_fake[D(G(z))]

where D is 1-Lipschitz

Advantages:
- Meaningful loss (correlates with quality)
- No saturation
- More stable training

Enforce Lipschitz:
- Weight clipping: clip θ to [-c, c]
- Gradient penalty (WGAN-GP): better
```

**WGAN-GP (Gradient Penalty):**
```
L_D = -E_real[D(x)] + E_fake[D(G(z))]
      + λ·E_x̂[(||∇_x̂ D(x̂)||₂ - 1)²]

where:
x̂ = ε·x_real + (1-ε)·x_fake, ε ~ U[0,1]
λ = 10 typically

Gradient penalty enforces 1-Lipschitz constraint softly
```

**Convergence Analysis:**
```
Original GAN:
- Theoretically: Reaches Nash equilibrium (D*,G*)
- Practically: Oscillates, mode collapse, instability

WGAN:
- Theoretical guarantee: G → data distribution
- Practically: More stable, slower

WGAN-GP:
- Best practical performance
- Lipschitz constraint well-enforced
- Typical: 5 critic steps per 1 generator step
```

#### Rigorous GAN Theory

**Theorem 44 (GAN Nash Equilibrium - Goodfellow et al., 2014):**
The minimax game:

min_G max_D V(D,G) = E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 - D(G(z)))]

has a **global optimum** at:
- D*(x) = p_data(x) / (p_data(x) + p_G(x))
- p_G = p_data (generator matches data distribution)

**Proof:**
For fixed G, optimal D* maximizes V(D,G):

V(D,G) = ∫_x [p_data(x)log D(x) + p_G(x)log(1-D(x))] dx

Taking derivative w.r.t. D(x) and setting to zero:
p_data(x)/D(x) - p_G(x)/(1-D(x)) = 0

Solving: D*(x) = p_data(x)/(p_data(x) + p_G(x))

Substituting back:
V(D*,G) = E_x[p_data log(p_data/(p_data+p_G)) + p_G log(p_G/(p_data+p_G))]
         = -log 4 + 2·JSD(p_data || p_G)

where JSD is Jensen-Shannon divergence.

Minimum when JSD(p_data || p_G) = 0, i.e., p_G = p_data. ∎

**Theorem 45 (GAN Training Instability - Arjovsky & Bottou, 2017):**
When discriminator is optimal (or near-optimal), the generator gradient **vanishes**:

∇_θ V(D*,G_θ) ≈ ∇_θ 2·JSD(p_data || p_G) = 0

when p_data and p_G have **non-overlapping supports**.

**Proof (informal):**
If supp(p_data) ∩ supp(p_G) = ∅:
- D can perfectly classify: D(x) = 1 for x ~ p_data, D(x) = 0 for x ~ p_G
- log(1 - D(G(z))) ≈ log(1 - 0) = 0 (constant!)
- Generator gradient: ∇_θ log(1-D(G(z))) ≈ 0

This happens in high dimensions where distributions are low-dimensional manifolds. ∎

**Consequence:** Standard GAN training is inherently unstable!

**Theorem 46 (Wasserstein Distance - Arjovsky et al., 2017):**
The Wasserstein-1 (Earth Mover's) distance is:

W_1(p_data, p_G) = inf_{γ∈Π(p_data,p_G)} E_{(x,y)~γ}[||x - y||]

where Π(p_data, p_G) are all joint distributions with marginals p_data and p_G.

**Kantorovich-Rubinstein duality:**
W_1(p_data, p_G) = sup_{||f||_L≤1} E_{x~p_data}[f(x)] - E_{x~p_G}[f(x)]

where ||f||_L ≤ 1 means f is 1-Lipschitz: |f(x) - f(y)| ≤ ||x - y||.

**WGAN objective:**
L_D = -E_{x~p_data}[D(x)] + E_{x~p_G}[D(x)]

approximates W_1 if D is 1-Lipschitz!

**Theorem 47 (WGAN Convergence Properties):**
Wasserstein distance has **better properties** than JS divergence:

(a) **Continuity:** W_1 is continuous everywhere
- JS divergence can be discontinuous (non-overlapping supports)

(b) **Weak convergence:** W_1(p_n, p) → 0 implies p_n ⇀ p (weak convergence)
- JS(p_n || p) doesn't imply convergence!

(c) **Informative gradients:** Even when supports don't overlap:
∇_θ W_1(p_data, p_G) ≠ 0 (provides learning signal)

**Example (1D case):**
- p_data = δ(x - 0) (point mass at 0)
- p_θ = δ(x - θ) (point mass at θ)

JS divergence:
- JS(p_data || p_θ) = log 2 for all θ ≠ 0 (constant! No gradient!)
- JS(p_data || p_0) = 0

Wasserstein:
- W_1(p_data, p_θ) = |θ| (linear! Always has gradient!)
- ∇_θ W_1 = sign(θ) (informative everywhere)

**Theorem 48 (Lipschitz Constraint Enforcement):**

(a) **Weight clipping (WGAN):**
Clip weights: W ∈ [-c, c]

Problems:
- Capacity reduction: limits function class
- Gradient pathology: pushes weights to boundaries
- Poor convergence: needs small c (≈ 0.01)

(b) **Gradient penalty (WGAN-GP):**
L_D = E_x̃[(||∇_x̃ D(x̃)||_2 - 1)²]

where x̃ = ε·x_real + (1-ε)·x_fake, ε ~ U[0,1]

**Why interpolation?** Optimal discriminator has ||∇D|| = 1 almost everywhere
between real and fake samples.

**Theorem 49 (WGAN-GP Convergence Rate):**
With gradient penalty λ and optimal critic steps per generator step r,
WGAN-GP converges at rate:

W_1(p_data, p_G) ≤ O(1/√T) + O(1/r) + O(√λ)

**Interpretation:**
- Need T = O(1/ε²) steps for ε-optimal
- Need r = O(1/ε) critic steps per generator step
- λ = 10 typical (tradeoff: too small → poor Lipschitz, too large → gradient penalty dominates)

**Practical recipe (Gulrajani et al., 2017):**
- r = 5 critic updates per 1 generator update
- λ = 10 for gradient penalty
- Learning rate: α = 10^(-4) for both D and G
- Optimizer: Adam with β₁ = 0, β₂ = 0.9

**Theorem 50 (Mode Collapse Analysis - Metz et al., 2017):**
Mode collapse occurs when generator maps multiple noise vectors to same output:

G(z₁) = G(z₂) for z₁ ≠ z₂

**Quantification:** Effective number of modes captured:
N_eff = exp(H[p_G])

where H is entropy. Full diversity: N_eff = N_data.

**Causes:**
(a) **Generator gradient:** ∇_θ L_G focuses on fooling D, not diversity
(b) **Discriminator saturation:** D → 0 or 1 → no gradient signal
(c) **Nash equilibrium instability:** No guarantee of reaching equilibrium

**Solutions:**
(a) **Minibatch discrimination:** D sees multiple samples → can detect mode collapse
(b) **Unrolled GAN:** Optimize G w.r.t. unrolled D (k steps ahead)
(c) **Spectral normalization:** Control Lipschitz constant of D

**Theorem 51 (Sample Complexity for GANs):**
To learn ε-optimal generator in W_1 distance with confidence 1-δ:

n = Ω((d/ε²)·log(1/δ))

where d is ambient dimension.

**Comparison to density estimation:**
- Density estimation: n = Ω((d/ε)^d) (exponential in d!)
- GANs: n = Ω(d/ε²) (polynomial!)

GANs can learn distributions in high dimensions with **polynomial sample complexity**!

**Caveat:** This is for W_1 distance. Other metrics may require more samples.

**Example:**
- Image dimension: d = 64×64×3 ≈ 12,000
- Target distance: ε = 0.1
- Confidence: δ = 0.05

GAN: n ≈ 12,000·100·3 ≈ 3.6M images
Density estimation: n ≈ (12,000·10)^12000 (intractable!)

**Theorem 52 (GAN vs VAE Trade-offs):**

**GANs:**
- Minimize: W_1(p_data, p_G) or JSD(p_data || p_G)
- Sample quality: High (sharp images)
- Sample diversity: Can suffer mode collapse
- Training: Unstable (adversarial)
- Likelihood: Cannot compute p_G(x)

**VAEs:**
- Maximize: ELBO ≤ log p(x)
- Sample quality: Lower (blurry images)
- Sample diversity: Good (covers all modes)
- Training: Stable (direct optimization)
- Likelihood: Can compute lower bound

**Formal difference:**
- GAN optimizes f-divergence (focuses on high-probability regions)
- VAE optimizes likelihood (penalizes all errors equally)

**Use Cases:**
```
Original GAN:
✗ Hard to train, unstable

Non-saturating GAN:
✓ Better gradients, still can be unstable

WGAN/WGAN-GP:
✓ Image generation (best quality)
✓ More stable training
✓ Meaningful loss values (W_1 distance)
✓ State-of-the-art for GANs
✓ When: Sample quality > sample diversity

VAE:
✓ When: Need stable training
✓ When: Need likelihood estimates
✓ When: Sample diversity critical
```

---

## Convergence Properties Summary

### Convexity and Convergence Rates

```
Loss Type        | Convexity | GD Rate      | SGD Rate     | Notes
-----------------|-----------|--------------|--------------|------------------
MSE (Linear)     | Convex    | O(exp(-k))   | O(1/k)       | Strongly convex
MAE              | Convex    | O(1/√k)      | O(1/√k)      | Non-smooth
Huber            | Convex    | O(exp(-k))   | O(1/k)       | Smooth, robust
Cross-Entropy    | Convex*   | O(1/k)       | O(1/√k)      | *if separable
Hinge            | Convex    | O(1/√k)      | O(1/√k)      | Non-smooth
Triplet          | Non-conv  | -            | O(1/√k)      | Mining crucial
ELBO             | Non-conv  | -            | Local        | VAE
GAN              | Non-conv  | -            | Oscillates   | Nash equilibrium

where k = iteration number
```

### Lipschitz Constants

```
Loss                     | Lipschitz Constant L
-------------------------|--------------------
MSE                      | λ_max(X^T X)
Cross-entropy            | 1/4 (for logistic)
Huber(δ)                | 1 (in linear region)
Hinge                    | 1

For Gradient Descent: α ≤ 1/L for convergence
```

### Practical Convergence Tips

```
1. **MSE**: Fast convergence, watch for outliers
2. **Cross-Entropy**: Standard for classification, use with softmax
3. **Huber**: Best of both worlds for regression
4. **Focal**: Need more iterations than cross-entropy
5. **Triplet**: Mining strategy critical, start easy
6. **ELBO**: KL annealing often helps
7. **GAN**: Use WGAN-GP, multiple critic steps
```

---

## Loss Function Selection Guide

### Decision Tree

```
Task: Regression?
├─ Yes: Outliers present?
│  ├─ Yes: Use Huber or MAE
│  └─ No: Use MSE
│
└─ No (Classification)
   ├─ Binary or Multi-class?
   │  ├─ Binary: Use Binary Cross-Entropy
   │  └─ Multi-class:
   │     ├─ Exclusive classes? Use Categorical Cross-Entropy
   │     └─ Non-exclusive? Use Binary Cross-Entropy per class
   │
   ├─ Class imbalance?
   │  ├─ Severe (>100:1): Use Focal Loss
   │  └─ Moderate: Use weighted Cross-Entropy
   │
   ├─ Need embeddings?
   │  └─ Use Triplet or Contrastive Loss
   │
   └─ Generative model?
      ├─ VAE: Use ELBO
      ├─ GAN: Use WGAN-GP
      └─ Autoregressive: Use Cross-Entropy
```

### Quick Reference

```
Scenario                           | Recommended Loss
-----------------------------------|------------------
Image classification               | Cross-Entropy + Label Smoothing
Object detection                   | Focal Loss (RetinaNet) or Smooth L1 (Faster R-CNN)
Semantic segmentation              | Cross-Entropy or Dice Loss
Image regression                   | MSE or Huber
Face recognition                   | Triplet or ArcFace
Self-supervised learning           | Contrastive (SimCLR)
Speech recognition                 | CTC
Machine translation                | Cross-Entropy (teacher forcing)
VAE                               | ELBO (reconstruction + KL)
GAN                               | WGAN-GP
Reinforcement learning             | Policy Gradient + Value Loss (A2C/PPO)
Ranking/Retrieval                  | Triplet or Listwise
Anomaly detection                  | Reconstruction error (MSE/MAE)
Time series forecasting            | MSE, MAE, or Quantile Loss
Ordinal regression                 | Ordinal Cross-Entropy
Multi-task learning                | Weighted sum + uncertainty weighting
```

---

## Practical Implementations of Theory

### Computing Proper Scoring Rules

**Implementation of proper scoring rules and calibration metrics:**

```python
import numpy as np
from scipy.stats import chi2
from scipy.special import softmax

def log_score(y_true, y_pred_proba):
    """
    Logarithmic scoring rule (negative log-likelihood).
    Strictly proper scoring rule.

    Args:
        y_true: Binary labels (0 or 1), shape (n,)
        y_pred_proba: Predicted probabilities for class 1, shape (n,)

    Returns:
        score: Average log score (lower is better)
    """
    epsilon = 1e-15  # Avoid log(0)
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)

    scores = -np.where(y_true == 1,
                       np.log(y_pred_proba),
                       np.log(1 - y_pred_proba))

    return np.mean(scores)

def brier_score(y_true, y_pred_proba):
    """
    Brier score (mean squared error of probabilities).
    Strictly proper scoring rule.

    Args:
        y_true: Binary labels (0 or 1), shape (n,)
        y_pred_proba: Predicted probabilities for class 1, shape (n,)

    Returns:
        score: Brier score (lower is better)
    """
    return np.mean((y_pred_proba - y_true) ** 2)

def spherical_score(y_true, y_pred_proba):
    """
    Spherical scoring rule.
    Proper scoring rule that normalizes predictions.

    Args:
        y_true: Binary labels (0 or 1), shape (n,)
        y_pred_proba: Predicted probabilities for class 1, shape (n,)

    Returns:
        score: Negative spherical score (lower is better)
    """
    # For binary: p' = [1-p, p], y' = [1-y, y]
    p0 = 1 - y_pred_proba
    p1 = y_pred_proba
    norm = np.sqrt(p0**2 + p1**2)

    scores = np.where(y_true == 1, p1 / norm, p0 / norm)

    return -np.mean(scores)  # Negative because higher is better

def expected_calibration_error(y_true, y_pred_proba, n_bins=10):
    """
    Expected Calibration Error (ECE).
    Measures calibration by binning predictions.

    Args:
        y_true: True labels, shape (n,)
        y_pred_proba: Predicted probabilities, shape (n,)
        n_bins: Number of bins for calibration

    Returns:
        ece: Expected calibration error
        bin_stats: Dictionary with per-bin statistics
    """
    # Create bins
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred_proba, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    bin_stats = []

    for i in range(n_bins):
        mask = bin_indices == i

        if np.sum(mask) == 0:
            continue

        bin_size = np.sum(mask)
        bin_confidence = np.mean(y_pred_proba[mask])
        bin_accuracy = np.mean(y_true[mask])

        bin_error = np.abs(bin_confidence - bin_accuracy)
        ece += (bin_size / len(y_true)) * bin_error

        bin_stats.append({
            'bin': i,
            'count': bin_size,
            'confidence': bin_confidence,
            'accuracy': bin_accuracy,
            'error': bin_error
        })

    return ece, bin_stats

def maximum_calibration_error(y_true, y_pred_proba, n_bins=10):
    """
    Maximum Calibration Error (MCE).
    Worst-case calibration across bins.

    Args:
        y_true: True labels, shape (n,)
        y_pred_proba: Predicted probabilities, shape (n,)
        n_bins: Number of bins

    Returns:
        mce: Maximum calibration error
    """
    _, bin_stats = expected_calibration_error(y_true, y_pred_proba, n_bins)

    if len(bin_stats) == 0:
        return 0.0

    return max(stat['error'] for stat in bin_stats)

def temperature_scaling(logits, y_true, T_init=1.0, lr=0.01, max_iter=100):
    """
    Temperature scaling for model calibration.
    Optimizes temperature T to minimize NLL on validation set.

    Args:
        logits: Model logits before softmax, shape (n, n_classes)
        y_true: True class labels, shape (n,)
        T_init: Initial temperature
        lr: Learning rate
        max_iter: Maximum iterations

    Returns:
        T_optimal: Optimal temperature
        nll_history: NLL at each iteration
    """
    T = T_init
    nll_history = []

    for iteration in range(max_iter):
        # Compute probabilities with current temperature
        scaled_logits = logits / T
        probs = softmax(scaled_logits, axis=1)

        # Compute negative log-likelihood
        epsilon = 1e-15
        probs = np.clip(probs, epsilon, 1 - epsilon)
        nll = -np.mean(np.log(probs[np.arange(len(y_true)), y_true]))
        nll_history.append(nll)

        # Gradient of NLL w.r.t. T
        # ∂NLL/∂T = (1/T²) Σ_i (z_i - z_{y_i})
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(len(y_true)), y_true] = 1

        grad_T = np.mean(np.sum((probs - one_hot) * logits, axis=1)) / (T ** 2)

        # Update temperature
        T -= lr * grad_T
        T = max(T, 0.01)  # Ensure T > 0

        # Check convergence
        if iteration > 0 and abs(nll_history[-1] - nll_history[-2]) < 1e-6:
            break

    return T, nll_history

# Example usage
if __name__ == "__main__":
    # Generate example data
    np.random.seed(42)
    n_samples = 1000

    # Ground truth labels
    y_true = np.random.binomial(1, 0.3, n_samples)

    # Overconfident predictions (miscalibrated)
    y_pred_overconfident = np.where(y_true == 1,
                                     np.random.beta(8, 2, n_samples),
                                     np.random.beta(2, 8, n_samples))

    # Well-calibrated predictions
    y_pred_calibrated = np.where(y_true == 1,
                                  np.random.beta(3, 2, n_samples),
                                  np.random.beta(2, 3, n_samples))

    print("=== Proper Scoring Rules ===")
    print(f"\nOverconfident model:")
    print(f"  Log score: {log_score(y_true, y_pred_overconfident):.4f}")
    print(f"  Brier score: {brier_score(y_true, y_pred_overconfident):.4f}")
    print(f"  Spherical score: {spherical_score(y_true, y_pred_overconfident):.4f}")

    print(f"\nCalibrated model:")
    print(f"  Log score: {log_score(y_true, y_pred_calibrated):.4f}")
    print(f"  Brier score: {brier_score(y_true, y_pred_calibrated):.4f}")
    print(f"  Spherical score: {spherical_score(y_true, y_pred_calibrated):.4f}")

    print("\n=== Calibration Metrics ===")
    ece_over, _ = expected_calibration_error(y_true, y_pred_overconfident)
    mce_over = maximum_calibration_error(y_true, y_pred_overconfident)

    ece_cal, _ = expected_calibration_error(y_true, y_pred_calibrated)
    mce_cal = maximum_calibration_error(y_true, y_pred_calibrated)

    print(f"\nOverconfident model:")
    print(f"  ECE: {ece_over:.4f}")
    print(f"  MCE: {mce_over:.4f}")

    print(f"\nCalibrated model:")
    print(f"  ECE: {ece_cal:.4f}")
    print(f"  MCE: {mce_cal:.4f}")
```

### Computing f-Divergences

**Implementation of various f-divergences:**

```python
import numpy as np
from scipy.special import kl_div, rel_entr

def compute_kl_divergence(P, Q, epsilon=1e-10):
    """
    Kullback-Leibler divergence KL(P || Q).

    Args:
        P: True distribution, shape (n,)
        Q: Approximate distribution, shape (n,)
        epsilon: Small value to avoid log(0)

    Returns:
        kl: KL divergence (nats)
    """
    P = np.asarray(P) + epsilon
    Q = np.asarray(Q) + epsilon

    # Normalize
    P = P / np.sum(P)
    Q = Q / np.sum(Q)

    return np.sum(rel_entr(P, Q))

def compute_reverse_kl(P, Q, epsilon=1e-10):
    """
    Reverse KL divergence KL(Q || P).

    Args:
        P: First distribution, shape (n,)
        Q: Second distribution, shape (n,)
        epsilon: Small value to avoid log(0)

    Returns:
        rkl: Reverse KL divergence (nats)
    """
    return compute_kl_divergence(Q, P, epsilon)

def compute_js_divergence(P, Q, epsilon=1e-10):
    """
    Jensen-Shannon divergence JS(P || Q).
    Symmetric and bounded version of KL divergence.

    Args:
        P: First distribution, shape (n,)
        Q: Second distribution, shape (n,)
        epsilon: Small value to avoid log(0)

    Returns:
        js: Jensen-Shannon divergence (nats)
    """
    P = np.asarray(P) + epsilon
    Q = np.asarray(Q) + epsilon

    # Normalize
    P = P / np.sum(P)
    Q = Q / np.sum(Q)

    # Mixture distribution
    M = 0.5 * (P + Q)

    return 0.5 * compute_kl_divergence(P, M, 0) + 0.5 * compute_kl_divergence(Q, M, 0)

def compute_hellinger_distance(P, Q, epsilon=1e-10):
    """
    Hellinger distance H(P, Q).
    Metric derived from squared Hellinger divergence.

    Args:
        P: First distribution, shape (n,)
        Q: Second distribution, shape (n,)
        epsilon: Small value for numerical stability

    Returns:
        hellinger: Hellinger distance [0, 1]
    """
    P = np.asarray(P) + epsilon
    Q = np.asarray(Q) + epsilon

    # Normalize
    P = P / np.sum(P)
    Q = Q / np.sum(Q)

    # Hellinger distance: (1/√2)||√P - √Q||
    return np.sqrt(np.sum((np.sqrt(P) - np.sqrt(Q)) ** 2)) / np.sqrt(2)

def compute_total_variation(P, Q, epsilon=1e-10):
    """
    Total Variation distance TV(P, Q).

    Args:
        P: First distribution, shape (n,)
        Q: Second distribution, shape (n,)
        epsilon: Small value for numerical stability

    Returns:
        tv: Total variation distance [0, 1]
    """
    P = np.asarray(P) + epsilon
    Q = np.asarray(Q) + epsilon

    # Normalize
    P = P / np.sum(P)
    Q = Q / np.sum(Q)

    return 0.5 * np.sum(np.abs(P - Q))

def compute_chi_squared(P, Q, epsilon=1e-10):
    """
    Chi-squared divergence χ²(P || Q).

    Args:
        P: True distribution, shape (n,)
        Q: Approximate distribution, shape (n,)
        epsilon: Small value to avoid division by zero

    Returns:
        chi_sq: Chi-squared divergence
    """
    P = np.asarray(P) + epsilon
    Q = np.asarray(Q) + epsilon

    # Normalize
    P = P / np.sum(P)
    Q = Q / np.sum(Q)

    return np.sum((P - Q) ** 2 / Q)

def compute_f_divergence(P, Q, f_func, epsilon=1e-10):
    """
    General f-divergence computation.

    Args:
        P: True distribution, shape (n,)
        Q: Approximate distribution, shape (n,)
        f_func: Convex function f(t) with f(1) = 0
        epsilon: Small value for numerical stability

    Returns:
        div: f-divergence
    """
    P = np.asarray(P) + epsilon
    Q = np.asarray(Q) + epsilon

    # Normalize
    P = P / np.sum(P)
    Q = Q / np.sum(Q)

    # Compute t = P(x)/Q(x) for each x
    t = P / Q

    # D_f(P || Q) = Σ Q(x) f(P(x)/Q(x))
    return np.sum(Q * f_func(t))

# Example usage
if __name__ == "__main__":
    # Example distributions
    P = np.array([0.4, 0.3, 0.2, 0.1])
    Q = np.array([0.3, 0.3, 0.25, 0.15])

    print("=== f-Divergences between P and Q ===")
    print(f"P = {P}")
    print(f"Q = {Q}")
    print()

    print(f"KL(P || Q):         {compute_kl_divergence(P, Q):.6f} nats")
    print(f"KL(Q || P):         {compute_reverse_kl(P, Q):.6f} nats")
    print(f"JS(P || Q):         {compute_js_divergence(P, Q):.6f} nats")
    print(f"Hellinger(P, Q):    {compute_hellinger_distance(P, Q):.6f}")
    print(f"TV(P, Q):           {compute_total_variation(P, Q):.6f}")
    print(f"χ²(P || Q):         {compute_chi_squared(P, Q):.6f}")

    print("\n=== Custom f-divergences ===")

    # KL: f(t) = t log t
    kl_custom = compute_f_divergence(P, Q, lambda t: t * np.log(t))
    print(f"KL (via f-div):     {kl_custom:.6f} nats")

    # Reverse KL: f(t) = -log t
    rkl_custom = compute_f_divergence(P, Q, lambda t: -np.log(t))
    print(f"Reverse KL (via f): {rkl_custom:.6f} nats")

    # Chi-squared: f(t) = (t - 1)²
    chi_custom = compute_f_divergence(P, Q, lambda t: (t - 1) ** 2)
    print(f"χ² (via f-div):     {chi_custom:.6f}")
```

### Visualizing Surrogate Losses

**Compare surrogate losses and their relationship to 0-1 loss:**

```python
import numpy as np
import matplotlib.pyplot as plt

def zero_one_loss(margin):
    """0-1 loss: 𝟙[margin ≤ 0]"""
    return (margin <= 0).astype(float)

def hinge_loss(margin):
    """Hinge loss: max(0, 1 - margin)"""
    return np.maximum(0, 1 - margin)

def logistic_loss(margin):
    """Logistic loss: log(1 + exp(-margin))"""
    return np.log(1 + np.exp(-margin))

def exponential_loss(margin):
    """Exponential loss: exp(-margin)"""
    return np.exp(-margin)

def squared_loss(margin):
    """Squared loss: (1 - margin)²"""
    return (1 - margin) ** 2

# Plot comparison
margin = np.linspace(-2, 3, 500)

plt.figure(figsize=(12, 7))

plt.plot(margin, zero_one_loss(margin), 'k-', linewidth=2, label='0-1 loss')
plt.plot(margin, hinge_loss(margin), 'r-', linewidth=2, label='Hinge')
plt.plot(margin, logistic_loss(margin), 'b-', linewidth=2, label='Logistic')
plt.plot(margin, exponential_loss(margin), 'g-', linewidth=2, label='Exponential')
plt.plot(margin, squared_loss(margin), 'm-', linewidth=2, label='Squared')

plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=1, color='gray', linestyle=':', alpha=0.3, label='Margin = 1')

plt.xlabel('Margin (y·f(x))', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Surrogate Losses for Binary Classification', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.ylim(-0.1, 3)
plt.xlim(-2, 3)

plt.tight_layout()
plt.savefig('surrogate_losses.png', dpi=300, bbox_inches='tight')
print("Saved surrogate_losses.png")

print("\n=== Loss Values at Key Margins ===")
margins_test = [-1, 0, 0.5, 1, 2]

for m in margins_test:
    print(f"\nMargin = {m}:")
    print(f"  0-1:         {zero_one_loss(np.array([m]))[0]:.4f}")
    print(f"  Hinge:       {hinge_loss(np.array([m]))[0]:.4f}")
    print(f"  Logistic:    {logistic_loss(np.array([m]))[0]:.4f}")
    print(f"  Exponential: {exponential_loss(np.array([m]))[0]:.4f}")
    print(f"  Squared:     {squared_loss(np.array([m]))[0]:.4f}")
```

## Custom Loss Functions

### Implementing Custom Losses

**PyTorch Template:**
```python
import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    """Template for custom loss function"""

    def __init__(self, param1=1.0, param2=0.5):
        super().__init__()
        self.param1 = param1
        self.param2 = param2

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: Model predictions (batch_size, ...)
            y_true: Ground truth (batch_size, ...)

        Returns:
            loss: Scalar loss value
        """
        # Implement loss computation
        loss = self.compute_loss(y_pred, y_true)

        return loss

    def compute_loss(self, y_pred, y_true):
        """Core loss computation"""
        # Example: Weighted MSE + MAE
        mse = torch.mean((y_pred - y_true) ** 2)
        mae = torch.mean(torch.abs(y_pred - y_true))

        loss = self.param1 * mse + self.param2 * mae
        return loss

# Usage
criterion = CustomLoss(param1=0.7, param2=0.3)
loss = criterion(predictions, targets)
loss.backward()
```

**TensorFlow/Keras Template:**
```python
import tensorflow as tf

def custom_loss(param1=1.0, param2=0.5):
    """Factory function for custom loss"""

    def loss_fn(y_true, y_pred):
        """
        Args:
            y_true: Ground truth tensor
            y_pred: Prediction tensor

        Returns:
            loss: Scalar tensor
        """
        # Ensure same shape
        y_true = tf.cast(y_true, y_pred.dtype)

        # Compute loss components
        mse = tf.reduce_mean(tf.square(y_pred - y_true))
        mae = tf.reduce_mean(tf.abs(y_pred - y_true))

        # Combine
        loss = param1 * mse + param2 * mae

        return loss

    return loss_fn

# Usage
model.compile(
    optimizer='adam',
    loss=custom_loss(param1=0.7, param2=0.3)
)
```

### Advanced: Composite Losses

**Example: Perceptual Loss (for Image Generation)**
```python
class PerceptualLoss(nn.Module):
    """Combines pixel-level and perceptual losses"""

    def __init__(self, feature_extractor, layer_weights=None):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.layer_weights = layer_weights or [1.0, 1.0, 1.0]

        # Freeze feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, generated, target):
        # Pixel loss (L1)
        pixel_loss = torch.mean(torch.abs(generated - target))

        # Feature loss
        gen_features = self.feature_extractor(generated)
        target_features = self.feature_extractor(target)

        feature_loss = 0
        for i, (gen_feat, target_feat) in enumerate(zip(gen_features, target_features)):
            feature_loss += self.layer_weights[i] * torch.mean(
                torch.abs(gen_feat - target_feat)
            )

        # Total loss
        total_loss = pixel_loss + 0.1 * feature_loss

        return total_loss, pixel_loss, feature_loss
```

### Validation and Testing

**Loss Sanity Checks:**
```python
def test_loss_function(loss_fn):
    """Test properties of custom loss"""

    # 1. Test gradient flow
    y_pred = torch.randn(10, 5, requires_grad=True)
    y_true = torch.randn(10, 5)

    loss = loss_fn(y_pred, y_true)
    loss.backward()

    assert y_pred.grad is not None, "No gradient!"
    print(f"✓ Gradient flows: {y_pred.grad.shape}")

    # 2. Test perfect prediction
    y_pred = y_true.clone().detach().requires_grad_(True)
    loss = loss_fn(y_pred, y_true)

    assert loss.item() < 1e-6, f"Loss should be ~0 for perfect pred: {loss.item()}"
    print(f"✓ Perfect prediction loss: {loss.item():.6f}")

    # 3. Test batch invariance
    single_loss = loss_fn(y_pred[:1], y_true[:1])
    batch_loss = loss_fn(y_pred, y_true)

    # Should be similar (unless using batch statistics)
    print(f"✓ Single vs batch loss: {single_loss.item():.4f} vs {batch_loss.item():.4f}")

    # 4. Test range
    random_pred = torch.randn_like(y_true)
    random_loss = loss_fn(random_pred, y_true)

    assert random_loss.item() > 0, "Loss should be positive"
    print(f"✓ Random prediction loss: {random_loss.item():.4f}")

# Test
test_loss_function(nn.MSELoss())
```

---

## References

### Regression Losses

1. **Huber, P. J.** (1964). "Robust estimation of a location parameter." *The Annals of Mathematical Statistics*, 35(1), 73-101.
   - Huber loss derivation

2. **Koenker, R., & Bassett Jr, G.** (1978). "Regression quantiles." *Econometrica*, 46(1), 33-50.
   - Quantile regression and pinball loss

### Classification Losses

3. **Bishop, C. M.** (2006). *Pattern recognition and machine learning*. Springer.
   - Cross-entropy and probabilistic losses

4. **Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P.** (2017). "Focal loss for dense object detection." *ICCV*.
   - Focal loss for class imbalance

5. **Müller, R., Kornblith, S., & Hinton, G. E.** (2019). "When does label smoothing help?" *NeurIPS*.
   - Label smoothing analysis

### Metric Learning

6. **Schroff, F., Kalinichenko, D., & Philbin, J.** (2015). "FaceNet: A unified embedding for face recognition and clustering." *CVPR*.
   - Triplet loss for face recognition

7. **Chen, T., Kornblith, S., Norouzi, M., & Hinton, G.** (2020). "A simple framework for contrastive learning of visual representations." *ICML*.
   - SimCLR contrastive loss

### Generative Models

8. **Kingma, D. P., & Welling, M.** (2014). "Auto-encoding variational bayes." *ICLR*.
   - ELBO and VAE

9. **Arjovsky, M., Chintala, S., & Bottou, L.** (2017). "Wasserstein generative adversarial networks." *ICML*.
   - Wasserstein loss for GANs

10. **Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. C.** (2017). "Improved training of wasserstein gans." *NeurIPS*.
    - WGAN with gradient penalty

### Theoretical Foundations

11. **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). *Deep learning*. MIT Press.
    - Comprehensive deep learning theory

12. **Bottou, L., Curtis, F. E., & Nocedal, J.** (2018). "Optimization methods for large-scale machine learning." *SIAM Review*, 60(2), 223-311.
    - Optimization and convergence analysis

---

*This comprehensive guide covers the mathematical foundations, convergence properties, and practical considerations for loss functions across all major machine learning tasks.*
