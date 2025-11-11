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

**Use Cases:**
```
✓ Object detection (RetinaNet)
✓ Extreme class imbalance
✓ When: Hard examples are important
✓ When: Easy negatives dominate
✗ When: Classes are balanced (use cross-entropy)
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

**Use Cases:**
```
✓ Face recognition (FaceNet)
✓ Metric learning
✓ Image retrieval
✓ When: Need embedding space with metric properties
✗ When: Number of classes is small (use softmax)
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

**Use Cases:**
```
✓ Self-supervised learning (SimCLR, MoCo)
✓ Representation learning
✓ When: Have pairs of similar/dissimilar samples
✓ When: Unsupervised or semi-supervised setting
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

**Use Cases:**
```
✓ Speech recognition
✓ OCR (handwriting recognition)
✓ When: Input and output lengths differ
✓ When: Alignment is unknown
✗ When: Need attention mechanism (use seq2seq instead)
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

**Use Cases:**
```
Original GAN:
✗ Hard to train, unstable

Non-saturating GAN:
✓ Better gradients, still can be unstable

WGAN/WGAN-GP:
✓ Image generation
✓ More stable training
✓ Meaningful loss values
✓ State-of-the-art for GANs
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
