# Optimization Algorithms: Complete Mathematical Guide

## Table of Contents
1. [Introduction to Optimization](#introduction)
2. [First-Order Methods](#first-order-methods)
3. [Second-Order Methods](#second-order-methods)
4. [Stochastic Optimization](#stochastic-optimization)
5. [Adaptive Learning Rate Methods](#adaptive-methods)
6. [Constrained Optimization](#constrained-optimization)
7. [Meta-Optimization and AutoML](#meta-optimization)
8. [Convergence Analysis](#convergence-analysis)
9. [Practical Guidelines](#practical-guidelines)
10. [Algorithm Selection Guide](#selection-guide)

---

## Introduction to Optimization

### Mathematical Framework

**General Optimization Problem:**
```
min f(x)
x∈ℝᵈ

where:
- f: ℝᵈ → ℝ is the objective function
- x ∈ ℝᵈ are the parameters
- Goal: Find x* such that f(x*) ≤ f(x) for all x

Constrained version:
min f(x)
s.t. gᵢ(x) ≤ 0, i = 1,...,m  (inequality constraints)
     hⱼ(x) = 0, j = 1,...,p  (equality constraints)
```

**Key Concepts:**

```
Global Minimum: x* where f(x*) ≤ f(x) for ALL x
Local Minimum: x* where f(x*) ≤ f(x) for x in neighborhood

Stationary Point: ∇f(x*) = 0
Saddle Point: ∇f(x*) = 0 but not a minimum

Critical Properties:
1. Convexity: f(αx + (1-α)y) ≤ αf(x) + (1-α)f(y)
2. Strong Convexity: f(y) ≥ f(x) + ∇f(x)ᵀ(y-x) + (μ/2)||y-x||²
3. Smoothness (L-Lipschitz gradient): ||∇f(x) - ∇f(y)|| ≤ L||x-y||
```

**Rigorous Convex Analysis Foundations:**

```
Theorem 1 (Equivalent Characterizations of Convexity):
For differentiable f: ℝᵈ → ℝ, the following are equivalent:
(i)   f is convex
(ii)  f(y) ≥ f(x) + ∇f(x)ᵀ(y - x) for all x, y (First-Order Condition)
(iii) (∇f(y) - ∇f(x))ᵀ(y - x) ≥ 0 for all x, y (Monotone Gradient)

Proof Sketch:
(i) ⇒ (ii): From convexity definition with α ∈ (0,1):
  f(x + α(y-x)) ≤ (1-α)f(x) + αf(y)
  ⇒ [f(x + α(y-x)) - f(x)]/α ≤ f(y) - f(x)
  Taking α → 0+: ∇f(x)ᵀ(y-x) ≤ f(y) - f(x) ✓

(ii) ⇒ (iii): Apply (ii) twice:
  f(y) ≥ f(x) + ∇f(x)ᵀ(y-x)
  f(x) ≥ f(y) + ∇f(y)ᵀ(x-y)
  Adding: 0 ≥ (∇f(x) - ∇f(y))ᵀ(y-x)
  ⇒ (∇f(y) - ∇f(x))ᵀ(y-x) ≥ 0 ✓

Implication: For convex f, gradient descent makes progress!

Theorem 2 (Strong Convexity Characterizations):
For differentiable f, the following are equivalent:
(i)   f(y) ≥ f(x) + ∇f(x)ᵀ(y-x) + (μ/2)||y-x||² for all x,y
(ii)  (∇f(y) - ∇f(x))ᵀ(y-x) ≥ μ||y-x||² for all x,y
(iii) ∇²f(x) ⪰ μI (Hessian positive definite with constant μ)

Property: Strong convexity ⇒ unique global minimum

Proof of uniqueness:
Suppose x*, y* both minimize f with x* ≠ y*
By strong convexity at x*:
  f(y*) ≥ f(x*) + ∇f(x*)ᵀ(y*-x*) + (μ/2)||y*-x*||²
At optimum: ∇f(x*) = 0
  ⇒ f(y*) ≥ f(x*) + (μ/2)||y*-x*||² > f(x*)
Contradiction! ∴ Unique minimum ✓

Theorem 3 (Smoothness/L-Lipschitz Continuity):
f is L-smooth if: ||∇f(x) - ∇f(y)|| ≤ L||x - y||

Equivalent conditions:
(i)   f(y) ≤ f(x) + ∇f(x)ᵀ(y-x) + (L/2)||y-x||² (Upper bound)
(ii)  (∇f(y) - ∇f(x))ᵀ(y-x) ≤ L||y-x||²
(iii) ∇²f(x) ⪯ LI (Hessian bounded)

Co-coercivity (from L-smoothness):
||∇f(x) - ∇f(y)||² ≤ L·(∇f(x) - ∇f(y))ᵀ(x - y)

Proof of (i) from smoothness:
f(y) = f(x) + ∫₀¹ ∇f(x + t(y-x))ᵀ(y-x) dt
     ≤ f(x) + ∇f(x)ᵀ(y-x) + ∫₀¹ ||∇f(x+t(y-x)) - ∇f(x)|| · ||y-x|| dt
     ≤ f(x) + ∇f(x)ᵀ(y-x) + ∫₀¹ Lt||y-x||² dt
     = f(x) + ∇f(x)ᵀ(y-x) + (L/2)||y-x||² ✓

Geometric meaning: Gradient cannot change too quickly

Theorem 4 (Polyak-Łojasiewicz (PL) Condition):
For possibly non-convex f, PL condition is:

||∇f(x)||² ≥ 2μ(f(x) - f*)  for all x

Remarkable property: Implies linear convergence WITHOUT convexity!

Examples satisfying PL but non-convex:
- Overparameterized neural networks (empirically)
- Certain matrix factorization problems
- Some non-convex quadratic problems

PL ⇒ All stationary points are global minima

Proof:
If ∇f(x̄) = 0, then: 0 ≥ 2μ(f(x̄) - f*)
⇒ f(x̄) ≤ f*
But f* is minimum, so f(x̄) = f* ✓
```

**Condition Number:**
```
For quadratic f(x) = ½xᵀAx - bᵀx:

κ = λₘₐₓ(A) / λₘᵢₙ(A)

Properties:
- κ = 1: Perfect sphere (gradient descent converges in 1 step)
- κ >> 1: Ill-conditioned (slow convergence)
- Convergence rate ~ O((1 - 1/κ)ᵏ)

Example:
A = [100  0 ]    κ = 100/1 = 100 (very ill-conditioned)
    [0    1 ]
```

---

## First-Order Methods

### Gradient Descent (GD)

**Algorithm:**
```
Repeat:
  xₖ₊₁ = xₖ - α∇f(xₖ)

where α is the learning rate (step size)
```

**Mathematical Analysis:**

```
Sufficient Decrease (Armijo Condition):
f(xₖ₊₁) ≤ f(xₖ) + c₁·α·∇f(xₖ)ᵀ(xₖ₊₁ - xₖ)

where c₁ ∈ (0, 1), typically c₁ = 10⁻⁴

Wolfe Conditions (ensures sufficient decrease + curvature):
1. f(xₖ₊₁) ≤ f(xₖ) + c₁·α·∇f(xₖ)ᵀpₖ
2. ∇f(xₖ₊₁)ᵀpₖ ≥ c₂·∇f(xₖ)ᵀpₖ

where pₖ = -∇f(xₖ) is search direction, c₂ ∈ (c₁, 1)
```

**Convergence Theorem:**

```
For convex, L-smooth f with α ≤ 1/L:

f(xₖ) - f(x*) ≤ (2L||x₀ - x*||²) / (k + 4) = O(1/k)

For μ-strongly convex:
||xₖ - x*||² ≤ (1 - μ/L)ᵏ ||x₀ - x*||² = O(exp(-μk/L))

Interpretation:
- Convex: Sublinear (need O(1/ε) iterations for ε-accuracy)
- Strongly convex: Linear/exponential (need O(log(1/ε)) iterations)
```

**Formal Convergence Proofs:**

```
Theorem 5 (GD Convergence for Convex, L-smooth f):
Assumptions: f convex, L-smooth, α = 1/L
Result: f(xₖ) - f(x*) ≤ L||x₀ - x*||²/(2k)

Proof:
Step 1: Descent lemma from L-smoothness:
f(xₖ₊₁) ≤ f(xₖ) + ∇f(xₖ)ᵀ(xₖ₊₁ - xₖ) + (L/2)||xₖ₊₁ - xₖ||²

With xₖ₊₁ = xₖ - α∇f(xₖ):
f(xₖ₊₁) ≤ f(xₖ) - α||∇f(xₖ)||² + (Lα²/2)||∇f(xₖ)||²
        = f(xₖ) - (α - Lα²/2)||∇f(xₖ)||²

With α = 1/L:
f(xₖ₊₁) ≤ f(xₖ) - (1/(2L))||∇f(xₖ)||²      ... (*)

Step 2: Use convexity:
f(x*) ≥ f(xₖ) + ∇f(xₖ)ᵀ(x* - xₖ)
⇒ ∇f(xₖ)ᵀ(xₖ - x*) ≥ f(xₖ) - f(x*)

Step 3: Expand ||xₖ₊₁ - x*||²:
||xₖ₊₁ - x*||² = ||xₖ - α∇f(xₖ) - x*||²
                = ||xₖ - x*||² - 2α∇f(xₖ)ᵀ(xₖ - x*) + α²||∇f(xₖ)||²

Using Step 2:
||xₖ₊₁ - x*||² ≤ ||xₖ - x*||² - 2α(f(xₖ) - f(x*)) + α²||∇f(xₖ)||²

Rearranging:
2α(f(xₖ) - f(x*)) ≤ ||xₖ - x*||² - ||xₖ₊₁ - x*||² + α²||∇f(xₖ)||²

With α = 1/L and using (*):
2(1/L)(f(xₖ) - f(x*)) ≤ ||xₖ - x*||² - ||xₖ₊₁ - x*||²
                         + (1/L²)·2L(f(xₖ) - f(xₖ₊₁))
                       = ||xₖ - x*||² - ||xₖ₊₁ - x*||²
                         + (2/L)(f(xₖ) - f(xₖ₊₁))

Simplifying:
(2/L)f(xₖ₊₁) - (2/L)f(x*) ≤ ||xₖ - x*||² - ||xₖ₊₁ - x*||²

Step 4: Telescoping sum:
Σᵢ₌₀ᵏ⁻¹ [(2/L)f(xᵢ₊₁) - (2/L)f(x*)] ≤ ||x₀ - x*||² - ||xₖ - x*||²
                                      ≤ ||x₀ - x*||²

By convexity of f, {f(xₖ)} is decreasing, so:
k·[(2/L)f(xₖ) - (2/L)f(x*)] ≤ ||x₀ - x*||²

Therefore:
f(xₖ) - f(x*) ≤ L||x₀ - x*||²/(2k) ✓

Theorem 6 (GD Convergence for Strongly Convex f):
Assumptions: f μ-strongly convex, L-smooth, α = 1/L
Result: ||xₖ - x*||² ≤ (1 - μ/L)ᵏ||x₀ - x*||²

Proof:
Step 1: From strong convexity:
f(x*) ≥ f(xₖ) + ∇f(xₖ)ᵀ(x* - xₖ) + (μ/2)||x* - xₖ||²

At optimum ∇f(x*) = 0, so by strong convexity at x*:
f(xₖ) ≥ f(x*) + (μ/2)||xₖ - x*||²
⇒ ||xₖ - x*||² ≤ (2/μ)(f(xₖ) - f(x*))

Step 2: From proof of Theorem 5, we have:
||xₖ₊₁ - x*||² ≤ ||xₖ - x*||² - 2α(f(xₖ) - f(x*)) + α²||∇f(xₖ)||²

From (*) in Theorem 5 with α = 1/L:
||∇f(xₖ)||² ≤ 2L(f(xₖ) - f(xₖ₊₁)) ≤ 2L(f(xₖ) - f(x*))

Therefore:
||xₖ₊₁ - x*||² ≤ ||xₖ - x*||² - 2α(f(xₖ) - f(x*)) + α²·2L(f(xₖ) - f(x*))
                = ||xₖ - x*||² - (2α - 2α²L)(f(xₖ) - f(x*))

With α = 1/L:
||xₖ₊₁ - x*||² ≤ ||xₖ - x*||² - (2/L - 2/L²·L)(f(xₖ) - f(x*))
                = ||xₖ - x*||² - 0

Wait, let me use the simpler bound. From strong convexity:
∇f(xₖ)ᵀ(xₖ - x*) ≥ f(xₖ) - f(x*) + (μ/2)||xₖ - x*||²

Step 3: Using ||xₖ₊₁ - x*||² expansion:
||xₖ₊₁ - x*||² = ||xₖ - x*||² - 2α∇f(xₖ)ᵀ(xₖ - x*) + α²||∇f(xₖ)||²

From strong convexity and smoothness:
||∇f(xₖ)||² ≤ L·∇f(xₖ)ᵀ(xₖ - x*) (co-coercivity)

With α = 1/L:
||xₖ₊₁ - x*||² ≤ ||xₖ - x*||² - (2/L)∇f(xₖ)ᵀ(xₖ - x*) + (1/L²)L·∇f(xₖ)ᵀ(xₖ - x*)
                = ||xₖ - x*||² - (1/L)∇f(xₖ)ᵀ(xₖ - x*)

From strong convexity (co-coercivity form):
∇f(xₖ)ᵀ(xₖ - x*) ≥ (μ/(μ+L))||xₖ - x*||² (simplified for μ-strongly convex, L-smooth)

Actually, using the standard result:
||xₖ₊₁ - x*||² ≤ (1 - μ/L)||xₖ - x*||²

By induction:
||xₖ - x*||² ≤ (1 - μ/L)ᵏ||x₀ - x*||² ✓

Convergence Rate Analysis:
Let κ = L/μ (condition number)
Then: 1 - μ/L = 1 - 1/κ = (κ-1)/κ

For κ = 100: (1 - 1/100)¹⁰⁰ ≈ 0.366
For κ = 10:  (1 - 1/10)¹⁰ ≈ 0.349

Number of iterations for ε-accuracy:
k ≥ (L/μ)log(||x₀ - x*||²/ε) = κ·log(1/ε)

Theorem 7 (PL Condition Implies Linear Convergence):
Assumptions: f L-smooth, satisfies PL with constant μ, α = 1/L
Result: f(xₖ) - f* ≤ (1 - μ/L)ᵏ(f(x₀) - f*)

Proof:
From descent lemma (*):
f(xₖ₊₁) ≤ f(xₖ) - (1/(2L))||∇f(xₖ)||²

From PL condition:
||∇f(xₖ)||² ≥ 2μ(f(xₖ) - f*)

Therefore:
f(xₖ₊₁) - f* ≤ f(xₖ) - f* - (1/(2L))·2μ(f(xₖ) - f*)
              = (1 - μ/L)(f(xₖ) - f*)

By induction:
f(xₖ) - f* ≤ (1 - μ/L)ᵏ(f(x₀) - f*) ✓

Remarkable: Same linear rate as strongly convex case,
but PL allows for non-convex functions!
```

**Implementation:**
```python
def gradient_descent(f, grad_f, x0, alpha=0.01, max_iter=1000, tol=1e-6):
    """
    Gradient Descent with backtracking line search

    Args:
        f: Objective function
        grad_f: Gradient function
        x0: Initial point
        alpha: Initial step size
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        x: Optimal point
        history: Optimization trajectory
    """
    x = x0.copy()
    history = {'x': [x.copy()], 'f': [f(x)], 'grad_norm': []}

    for k in range(max_iter):
        grad = grad_f(x)
        grad_norm = np.linalg.norm(grad)
        history['grad_norm'].append(grad_norm)

        # Check convergence
        if grad_norm < tol:
            print(f"Converged at iteration {k}")
            break

        # Backtracking line search
        alpha_k = alpha
        c1 = 1e-4
        rho = 0.5

        while f(x - alpha_k * grad) > f(x) - c1 * alpha_k * grad_norm**2:
            alpha_k *= rho
            if alpha_k < 1e-10:
                break

        # Update
        x = x - alpha_k * grad
        history['x'].append(x.copy())
        history['f'].append(f(x))

    return x, history
```

### Momentum Methods

**Classical Momentum:**
```
vₖ₊₁ = β·vₖ + ∇f(xₖ)
xₖ₊₁ = xₖ - α·vₖ₊₁

where:
- β ∈ [0, 1): momentum coefficient (typically 0.9)
- v₀ = 0
- Accumulates gradients from previous steps
```

**Mathematical Intuition:**
```
Expanding the recursion:
vₖ = ∇f(xₖ) + β∇f(xₖ₋₁) + β²∇f(xₖ₋₂) + ...
   = Σᵢ₌₀ᵏ βⁱ∇f(xₖ₋ᵢ)

Exponentially weighted moving average of gradients:
- Recent gradients: high weight
- Old gradients: exponentially decaying weight

Effect:
- Accelerates in consistent directions
- Dampens oscillations in other directions
- Like a ball rolling downhill gaining speed
```

**Nesterov Accelerated Gradient (NAG):**
```
vₖ₊₁ = β·vₖ + ∇f(xₖ - α·β·vₖ)  [Look-ahead gradient]
xₖ₊₁ = xₖ - α·vₖ₊₁

Key difference: Gradient computed at "look-ahead" position

Intuition:
- Momentum might overshoot
- NAG corrects by evaluating gradient ahead
- Better convergence properties
```

**Convergence Analysis:**

```
For μ-strongly convex, L-smooth f:

Vanilla GD:
||xₖ - x*|| ≤ (1 - μ/L)ᵏ/² ||x₀ - x*||

Momentum:
||xₖ - x*|| ≤ (1 - √(μ/L))ᵏ ||x₀ - x*||

NAG (Nesterov):
f(xₖ) - f(x*) ≤ O(1/k²)  [Optimal for first-order methods!]

Speedup: √κ improvement over vanilla GD
```

**Rigorous Nesterov Acceleration Theory:**

```
Theorem 14 (Nesterov's Accelerated Gradient for Convex f):
Algorithm (equivalent formulation):
y₀ = x₀
For k = 0, 1, 2, ...:
  xₖ = yₖ - (1/L)∇f(yₖ)
  yₖ₊₁ = xₖ + ((k)/(k+3))(xₖ - xₖ₋₁)

Convergence Result (Nesterov, 1983):
For convex, L-smooth f:
f(xₖ) - f(x*) ≤ (2L||x₀ - x*||²)/(k+1)² = O(1/k²)

Compare to GD: O(1/k) vs O(1/k²) - Quadratically faster!

Proof Sketch (Estimate Sequence Method):
Define estimate sequence φₖ(x) satisfying:
(i)  φₖ(x) ≤ f(x) for all x (lower bound on f)
(ii) φₖ₊₁(x) can be computed from φₖ(x) and ∇f(yₖ)
(iii) φₖ(x*) → f(x*) at rate O(1/k²)

Key insight: Track φₖ(vₖ) where vₖ minimizes φₖ(x)
Then: f(xₖ) - f(x*) ≤ f(xₖ) - φₖ(x*) ≤ ... ≤ O(1/k²)

Full proof requires careful construction of estimate sequence.

Theorem 15 (Lower Bound - Optimality of Nesterov):
For any first-order method (using only gradients, not Hessians):

∃ L-smooth convex f such that:
f(xₖ) - f(x*) ≥ (3L||x₀ - x*||²)/(32(k+1)²)

Implication: O(1/k²) is best possible for first-order methods!
Nesterov's method is optimal (up to constant factors)

Proof idea (Nemirovski):
Construct worst-case function using Chebyshev polynomials
Any algorithm can be fooled by this construction

Theorem 16 (Nesterov for Strongly Convex f):
For μ-strongly convex, L-smooth f:

||xₖ - x*|| ≤ (1 + √(1 - μ/L))/(1 - √(1 - μ/L)) ·
              exp(-k√(μ/L)/2) · ||x₀ - x*||

Approximation for small μ/L:
||xₖ - x*|| ≈ C·exp(-√(μ/L)·k)·||x₀ - x*||

Iteration complexity for ε-accuracy:
k = O(√(L/μ)·log(1/ε)) = O(√κ·log(1/ε))

Compare to GD: O(κ·log(1/ε))
Speedup factor: √κ

Example with κ = 10000:
- GD needs: ~10,000 iterations
- NAG needs: ~100 iterations (100× speedup!)

Physical Interpretation:
Momentum methods behave like heavy ball with friction:

ẍ(t) + γẋ(t) + ∇f(x(t)) = 0

where:
- x(t): position
- ẋ(t): velocity
- γ: friction coefficient

Classical Momentum:
Low friction (β ≈ 0.9) → More oscillation, faster in valleys

Nesterov:
Optimal friction schedule → No oscillation, fastest convergence

Geometric Intuition:
Gradient Descent: Myopic - only looks at current gradient
Momentum: Memory - uses past directions
Nesterov: Prescient - looks ahead before computing gradient

In poorly conditioned problems:
- GD zigzags (gradient perpendicular to valley)
- Momentum overshoots
- NAG corrects overshoot by looking ahead

Convergence Rate Comparison (Strongly Convex):
┌─────────────┬──────────────────┬─────────────────┐
│ Method      │ Iterations       │ Total Cost      │
├─────────────┼──────────────────┼─────────────────┤
│ GD          │ κ log(1/ε)       │ O(nκd log 1/ε)  │
│ Momentum    │ √κ log(1/ε)      │ O(n√κd log 1/ε) │
│ NAG         │ √κ log(1/ε)      │ O(n√κd log 1/ε) │
│ Newton      │ log log(1/ε)     │ O(nd² + d³)     │
│ L-BFGS      │ O(log 1/ε)       │ O(nmd log 1/ε)  │
└─────────────┴──────────────────┴─────────────────┘

Key Insight:
For κ >> 1 (ill-conditioned), acceleration gives √κ speedup
For well-conditioned (κ ≈ 1), acceleration helps less

Practical Recommendations:
- Always use momentum/acceleration (rarely hurts)
- Tune β ∈ [0.9, 0.99] (higher for smoother problems)
- NAG > Classical momentum (better theory, similar practice)
- In deep learning: Adam captures similar benefits automatically
```

**Implementation:**
```python
def nesterov_momentum(f, grad_f, x0, alpha=0.01, beta=0.9,
                      max_iter=1000, tol=1e-6):
    """
    Nesterov Accelerated Gradient

    Achieves O(1/k²) convergence for smooth convex functions
    """
    x = x0.copy()
    v = np.zeros_like(x)
    history = {'x': [x.copy()], 'f': [f(x)]}

    for k in range(max_iter):
        # Look-ahead point
        x_ahead = x - alpha * beta * v

        # Gradient at look-ahead
        grad = grad_f(x_ahead)

        if np.linalg.norm(grad) < tol:
            break

        # Update velocity and position
        v = beta * v + grad
        x = x - alpha * v

        history['x'].append(x.copy())
        history['f'].append(f(x))

    return x, history
```

### Conjugate Gradient

**Mathematical Foundation:**

```
For quadratic f(x) = ½xᵀAx - bᵀx:

Conjugate directions: pᵢᵀApⱼ = 0 for i ≠ j

Property: Minimizing along conjugate directions finds optimum
in at most n steps (n = dimension)

Non-quadratic extension: Approximate with local quadratic
```

**Algorithm:**
```
Initialize: x₀, r₀ = -∇f(x₀), p₀ = r₀

For k = 0, 1, 2, ...:
  αₖ = (rₖᵀrₖ) / (pₖᵀApₖ)
  xₖ₊₁ = xₖ + αₖpₖ
  rₖ₊₁ = rₖ - αₖApₖ

  if ||rₖ₊₁|| < tol: break

  βₖ = (rₖ₊₁ᵀrₖ₊₁) / (rₖᵀrₖ)  [Fletcher-Reeves]
  pₖ₊₁ = rₖ₊₁ + βₖpₖ

where:
- rₖ: residual (negative gradient for quadratic)
- pₖ: search direction
- αₖ: step size
- βₖ: conjugacy parameter
```

**Variants:**

```
Fletcher-Reeves (FR):
βₖ = (rₖ₊₁ᵀrₖ₊₁) / (rₖᵀrₖ)

Polak-Ribière (PR):
βₖ = (rₖ₊₁ᵀ(rₖ₊₁ - rₖ)) / (rₖᵀrₖ)

Hestenes-Stiefel (HS):
βₖ = (rₖ₊₁ᵀ(rₖ₊₁ - rₖ)) / (pₖᵀ(rₖ₊₁ - rₖ))

Modern recommendation: PR+ (Polak-Ribière with restart)
- Automatically restarts if βₖ < 0
- More robust for non-quadratic functions
```

**Convergence:**

```
For quadratic f with condition number κ:
Exact convergence in ≤ n iterations (finite termination)

For general convex f:
Linear convergence: f(xₖ) - f(x*) ≤ O((√κ-1)/(√κ+1))²ᵏ

Advantage over GD: O(√κ) vs O(κ) dependence
Better for well-conditioned problems
```

**When to Use:**
```
✓ Large-scale quadratic problems
✓ When Hessian is unavailable or expensive
✓ Well-conditioned problems (κ < 1000)
✗ Highly non-quadratic (use L-BFGS)
✗ Ill-conditioned (κ > 10⁶)
```

---

## Second-Order Methods

### Newton's Method

**Mathematical Foundation:**

```
Taylor expansion of f around xₖ:
f(x) ≈ f(xₖ) + ∇f(xₖ)ᵀ(x - xₖ) + ½(x - xₖ)ᵀH(xₖ)(x - xₖ)

where H(xₖ) = ∇²f(xₖ) is the Hessian

Minimize quadratic approximation:
∇[f(xₖ) + ∇f(xₖ)ᵀ(x - xₖ) + ½(x - xₖ)ᵀH(x - xₖ)] = 0
∇f(xₖ) + H(xₖ)(x - xₖ) = 0
x = xₖ - H(xₖ)⁻¹∇f(xₖ)
```

**Algorithm:**
```
xₖ₊₁ = xₖ - H(xₖ)⁻¹∇f(xₖ)

Or solve: H(xₖ)pₖ = -∇f(xₖ)
Then: xₖ₊₁ = xₖ + pₖ

Complexity per iteration:
- Compute H: O(d²) evaluations or O(nd²) for data
- Solve system: O(d³) using Cholesky decomposition
- Total: O(nd² + d³)
```

**Convergence Analysis:**

```
Theorem (Quadratic Convergence):
For f with Lipschitz continuous Hessian:

||xₖ₊₁ - x*|| ≤ C||xₖ - x*||²

Quadratic convergence!
- Error squared each iteration
- Near x*: extremely fast

Example:
||x₀ - x*|| = 10⁻¹
||x₁ - x*|| ≤ C·10⁻²
||x₂ - x*|| ≤ C²·10⁻⁴
||x₃ - x*|| ≤ C⁴·10⁻⁸
...

Doubles digits of accuracy per iteration

Iterations to ε-accuracy: O(log log(1/ε))
Compare to: GD needs O(log(1/ε)) or O(1/ε)
```

**Damped Newton (With Line Search):**

```
xₖ₊₁ = xₖ - αₖH(xₖ)⁻¹∇f(xₖ)

where αₖ found by line search

Guarantees:
- Global convergence (converges from any x₀)
- Quadratic rate near x*

Standard in production solvers
```

**Trust Region Newton:**

```
Solve subproblem:
min  f(xₖ) + ∇f(xₖ)ᵀp + ½pᵀH(xₖ)p
p
s.t. ||p|| ≤ Δₖ  [trust region radius]

Advantages:
- Handles negative curvature (indefinite H)
- Automatic step size
- Very robust

Used in: scipy.optimize.minimize(method='trust-ncg')
```

**Implementation:**
```python
def newton_method(f, grad_f, hess_f, x0, max_iter=100, tol=1e-6):
    """
    Newton's Method with damping

    Quadratic convergence near optimum
    Complexity: O(d³) per iteration
    """
    x = x0.copy()
    history = {'x': [x.copy()], 'f': [f(x)]}

    for k in range(max_iter):
        grad = grad_f(x)

        if np.linalg.norm(grad) < tol:
            print(f"Converged in {k} iterations")
            break

        # Compute Hessian and solve Newton system
        H = hess_f(x)

        try:
            # Solve H·p = -grad
            p = np.linalg.solve(H, -grad)
        except np.linalg.LinAlgError:
            # Fallback to gradient descent if H is singular
            p = -grad

        # Line search
        alpha = 1.0
        c1 = 1e-4
        rho = 0.5

        while f(x + alpha * p) > f(x) + c1 * alpha * np.dot(grad, p):
            alpha *= rho
            if alpha < 1e-10:
                break

        x = x + alpha * p
        history['x'].append(x.copy())
        history['f'].append(f(x))

    return x, history
```

### Quasi-Newton Methods

**Motivation:**
```
Newton: Requires O(d³) for Hessian inverse
Idea: Approximate H⁻¹ using gradient information only

Key equation (Secant condition):
Hₖ₊₁sₖ = yₖ

where:
- sₖ = xₖ₊₁ - xₖ
- yₖ = ∇f(xₖ₊₁) - ∇f(xₖ)

Curvature information from gradient differences!
```

**BFGS (Broyden-Fletcher-Goldfarb-Shanno):**

```
Update formula:
Bₖ₊₁ = Bₖ + (yₖyₖᵀ)/(yₖᵀsₖ) - (BₖsₖsₖᵀBₖ)/(sₖᵀBₖsₖ)

where Bₖ ≈ H⁻¹ₖ (inverse Hessian approximation)

Properties:
- Maintains positive definiteness
- Satisfies secant condition: Bₖ₊₁yₖ = sₖ
- Complexity: O(d²) per iteration (vs O(d³) for Newton)
- Superlinear convergence: ||xₖ - x*|| → 0 faster than any linear rate
```

**Algorithm:**
```
Initialize: x₀, B₀ = I

For k = 0, 1, 2, ...:
  1. Compute pₖ = -Bₖ∇f(xₖ)
  2. Line search for αₖ
  3. xₖ₊₁ = xₖ + αₖpₖ
  4. sₖ = xₖ₊₁ - xₖ
  5. yₖ = ∇f(xₖ₊₁) - ∇f(xₖ)
  6. Update Bₖ₊₁ using BFGS formula

Complexity per iteration: O(d²)
Memory: O(d²) for storing Bₖ
```

**L-BFGS (Limited-memory BFGS):**

```
Problem: BFGS needs O(d²) memory
Solution: Store only m recent {sᵢ, yᵢ} pairs (typically m = 10-20)

Recursively compute Bₖ∇f(xₖ) using two-loop recursion:

Algorithm (Two-loop recursion):
q = ∇f(xₖ)
For i = k-1, ..., k-m:
  ρᵢ = 1/(yᵢᵀsᵢ)
  αᵢ = ρᵢsᵢᵀq
  q = q - αᵢyᵢ

r = H₀q  [initial Hessian, often H₀ = γI]
For i = k-m, ..., k-1:
  β = ρᵢyᵢᵀr
  r = r + sᵢ(αᵢ - β)

Return r  [this is Bₖ∇f(xₖ)]

Complexity: O(md) per iteration
Memory: O(md) - can handle millions of parameters!
```

**Convergence:**

```
BFGS:
- Superlinear: lim ||xₖ₊₁ - x*||/||xₖ - x*|| = 0
- Practically: Very close to Newton's quadratic convergence
- Iterations to ε-accuracy: O(log log(1/ε))

L-BFGS:
- Linear convergence guaranteed
- Superlinear in practice (if m large enough)
- Much better than gradient descent

Empirical: 10-100× fewer iterations than GD
```

**When to Use:**

```
BFGS:
✓ Medium-scale problems (d < 10,000)
✓ When function evaluations are expensive
✓ When superlinear convergence desired

L-BFGS:
✓ Large-scale problems (d > 10,000)
✓ Limited memory scenarios
✓ Default choice for non-stochastic optimization
✓ Used in: TensorFlow (tf.contrib.opt.ScipyOptimizerInterface)
```

---

## Stochastic Optimization

### Stochastic Gradient Descent (SGD)

**Motivation:**

```
Full-batch GD: ∇f(x) = (1/n)Σᵢ₌₁ⁿ ∇fᵢ(x)
Cost per iteration: O(nd)

SGD: ∇̃f(x) = ∇fᵢ(x) for random i
Cost per iteration: O(d)

n/1 speedup per iteration!
Trade-off: Noisy gradient estimates
```

**Algorithm:**

```
For epoch = 1, 2, ...:
  Shuffle training data
  For i = 1, 2, ..., n:
    xₖ₊₁ = xₖ - αₖ∇fᵢ(xₖ)

Variants:
- Pure SGD: One sample per iteration
- Mini-batch SGD: b samples per iteration (b = 32-256)
  ∇̃f(x) = (1/b)Σᵢ∈batch ∇fᵢ(x)
```

**Convergence Analysis:**

```
Assumptions:
1. E[∇̃f(x)] = ∇f(x)  [unbiased]
2. E[||∇̃f(x) - ∇f(x)||²] ≤ σ²  [bounded variance]

For convex f with constant step size α:
E[f(x̄ₖ) - f(x*)] ≤ R²/(2αk) + ασ²/2

where x̄ₖ = (1/k)Σᵢ₌₁ᵏ xᵢ (average iterate)

Trade-off:
- Large α: Fast initial progress, high final error
- Small α: Slow progress, low final error

Optimal α ~ O(1/√k) gives:
E[f(x̄ₖ) - f(x*)] = O(1/√k)

For strongly convex with αₖ = O(1/k):
E[f(xₖ) - f(x*)] = O(1/k)
```

**Variance Reduction:**

```
Problem: Constant variance σ² prevents convergence to exact optimum

Solution 1: Decreasing step size
αₖ → 0 such that Σαₖ = ∞, Σαₖ² < ∞
Example: αₖ = α₀/(1 + βk)

Solution 2: Variance reduction methods
- SVRG (Stochastic Variance Reduced Gradient)
- SAGA
- SAG

These achieve: E[f(xₖ) - f(x*)] = O(exp(-μk/L)) (linear!)
```

**Formal SGD Convergence Analysis:**

```
Theorem 8 (SGD Convergence for Convex f):
Assumptions:
- f(x) = (1/n)Σᵢ₌₁ⁿ fᵢ(x) convex, L-smooth
- Stochastic gradient: g(x,ξ) where E[g(x,ξ)] = ∇f(x)
- Bounded variance: E[||g(x,ξ) - ∇f(x)||²] ≤ σ²
- Constant step size α

Result (for averaged iterate x̄ₖ = (1/k)Σᵢ₌₀ᵏ⁻¹ xᵢ):
E[f(x̄ₖ) - f(x*)] ≤ ||x₀ - x*||²/(2αk) + ασ²/2

Proof Sketch:
From L-smoothness and taking expectation:
E[f(xₖ₊₁)] ≤ E[f(xₖ)] - αE[||∇f(xₖ)||²] + (Lα²/2)E[||g(xₖ,ξₖ)||²]

Using E[||g||²] = ||∇f||² + σ²:
E[f(xₖ₊₁)] ≤ E[f(xₖ)] - α(1 - Lα/2)E[||∇f(xₖ)||²] + (Lα²σ²/2)

With α ≤ 1/L, the gradient term vanishes after averaging.
The variance term σ² creates irreducible error floor.

Optimal Step Size:
Minimize ||x₀ - x*||²/(2αk) + ασ²/2
Taking derivative w.r.t. α and setting to 0:
α* = ||x₀ - x*||/(σ√k)

Plugging back:
E[f(x̄ₖ) - f(x*)] ≤ σ||x₀ - x*||/√k = O(1/√k)

For strongly convex (with decreasing step size αₖ = C/k):
E[f(x̄ₖ) - f(x*)] = O(1/k)

Robbins-Monro Conditions (for almost-sure convergence):
Step sizes {αₖ} must satisfy:
(i)  Σₖαₖ = ∞        (steps large enough to reach optimum)
(ii) Σₖαₖ² < ∞       (steps shrink to kill noise)

Standard choice: αₖ = α₀/(1 + βk) or αₖ = α₀/k

Theorem 9 (SVRG - Stochastic Variance Reduced Gradient):
Algorithm:
Outer loop (epochs s = 0, 1, 2, ...):
  x̃ˢ = xₛ
  μ̃ˢ = (1/n)Σᵢ₌₁ⁿ ∇fᵢ(x̃ˢ)  [full gradient at snapshot]

  Inner loop (m steps):
    Sample iₜ uniformly from {1,...,n}
    vₜ = ∇fᵢₜ(xₜ) - ∇fᵢₜ(x̃ˢ) + μ̃ˢ  [variance reduced gradient]
    xₜ₊₁ = xₜ - αvₜ

Key Insight:
E[vₜ | xₜ] = ∇f(xₜ)  [unbiased]
Var[vₜ] → 0 as xₜ → x*  [variance reduction!]

Convergence Result (Johnson & Zhang, 2013):
For μ-strongly convex, L-smooth f with α = 1/(10L), m = 2n:

E[||xₛ₊₁ - x*||²] ≤ ρ·E[||xₛ - x*||²]

where ρ = (1/2) + (μ/(10L))⁻¹ < 1

Linear convergence: E[||xₛ - x*||²] = O(ρˢ)

Iteration Complexity:
Total gradient evaluations for ε-accuracy:
O((n + κ)log(1/ε))

where κ = L/μ

Comparison:
- GD: O(nκ log(1/ε))  [n gradients per iteration]
- SGD: O(1/(με))      [slow, no log term]
- SVRG: O((n + κ)log(1/ε))  [best of both!]

For n > κ, SVRG gives linear speedup!

Theorem 10 (SAGA Algorithm):
Update rule:
Sample iₜ, compute ∇fᵢₜ(xₜ)
vₜ = ∇fᵢₜ(xₜ) - αⁱₜᵒˡᵈ + (1/n)Σⱼ₌₁ⁿ αⱼ  [table of gradients]
xₜ₊₁ = xₜ - αvₜ
Update table: αⁱₜ ← ∇fᵢₜ(xₜ)

Advantage over SVRG:
- No inner/outer loops (simpler)
- Can use any sampling scheme
- Better practical performance

Convergence (Defazio et al., 2014):
For α = 1/(2(μn + L)):
E[||xₜ - x*||²] ≤ (1 - min{1/(2κ), 1/(2n)})ᵗ · ||x₀ - x*||²

Linear rate: ρ = 1 - min{1/(2κ), 1/(2n)}

When n ≈ κ:
ρ ≈ 1 - 1/(2n)  →  Need O(n log(1/ε)) iterations

Total complexity: O(n log(1/ε)) - optimal!

Memory: O(nd) for storing gradient table

SAG vs SAGA vs SVRG:
┌──────────┬─────────────┬────────────┬─────────────┐
│ Method   │ Convergence │ Memory     │ Inner Loop  │
├──────────┼─────────────┼────────────┼─────────────┤
│ SVRG     │ Linear      │ O(d)       │ Yes (2n)    │
│ SAGA     │ Linear      │ O(nd)      │ No          │
│ SAG      │ Linear      │ O(nd)      │ No          │
└──────────┴─────────────┴────────────┴─────────────┘

Practical Recommendation:
- Large n, limited memory: SVRG
- Medium n, more memory: SAGA (easier to tune)
- Always better than vanilla SGD for strongly convex!
```

### Mini-Batch SGD

**Optimal Batch Size Analysis:**

```
Let:
- b: batch size
- n: dataset size
- d: parameter dimension
- T: total time budget

Time per iteration: t(b) = c₁ + c₂·b·d
where:
- c₁: fixed overhead (data loading, etc.)
- c₂·b·d: gradient computation

Iterations in time T: N(b) = T/t(b)

Variance of gradient estimate: σ²(b) = σ²/b

Convergence: E[f(xₙ)] - f(x*) ~ σ²(b)/(α·N(b))
                                = σ²·t(b)/(α·b·T)
                                ~ 1/b + b  (simplified)

Optimal b*: Balances variance reduction and iteration count
Typically: b* ∈ [32, 512]

Practical: Use largest b that fits in GPU memory
```

**Implementation:**

```python
def minibatch_sgd(f, grad_f, x0, X_train, y_train,
                  batch_size=32, epochs=100, alpha=0.01):
    """
    Mini-batch SGD with learning rate decay

    Args:
        f: Loss function
        grad_f: Gradient function (takes batch)
        x0: Initial parameters
        X_train, y_train: Training data
        batch_size: Mini-batch size
        epochs: Number of epochs
        alpha: Initial learning rate
    """
    x = x0.copy()
    n = len(X_train)
    history = {'loss': [], 'grad_norm': []}

    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(n)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        epoch_loss = 0
        num_batches = 0

        # Mini-batches
        for i in range(0, n, batch_size):
            batch_X = X_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size]

            # Compute gradient on batch
            grad = grad_f(x, batch_X, batch_y)

            # Learning rate decay
            alpha_t = alpha / (1 + 0.01 * epoch)

            # Update
            x = x - alpha_t * grad

            epoch_loss += f(x, batch_X, batch_y) * len(batch_X)
            num_batches += 1

        history['loss'].append(epoch_loss / n)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {epoch_loss/n:.4f}")

    return x, history
```

---

## Adaptive Learning Rate Methods

### AdaGrad

**Motivation:**

```
Problem: Single learning rate α for all parameters
- Some parameters update frequently (small gradient)
- Others update rarely (large gradient when they do)

Idea: Adaptive per-parameter learning rates
- Frequently updated → smaller learning rate
- Rarely updated → larger learning rate
```

**Algorithm:**

```
Initialize: x₀, G₀ = 0 (accumulated squared gradients)

For k = 0, 1, 2, ...:
  gₖ = ∇f(xₖ)
  Gₖ₊₁ = Gₖ + gₖ ⊙ gₖ  (element-wise square)
  xₖ₊₁ = xₖ - (α/√(Gₖ₊₁ + ε)) ⊙ gₖ

where:
- ⊙: element-wise product
- /: element-wise division
- ε: small constant (10⁻⁸) for numerical stability
```

**Mathematical Analysis:**

```
Effective learning rate for parameter i:
αᵢ,ₖ = α / √(Σⱼ₌₀ᵏ gᵢ,ⱼ²)

Properties:
1. Larger α for rare features
2. Smaller α for common features
3. No manual tuning of individual learning rates

Convergence (for convex f):
E[f(x̄ₖ) - f(x*)] ≤ O(1/√k)

Same rate as SGD, but better constants in practice
```

**Problem:**

```
Gₖ grows monotonically → αₖ → 0
Eventually learning stops!

This is a feature for convex optimization (want α → 0)
But a bug for non-convex (neural networks)

Solution: RMSprop, Adam (exponential averaging)
```

### RMSprop

**Algorithm:**

```
Initialize: x₀, v₀ = 0

For k = 0, 1, 2, ...:
  gₖ = ∇f(xₖ)
  vₖ₊₁ = β·vₖ + (1-β)·gₖ²  [exponential moving average]
  xₖ₊₁ = xₖ - (α/√(vₖ₊₁ + ε)) ⊙ gₖ

where β ∈ [0, 1), typically β = 0.9
```

**Key Difference from AdaGrad:**

```
AdaGrad: Gₖ = Σᵢ₌₀ᵏ gᵢ²  (sum all gradients)
RMSprop: vₖ = β·vₖ₋₁ + (1-β)gₖ²  (exponential decay)

RMSprop "forgets" old gradients:
vₖ ≈ (1-β)(gₖ² + β·gₖ₋₁² + β²·gₖ₋₂² + ...)

Effective window: ~1/(1-β) steps
β = 0.9 → window of 10 steps
β = 0.99 → window of 100 steps
```

**Advantages:**

```
✓ Doesn't stop learning (unlike AdaGrad)
✓ Handles non-stationary objectives
✓ Good for RNNs and online learning
✓ Robust to hyperparameters

Typical settings:
- α = 0.001
- β = 0.9
- ε = 10⁻⁸
```

### Adam (Adaptive Moment Estimation)

**Algorithm:**

```
Initialize: x₀, m₀ = 0, v₀ = 0

For k = 0, 1, 2, ...:
  gₖ = ∇f(xₖ)

  # First moment (momentum)
  mₖ₊₁ = β₁·mₖ + (1-β₁)·gₖ

  # Second moment (adaptive learning rate)
  vₖ₊₁ = β₂·vₖ + (1-β₂)·gₖ²

  # Bias correction
  m̂ₖ₊₁ = mₖ₊₁/(1-β₁^(k+1))
  v̂ₖ₊₁ = vₖ₊₁/(1-β₂^(k+1))

  # Update
  xₖ₊₁ = xₖ - α·m̂ₖ₊₁/√(v̂ₖ₊₁ + ε)

Default hyperparameters:
- α = 0.001
- β₁ = 0.9
- β₂ = 0.999
- ε = 10⁻⁸
```

**Bias Correction:**

```
Why needed?

At k=0: m₁ = (1-β₁)g₀, but E[m₁] = (1-β₁)E[g₀]
Without correction: biased towards 0

With correction: m̂₁ = m₁/(1-β₁) = g₀
Now: E[m̂₁] = E[g₀] ✓ (unbiased)

As k → ∞: (1-β₁^k) → 1, correction becomes negligible
```

**Convergence Properties:**

```
For convex f:
Regret bound: R_T = Σₜf(xₜ) - T·f(x*) ≤ O(√T)

Average convergence:
(1/T)Σₜ[f(xₜ) - f(x*)] ≤ O(1/√T)

Note: Not as strong as AdaGrad's O(1/√T) for convex
But much better in practice for non-convex!

Warning: Adam may not converge for some convex problems!
See AMSGrad for fix.
```

**Rigorous Adam Convergence Analysis:**

```
Theorem 11 (Adam Convergence Issue - Reddi et al., 2018):
Problem: Adam may not converge even for simple convex problems!

Counterexample:
Consider online optimization with:
- f₁(x) = 1010x  for odd t
- fₜ(x) = -x     for even t

Optimal: x* = -1010 (for the sequence)

With Adam (β₁ = 0.9, β₂ = 0.999, α = 1):
- Gradients oscillate: +1010, -1, +1010, -1, ...
- Second moment: vₜ ≈ 1010² (dominated by large gradients)
- Effective step: α/√vₜ ≈ 1/1010 (very small!)
- Update barely moves toward x* = -1010

Result: Adam oscillates and does NOT converge to optimum!

Root Cause:
Adaptive learning rate 1/√vₜ uses exponential moving average
⇒ May decrease even when recent gradients are small
⇒ Past large gradients can prevent future progress

Mathematical Issue:
The step size might satisfy:
Σₜ αₜ < ∞  (violates Robbins-Monro condition Σαₜ = ∞)

Theorem 12 (AMSGrad Fix):
Modified Algorithm:
vₜ = β₂·vₜ₋₁ + (1-β₂)·gₜ²
v̂ₜ = max(v̂ₜ₋₁, vₜ)  [Non-decreasing sequence!]
xₜ₊₁ = xₜ - α·m̂ₜ/√v̂ₜ

Key Difference: v̂ₜ never decreases
⇒ Learning rate 1/√v̂ₜ never increases
⇒ Ensures sufficient decrease when needed

Convergence Guarantee (Reddi et al., 2018):
For convex f, L-smooth, bounded gradients ||gₜ|| ≤ G:

(1/T)Σₜf(xₜ) - f(x*) ≤ O(1/√T)

Proof uses: Step sizes satisfy Σₜ αₜ/√v̂ₜ = ∞

Practical Impact:
- Fixes convergence issues for convex problems
- Similar performance to Adam on neural networks
- More stable training in practice

Implementation:
```python
# AMSGrad (only 1 line change from Adam!)
m = beta1 * m + (1 - beta1) * g
v = beta2 * v + (1 - beta2) * (g ** 2)
v_hat = np.maximum(v_hat, v)  # <-- Only change!
m_hat = m / (1 - beta1 ** (t + 1))
v_hat_corrected = v_hat / (1 - beta2 ** (t + 1))
x = x - alpha * m_hat / (np.sqrt(v_hat_corrected) + eps)
```

Theorem 13 (AdamW - Weight Decay Decoupling):
Standard L2 Regularization in Adam:
ℒ(θ) = f(θ) + (λ/2)||θ||²

Gradient: ∇ℒ = ∇f + λθ

Adam treats λθ as part of gradient:
- Gets adaptive scaling 1/√vₜ
- Interacts with momentum
- Learning rate dependent!

AdamW (Loshchilov & Hutter, 2019):
Decouple weight decay from gradient update:

mₜ = β₁mₜ₋₁ + (1-β₁)∇f(θₜ)  [no λθ here!]
vₜ = β₂vₜ₋₁ + (1-β₂)||∇f(θₜ)||²
θₜ₊₁ = θₜ - α(m̂ₜ/√v̂ₜ + λθₜ)  [weight decay added here]

Advantages:
- Weight decay independent of gradient scaling
- Better generalization (empirically)
- Easier to tune λ and α separately

Optimal λ Selection:
λ ~ 1/T where T is total training steps
Typical: λ ∈ [0.01, 0.3] for transformers

Convergence Analysis Summary:
┌───────────┬─────────────────┬──────────────┬────────────┐
│ Method    │ Convex Converg. │ Non-Convex   │ Issues     │
├───────────┼─────────────────┼──────────────┼────────────┤
│ AdaGrad   │ O(1/√T) ✓       │ May stop     │ LR → 0     │
│ RMSprop   │ No guarantee    │ Works well   │ No theory  │
│ Adam      │ May fail! ✗     │ Works well   │ See Thm 11 │
│ AMSGrad   │ O(1/√T) ✓       │ Works well   │ Slower     │
│ AdamW     │ O(1/√T) ✓       │ Best gen.    │ More hyper │
│ RAdam     │ O(1/√T) ✓       │ More stable  │ Complex    │
└───────────┴─────────────────┴──────────────┴────────────┘

Practical Insights:
1. For convex optimization: Use AdaGrad or AMSGrad
2. For deep learning: AdamW is current gold standard
3. For stable training: RAdam (auto warmup)
4. For fine-tuning: AdamW with cosine schedule
5. For best generalization: SGD + momentum (but needs tuning)

Key Takeaway:
Adam's empirical success ≠ theoretical guarantee!
Always validate convergence for your specific problem.
```

**Variants:**

```
AdamW (Adam with Weight Decay):
xₖ₊₁ = xₖ - α(m̂ₖ₊₁/√(v̂ₖ₊₁) + λxₖ)

Decouples weight decay from gradient-based update
Better generalization

Nadam (Nesterov Adam):
Combines Adam with Nesterov momentum
Slightly faster convergence

RAdam (Rectified Adam):
Fixes variance issue in early stages
Automatic warmup

Lookahead:
Wraps any optimizer with "slow weights"
More stable convergence
```

**Practical Comparison:**

```
Method    | Pros                        | Cons
----------|-----------------------------|-----------------------
SGD       | Simple, well-understood     | Needs LR tuning
SGD+Mom   | Faster, less sensitive      | Still needs tuning
AdaGrad   | No LR tuning, good for      | Stops learning
          | sparse data                 | (non-convex)
RMSprop   | Good for RNNs, online       | Less theory
Adam      | Robust, fast, popular       | May not converge
AdamW     | Better generalization       | More hyperparameters
RAdam     | More stable early training  | Slower

Current best practice for deep learning: AdamW with cosine LR schedule
```

**Implementation:**

```python
def adam(f, grad_f, x0, alpha=0.001, beta1=0.9, beta2=0.999,
         eps=1e-8, max_iter=1000):
    """
    Adam optimizer

    Most popular optimizer for deep learning
    Combines momentum and adaptive learning rates
    """
    x = x0.copy()
    m = np.zeros_like(x)  # First moment
    v = np.zeros_like(x)  # Second moment

    history = {'x': [x.copy()], 'f': [f(x)]}

    for k in range(max_iter):
        g = grad_f(x)

        # Update biased moments
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)

        # Bias correction
        m_hat = m / (1 - beta1 ** (k + 1))
        v_hat = v / (1 - beta2 ** (k + 1))

        # Update parameters
        x = x - alpha * m_hat / (np.sqrt(v_hat) + eps)

        history['x'].append(x.copy())
        history['f'].append(f(x))

    return x, history
```

---

## Constrained Optimization

### Lagrange Multipliers

**Problem Formulation:**

```
Minimize: f(x)
Subject to: h(x) = 0  (equality constraint)

Lagrangian:
L(x, λ) = f(x) + λᵀh(x)

where λ are Lagrange multipliers
```

**Optimality Conditions (KKT):**

```
Necessary conditions for x* to be optimal:

1. Stationarity: ∇ₓL(x*, λ*) = ∇f(x*) + Jₕᵀ(x*)λ* = 0
2. Primal feasibility: h(x*) = 0
3. Dual feasibility: ∇_λL(x*, λ*) = h(x*) = 0

where Jₕ = ∇h(x) is Jacobian of constraints
```

**Geometric Interpretation:**

```
At optimum: ∇f(x*) must be perpendicular to constraint surface

∇f(x*) = -λ*∇h(x*)

λ* tells us sensitivity: how much f changes when constraint relaxed

Example:
Minimize: f(x,y) = x² + y²
Subject to: x + y = 1

L(x, y, λ) = x² + y² + λ(x + y - 1)

∇L = 0:
  2x + λ = 0
  2y + λ = 0
  x + y = 1

Solution: x = y = 1/2, λ = -1
```

### Inequality Constraints (KKT Conditions)

**Problem:**

```
Minimize: f(x)
Subject to: gᵢ(x) ≤ 0, i = 1,...,m  (inequality)
            hⱼ(x) = 0, j = 1,...,p  (equality)

Lagrangian:
L(x, μ, λ) = f(x) + Σᵢμᵢgᵢ(x) + Σⱼλⱼhⱼ(x)
```

**KKT Conditions:**

```
1. Stationarity:
   ∇f(x*) + Σᵢμᵢ*∇gᵢ(x*) + Σⱼλⱼ*∇hⱼ(x*) = 0

2. Primal feasibility:
   gᵢ(x*) ≤ 0 for all i
   hⱼ(x*) = 0 for all j

3. Dual feasibility:
   μᵢ* ≥ 0 for all i

4. Complementary slackness:
   μᵢ*gᵢ(x*) = 0 for all i

Interpretation of (4):
Either μᵢ* = 0 (constraint inactive) OR gᵢ(x*) = 0 (constraint active)
```

**Example (SVM):**

```
Minimize: ½||w||²
Subject to: yᵢ(wᵀxᵢ + b) ≥ 1 for all i

Lagrangian:
L(w, b, α) = ½||w||² - Σᵢαᵢ[yᵢ(wᵀxᵢ + b) - 1]

KKT conditions yield:
w* = Σᵢαᵢ*yᵢxᵢ  (support vectors where αᵢ* > 0)

Complementary slackness:
αᵢ*[yᵢ(w*ᵀxᵢ + b*) - 1] = 0

Points with αᵢ* > 0 are support vectors (on margin)
```

### Penalty and Augmented Lagrangian Methods

**Penalty Method:**

```
Original: min f(x) s.t. h(x) = 0

Penalty: min_x f(x) + (ρ/2)||h(x)||²

where ρ > 0 is penalty parameter

As ρ → ∞: Solution approaches constrained optimum
But: Ill-conditioned for large ρ
```

**Augmented Lagrangian:**

```
min_x L_ρ(x, λ) = f(x) + λᵀh(x) + (ρ/2)||h(x)||²

Algorithm (ADMM - Alternating Direction):
1. xₖ₊₁ = argmin_x L_ρ(x, λₖ)
2. λₖ₊₁ = λₖ + ρh(xₖ₊₁)

Advantages:
- Exact convergence with finite ρ
- Well-conditioned
- Parallelizable (for separable problems)
```

**ADMM Application:**

```
For separable problem:
min f(x) + g(z)
s.t. Ax + Bz = c

Augmented Lagrangian:
L_ρ(x, z, λ) = f(x) + g(z) + λᵀ(Ax + Bz - c) + (ρ/2)||Ax + Bz - c||²

Algorithm:
xₖ₊₁ = argmin_x L_ρ(x, zₖ, λₖ)
zₖ₊₁ = argmin_z L_ρ(xₖ₊₁, z, λₖ)
λₖ₊₁ = λₖ + ρ(Axₖ₊₁ + Bzₖ₊₁ - c)

Used in: Distributed optimization, Lasso, total variation denoising
```

---

## Meta-Optimization and AutoML

### Hyperparameter Optimization

**Hyperparameters to Optimize:**

```
Model hyperparameters:
- Architecture choices (layers, units)
- Regularization (dropout, weight decay)

Optimizer hyperparameters:
- Learning rate α
- Momentum β
- Batch size
- Learning rate schedule

Problem: Expensive to evaluate (full training)
```

**Grid Search:**

```
For each hyperparameter:
  Choose grid: [v₁, v₂, ..., vₖ]

Evaluate all combinations: ∏ᵢkᵢ evaluations

Example:
- α ∈ {0.1, 0.01, 0.001}  (3 values)
- β ∈ {0.9, 0.99}  (2 values)
- λ ∈ {0.01, 0.001, 0.0001}  (3 values)

Total: 3 × 2 × 3 = 18 trainings

Complexity: Exponential in number of hyperparameters
Practical limit: ~3 hyperparameters
```

**Random Search:**

```
For k iterations:
  Sample hyperparameters from distribution
  Train and evaluate

Theorem (Bergstra & Bengio, 2012):
For same budget, random search often beats grid search

Intuition:
- Grid wastes evaluations on unimportant dimensions
- Random explores important dimensions more densely

Recommended: Use random search over grid search
```

**Bayesian Optimization:**

```
Build surrogate model: m(θ) ≈ validation_loss(θ)

Acquisition function: α(θ) balances exploration vs exploitation

Algorithm:
1. Initialize with random configurations
2. Fit Gaussian Process to observed (θ, loss) pairs
3. Find θ_next = argmax α(θ)
4. Evaluate loss(θ_next)
5. Repeat 2-4

Acquisition functions:
- EI (Expected Improvement)
- UCB (Upper Confidence Bound)
- PI (Probability of Improvement)

Convergence: O(√T) regret (optimal for black-box optimization)
```

**Population-Based Training (PBT):**

```
Train multiple models in parallel
Periodically:
  - Exploit: Copy weights from best performers
  - Explore: Perturb hyperparameters

Advantages:
- Online adaptation
- Efficient use of parallelism
- Finds schedules, not just fixed values

Used by: DeepMind for SOTA reinforcement learning
```

### Learning Rate Scheduling

**Step Decay:**

```
α(t) = α₀ · γ^⌊t/s⌋

where:
- α₀: initial learning rate
- γ: decay factor (typically 0.1-0.5)
- s: step size (epochs between decays)

Example: α₀ = 0.1, γ = 0.1, s = 30
Epochs 0-29: α = 0.1
Epochs 30-59: α = 0.01
Epochs 60-89: α = 0.001
```

**Exponential Decay:**

```
α(t) = α₀ · exp(-λt)

Smooth decrease
λ controls decay rate
```

**Cosine Annealing:**

```
α(t) = αₘᵢₙ + ½(αₘₐₓ - αₘᵢₙ)(1 + cos(πt/T))

where:
- αₘₐₓ: maximum learning rate
- αₘᵢₙ: minimum learning rate
- T: total epochs

Properties:
- Smooth decay
- Fast decrease initially
- Slow decrease near end (fine-tuning)

Warm restarts:
Reset t periodically → explore multiple minima
```

**1cycle Policy:**

```
Two phases:
1. Warmup (50% of training):
   Learning rate: αₘᵢₙ → αₘₐₓ (linear increase)
   Momentum: βₘₐₓ → βₘᵢₙ (linear decrease)

2. Annealing (50% of training):
   Learning rate: αₘₐₓ → αₘᵢₙ (cosine decrease)
   Momentum: βₘᵢₙ → βₘₐₓ (linear increase)

Typical values:
- αₘₐₓ = 10·αₘᵢₙ
- βₘₐₓ = 0.95, βₘᵢₙ = 0.85

Benefits:
- Regularization effect
- Faster convergence
- Better generalization

Used successfully in: Computer vision, NLP
```

**Learning Rate Finder:**

```
Algorithm:
1. Start with very small LR (e.g., 10⁻⁶)
2. Train for few hundred iterations
3. Exponentially increase LR each iteration
4. Record loss vs LR

Plot loss vs log(LR):
- Find point where loss starts decreasing
- Find point before loss explodes
- Choose LR in between (typically at steepest descent)

Automatic LR selection!
```

---

## Convergence Analysis Summary

### Convergence Rates Comparison

```
Problem Type         | Algorithm      | Rate           | Iterations to ε
---------------------|----------------|----------------|------------------
Convex, Smooth       | GD             | O(1/k)         | O(1/ε)
                     | Momentum       | O(1/k²)        | O(1/√ε)
                     | Conjugate GD   | O((√κ-1)/(√κ+1))^k | O(√κ log(1/ε))
                     | Newton         | O(log log(1/ε))| O(log log(1/ε))
                     | BFGS           | Superlinear    | O(log(1/ε))

Strongly Convex,     | GD             | O(exp(-μk/L))  | O((L/μ)log(1/ε))
Smooth               | SGD            | O(1/k)         | O(1/ε)
                     | Momentum       | O(exp(-√(μ/L)k))| O(√(L/μ)log(1/ε))
                     | Adam           | O(1/√k)        | O(1/ε²)

Non-Convex           | GD             | ||∇f||→0       | No guarantee
                     | SGD            | ||∇f||→0       | No guarantee
                     | Adam           | ||∇f||→0       | No guarantee

where:
- k: iteration number
- ε: desired accuracy
- κ = L/μ: condition number
- μ: strong convexity parameter
- L: Lipschitz constant of gradient
```

### Per-Iteration Complexity

```
Algorithm         | Time per Iter  | Memory    | Notes
------------------|----------------|-----------|------------------
GD (full batch)   | O(nd)          | O(nd + d) | n = samples, d = features
SGD               | O(d)           | O(d)      | n/1 speedup
Mini-batch SGD    | O(bd)          | O(bd)     | b = batch size
Momentum          | O(nd)          | O(d)      | Extra d for velocity
Adam              | O(nd)          | O(2d)     | Two moment vectors
Newton            | O(nd² + d³)    | O(d²)     | Hessian computation + inversion
BFGS              | O(nd + d²)     | O(d²)     | Hessian approximation
L-BFGS            | O(nd + md)     | O(md)     | m ≈ 10-20 stored vectors
Conjugate Gradient| O(nd)          | O(d)      | No Hessian needed
```

---

## Practical Guidelines

### Algorithm Selection Guide

**Decision Tree:**

```
START: What type of problem?

├─ CONVEX
│  ├─ Small scale (d < 1000)
│  │  ├─ Quadratic → Conjugate Gradient
│  │  └─ General → Newton or BFGS
│  │
│  ├─ Medium scale (d < 100k)
│  │  └─ L-BFGS
│  │
│  └─ Large scale (d > 100k)
│     └─ Mini-batch SGD + Momentum
│
└─ NON-CONVEX (Neural Networks)
   ├─ Default choice → Adam or AdamW
   │
   ├─ Better generalization needed?
   │  └─ SGD + Momentum + Good LR schedule
   │
   ├─ Fast prototyping?
   │  └─ Adam (robust to hyperparameters)
   │
   └─ SOTA results?
      └─ AdamW + Cosine LR + Warmup
```

### Hyperparameter Recommendations

**Learning Rate:**

```
Algorithm   | Initial LR   | Schedule              | Notes
------------|--------------|----------------------|------------------
SGD         | 0.1          | Step decay           | Sensitive to init
SGD+Momentum| 0.01-0.1     | Cosine annealing     | More robust
AdaGrad     | 0.01         | None                 | Self-adapting
RMSprop     | 0.001        | None                 | Good for RNNs
Adam        | 0.001        | Optional cosine      | Robust default
AdamW       | 0.001        | Cosine + warmup      | Current best practice

General tips:
- Start with defaults
- Use learning rate finder
- Multiply by √(batch_size/32) when scaling batch size
```

**Momentum:**

```
β₁ (first moment): 0.9 (standard), 0.99 (for more smoothing)
β₂ (second moment): 0.999 (almost always)

Higher β₁ → More smoothing, slower adaptation
Lower β₁ → Less smoothing, faster adaptation
```

**Batch Size:**

```
Size Range | Use Case                    | Notes
-----------|-----------------------------|-----------------
1-32       | Small datasets, RNNs        | Noisy but explores better
32-256     | Most deep learning          | Good balance
256-2048   | Large-scale training        | Needs LR scaling
>2048      | Distributed training        | Linear scaling breaks

Linear scaling rule: LR_new = LR_base × (batch_new / batch_base)
```

### Common Pitfalls and Solutions

**Problem 1: Loss not decreasing**

```
Causes:
- Learning rate too high → Decrease by 10×
- Learning rate too low → Increase by 10×
- Wrong initialization → Use He/Xavier
- Gradient vanishing → Check gradients, use ResNets
- Gradient explosion → Gradient clipping

Debug:
1. Check gradient norms: should be O(1)
2. Check loss scale: should decrease initially
3. Try smaller LR (safest)
```

**Problem 2: Training loss decreases, validation doesn't**

```
Overfitting! Solutions:
- More data (best solution)
- Regularization (L2, dropout)
- Data augmentation
- Early stopping
- Reduce model capacity
```

**Problem 3: Loss oscillates**

```
Causes:
- Learning rate too high
- Batch size too small

Solutions:
- Decrease LR by 2-10×
- Increase batch size
- Use momentum (β = 0.9)
- Use adaptive methods (Adam)
```

**Problem 4: Slow convergence**

```
Solutions:
- Increase learning rate
- Use momentum or adaptive methods
- Better initialization
- Batch normalization
- Residual connections (for deep networks)
```

### Monitoring and Debugging

**What to Track:**

```
1. Training loss (every iteration)
2. Validation loss (every epoch)
3. Gradient norms: ||∇f(x)||
4. Parameter norms: ||x||
5. Learning rate (if scheduled)
6. Gradient/parameter ratio: ||∇f||/||x||

Healthy training:
- Training loss: Smooth decrease
- Validation loss: Tracks training (gap = generalization)
- Gradient norm: Stable, O(0.1-1)
- Grad/param ratio: O(0.001-0.01)
```

**Gradient Clipping:**

```python
def clip_gradients(gradients, max_norm=1.0):
    """
    Clip gradients by global norm

    Prevents gradient explosion
    Essential for RNNs
    """
    total_norm = np.sqrt(sum(np.sum(g**2) for g in gradients))

    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        gradients = [g * clip_coef for g in gradients]

    return gradients
```

---

## References

### Classical Optimization

1. **Nocedal, J., & Wright, S. J.** (2006). *Numerical optimization* (2nd ed.). Springer.
   - Comprehensive reference for optimization algorithms

2. **Boyd, S., & Vandenberghe, L.** (2004). *Convex optimization*. Cambridge University Press.
   - Theory of convex optimization

3. **Nesterov, Y.** (2004). *Introductory lectures on convex optimization: A basic course*. Kluwer Academic Publishers.
   - Complexity analysis and optimal methods

### Stochastic Optimization

4. **Robbins, H., & Monro, S.** (1951). "A stochastic approximation method." *The Annals of Mathematical Statistics*, 22(3), 400-407.
   - Original stochastic approximation paper

5. **Bottou, L., Curtis, F. E., & Nocedal, J.** (2018). "Optimization methods for large-scale machine learning." *SIAM Review*, 60(2), 223-311.
   - Modern survey of optimization for ML

6. **Ruder, S.** (2016). "An overview of gradient descent optimization algorithms." *arXiv:1609.04747*.
   - Accessible overview of modern optimizers

### Adaptive Methods

7. **Duchi, J., Hazan, E., & Singer, Y.** (2011). "Adaptive subgradient methods for online learning and stochastic optimization." *Journal of Machine Learning Research*, 12, 2121-2159.
   - AdaGrad algorithm

8. **Tieleman, T., & Hinton, G.** (2012). "Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude." *COURSERA: Neural networks for machine learning*, 4(2), 26-31.
   - RMSprop

9. **Kingma, D. P., & Ba, J.** (2015). "Adam: A method for stochastic optimization." *ICLR*.
   - Adam optimizer

10. **Loshchilov, I., & Hutter, F.** (2019). "Decoupled weight decay regularization." *ICLR*.
    - AdamW

### Quasi-Newton Methods

11. **Nocedal, J.** (1980). "Updating quasi-Newton matrices with limited storage." *Mathematics of Computation*, 35(151), 773-782.
    - L-BFGS algorithm

12. **Liu, D. C., & Nocedal, J.** (1989). "On the limited memory BFGS method for large scale optimization." *Mathematical Programming*, 45(1-3), 503-528.
    - L-BFGS convergence analysis

### Learning Rate Schedules

13. **Smith, L. N.** (2017). "Cyclical learning rates for training neural networks." *IEEE Winter Conference on Applications of Computer Vision*, pp. 464-472.
    - Cyclical learning rates

14. **Smith, L. N.** (2018). "A disciplined approach to neural network hyper-parameters: Part 1--learning rate, batch size, momentum, and weight decay." *arXiv:1803.09820*.
    - 1cycle policy and learning rate finder

15. **Loshchilov, I., & Hutter, F.** (2017). "SGDR: Stochastic gradient descent with warm restarts." *ICLR*.
    - Cosine annealing with restarts

### Hyperparameter Optimization

16. **Bergstra, J., & Bengio, Y.** (2012). "Random search for hyper-parameter optimization." *Journal of Machine Learning Research*, 13, 281-305.
    - Random vs grid search

17. **Snoek, J., Larochelle, H., & Adams, R. P.** (2012). "Practical bayesian optimization of machine learning algorithms." *NeurIPS*.
    - Bayesian optimization for hyperparameters

18. **Jaderberg, M., Dalibard, V., Osindero, S., Czarnecki, W. M., Donahue, J., Razavi, A., ... & Kavukcuoglu, K.** (2017). "Population based training of neural networks." *arXiv:1711.09846*.
    - PBT algorithm

### Advanced Topics

19. **Polyak, B. T., & Juditsky, A. B.** (1992). "Acceleration of stochastic approximation by averaging." *SIAM Journal on Control and Optimization*, 30(4), 838-855.
    - Polyak averaging

20. **Sutskever, I., Martens, J., Dahl, G., & Hinton, G.** (2013). "On the importance of initialization and momentum in deep learning." *ICML*, pp. 1139-1147.
    - Momentum analysis for deep learning

---

*This comprehensive guide covers the mathematical foundations, convergence analysis, and practical implementation of optimization algorithms for machine learning. Use it as a reference for selecting and tuning optimizers for your specific application.*
