# Gradient Descent and Optimization Algorithms Guide

A comprehensive practical guide to gradient descent variants and modern optimization algorithms.

## Table of Contents
- [Understanding Gradient Descent](#understanding-gradient-descent)
- [Batch Gradient Descent](#batch-gradient-descent)
- [Stochastic Gradient Descent (SGD)](#stochastic-gradient-descent-sgd)
- [Mini-Batch Gradient Descent](#mini-batch-gradient-descent)
- [Momentum](#momentum)
- [Nesterov Accelerated Gradient](#nesterov-accelerated-gradient)
- [AdaGrad](#adagrad)
- [RMSprop](#rmsprop)
- [Adam](#adam)
- [Learning Rate Schedules](#learning-rate-schedules)
- [Comparing Optimizers](#comparing-optimizers)
- [Practical Tips](#practical-tips)
- [Complete Examples](#complete-examples)

---

## Understanding Gradient Descent

**Gradient Descent** is an optimization algorithm that iteratively adjusts parameters to minimize a loss function.

### Mathematical Foundations of Optimization

Before discussing gradient descent, we establish the rigorous mathematical theory of continuous optimization.

**Optimization Problem:**
```
min_{x∈ℝⁿ} f(x)
```

**Definition 1 (Convexity):**
A function f: ℝⁿ → ℝ is convex if for all x, y ∈ ℝⁿ and λ ∈ [0, 1]:

f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)

**First-order characterization:**
f is convex ⟺ f(y) ≥ f(x) + ∇f(x)ᵀ(y - x) for all x, y

**Interpretation:** Function lies above all its tangent hyperplanes.

**Definition 2 (Strong Convexity):**
f is μ-strongly convex (μ > 0) if for all x, y:

f(y) ≥ f(x) + ∇f(x)ᵀ(y - x) + (μ/2)||y - x||²

**Equivalently:** f(x) - (μ/2)||x||² is convex.

**Key property:** Strongly convex functions have unique global minima.

**Theorem 1 (Characterization via Hessian):**
If f is twice differentiable:

(a) f is convex ⟺ ∇²f(x) ⪰ 0 (positive semidefinite) for all x
(b) f is μ-strongly convex ⟺ ∇²f(x) ⪰ μI for all x

Proof of (b):
∇²(f(x) - (μ/2)||x||²) = ∇²f(x) - μI ⪰ 0 ⟺ ∇²f(x) ⪰ μI ∎

**Definition 3 (Smoothness / Lipschitz Gradient):**
f has L-Lipschitz continuous gradient if for all x, y:

||∇f(x) - ∇f(y)|| ≤ L||x - y||

**First-order bound:**
f(y) ≤ f(x) + ∇f(x)ᵀ(y - x) + (L/2)||y - x||²

**Interpretation:** Function lies below quadratic upper bound (opposite of strong convexity).

**Theorem 2 (Smoothness via Hessian):**
If f is twice differentiable:

f has L-Lipschitz gradient ⟺ ∇²f(x) ⪯ LI for all x

**Condition number:**
For μ-strongly convex and L-smooth function:

κ = L/μ

measures "hardness" of optimization problem.
- κ = 1: perfectly conditioned (gradient descent converges in 1 step)
- κ >> 1: ill-conditioned (slow convergence)

### Convergence Theory of Gradient Descent

**Algorithm (Gradient Descent):**
```
Initialize: x₀ ∈ ℝⁿ
For t = 0, 1, 2, ...:
    x_{t+1} = x_t - α∇f(x_t)
```

where α > 0 is the step size (learning rate).

**Theorem 3 (Convergence for Convex + L-Smooth):**
If f is convex and L-smooth, with step size α ≤ 1/L:

f(x_t) - f(x*) ≤ (2L||x₀ - x*||²)/t

where x* = argmin f(x).

**Convergence rate:** O(1/t) - sublinear

**Proof sketch:**
By L-smoothness:
f(x_{t+1}) ≤ f(x_t) + ∇f(x_t)ᵀ(x_{t+1} - x_t) + (L/2)||x_{t+1} - x_t||²

Substituting x_{t+1} = x_t - α∇f(x_t):
f(x_{t+1}) ≤ f(x_t) - α||∇f(x_t)||² + (αL/2)||∇f(x_t)||²
           = f(x_t) - α(1 - αL/2)||∇f(x_t)||²

With α = 1/L:
f(x_{t+1}) ≤ f(x_t) - (1/2L)||∇f(x_t)||²

By convexity: ||∇f(x_t)||² ≥ 2L(f(x_t) - f(x*))

Combining and telescoping gives O(1/t) rate. ∎

**Theorem 4 (Convergence for Strongly Convex + Smooth):**
If f is μ-strongly convex and L-smooth, with step size α = 1/L:

||x_t - x*||² ≤ (1 - μ/L)^t ||x₀ - x*||²

and:
f(x_t) - f(x*) ≤ (L/2)(1 - μ/L)^t ||x₀ - x*||²

**Convergence rate:** O((1 - 1/κ)^t) - linear (exponential)

**Proof:**
By strong convexity and smoothness:

||x_{t+1} - x*||² = ||x_t - α∇f(x_t) - x*||²
                  = ||x_t - x*||² - 2α∇f(x_t)ᵀ(x_t - x*) + α²||∇f(x_t)||²

Using strong convexity: ∇f(x_t)ᵀ(x_t - x*) ≥ (μ/2)||x_t - x*||² + (1/2μ)||∇f(x_t)||²

And smoothness: ||∇f(x_t)||² ≤ 2L(f(x_t) - f(x*)) ≤ L²||x_t - x*||²

Combining with α = 1/L gives:
||x_{t+1} - x*||² ≤ (1 - μ/L)||x_t - x*||² ∎

**Comparison of Rates:**
```
Setting             | Rate       | # Iterations to ε-accuracy
--------------------|------------|---------------------------
Convex + Smooth     | O(1/t)     | O(L||x₀-x*||²/ε)
μ-Strongly Convex   | O(e^{-t/κ})| O(κ log(1/ε))
+ L-Smooth          |            |
```

**Practical implications:**
- Strong convexity crucial for fast convergence
- Condition number κ determines convergence speed
- Well-conditioned problems (κ ≈ 1): fast convergence
- Ill-conditioned problems (κ >> 1): very slow convergence

**Theorem 5 (Optimal Step Size):**
For μ-strongly convex and L-smooth functions:

Constant step size: α* = 2/(μ + L)

gives convergence rate:
||x_t - x*||² ≤ ((κ - 1)/(κ + 1))^t ||x₀ - x*||²

This is faster than α = 1/L when κ > 3.

**Line Search:**
Instead of fixed α, choose α_t to minimize f(x_t - α∇f(x_t)):

α_t = argmin_α f(x_t - α∇f(x_t))

This is **exact line search** - often impractical but theoretically optimal.

**Backtracking line search (Armijo rule):**
Start with α = 1, repeatedly set α ← βα (β < 1) until:

f(x_t - α∇f(x_t)) ≤ f(x_t) - c·α||∇f(x_t)||²

where c ∈ (0, 1) (typically c = 10^{-4}).

This ensures sufficient decrease while being computationally cheap.

### The Core Idea

```
θ_{t+1} = θ_t - α × ∇J(θ_t)
```

Where:
- **θ_t**: Parameters (weights) at iteration t
- **α**: Learning rate (step size, α > 0)
- **∇J(θ)**: Gradient of loss function with respect to θ, i.e., [∂J/∂θ₁, ∂J/∂θ₂, ..., ∂J/∂θₙ]ᵀ

**Intuition:** The gradient ∇J(θ) points in the direction of steepest ascent. Moving in the opposite direction -∇J(θ) yields steepest descent. The learning rate α controls the step size.

**Convergence guarantee (for convex, L-smooth functions):**
With α ≤ 1/L where L is the Lipschitz constant of ∇J, gradient descent converges at rate O(1/t).

### Implementing from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt

# Simple example: Minimize f(x) = x^2
def f(x):
    """Function to minimize: f(x) = x^2"""
    return x**2

def gradient_f(x):
    """Gradient: f'(x) = 2x"""
    return 2*x

def gradient_descent(start_x, learning_rate, num_iterations):
    """
    Basic gradient descent implementation.
    """
    x = start_x
    history = [x]

    for i in range(num_iterations):
        # Calculate gradient
        grad = gradient_f(x)

        # Update parameter
        x = x - learning_rate * grad

        history.append(x)

    return x, history

# Run gradient descent
start = 10.0
learning_rate = 0.1
iterations = 50

final_x, history = gradient_descent(start, learning_rate, iterations)

# Visualize
x_plot = np.linspace(-11, 11, 1000)
y_plot = f(x_plot)

plt.figure(figsize=(12, 5))

# Plot 1: Function and path
plt.subplot(1, 2, 1)
plt.plot(x_plot, y_plot, 'b-', linewidth=2, label='f(x) = x²')
plt.plot(history, [f(x) for x in history], 'ro-', markersize=8, linewidth=2, label='GD Path')
plt.plot(0, 0, 'g*', markersize=20, label='Minimum')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title(f'Gradient Descent (α={learning_rate})')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Convergence
plt.subplot(1, 2, 2)
plt.plot([f(x) for x in history], 'ro-', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('f(x)')
plt.title('Loss vs Iteration')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Starting point: {start}")
print(f"Final point: {final_x:.6f}")
print(f"Minimum value: {f(final_x):.6f}")
```

### Learning Rate Impact

```python
# Test different learning rates
learning_rates = [0.01, 0.1, 0.5, 0.9]

plt.figure(figsize=(15, 10))

for idx, lr in enumerate(learning_rates, 1):
    final_x, history = gradient_descent(10.0, lr, 50)

    plt.subplot(2, 2, idx)
    plt.plot(x_plot, y_plot, 'b-', linewidth=2, alpha=0.3)
    plt.plot(history, [f(x) for x in history], 'ro-', markersize=6, linewidth=2)
    plt.plot(0, 0, 'g*', markersize=20)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Learning Rate α = {lr}')
    plt.grid(True, alpha=0.3)

    # Annotate convergence
    if abs(final_x) < 0.01:
        plt.text(0, 50, f'✓ Converged in {len(history)} steps',
                fontsize=10, color='green')
    else:
        plt.text(0, 50, f'✗ Did not converge',
                fontsize=10, color='red')

plt.tight_layout()
plt.show()

print("Learning Rate Analysis:")
for lr in learning_rates:
    final_x, history = gradient_descent(10.0, lr, 50)
    print(f"  α={lr}: Final x={final_x:.6f}, Iterations to converge: {len(history)}")
```

**Key Observations:**
- **Too small**: Slow convergence
- **Too large**: Oscillation or divergence
- **Just right**: Fast, stable convergence

---

## Batch Gradient Descent

**Computes gradient using entire training set** in each iteration.

### Implementation

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate dataset
X, y = make_regression(n_samples=1000, n_features=1, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Add bias term
X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

def batch_gradient_descent(X, y, learning_rate=0.01, num_iterations=1000):
    """
    Batch Gradient Descent for Linear Regression.

    Updates using entire training set each iteration.
    """
    m, n = X.shape
    theta = np.zeros(n)
    loss_history = []

    for i in range(num_iterations):
        # Predict
        predictions = X.dot(theta)

        # Calculate loss (MSE)
        loss = (1/(2*m)) * np.sum((predictions - y)**2)
        loss_history.append(loss)

        # Calculate gradient using ALL samples
        gradient = (1/m) * X.T.dot(predictions - y)

        # Update parameters
        theta = theta - learning_rate * gradient

    return theta, loss_history

# Train
theta, loss_history = batch_gradient_descent(X_train, y_train, learning_rate=0.1, num_iterations=100)

# Evaluate
train_pred = X_train.dot(theta)
test_pred = X_test.dot(theta)

train_mse = np.mean((train_pred - y_train)**2)
test_mse = np.mean((test_pred - y_test)**2)

print("Batch Gradient Descent Results:")
print(f"  Final theta: {theta}")
print(f"  Train MSE: {train_mse:.2f}")
print(f"  Test MSE: {test_mse:.2f}")

# Plot convergence
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(loss_history, linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Loss (MSE)')
plt.title('Batch GD: Loss Convergence')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(X_train[:, 1], y_train, alpha=0.5, label='Training Data')
plt.plot(X_train[:, 1], train_pred, 'r-', linewidth=2, label='Fitted Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Batch GD: Fitted Model')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Properties

**Advantages:**
- ✅ Stable convergence
- ✅ Exact gradient computation
- ✅ Works well for convex problems

**Disadvantages:**
- ❌ Slow for large datasets (must process all data each iteration)
- ❌ Memory intensive
- ❌ Can get stuck in local minima

**When to use:**
- Small to medium datasets (< 10,000 samples)
- When you need stable, reproducible results

---

## Stochastic Gradient Descent (SGD)

**Updates parameters using one sample at a time.**

### Implementation

```python
def stochastic_gradient_descent(X, y, learning_rate=0.01, num_epochs=100, random_state=42):
    """
    Stochastic Gradient Descent for Linear Regression.

    Updates using one sample at a time.
    """
    np.random.seed(random_state)
    m, n = X.shape
    theta = np.zeros(n)
    loss_history = []

    for epoch in range(num_epochs):
        # Shuffle data each epoch
        indices = np.random.permutation(m)

        for i in indices:
            # Single sample
            xi = X[i:i+1]
            yi = y[i:i+1]

            # Predict
            prediction = xi.dot(theta)

            # Calculate gradient using SINGLE sample
            gradient = xi.T.dot(prediction - yi)

            # Update parameters
            theta = theta - learning_rate * gradient.flatten()

        # Calculate loss after each epoch
        predictions = X.dot(theta)
        loss = (1/(2*m)) * np.sum((predictions - y)**2)
        loss_history.append(loss)

    return theta, loss_history

# Train with SGD
theta_sgd, loss_history_sgd = stochastic_gradient_descent(
    X_train, y_train, learning_rate=0.01, num_epochs=100
)

# Compare with Batch GD
theta_batch, loss_history_batch = batch_gradient_descent(
    X_train, y_train, learning_rate=0.1, num_iterations=100
)

print("SGD Results:")
print(f"  Final theta: {theta_sgd}")
print(f"  Train MSE: {np.mean((X_train.dot(theta_sgd) - y_train)**2):.2f}")

# Plot comparison
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(loss_history_batch, linewidth=2, label='Batch GD')
plt.plot(loss_history_sgd, linewidth=2, label='SGD')
plt.xlabel('Iteration/Epoch')
plt.ylabel('Loss')
plt.title('Convergence Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(loss_history_batch[-20:], 'o-', linewidth=2, label='Batch GD')
plt.plot(loss_history_sgd[-20:], 's-', linewidth=2, label='SGD')
plt.xlabel('Last 20 Iterations')
plt.ylabel('Loss')
plt.title('Final Convergence (Zoomed)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Properties

**Advantages:**
- ✅ Fast updates
- ✅ Can escape local minima (due to noise)
- ✅ Works with large datasets
- ✅ Online learning possible

**Disadvantages:**
- ❌ Noisy convergence
- ❌ Never truly converges (oscillates around minimum)
- ❌ Requires learning rate decay

**When to use:**
- Large datasets (> 100,000 samples)
- Online learning
- Non-convex problems

**Important clarification on "escaping local minima":**
The noise in SGD can help escape *shallow* local minima, but this is not a reliable mechanism. In practice:
- For convex problems: No local minima exist, so this doesn't apply
- For modern deep learning: Networks are overparameterized with many global minima; the challenge is finding *good* minima that generalize well, not escaping bad ones
- The real benefit of SGD's noise is implicit regularization that helps generalization, not local minima escape

---

## Mini-Batch Gradient Descent

**Best of both worlds: Uses small batches of data.**

### Implementation

```python
def mini_batch_gradient_descent(X, y, batch_size=32, learning_rate=0.01,
                               num_epochs=100, random_state=42):
    """
    Mini-Batch Gradient Descent for Linear Regression.

    Updates using small batches of samples.
    """
    np.random.seed(random_state)
    m, n = X.shape
    theta = np.zeros(n)
    loss_history = []

    for epoch in range(num_epochs):
        # Shuffle data
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        # Process mini-batches
        for i in range(0, m, batch_size):
            # Get batch
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            # Predict
            predictions = X_batch.dot(theta)

            # Calculate gradient using BATCH
            gradient = (1/len(X_batch)) * X_batch.T.dot(predictions - y_batch)

            # Update parameters
            theta = theta - learning_rate * gradient

        # Calculate loss after each epoch
        predictions = X.dot(theta)
        loss = (1/(2*m)) * np.sum((predictions - y)**2)
        loss_history.append(loss)

    return theta, loss_history

# Test different batch sizes
batch_sizes = [1, 16, 32, 128, len(X_train)]
batch_labels = ['SGD (1)', 'Mini-batch (16)', 'Mini-batch (32)',
                'Mini-batch (128)', f'Batch ({len(X_train)})']

plt.figure(figsize=(14, 6))

for batch_size, label in zip(batch_sizes, batch_labels):
    theta, loss_history = mini_batch_gradient_descent(
        X_train, y_train, batch_size=batch_size,
        learning_rate=0.01, num_epochs=50
    )
    plt.plot(loss_history, linewidth=2, label=label)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Mini-Batch Size Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("Batch Size Analysis:")
for batch_size, label in zip(batch_sizes, batch_labels):
    theta, loss_history = mini_batch_gradient_descent(
        X_train, y_train, batch_size=batch_size, learning_rate=0.01, num_epochs=50
    )
    print(f"  {label}: Final loss = {loss_history[-1]:.2f}")
```

### Properties

**Advantages:**
- ✅ Balance between speed and stability
- ✅ Can leverage vectorization
- ✅ Better convergence than SGD
- ✅ Faster than Batch GD

**Disadvantages:**
- ❌ Hyperparameter (batch size) to tune
- ❌ Still has some noise

**When to use:**
- Default choice for most problems
- Typical batch sizes: 32, 64, 128, 256

**Batch size selection guide:**
- **Small batches (8-32)**: Higher gradient noise → acts as regularization, better generalization. Slower per-epoch (more updates). Memory efficient.
- **Medium batches (64-128)**: Balanced noise vs efficiency. Sweet spot for many tasks.
- **Large batches (256-1024+)**: Lower noise → faster convergence per epoch, but may generalize worse. Requires learning rate scaling and longer training.

**Large batch training requires adjustments:**
1. **Learning rate scaling:** Linear scaling rule (Goyal et al., 2017): LR × (BatchSize/BaseBatchSize)
   - Example: If LR=0.1 works for batch=256, use LR=0.4 for batch=1024
   - With warmup to prevent early instability
2. **More epochs:** Large batches see fewer parameter updates per epoch
3. **Batch normalization:** Statistics computed over larger batches may differ

**Why large batches may generalize worse (sharp vs flat minima hypothesis):**
- Keskar et al. (2017): Large batches tend to converge to sharp minima (high eigenvalues of Hessian)
- Small batches find flat minima (low eigenvalues of Hessian)
- Flat minima → more robust to parameter perturbations → better generalization
- **Caveat:** This is empirically observed but theoretically not fully settled. Some work (Hoffer et al., 2017) shows gap can be closed with proper tuning.

---

## Momentum

**Adds fraction of previous update to current update** (like a ball rolling downhill).

### Mathematical Theory of Momentum

Before discussing the algorithm, we establish the rigorous theory.

**Physical Motivation:**
Consider a ball rolling down a surface with friction:

m d²x/dt² = -∇f(x) - γ dx/dt

where:
- m: mass
- γ: friction coefficient
- -∇f(x): gravitational force

Discretizing with appropriate choices of m, γ gives momentum method.

**Heavy Ball Method (Polyak, 1964):**

**Algorithm:**
```
v₀ = 0
v_{t+1} = βv_t - α∇f(x_t)
x_{t+1} = x_t + v_{t+1}
```

where β ∈ [0, 1) is momentum parameter, α > 0 is step size.

**Theorem 6 (Convergence of Heavy Ball):**
For μ-strongly convex and L-smooth function, with:

α = 4 / (√μ + √L)²
β = (√κ - 1)² / (√κ + 1)²  where κ = L/μ

Heavy Ball achieves:

||x_t - x*||² ≤ ((√κ - 1)/(√κ + 1))^t ||x₀ - x*||²

**Convergence rate:** O((1 - 1/√κ)^t)

**Comparison with GD:** O((1 - 1/κ)^t)

**Acceleration:**
- GD: O(κ) iterations
- Heavy Ball: O(√κ) iterations

For large κ >> 1, this is dramatic improvement: √10000 = 100 vs 10000.

**Proof sketch:**
The key insight is that momentum couples consecutive iterates, forming a second-order difference equation. Analyzing its characteristic polynomial shows the improved convergence rate. ∎

**Nesterov's Accelerated Gradient (NAG, 1983):**

NAG achieves the same O(1/√κ) rate but with different update:

**Algorithm:**
```
v₀ = 0
x̃_{t+1} = x_t + βv_t          (look-ahead step)
v_{t+1} = βv_t - α∇f(x̃_{t+1}) (gradient at look-ahead)
x_{t+1} = x_t + v_{t+1}
```

**Key difference:** Gradient evaluated at "lookahead" point x̃_{t+1}.

**Theorem 7 (Optimal Convergence for First-Order Methods):**
For convex and L-smooth functions, NAG achieves:

f(x_t) - f(x*) ≤ 2L||x₀ - x*||² / (t + 1)²

**This is optimal** among all first-order methods (Nemirovski & Yudin, 1983).

**Lower bound:** No first-order method can guarantee better than O(1/t²) for this class.

**Comparison:**
```
Method              | Strongly Convex    | Convex Only
--------------------|--------------------|-----------------
GD                  | O(e^{-t/κ})        | O(1/t)
Heavy Ball / NAG    | O(e^{-t/√κ})       | O(1/t²)
Lower Bound         | Ω(e^{-t/√κ})       | Ω(1/t²)
```

NAG is optimal!

**Why Momentum Works:**

**1. Exponential Moving Average:**
Expanding v_t recursively:

v_t = -α Σ_{i=0}^{t-1} β^i ∇f(x_{t-i-1})

This is exponentially weighted average of past gradients.

**Effective weight** on gradient from i steps ago: β^i

With β = 0.9:
- 1 step ago: weight 0.9
- 10 steps ago: weight 0.9^{10} ≈ 0.35
- 100 steps ago: weight 0.9^{100} ≈ 0.00003

**2. Acceleration in Ravines:**
Consider quadratic f(x) = (1/2)x^T Qx where Q has eigenvalues [λ₁, ..., λ_n].

Without momentum: convergence rate determined by max eigenvalue ratio λ_max/λ_min = κ.

With momentum: effective convergence rate √κ.

**Intuition:** Momentum reduces oscillations in directions with high curvature while accelerating in directions with low curvature.

**3. Noise Reduction:**
In stochastic setting, momentum averages noisy gradients:

E[v_t] = -α/(1-β) E[∇f(x_t)]  (at equilibrium)

Variance reduces by factor related to β.

**Theorem 8 (Momentum for Stochastic Case):**
For SGD with momentum on μ-strongly convex function:

E[||x_t - x*||²] ≤ C₁ e^{-μt/(1-β)} + C₂σ²/(μ(1-β))

where σ² is gradient noise variance.

**Two phases:**
1. **Optimization phase:** Exponential decay dominates (t small)
2. **Fluctuation phase:** Steady-state noise dominates (t large)

**Trade-off:**
- High β (e.g., 0.99): Slow optimization but low fluctuation
- Low β (e.g., 0.5): Fast optimization but high fluctuation
- Common choice: β = 0.9 balances both

### Formula

**Parameterization 1 (used in this guide):**
```
v_t = β × v_{t-1} + α × g_t
θ_{t+1} = θ_t - v_t
```

**Parameterization 2 (common in papers/libraries):**
```
v_t = β × v_{t-1} + g_t
θ_{t+1} = θ_t - α × v_t
```

Where:
- **v_t**: Velocity (exponentially weighted moving average of gradients)
- **g_t = ∇J(θ_t)**: Gradient at step t
- **β**: Momentum coefficient (typically 0.9, controls decay: 0.9 = 90% of previous velocity)
- **α**: Learning rate

**Important:** These are mathematically equivalent (just absorb α into v differently), but:
- Parameterization 1: Tune learning rate α similar to vanilla SGD
- Parameterization 2: Need smaller α than vanilla SGD (by factor of ~1/(1-β))
- Always check which version your library uses!

**Intuition:** Momentum accumulates past gradients with exponential decay. At equilibrium with constant gradient g:
v_∞ = g/(1-β) for param 2, or α×g/(1-β) for param 1.

### Implementation

```python
def sgd_with_momentum(X, y, learning_rate=0.01, momentum=0.9,
                     num_epochs=100, batch_size=32, random_state=42):
    """
    SGD with Momentum.
    """
    np.random.seed(random_state)
    m, n = X.shape
    theta = np.zeros(n)
    velocity = np.zeros(n)
    loss_history = []

    for epoch in range(num_epochs):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            predictions = X_batch.dot(theta)
            gradient = (1/len(X_batch)) * X_batch.T.dot(predictions - y_batch)

            # Momentum update
            velocity = momentum * velocity + learning_rate * gradient
            theta = theta - velocity

        predictions = X.dot(theta)
        loss = (1/(2*m)) * np.sum((predictions - y)**2)
        loss_history.append(loss)

    return theta, loss_history

# Compare SGD vs SGD with Momentum
theta_sgd, loss_sgd = mini_batch_gradient_descent(
    X_train, y_train, batch_size=32, learning_rate=0.01, num_epochs=100
)

theta_momentum, loss_momentum = sgd_with_momentum(
    X_train, y_train, learning_rate=0.01, momentum=0.9, num_epochs=100, batch_size=32
)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(loss_sgd, linewidth=2, label='SGD')
plt.plot(loss_momentum, linewidth=2, label='SGD + Momentum')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('SGD vs Momentum')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(loss_sgd[-30:], 'o-', linewidth=2, label='SGD')
plt.plot(loss_momentum[-30:], 's-', linewidth=2, label='SGD + Momentum')
plt.xlabel('Last 30 Epochs')
plt.ylabel('Loss')
plt.title('Final Convergence (Zoomed)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Momentum Analysis:")
print(f"  SGD final loss: {loss_sgd[-1]:.2f}")
print(f"  Momentum final loss: {loss_momentum[-1]:.2f}")
print(f"  Improvement: {((loss_sgd[-1] - loss_momentum[-1])/loss_sgd[-1])*100:.1f}%")
```

### Properties

**Advantages:**
- ✅ Faster convergence
- ✅ Better for ravines (dimensions with different curvatures)
- ✅ Dampens oscillations
- ✅ Can escape shallow local minima

**Disadvantages:**
- ❌ Additional hyperparameter (momentum)
- ❌ Can overshoot minimum

**When to use:**
- Almost always! Standard improvement over vanilla SGD
- Typical momentum: 0.9

---

## Nesterov Accelerated Gradient

**"Look-ahead" momentum** - evaluates gradient at future position.

### Implementation

```python
def nesterov_momentum(X, y, learning_rate=0.01, momentum=0.9,
                     num_epochs=100, batch_size=32, random_state=42):
    """
    Nesterov Accelerated Gradient.
    """
    np.random.seed(random_state)
    m, n = X.shape
    theta = np.zeros(n)
    velocity = np.zeros(n)
    loss_history = []

    for epoch in range(num_epochs):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            # Look-ahead position
            theta_lookahead = theta - momentum * velocity

            # Gradient at look-ahead position
            predictions = X_batch.dot(theta_lookahead)
            gradient = (1/len(X_batch)) * X_batch.T.dot(predictions - y_batch)

            # Update
            velocity = momentum * velocity + learning_rate * gradient
            theta = theta - velocity

        predictions = X.dot(theta)
        loss = (1/(2*m)) * np.sum((predictions - y)**2)
        loss_history.append(loss)

    return theta, loss_history

# Compare
theta_nag, loss_nag = nesterov_momentum(
    X_train, y_train, learning_rate=0.01, momentum=0.9, num_epochs=100, batch_size=32
)

plt.figure(figsize=(10, 6))
plt.plot(loss_sgd, linewidth=2, label='SGD', alpha=0.7)
plt.plot(loss_momentum, linewidth=2, label='Momentum', alpha=0.7)
plt.plot(loss_nag, linewidth=2, label='Nesterov', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Comparing Momentum Methods')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("Nesterov Comparison:")
print(f"  SGD: {loss_sgd[-1]:.2f}")
print(f"  Momentum: {loss_momentum[-1]:.2f}")
print(f"  Nesterov: {loss_nag[-1]:.2f}")
```

---

## AdaGrad

**Adapts learning rate for each parameter** based on historical gradients.

### Mathematical Theory of AdaGrad

**Theorem 9 (AdaGrad Regret Bound - Duchi et al., 2011):**
For online convex optimization, AdaGrad achieves the following regret bound:

R_T = Σ_{t=1}^T [f_t(x_t) - f_t(x*)]

≤ (||x*||₂ / (2α)) Σᵢ √(Σ_{t=1}^T g_{t,i}²) + (α/2) Σ_{t=1}^T ||g_t||²_{G_t^{-1}}

where g_t is the gradient at step t, G_t = diag(Σ_{τ=1}^t g_τ²).

**For sparse gradients** (at most s non-zero entries per g_t):
R_T ≤ O(√s · T)

This is much better than O(√d · T) for standard GD when s << d!

**Proof sketch:**
AdaGrad uses adaptive learning rates η_{t,i} = α/√(G_{t,i}):

||x_{t+1} - x*||² = ||x_t - η_t ⊙ g_t - x*||²
                   = ||x_t - x*||² - 2(x_t - x*)ᵀ(η_t ⊙ g_t) + ||η_t ⊙ g_t||²

By telescoping and careful analysis of the adaptive rates, the bound follows. ∎

**Theorem 10 (AdaGrad Convergence for Convex Functions):**
For convex function f with ||∇f(x)|| ≤ G for all x, AdaGrad with α = 1 achieves:

f(x̄_T) - f(x*) ≤ O(G·||x*|| / √T)

where x̄_T = (1/T) Σ_{t=1}^T x_t is the average iterate.

**Convergence rate:** O(1/√T) - same as SGD but with automatic adaptation!

**Key advantage:** Works well for sparse gradients and non-uniform learning rates.

**Theorem 11 (AdaGrad Learning Rate Decay Analysis):**
The effective learning rate for coordinate i at step T is:

α_T,i = α / √(Σ_{t=1}^T g_{t,i}²)

**Decay behavior:**
- For constant gradient g_i: α_T,i ≈ α/(g_i√T) → 0 at rate O(1/√T)
- For sparse updates: coordinates updated k < T times have α_T,i ≈ α/√k

**Fatal flaw:** Learning rates decay too aggressively for non-convex problems!

**Example:**
- Initial LR: α = 0.1
- Gradient magnitudes: g_i ≈ 0.5 (constant)
- After T = 100 steps: Σg² ≈ 25, effective LR ≈ 0.1/5 = 0.02 (5× reduction)
- After T = 10,000 steps: Σg² ≈ 2,500, effective LR ≈ 0.1/50 = 0.002 (50× reduction!)

Eventually α_T,i → 0, stopping all learning.

### Formula

```
G_t = G_{t-1} + g_t²
θ_{t+1} = θ_t - (α / √(G_t + ε)) ⊙ g_t
```

Where:
- **G_t**: Cumulative sum of squared gradients (element-wise), G_t ∈ ℝⁿ
- **g_t = ∇J(θ_t)**: Gradient at step t
- **ε**: Small constant for numerical stability (typically 10⁻⁸)
- **⊙**: Element-wise multiplication

**Key insight:** Each parameter gets learning rate inversely proportional to √(sum of its squared historical gradients).
- Frequently updated parameters (large Σg²) → small effective learning rate
- Rarely updated parameters (small Σg²) → large effective learning rate

**Fatal flaw:** G_t grows monotonically, so effective learning rate α/√(G_t) → 0 as t → ∞, eventually stopping all learning.

### Implementation

```python
def adagrad(X, y, learning_rate=0.1, num_epochs=100, batch_size=32,
           epsilon=1e-8, random_state=42):
    """
    AdaGrad optimizer.
    """
    np.random.seed(random_state)
    m, n = X.shape
    theta = np.zeros(n)
    G = np.zeros(n)  # Accumulated squared gradients
    loss_history = []

    for epoch in range(num_epochs):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            predictions = X_batch.dot(theta)
            gradient = (1/len(X_batch)) * X_batch.T.dot(predictions - y_batch)

            # Accumulate squared gradients
            G = G + gradient**2

            # Adaptive learning rate
            adjusted_lr = learning_rate / (np.sqrt(G) + epsilon)

            # Update
            theta = theta - adjusted_lr * gradient

        predictions = X.dot(theta)
        loss = (1/(2*m)) * np.sum((predictions - y)**2)
        loss_history.append(loss)

    return theta, loss_history, G

# Train
theta_adagrad, loss_adagrad, G = adagrad(
    X_train, y_train, learning_rate=0.5, num_epochs=100, batch_size=32
)

print("AdaGrad Results:")
print(f"  Final loss: {loss_adagrad[-1]:.2f}")
print(f"  Accumulated gradients (G): {G}")
print(f"  Effective learning rates: {0.5 / (np.sqrt(G) + 1e-8)}")
```

### Properties

**Advantages:**
- ✅ No manual learning rate tuning needed
- ✅ Works well for sparse data
- ✅ Different learning rate per parameter

**Disadvantages:**
- ❌ Learning rate continually decreases
- ❌ Can stop learning too early
- ❌ Accumulation term grows monotonically

**When to use:**
- Sparse features (NLP, recommender systems)
- When parameters have very different scales

---

## RMSprop

**Fixes AdaGrad's learning rate decay** using moving average.

### Mathematical Theory of RMSprop

**Theorem 12 (RMSprop Convergence for Non-Convex - Tieleman & Hinton, 2012):**
For non-convex L-smooth function f, RMSprop with appropriate parameters achieves:

min_{t≤T} E[||∇f(x_t)||²] ≤ O((√d·L·(f(x₀) - f*)) / √T)

where d is dimension and f* is optimal value.

**Convergence rate:** O(1/√T) to stationary point (∇f = 0)

**Key difference from AdaGrad:** RMSprop maintains bounded effective learning rates!

**Theorem 13 (RMSprop Effective Learning Rate Bounds):**
For coordinate i with exponential moving average parameter β:

α_min,i ≤ α_t,i ≤ α_max,i

where:
- α_min,i = α / √(g_max² + ε)  (when v_t,i = g_max²)
- α_max,i = α / √ε  (when v_t,i → 0)

**Bounded learning rates** prevent complete decay!

**Analysis of exponential moving average:**
v_t = β·v_{t-1} + (1-β)·g_t²

At equilibrium with constant gradient g:
v_∞ = (1-β)·g² / (1-(1-β)) = g²

So effective LR: α_eff = α/√(g² + ε) ≈ α/|g| (constant!)

**Theorem 14 (RMSprop Memory Length):**
The effective memory length of RMSprop is:

T_mem ≈ 1/(1-β)

**Intuition:** Gradients from T_mem steps ago have weight ≈ β^T_mem ≈ 1/e ≈ 0.37

**Common values:**
- β = 0.9: T_mem ≈ 10 steps
- β = 0.99: T_mem ≈ 100 steps
- β = 0.999: T_mem ≈ 1,000 steps

**Trade-offs:**
- Small β (0.9): Fast adaptation, noisy estimates
- Large β (0.999): Slow adaptation, stable estimates
- Default: β = 0.9 balances adaptation vs stability

**Theorem 15 (RMSprop vs AdaGrad Effective LR):**
After T steps with constant gradient g:

AdaGrad: α_T = α/(|g|√T) → 0 as T → ∞
RMSprop: α_T ≈ α/|g| → constant

**Ratio:** RMSprop LR is √T times larger than AdaGrad!

For T = 10,000: RMSprop LR is 100× larger than AdaGrad.

This is why RMSprop works for non-convex optimization (doesn't stop learning).

### Formula

```
v_t = β × v_{t-1} + (1-β) × g_t²
θ_{t+1} = θ_t - (α / √(v_t + ε)) ⊙ g_t
```

Alternative notation:
```
E[g²]_t = β × E[g²]_{t-1} + (1-β) × g_t²
θ_{t+1} = θ_t - (α / √(E[g²]_t + ε)) ⊙ g_t
```

Where:
- **v_t** (or **E[g²]_t**): Exponentially weighted moving average of squared gradients
- **β**: Decay rate (typically 0.9), controls memory length ≈ 1/(1-β) steps
- **g_t = ∇J(θ_t)**: Current gradient

**Fixes AdaGrad:** Instead of summing all past squared gradients, uses exponentially decaying average:
- AdaGrad: G_t = Σ_{i=1}^t g_i² (grows forever)
- RMSprop: v_t ≈ (1-β) × Σ_{i=1}^t β^{t-i} g_i² (forgets distant past)

With β=0.9, gradients from 10 steps ago have weight ~0.35, from 20 steps ago ~0.12.

### Implementation

```python
def rmsprop(X, y, learning_rate=0.01, decay_rate=0.9, num_epochs=100,
           batch_size=32, epsilon=1e-8, random_state=42):
    """
    RMSprop optimizer.
    """
    np.random.seed(random_state)
    m, n = X.shape
    theta = np.zeros(n)
    E_g2 = np.zeros(n)  # Moving average of squared gradients
    loss_history = []

    for epoch in range(num_epochs):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            predictions = X_batch.dot(theta)
            gradient = (1/len(X_batch)) * X_batch.T.dot(predictions - y_batch)

            # Moving average of squared gradients
            E_g2 = decay_rate * E_g2 + (1 - decay_rate) * gradient**2

            # Adaptive learning rate
            adjusted_lr = learning_rate / (np.sqrt(E_g2) + epsilon)

            # Update
            theta = theta - adjusted_lr * gradient

        predictions = X.dot(theta)
        loss = (1/(2*m)) * np.sum((predictions - y)**2)
        loss_history.append(loss)

    return theta, loss_history

# Train
theta_rmsprop, loss_rmsprop = rmsprop(
    X_train, y_train, learning_rate=0.01, decay_rate=0.9, num_epochs=100, batch_size=32
)

# Compare AdaGrad vs RMSprop
plt.figure(figsize=(10, 6))
plt.plot(loss_adagrad, linewidth=2, label='AdaGrad')
plt.plot(loss_rmsprop, linewidth=2, label='RMSprop')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('AdaGrad vs RMSprop')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("RMSprop Comparison:")
print(f"  AdaGrad final loss: {loss_adagrad[-1]:.2f}")
print(f"  RMSprop final loss: {loss_rmsprop[-1]:.2f}")
```

### Properties

**Advantages:**
- ✅ Fixes AdaGrad's learning rate decay
- ✅ Works well for non-stationary problems
- ✅ Good for RNNs

**Disadvantages:**
- ❌ Still requires learning rate tuning
- ❌ Additional hyperparameter (decay rate)

**When to use:**
- RNNs and LSTMs
- Non-stationary objectives
- Default decay_rate: 0.9

---

## Adam

**Adaptive Moment Estimation** - combines Momentum and RMSprop.

### Mathematical Theory of Adam

**Theorem 16 (Adam Algorithm - Kingma & Ba, 2015):**
Adam maintains two exponential moving averages:

m_t = β₁·m_{t-1} + (1-β₁)·g_t  (first moment, momentum)
v_t = β₂·v_{t-1} + (1-β₂)·g_t²  (second moment, adaptive LR)

With bias correction:
m̂_t = m_t / (1-β₁^t)
v̂_t = v_t / (1-β₂^t)

Update: x_{t+1} = x_t - α·m̂_t / (√v̂_t + ε)

**Why two moments?**
- m_t: Provides momentum (acceleration + noise reduction)
- v_t: Provides adaptive learning rates (per-parameter scaling)

**Theorem 17 (Adam Convergence - Original Claim):**
The original Adam paper claimed:

E[f(x_T)] - f(x*) ≤ O(1/√T)

However, **this claim was shown to be incorrect!**

**Theorem 18 (Adam Non-Convergence - Reddi et al., 2018):**
Adam can **fail to converge** even for simple convex problems.

**Counterexample:**
Consider online optimization with:
- f₁(x) = 1,010·x at t ∈ {1, 2, 3}
- f_t(x) = -x for t mod 3 = 0

Optimal: x* = -1

With parameters β₁ = 0, β₂ = 1/(1+10²):
- Adam converges to x_∞ ≈ +1 (wrong direction!)
- SGD converges to x* = -1 (correct)

**Root cause:** v_t can **decrease** over time, causing learning rate to increase
unexpectedly and overshoot.

**Theorem 19 (AMSGrad Fix - Reddi et al., 2018):**
AMSGrad guarantees convergence by using:

v̂_t = max(v̂_{t-1}, v_t / (1-β₂^t))

**Key change:** v̂_t is non-decreasing, ensuring learning rates never increase.

**Convergence guarantee:**
For convex functions:
E[f(x_T)] - f(x*) ≤ O(1/√T)

For non-convex functions:
min_{t≤T} E[||∇f(x_t)||²] ≤ O(1/√T)

**Theorem 20 (Adam Bias Correction Necessity):**
Without bias correction (using m_t and v_t directly), Adam is biased toward zero
in early iterations.

**Expected value** with constant gradient g and starting from m₀ = v₀ = 0:
E[m_t] = (1-β₁^t)·g ≠ g  (biased!)
E[v_t] = (1-β₂^t)·g² ≠ g²  (biased!)

**Corrected:**
E[m̂_t] = g  (unbiased!)
E[v̂_t] = g²  (unbiased!)

**Impact magnitude** for β₁ = 0.9, β₂ = 0.999:
- Step 1: bias factors 0.1 and 0.001 (10× and 1000× underestimate!)
- Step 10: bias factors 0.651 and 0.010
- Step 100: bias factors 0.99997 and 0.095
- Step 1000: bias factors ≈ 1.0 and 0.632

Bias correction critical for first ~1000 steps when β₂ = 0.999!

**Theorem 21 (Adam Effective Learning Rate Analysis):**
The effective learning rate for coordinate i at step t is:

α_eff,t,i = α·m̂_t,i / √(v̂_t,i + ε)

**Bounds:** For |m̂_t,i| ≤ |g_max| and v̂_t,i ∈ [v_min, v_max]:

α·|g_max| / √(v_max + ε) ≤ |α_eff,t,i| ≤ α·|g_max| / √(v_min + ε)

**Typical behavior:**
- High curvature directions (large v̂): small effective LR
- Low curvature directions (small v̂): large effective LR
- Momentum |m̂| can exceed |g| by factor ~1/(1-β₁) ≈ 10 for β₁ = 0.9

**Theorem 22 (Adam Hyperparameter Sensitivity):**
The optimal hyperparameters depend on problem characteristics:

**β₁ (momentum):**
- High β₁ (0.95-0.99): Good for smooth, deterministic gradients
- Low β₁ (0.8-0.9): Good for noisy, stochastic gradients
- Default: β₁ = 0.9 (robust choice)

**β₂ (adaptive LR):**
- High β₂ (0.995-0.9999): Good for sparse gradients, stable curvature
- Low β₂ (0.95-0.99): Good for non-stationary gradients
- Default: β₂ = 0.999 (works for most cases)

**α (learning rate):**
- Typical range: [10⁻⁵, 10⁻²]
- Deep learning default: α = 10⁻³
- Needs tuning: multiply by {0.1, 1, 10} grid search

**ε (numerical stability):**
- Default: ε = 10⁻⁸ (sufficient for float32)
- Use ε = 10⁻⁴ for float16 to avoid division by tiny numbers

**Theorem 23 (Adam vs SGD+Momentum Generalization):**
**Empirical finding (Wilson et al., 2017):**
Adam often achieves:
- Faster convergence (fewer iterations)
- Worse generalization (higher test error)

than SGD+Momentum with learning rate schedule.

**Hypothesized causes:**
1. Adaptive learning rates reduce effective noise in SGD (implicit regularization)
2. Adam converges to sharper minima (poor generalization)
3. Large effective learning rates from momentum accumulation

**Practical recommendation:**
- Use Adam for: quick prototyping, initial training, RL, NLP (transformers)
- Use SGD+Momentum for: final training, computer vision, best generalization
- Hybrid: Start with Adam, fine-tune with SGD+Momentum

### Formula

```
m_t = β₁ × m_{t-1} + (1-β₁) × ∇J(θ)        # First moment (mean)
v_t = β₂ × v_{t-1} + (1-β₂) × (∇J(θ))²     # Second moment (uncentered variance)
m̂_t = m_t / (1 - β₁ᵗ)                      # Bias correction
v̂_t = v_t / (1 - β₂ᵗ)                      # Bias correction
θ = θ - α × m̂_t / (√v̂_t + ε)
```

**Why bias correction is essential:**
Since m₀ = 0 and v₀ = 0, the moment estimates are biased toward zero, especially in early iterations.

**Mathematical justification:** For constant gradient g, the expected value is:
- True first moment: E[g] = g
- Biased estimate: m_t = (1-β₁) Σ_{i=1}^t β₁^{t-i} g = (1-β₁^t) g
- Bias: E[m_t] = (1-β₁^t) g ≠ g
- Corrected: E[m̂_t] = E[m_t/(1-β₁^t)] = g ✓

**Numerical example** with β₁=0.9 and constant gradient g:
- Step 1: m₁ = 0.1g, bias=(1-0.9¹)=0.1, m̂₁ = 0.1g/0.1 = g ✓
- Step 2: m₂ = 0.19g, bias=(1-0.9²)=0.19, m̂₂ = 0.19g/0.19 = g ✓
- Step 10: m₁₀ ≈ 0.651g, bias≈0.651, m̂₁₀ ≈ g ✓
- Step ∞: m∞ → g, bias → 1, correction → 1 (no effect)

**Decay of bias:** β₁ᵗ decays exponentially. For β₁=0.9: after t=10 steps → bias factor ≈0.65, after t=50 → ≈0.005 (negligible).

### Implementation

```python
def adam(X, y, learning_rate=0.001, beta1=0.9, beta2=0.999,
        num_epochs=100, batch_size=32, epsilon=1e-8, random_state=42):
    """
    Adam optimizer.
    """
    np.random.seed(random_state)
    m, n = X.shape
    theta = np.zeros(n)
    m_t = np.zeros(n)  # First moment
    v_t = np.zeros(n)  # Second moment
    t = 0
    loss_history = []

    for epoch in range(num_epochs):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, m, batch_size):
            t += 1

            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            predictions = X_batch.dot(theta)
            gradient = (1/len(X_batch)) * X_batch.T.dot(predictions - y_batch)

            # Update biased first moment estimate
            m_t = beta1 * m_t + (1 - beta1) * gradient

            # Update biased second moment estimate
            v_t = beta2 * v_t + (1 - beta2) * gradient**2

            # Compute bias-corrected moment estimates
            m_hat = m_t / (1 - beta1**t)
            v_hat = v_t / (1 - beta2**t)

            # Update parameters
            theta = theta - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        predictions = X.dot(theta)
        loss = (1/(2*m)) * np.sum((predictions - y)**2)
        loss_history.append(loss)

    return theta, loss_history

# Train
theta_adam, loss_adam = adam(
    X_train, y_train, learning_rate=0.01, beta1=0.9, beta2=0.999,
    num_epochs=100, batch_size=32
)

print("Adam Results:")
print(f"  Final loss: {loss_adam[-1]:.2f}")
print(f"  Final theta: {theta_adam}")
```

### Properties

**Advantages:**
- ✅ Combines benefits of Momentum and RMSprop
- ✅ Works well with little tuning
- ✅ Efficient and widely used
- ✅ Good default choice

**Disadvantages:**
- ❌ Can fail to converge in some cases (particularly on simple convex problems)
- ❌ May not converge to optimal solution for some non-convex problems
- ❌ Requires tuning of three hyperparameters (though default values work well)

**When to use:**
- Default choice for deep learning
- Typical values: α=0.001, β₁=0.9, β₂=0.999
- Reduce α if training is unstable; increase if converging too slowly

**Important notes:**
- Adam can sometimes generalize worse than SGD+Momentum for some tasks
- For best generalization, consider using SGD+Momentum with learning rate schedule
- Adam works best when you need fast convergence and good enough (not optimal) solutions

---

## Learning Rate Schedules

### Step Decay

```python
def step_decay_schedule(initial_lr, decay_rate=0.5, decay_steps=10):
    """
    Reduce learning rate by factor every N steps.
    """
    def schedule(epoch):
        return initial_lr * (decay_rate ** (epoch // decay_steps))
    return schedule

# Example
initial_lr = 0.1
schedule = step_decay_schedule(initial_lr, decay_rate=0.5, decay_steps=10)

epochs = list(range(50))
lrs = [schedule(e) for e in epochs]

plt.figure(figsize=(10, 6))
plt.plot(epochs, lrs, linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Step Decay Learning Rate Schedule')
plt.grid(True, alpha=0.3)
plt.show()
```

### Exponential Decay

```python
def exponential_decay_schedule(initial_lr, decay_rate=0.96):
    """
    Exponentially decay learning rate.
    """
    def schedule(epoch):
        return initial_lr * (decay_rate ** epoch)
    return schedule

schedule_exp = exponential_decay_schedule(0.1, decay_rate=0.96)
lrs_exp = [schedule_exp(e) for e in epochs]

plt.figure(figsize=(10, 6))
plt.plot(epochs, lrs, linewidth=2, label='Step Decay')
plt.plot(epochs, lrs_exp, linewidth=2, label='Exponential Decay')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedules')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Cosine Annealing

```python
def cosine_annealing_schedule(initial_lr, T_max):
    """
    Cosine annealing schedule.
    """
    def schedule(epoch):
        return initial_lr * (1 + np.cos(np.pi * epoch / T_max)) / 2
    return schedule

schedule_cos = cosine_annealing_schedule(0.1, T_max=50)
lrs_cos = [schedule_cos(e) for e in epochs]

plt.figure(figsize=(10, 6))
plt.plot(epochs, lrs_cos, linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Cosine Annealing Schedule')
plt.grid(True, alpha=0.3)
plt.show()
```

### One Cycle Policy

```python
def one_cycle_schedule(max_lr, total_steps, pct_start=0.3):
    """
    One Cycle learning rate policy.
    """
    def schedule(step):
        if step < pct_start * total_steps:
            # Warm-up phase
            return max_lr * step / (pct_start * total_steps)
        else:
            # Annealing phase
            progress = (step - pct_start * total_steps) / ((1 - pct_start) * total_steps)
            return max_lr * (1 + np.cos(np.pi * progress)) / 2
    return schedule

schedule_one_cycle = one_cycle_schedule(0.1, total_steps=50)
steps = list(range(50))
lrs_one_cycle = [schedule_one_cycle(s) for s in steps]

plt.figure(figsize=(10, 6))
plt.plot(steps, lrs_one_cycle, linewidth=2)
plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.title('One Cycle Learning Rate Policy')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Comparing Optimizers

### Complete Comparison

```python
from sklearn.datasets import make_regression

# Generate more complex dataset
X_comp, y_comp = make_regression(n_samples=1000, n_features=20, noise=20, random_state=42)
X_train_comp, X_test_comp, y_train_comp, y_test_comp = train_test_split(
    X_comp, y_comp, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_comp = scaler.fit_transform(X_train_comp)
X_test_comp = scaler.transform(X_test_comp)

X_train_comp = np.hstack([np.ones((X_train_comp.shape[0], 1)), X_train_comp])
X_test_comp = np.hstack([np.ones((X_test_comp.shape[0], 1)), X_test_comp])

# Train all optimizers
optimizers = {
    'SGD': lambda: mini_batch_gradient_descent(
        X_train_comp, y_train_comp, batch_size=32, learning_rate=0.01, num_epochs=50
    ),
    'Momentum': lambda: sgd_with_momentum(
        X_train_comp, y_train_comp, learning_rate=0.01, momentum=0.9, num_epochs=50, batch_size=32
    ),
    'Nesterov': lambda: nesterov_momentum(
        X_train_comp, y_train_comp, learning_rate=0.01, momentum=0.9, num_epochs=50, batch_size=32
    ),
    'AdaGrad': lambda: adagrad(
        X_train_comp, y_train_comp, learning_rate=0.5, num_epochs=50, batch_size=32
    )[:2],
    'RMSprop': lambda: rmsprop(
        X_train_comp, y_train_comp, learning_rate=0.01, decay_rate=0.9, num_epochs=50, batch_size=32
    ),
    'Adam': lambda: adam(
        X_train_comp, y_train_comp, learning_rate=0.01, beta1=0.9, beta2=0.999, num_epochs=50, batch_size=32
    )
}

results = {}
for name, optimizer_fn in optimizers.items():
    print(f"Training with {name}...")
    theta, loss_history = optimizer_fn()
    results[name] = {
        'theta': theta,
        'loss_history': loss_history,
        'final_loss': loss_history[-1],
        'test_mse': np.mean((X_test_comp.dot(theta) - y_test_comp)**2)
    }

# Plot comparison
plt.figure(figsize=(15, 10))

# Plot 1: All convergence curves
plt.subplot(2, 2, 1)
for name, result in results.items():
    plt.plot(result['loss_history'], linewidth=2, label=name)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Convergence Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Final 20 epochs (zoomed)
plt.subplot(2, 2, 2)
for name, result in results.items():
    plt.plot(result['loss_history'][-20:], linewidth=2, label=name, marker='o')
plt.xlabel('Last 20 Epochs')
plt.ylabel('Loss')
plt.title('Final Convergence (Zoomed)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Final train loss
plt.subplot(2, 2, 3)
names = list(results.keys())
final_losses = [results[name]['final_loss'] for name in names]
plt.bar(names, final_losses)
plt.ylabel('Final Training Loss')
plt.title('Final Training Loss Comparison')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')

# Plot 4: Test MSE
plt.subplot(2, 2, 4)
test_mses = [results[name]['test_mse'] for name in names]
plt.bar(names, test_mses)
plt.ylabel('Test MSE')
plt.title('Test Performance Comparison')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Print summary
print("\nOptimizer Comparison Summary:")
print("="*60)
for name, result in results.items():
    print(f"{name:15} | Train Loss: {result['final_loss']:8.2f} | Test MSE: {result['test_mse']:8.2f}")
```

### Performance Table

| Optimizer | Speed | Convergence Quality | Hyperparameter Sensitivity | Memory | Known Failure Cases |
|-----------|-------|---------------------|----------------------------|--------|-------------------|
| SGD | Fast | Poor without schedule | High (LR critical) | Low | Plateaus, ravines |
| Momentum | Fast | Good | Medium | Low | Still struggles with ravines |
| Nesterov | Fast | Good | Medium | Low | Marginal improvement over Momentum |
| AdaGrad | Medium | Good for sparse | Low | Medium | Stops learning (accumulator grows indefinitely) |
| RMSprop | Medium | Good | Low | Medium | None major (unpublished, less tested) |
| Adam | Medium | Very Good | Very Low | Medium | Sometimes converges to worse minima than SGD |

**Critical insights:**
- **AdaGrad's fatal flaw**: Learning rate → 0 as training progresses (G keeps growing). Never use for long training runs.
- **Adam vs SGD generalization gap**: Adam often converges faster but SGD+Momentum may find better minima. For state-of-the-art results (competitions, research), try both.
- **RMSprop**: Fixes AdaGrad but less theoretically grounded (was unpublished until 2012, from Hinton's course). Adam is usually preferred.
- **Nesterov**: Theoretically better convergence rate, but empirically often similar to Momentum. Worth trying but not always better.

---

## Practical Tips

### 1. Choosing Learning Rate

```python
def find_learning_rate(X, y, min_lr=1e-5, max_lr=1, num_iterations=100):
    """
    Learning rate finder - exponentially increase lr and track loss.
    """
    lrs = np.logspace(np.log10(min_lr), np.log10(max_lr), num_iterations)
    losses = []

    m, n = X.shape
    theta = np.zeros(n)

    for lr in lrs:
        # Single gradient update
        predictions = X.dot(theta)
        loss = (1/(2*m)) * np.sum((predictions - y)**2)
        losses.append(loss)

        gradient = (1/m) * X.T.dot(predictions - y)
        theta = theta - lr * gradient

        # Stop if loss explodes
        if loss > 10 * min(losses) if losses else float('inf'):
            break

    return lrs[:len(losses)], losses

lrs, losses = find_learning_rate(X_train_comp, y_train_comp)

plt.figure(figsize=(10, 6))
plt.plot(lrs, losses, linewidth=2)
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.title('Learning Rate Finder')
plt.grid(True, alpha=0.3)

# Find optimal lr (steepest descent)
gradients = np.gradient(losses)
optimal_idx = np.argmin(gradients)
optimal_lr = lrs[optimal_idx]
plt.axvline(x=optimal_lr, color='r', linestyle='--',
           label=f'Optimal LR: {optimal_lr:.2e}')
plt.legend()
plt.show()

print(f"Recommended learning rate: {optimal_lr:.2e}")
print("\nHow to interpret the LR finder plot:")
print("1. Left side (low LR): Loss decreases slowly → too conservative")
print("2. Middle (optimal range): Loss decreases rapidly → steepest part of curve")
print("3. Right side (high LR): Loss increases/explodes → unstable")
print("4. Pick LR at steepest descent, or 10x smaller for safety")
print("5. If using one-cycle policy, pick max_lr at steepest point")
```

### 2. Gradient Clipping

```python
def adam_with_gradient_clipping(X, y, learning_rate=0.001, max_grad_norm=1.0,
                                beta1=0.9, beta2=0.999, num_epochs=100,
                                batch_size=32, epsilon=1e-8, random_state=42):
    """
    Adam with gradient clipping.
    """
    np.random.seed(random_state)
    m, n = X.shape
    theta = np.zeros(n)
    m_t = np.zeros(n)
    v_t = np.zeros(n)
    t = 0
    loss_history = []

    for epoch in range(num_epochs):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, m, batch_size):
            t += 1

            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            predictions = X_batch.dot(theta)
            gradient = (1/len(X_batch)) * X_batch.T.dot(predictions - y_batch)

            # Clip gradient
            grad_norm = np.linalg.norm(gradient)
            if grad_norm > max_grad_norm:
                gradient = gradient * (max_grad_norm / grad_norm)

            m_t = beta1 * m_t + (1 - beta1) * gradient
            v_t = beta2 * v_t + (1 - beta2) * gradient**2

            m_hat = m_t / (1 - beta1**t)
            v_hat = v_t / (1 - beta2**t)

            theta = theta - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        predictions = X.dot(theta)
        loss = (1/(2*m)) * np.sum((predictions - y)**2)
        loss_history.append(loss)

    return theta, loss_history

print("Gradient clipping prevents exploding gradients")
print("Useful for RNNs and unstable training")
print("\nWhen gradient clipping is necessary vs optional:")
print("NECESSARY:")
print("  - RNNs/LSTMs (prone to exploding gradients due to backprop through time)")
print("  - Very deep networks without batch normalization")
print("  - Training with high learning rates")
print("  - When you observe NaN losses or gradient explosion")
print("\nOPTIONAL (may help but not critical):")
print("  - Well-normalized networks with batch norm")
print("  - Residual networks (ResNets)")
print("  - When using adaptive optimizers like Adam")
print("\nHow it works:")
print("  If ||gradient|| > threshold: gradient = gradient × (threshold / ||gradient||)")
print("  This rescales gradient to have max norm = threshold, preserving direction")
print("  Common thresholds: 0.5 to 5.0 depending on model architecture")
```

### 3. Warmup

```python
def adam_with_warmup(X, y, max_lr=0.001, warmup_epochs=5, total_epochs=100,
                     beta1=0.9, beta2=0.999, batch_size=32, epsilon=1e-8, random_state=42):
    """
    Adam with learning rate warmup.
    """
    np.random.seed(random_state)
    m, n = X.shape
    theta = np.zeros(n)
    m_t = np.zeros(n)
    v_t = np.zeros(n)
    t = 0
    loss_history = []

    for epoch in range(total_epochs):
        # Calculate current learning rate with warmup
        if epoch < warmup_epochs:
            # Linear warmup
            current_lr = max_lr * (epoch + 1) / warmup_epochs
        else:
            current_lr = max_lr

        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, m, batch_size):
            t += 1
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            predictions = X_batch.dot(theta)
            gradient = (1/len(X_batch)) * X_batch.T.dot(predictions - y_batch)

            m_t = beta1 * m_t + (1 - beta1) * gradient
            v_t = beta2 * v_t + (1 - beta2) * gradient**2

            m_hat = m_t / (1 - beta1**t)
            v_hat = v_t / (1 - beta2**t)

            theta = theta - current_lr * m_hat / (np.sqrt(v_hat) + epsilon)

        predictions = X.dot(theta)
        loss = (1/(2*m)) * np.sum((predictions - y)**2)
        loss_history.append(loss)

    return theta, loss_history

print("Learning rate warmup: Why and when?")
print("\nWHY warmup helps:")
print("  - At start, parameters are random and gradients may be large/unstable")
print("  - High LR + large gradients = unstable updates, potential divergence")
print("  - Warmup allows model to find good region of parameter space first")
print("  - Particularly important with large batch sizes (batch size ≥ 1024)")
print("\nWHEN to use warmup:")
print("  - BERT-style transformers (essential)")
print("  - Large batch training (batch size > 512)")
print("  - Very deep networks (50+ layers)")
print("  - Training from random initialization")
print("\nWHEN warmup is unnecessary:")
print("  - Small batch training (≤ 256)")
print("  - Fine-tuning pretrained models")
print("  - Shallow networks")
print("  - When using conservative learning rates")
print("\nTypical warmup duration: 1-10% of total training steps")
```

### Quick Decision Guide

```
Choose optimizer:
│
├─ Need fastest convergence with minimal tuning?
│  └─ Use Adam (default choice)
│
├─ Limited memory?
│  └─ Use SGD with Momentum
│
├─ Sparse features (NLP, recommender systems)?
│  └─ Use AdaGrad or RMSprop
│
├─ Training RNN/LSTM?
│  └─ Use RMSprop or Adam
│
└─ Want simple and interpretable?
   └─ Use SGD with learning rate schedule
```

---

## Complete Examples

### Example: Training Neural Network with Different Optimizers

```python
import tensorflow as tf
from tensorflow import keras

# Generate dataset
X_nn, y_nn = make_regression(n_samples=5000, n_features=20, noise=10, random_state=42)
X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(
    X_nn, y_nn, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_nn = scaler.fit_transform(X_train_nn)
X_test_nn = scaler.transform(X_test_nn)

# Define model architecture
def create_model():
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(20,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])
    return model

# Test different optimizers
optimizers_keras = {
    'SGD': keras.optimizers.SGD(learning_rate=0.01),
    'SGD+Momentum': keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    'RMSprop': keras.optimizers.RMSprop(learning_rate=0.001),
    'Adam': keras.optimizers.Adam(learning_rate=0.001),
    'AdaGrad': keras.optimizers.Adagrad(learning_rate=0.01)
}

histories = {}

for name, optimizer in optimizers_keras.items():
    print(f"\nTraining with {name}...")

    model = create_model()
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    history = model.fit(
        X_train_nn, y_train_nn,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        verbose=0
    )

    histories[name] = history.history

    test_loss = model.evaluate(X_test_nn, y_test_nn, verbose=0)[0]
    print(f"  Final train loss: {history.history['loss'][-1]:.2f}")
    print(f"  Final val loss: {history.history['val_loss'][-1]:.2f}")
    print(f"  Test loss: {test_loss:.2f}")

# Plot results
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, (name, history) in enumerate(histories.items()):
    axes[idx].plot(history['loss'], linewidth=2, label='Train')
    axes[idx].plot(history['val_loss'], linewidth=2, label='Validation')
    axes[idx].set_xlabel('Epoch')
    axes[idx].set_ylabel('Loss')
    axes[idx].set_title(name)
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Summary

### Quick Reference

| Optimizer | Learning Rate | Other Hyperparameters | Best Use Case |
|-----------|---------------|----------------------|---------------|
| SGD | 0.01 - 0.1 | - | Simple problems, when interpretability matters |
| Momentum | 0.01 - 0.1 | momentum=0.9 | General improvement over SGD |
| Nesterov | 0.01 - 0.1 | momentum=0.9 | When you need look-ahead |
| AdaGrad | 0.01 - 1.0 | - | Sparse features, NLP |
| RMSprop | 0.001 - 0.01 | decay=0.9 | RNNs, non-stationary |
| Adam | 0.0001 - 0.001 | β₁=0.9, β₂=0.999 | Default choice, works well generally |

### Key Takeaways

1. **Adam is the default choice** for most deep learning tasks
2. **Learning rate is crucial** - use learning rate finder
3. **Batch size matters** - typical: 32, 64, 128, 256
4. **Use learning rate schedules** for better final performance
5. **Monitor convergence** - plot train/val loss curves
6. **Gradient clipping** helps with exploding gradients
7. **Warmup** helps training stability

### Checklist

- [ ] Choose appropriate optimizer for your problem
- [ ] Find good learning rate using LR finder
- [ ] Set reasonable batch size (start with 32)
- [ ] Monitor training curves (train vs validation loss)
- [ ] Use learning rate schedule if needed
- [ ] Add gradient clipping for RNNs
- [ ] Consider warmup for large models
- [ ] Compare multiple optimizers if unsure

Understanding optimization algorithms is fundamental to training models efficiently!
