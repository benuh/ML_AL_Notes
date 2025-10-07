# 📐 Mathematics for Machine Learning

## Complete Mathematical Foundation for ML/AI

This comprehensive guide covers all essential mathematics needed to deeply understand and implement machine learning algorithms.

---

## 📋 Table of Contents

1. [Linear Algebra](#linear-algebra)
2. [Calculus](#calculus)
3. [Probability Theory](#probability-theory)
4. [Statistics](#statistics)
5. [Optimization](#optimization)
6. [Information Theory](#information-theory)

---

## 🔢 Linear Algebra

Linear algebra is the foundation of machine learning. Neural networks, dimensionality reduction, and most ML algorithms rely heavily on linear algebra operations.

### Vectors and Matrices

**Vectors** (1D arrays):
```
v = [1, 2, 3]

Geometric interpretation:
- Direction and magnitude in space
- Point in n-dimensional space

ML applications:
- Feature vector: [age, income, credit_score]
- Word embedding: 300-dimensional vector representing a word
- Model parameters: weights and biases
```

**Matrices** (2D arrays):
```
A = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]

Shape: 3×3 (rows × columns)

ML applications:
- Dataset: Each row is a sample, each column is a feature
- Weight matrix in neural network layer
- Covariance matrix
- Transformation matrix
```

### Key Operations

**Dot Product** (scalar result):
```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

dot_product = np.dot(a, b)
# = 1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32

# Alternative notation
dot_product = a @ b
dot_product = np.sum(a * b)

# Geometric meaning:
# dot(a, b) = |a| × |b| × cos(θ)
# Measures similarity between vectors

# ML application: Similarity between embeddings
similarity = word_embedding_1 @ word_embedding_2
```

**Matrix Multiplication**:
```python
# A (m×n) × B (n×p) = C (m×p)

A = np.array([[1, 2],
              [3, 4],
              [5, 6]])  # 3×2

B = np.array([[7, 8, 9],
              [10, 11, 12]])  # 2×3

C = A @ B  # 3×3

# C[i,j] = sum of (row i of A) × (column j of B)
# C[0,0] = 1×7 + 2×10 = 27
# C[0,1] = 1×8 + 2×11 = 30
# C[0,2] = 1×9 + 2×12 = 33

# ML application: Forward pass in neural network
# X (batch_size, input_dim) @ W (input_dim, hidden_dim) = H (batch_size, hidden_dim)
```

**Matrix Transpose**:
```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])  # 2×3

A_T = A.T  # 3×2
# [[1, 4],
#  [2, 5],
#  [3, 6]]

# Properties:
# (A^T)^T = A
# (AB)^T = B^T A^T

# ML application: Backpropagation
# dL/dW = X^T @ dL/dY
```

### Advanced Concepts

**Eigenvalues and Eigenvectors**:
```python
# Definition: Av = λv
# v is eigenvector, λ is eigenvalue

A = np.array([[4, 2],
              [1, 3]])

eigenvalues, eigenvectors = np.linalg.eig(A)

# eigenvalues: [5, 2]
# eigenvectors: [[0.89, -0.71],
#                [0.45,  0.71]]

# Meaning: v is a direction that only gets scaled (not rotated) by A

# ML applications:
# 1. Principal Component Analysis (PCA)
#    - Find directions of maximum variance
#    - Eigenvectors of covariance matrix

# 2. PageRank algorithm
#    - Dominant eigenvector of web graph

# 3. Spectral clustering
#    - Eigenvectors of graph Laplacian
```

**Singular Value Decomposition (SVD)**:
```python
# Any matrix A (m×n) can be decomposed:
# A = U Σ V^T

A = np.random.rand(4, 3)
U, S, VT = np.linalg.svd(A, full_matrices=False)

# U (4×3): Left singular vectors (orthonormal)
# S (3,): Singular values (diagonal of Σ)
# VT (3×3): Right singular vectors (orthonormal)

# Reconstruction:
A_reconstructed = U @ np.diag(S) @ VT
# A_reconstructed ≈ A (within numerical precision)

# ML applications:

# 1. Low-rank approximation
def low_rank_approximation(A, k):
    """Keep only top k singular values"""
    U, S, VT = np.linalg.svd(A, full_matrices=False)
    return U[:, :k] @ np.diag(S[:k]) @ VT[:k, :]

# 2. PCA implementation
def pca_via_svd(X, n_components):
    """PCA using SVD"""
    # Center data
    X_centered = X - X.mean(axis=0)

    # SVD
    U, S, VT = np.linalg.svd(X_centered, full_matrices=False)

    # Principal components
    components = VT[:n_components]

    # Transformed data
    X_transformed = X_centered @ components.T

    return X_transformed, components

# 3. Recommender systems (Matrix Factorization)
# User-item matrix ≈ U @ Σ @ V^T
# Latent factors representation
```

**Matrix Norms**:
```python
A = np.array([[1, 2],
              [3, 4]])

# Frobenius norm (most common)
frobenius_norm = np.linalg.norm(A, 'fro')
# = sqrt(1² + 2² + 3² + 4²) = sqrt(30) ≈ 5.48

# L1 norm (maximum absolute column sum)
l1_norm = np.linalg.norm(A, 1)  # 6

# L2 norm (largest singular value)
l2_norm = np.linalg.norm(A, 2)  # ≈ 5.46

# ML application: Regularization
# L2 regularization: minimize ||w||²
# Frobenius norm for weight decay: minimize ||W||_F²
```

### Practical ML Examples

**Linear Regression (Matrix Form)**:
```python
# y = Xw + noise
# Closed-form solution: w = (X^T X)^(-1) X^T y

def linear_regression_normal_equation(X, y):
    """
    Solve linear regression analytically

    X: (n_samples, n_features)
    y: (n_samples,)
    """
    # Add intercept term (column of ones)
    X_with_intercept = np.column_stack([np.ones(len(X)), X])

    # Normal equation
    XTX = X_with_intercept.T @ X_with_intercept
    XTy = X_with_intercept.T @ y

    # Solve: w = (X^T X)^(-1) X^T y
    w = np.linalg.solve(XTX, XTy)

    return w

# Example
X = np.random.rand(100, 5)
y = X @ np.array([1, 2, 3, 4, 5]) + np.random.randn(100) * 0.1

weights = linear_regression_normal_equation(X, y)
print(f"Learned weights: {weights}")
# Should be close to [1, 2, 3, 4, 5]
```

**PCA (Dimensionality Reduction)**:
```python
def pca(X, n_components=2):
    """
    Principal Component Analysis

    Steps:
    1. Center data
    2. Compute covariance matrix
    3. Find eigenvectors (principal components)
    4. Project data onto top components
    """
    # 1. Center data (mean = 0)
    X_centered = X - np.mean(X, axis=0)

    # 2. Covariance matrix
    # Cov = (1/n) X^T X
    n_samples = X_centered.shape[0]
    cov_matrix = (X_centered.T @ X_centered) / n_samples

    # 3. Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort by eigenvalue (descending)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 4. Select top k eigenvectors
    principal_components = eigenvectors[:, :n_components]

    # 5. Project data
    X_transformed = X_centered @ principal_components

    # Explained variance
    explained_variance_ratio = eigenvalues[:n_components] / eigenvalues.sum()

    return X_transformed, principal_components, explained_variance_ratio

# Example: Compress 784-dimensional MNIST to 2D
from sklearn.datasets import load_digits
digits = load_digits()
X, y = digits.data, digits.target

X_2d, components, var_ratio = pca(X, n_components=2)

print(f"Explained variance: {var_ratio.sum():.2%}")
# Typically 20-30% for 2 components

import matplotlib.pyplot as plt
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10')
plt.xlabel(f'PC1 ({var_ratio[0]:.1%})')
plt.ylabel(f'PC2 ({var_ratio[1]:.1%})')
plt.title('MNIST Digits in 2D (PCA)')
plt.colorbar()
plt.show()
```

**Neural Network Forward Pass**:
```python
def neural_network_forward(X, weights, biases):
    """
    2-layer neural network using matrix operations

    X: (batch_size, input_dim)
    weights: list of weight matrices
    biases: list of bias vectors
    """
    # Layer 1: X (n, d_in) @ W1 (d_in, d_hidden) = H (n, d_hidden)
    z1 = X @ weights[0] + biases[0]
    a1 = np.maximum(0, z1)  # ReLU activation

    # Layer 2: H (n, d_hidden) @ W2 (d_hidden, d_out) = Y (n, d_out)
    z2 = a1 @ weights[1] + biases[1]

    # Softmax for classification
    exp_scores = np.exp(z2 - np.max(z2, axis=1, keepdims=True))  # Numerical stability
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    return probs

# Example: MNIST classifier
batch_size = 32
input_dim = 784  # 28×28 images
hidden_dim = 128
output_dim = 10  # digits 0-9

W1 = np.random.randn(input_dim, hidden_dim) * 0.01
b1 = np.zeros(hidden_dim)
W2 = np.random.randn(hidden_dim, output_dim) * 0.01
b2 = np.zeros(output_dim)

X_batch = np.random.randn(batch_size, input_dim)
predictions = neural_network_forward(X_batch, [W1, W2], [b1, b2])

print(f"Predictions shape: {predictions.shape}")  # (32, 10)
print(f"Sample prediction: {predictions[0]}")  # Probabilities sum to 1
```

---

## 📈 Calculus

Calculus enables optimization - the core of training ML models. Understanding derivatives and gradients is essential for backpropagation.

### Derivatives

**Definition**: Rate of change
```
f'(x) = lim (h→0) [f(x+h) - f(x)] / h

Geometric meaning: Slope of tangent line
ML meaning: How much loss changes when we change a parameter
```

**Common Derivatives**:
```python
# Power rule: d/dx [x^n] = n × x^(n-1)
# d/dx [x²] = 2x
# d/dx [x³] = 3x²

# Exponential: d/dx [e^x] = e^x

# Logarithm: d/dx [ln(x)] = 1/x

# Trigonometric:
# d/dx [sin(x)] = cos(x)
# d/dx [cos(x)] = -sin(x)

# Chain rule: d/dx [f(g(x))] = f'(g(x)) × g'(x)
# Example: d/dx [sin(x²)] = cos(x²) × 2x

# Product rule: d/dx [f(x)g(x)] = f'(x)g(x) + f(x)g'(x)

# Quotient rule: d/dx [f(x)/g(x)] = [f'(x)g(x) - f(x)g'(x)] / g(x)²
```

**Numerical Differentiation** (for verification):
```python
def numerical_derivative(f, x, h=1e-5):
    """Compute derivative using finite differences"""
    return (f(x + h) - f(x - h)) / (2 * h)

# Example
f = lambda x: x**2
f_prime_numerical = numerical_derivative(f, x=3.0)
f_prime_analytical = 2 * 3.0

print(f"Numerical: {f_prime_numerical:.6f}")  # 6.000000
print(f"Analytical: {f_prime_analytical:.6f}")  # 6.000000
```

### Gradients (Multivariate Derivatives)

**Gradient**: Vector of partial derivatives
```
f(x, y) = x² + y²

Gradient: ∇f = [∂f/∂x, ∂f/∂y] = [2x, 2y]

Meaning: Direction of steepest increase
ML meaning: Which direction to move parameters to increase loss
           (we move opposite direction to decrease loss!)
```

**Gradient Descent**:
```python
def gradient_descent(f, grad_f, x0, learning_rate=0.1, num_iterations=100,
                    tolerance=1e-6, verbose=False):
    """
    Minimize function f using gradient descent

    Parameters:
    -----------
    f: function to minimize
    grad_f: gradient of f
    x0: initial point (numpy array)
    learning_rate: step size α (also called learning rate)
    tolerance: convergence threshold

    Convergence Conditions (for convex f with L-Lipschitz gradient):
    - If α ≤ 1/L: Guaranteed convergence to global minimum
    - Convergence rate: O(1/k) where k is iteration number

    For strongly convex functions (with parameter μ):
    - Convergence rate improves to O(exp(-k·μ/L))
    """
    x = x0.copy()
    history = [x.copy()]
    grad_norms = []

    for i in range(num_iterations):
        # Compute gradient
        grad = grad_f(x)
        grad_norm = np.linalg.norm(grad)
        grad_norms.append(grad_norm)

        # Check convergence
        if grad_norm < tolerance:
            if verbose:
                print(f"Converged at iteration {i}, gradient norm: {grad_norm:.2e}")
            break

        # Update: x^(k+1) = x^(k) - α∇f(x^(k))
        x = x - learning_rate * grad

        history.append(x.copy())

        if verbose and i % 10 == 0:
            print(f"Iter {i}: f(x) = {f(x):.6f}, ||∇f|| = {grad_norm:.2e}")

    return x, np.array(history), np.array(grad_norms)

# Example: Minimize f(x,y) = x² + y²
f = lambda x: x[0]**2 + x[1]**2
grad_f = lambda x: np.array([2*x[0], 2*x[1]])

x_min, history = gradient_descent(f, grad_f, x0=np.array([5.0, 5.0]))

print(f"Minimum found at: {x_min}")  # Should be near [0, 0]
print(f"Minimum value: {f(x_min):.6f}")  # Should be near 0

# Visualize optimization path
import matplotlib.pyplot as plt

x = np.linspace(-6, 6, 100)
y = np.linspace(-6, 6, 100)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2

plt.contour(X, Y, Z, levels=20)
plt.plot(history[:, 0], history[:, 1], 'ro-', markersize=3)
plt.plot(history[0, 0], history[0, 1], 'go', markersize=10, label='Start')
plt.plot(history[-1, 0], history[-1, 1], 'r*', markersize=15, label='End')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gradient Descent Optimization')
plt.legend()
plt.show()
```

### Backpropagation (Chain Rule in Action)

**Chain Rule for Neural Networks**:
```python
# Network: x → f → g → h → loss
# x → f → a
# a → g → b
# b → h → loss

# Chain rule:
# dL/dx = (dL/db) × (db/da) × (da/dx)

# Backpropagation computes these efficiently!

def backpropagation_example():
    """
    Simple 2-layer network with explicit backprop

    x → W1 → a → ReLU → h → W2 → y → Loss
    """
    # Forward pass
    x = np.array([[1.0, 2.0]])  # (1, 2)
    W1 = np.array([[0.1, 0.2, 0.3],
                   [0.4, 0.5, 0.6]])  # (2, 3)
    W2 = np.array([[0.7],
                   [0.8],
                   [0.9]])  # (3, 1)
    target = np.array([[1.0]])

    # Layer 1
    a = x @ W1  # (1, 3)
    # a = [1×0.1 + 2×0.4, 1×0.2 + 2×0.5, 1×0.3 + 2×0.6]
    #   = [0.9, 1.2, 1.5]

    # ReLU
    h = np.maximum(0, a)  # (1, 3)
    # h = [0.9, 1.2, 1.5] (all positive, no change)

    # Layer 2
    y = h @ W2  # (1, 1)
    # y = [0.9×0.7 + 1.2×0.8 + 1.5×0.9]
    #   = [0.63 + 0.96 + 1.35] = [2.94]

    # Loss (MSE)
    loss = 0.5 * (y - target)**2
    # loss = 0.5 × (2.94 - 1.0)² = 0.5 × 3.76 = 1.88

    print(f"Forward pass:")
    print(f"  a = {a}")
    print(f"  h = {h}")
    print(f"  y = {y}")
    print(f"  loss = {loss}")

    # Backward pass (compute gradients)

    # dL/dy
    dL_dy = y - target  # (1, 1)
    # dL/dy = 2.94 - 1.0 = 1.94

    # dL/dW2 = h^T @ dL/dy
    dL_dW2 = h.T @ dL_dy  # (3, 1)
    # = [[0.9], [1.2], [1.5]] × 1.94
    # = [[1.746], [2.328], [2.910]]

    # dL/dh = dL/dy @ W2^T
    dL_dh = dL_dy @ W2.T  # (1, 3)
    # = 1.94 × [0.7, 0.8, 0.9]
    # = [1.358, 1.552, 1.746]

    # dL/da (through ReLU derivative)
    # ReLU'(x) = 1 if x > 0 else 0
    dL_da = dL_dh * (a > 0)  # (1, 3)
    # = [1.358, 1.552, 1.746] × [1, 1, 1]
    # = [1.358, 1.552, 1.746]

    # dL/dW1 = x^T @ dL/da
    dL_dW1 = x.T @ dL_da  # (2, 3)
    # = [[1], [2]] × [1.358, 1.552, 1.746]
    # = [[1.358, 1.552, 1.746],
    #    [2.716, 3.104, 3.492]]

    print(f"\nBackward pass (gradients):")
    print(f"  dL/dW2 =\n{dL_dW2}")
    print(f"  dL/dW1 =\n{dL_dW1}")

    # Gradient descent update
    learning_rate = 0.01
    W1_new = W1 - learning_rate * dL_dW1
    W2_new = W2 - learning_rate * dL_dW2

    print(f"\nUpdated weights:")
    print(f"  W1_new =\n{W1_new}")
    print(f"  W2_new =\n{W2_new}")

backpropagation_example()
```

**Automatic Differentiation** (PyTorch):
```python
import torch

# PyTorch computes gradients automatically!

# Forward pass
x = torch.tensor([[1.0, 2.0]], requires_grad=False)
W1 = torch.tensor([[0.1, 0.2, 0.3],
                   [0.4, 0.5, 0.6]], requires_grad=True)
W2 = torch.tensor([[0.7],
                   [0.8],
                   [0.9]], requires_grad=True)
target = torch.tensor([[1.0]])

a = x @ W1
h = torch.relu(a)
y = h @ W2
loss = 0.5 * (y - target)**2

# Backward pass (automatic!)
loss.backward()

print(f"Gradients (automatic):")
print(f"  W1.grad =\n{W1.grad}")
print(f"  W2.grad =\n{W2.grad}")

# Update weights
learning_rate = 0.01
with torch.no_grad():
    W1 -= learning_rate * W1.grad
    W2 -= learning_rate * W2.grad

    # Clear gradients for next iteration
    W1.grad.zero_()
    W2.grad.zero_()
```

### Important ML Derivatives

**Sigmoid**:
```python
# σ(x) = 1 / (1 + e^(-x))
# σ'(x) = σ(x) × (1 - σ(x))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Efficient for backprop: Already have σ(x) from forward pass!
```

**Softmax**:
```python
# Softmax: S_i = e^(x_i) / Σ e^(x_j)

# Derivative w.r.t. x_i:
# ∂S_i/∂x_i = S_i × (1 - S_i)
# ∂S_i/∂x_j = -S_i × S_j  (i ≠ j)

# Combined with Cross-Entropy loss:
# ∂L/∂x = S - y  (beautiful simplification!)
# where y is one-hot encoded target

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # Numerical stability
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Cross-entropy loss
def cross_entropy(predictions, targets):
    return -np.sum(targets * np.log(predictions + 1e-8))

# Gradient (simplified!)
def softmax_cross_entropy_gradient(predictions, targets):
    return predictions - targets

# Example
logits = np.array([[2.0, 1.0, 0.1]])
target = np.array([[1, 0, 0]])  # First class

probs = softmax(logits)
loss = cross_entropy(probs, target)
grad = softmax_cross_entropy_gradient(probs, target)

print(f"Probabilities: {probs}")
print(f"Loss: {loss:.4f}")
print(f"Gradient: {grad}")
```

**ReLU**:
```python
# ReLU(x) = max(0, x)
# ReLU'(x) = 1 if x > 0 else 0

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Dead ReLU problem: If x < 0 always, gradient is 0, neuron never updates
# Solutions: Leaky ReLU, PReLU, ELU
```

---

## 🎲 Probability Theory

Probability underpins machine learning - from Bayesian inference to generative models to uncertainty quantification.

### Fundamentals

**Probability Axioms**:
```
1. P(A) ≥ 0  (non-negative)
2. P(Ω) = 1  (total probability = 1)
3. P(A ∪ B) = P(A) + P(B) if A and B are disjoint

Sample space Ω: All possible outcomes
Event A: Subset of Ω
```

**Conditional Probability**:
```
P(A|B) = P(A ∩ B) / P(B)

"Probability of A given B has occurred"

Example: Email spam detection
P(spam | contains "FREE") = P(spam ∩ contains "FREE") / P(contains "FREE")
```

**Bayes' Theorem**:
```
P(A|B) = P(B|A) × P(A) / P(B)

Where:
- P(A|B): Posterior probability (what we want to compute)
- P(B|A): Likelihood (probability of observing B given A)
- P(A): Prior probability (initial belief about A)
- P(B): Evidence or marginal likelihood (normalization constant)

In ML terms:
P(θ | D) = P(D | θ) × P(θ) / P(D)

Posterior = Likelihood × Prior / Evidence

Where:
- θ: Model parameters (hypothesis)
- D: Observed data
- P(θ | D): Posterior distribution over parameters given data
- P(D | θ): Likelihood of data given parameters
- P(θ): Prior distribution over parameters
- P(D) = ∫ P(D | θ) P(θ) dθ: Marginal likelihood (often intractable)

This is the foundation of Bayesian machine learning!

Key Properties:
1. Law of Total Probability: P(B) = Σ P(B|A_i) P(A_i)
2. Bayes' Rule is exact, not an approximation
3. Prior × Likelihood = Unnormalized Posterior
4. Conjugate priors simplify posterior computation
```

**Example: Medical Diagnosis**:
```python
def bayes_theorem_example():
    """
    Disease testing with Bayes' Theorem

    Disease prevalence: 1% (prior)
    Test sensitivity: 95% (true positive rate)
    Test specificity: 90% (true negative rate)

    Question: If test is positive, what's probability of having disease?
    """
    # Prior
    P_disease = 0.01
    P_no_disease = 0.99

    # Likelihood
    P_positive_given_disease = 0.95  # Sensitivity
    P_positive_given_no_disease = 0.10  # 1 - Specificity

    # Total probability of positive test
    P_positive = (P_positive_given_disease * P_disease +
                  P_positive_given_no_disease * P_no_disease)

    # Bayes' theorem
    P_disease_given_positive = (P_positive_given_disease * P_disease) / P_positive

    print(f"Probability of disease given positive test: {P_disease_given_positive:.1%}")
    # Only 8.7%! (because disease is rare)

    # Intuition: Many false positives because disease is rare

bayes_theorem_example()
# Output: Probability of disease given positive test: 8.7%
```

### Random Variables

**Discrete Random Variable**:
```python
# Probability Mass Function (PMF)

# Example: Rolling a die
outcomes = [1, 2, 3, 4, 5, 6]
probabilities = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]

# Expected value (mean)
E_X = sum(x * p for x, p in zip(outcomes, probabilities))
# E[X] = 1×(1/6) + 2×(1/6) + ... + 6×(1/6) = 3.5

# Variance
Var_X = sum((x - E_X)**2 * p for x, p in zip(outcomes, probabilities))
# Var[X] = (1-3.5)²×(1/6) + ... + (6-3.5)²×(1/6) ≈ 2.92

print(f"Expected value: {E_X}")
print(f"Variance: {Var_X:.2f}")
print(f"Standard deviation: {np.sqrt(Var_X):.2f}")
```

**Continuous Random Variable**:
```python
# Probability Density Function (PDF)

# Normal distribution: N(μ, σ²)
def normal_pdf(x, mu=0, sigma=1):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)

# Visualize
x = np.linspace(-5, 5, 1000)
y = normal_pdf(x, mu=0, sigma=1)

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title('Standard Normal Distribution N(0,1)')
plt.grid(True)
plt.show()

# Cumulative Distribution Function (CDF)
from scipy.stats import norm

# P(X ≤ x)
cdf_value = norm.cdf(1.96)  # P(X ≤ 1.96) ≈ 0.975 (95%)
print(f"P(X ≤ 1.96) = {cdf_value:.4f}")

# Inverse CDF (quantile function)
quantile = norm.ppf(0.95)  # Value where 95% of data is below
print(f"95th percentile: {quantile:.4f}")  # ≈ 1.645
```

### Common Distributions

**Bernoulli Distribution** (Single coin flip):
```python
# X ∈ {0, 1}
# P(X=1) = p

p = 0.7  # Probability of heads

X = np.random.binomial(n=1, p=p, size=1000)  # 1000 flips

print(f"Proportion of heads: {X.mean():.3f}")  # ≈ 0.700
# E[X] = p = 0.7
# Var[X] = p(1-p) = 0.7×0.3 = 0.21

# ML application: Binary classification output
```

**Binomial Distribution** (n coin flips):
```python
# Number of successes in n trials
# X ~ Binomial(n, p)

n = 10  # 10 flips
p = 0.5  # Fair coin

X = np.random.binomial(n=n, p=p, size=10000)

plt.hist(X, bins=range(12), density=True, alpha=0.7, edgecolor='black')
plt.xlabel('Number of heads')
plt.ylabel('Probability')
plt.title(f'Binomial Distribution (n={n}, p={p})')
plt.show()

# E[X] = np = 5
# Var[X] = np(1-p) = 2.5
```

**Normal (Gaussian) Distribution**:
```python
# X ~ N(μ, σ²)
# Most important distribution in ML!

mu, sigma = 0, 1
X = np.random.normal(mu, sigma, size=10000)

plt.hist(X, bins=50, density=True, alpha=0.7, edgecolor='black')
x = np.linspace(-4, 4, 100)
plt.plot(x, norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='PDF')
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Normal Distribution N(0,1)')
plt.legend()
plt.show()

# Properties:
# - 68% of data within 1 std
# - 95% within 2 std
# - 99.7% within 3 std

# ML applications:
# - Weight initialization: N(0, 1/sqrt(n))
# - Noise model in regression
# - Gaussian processes
# - Variational autoencoders
```

**Multivariate Normal Distribution**:
```python
# X ~ N(μ, Σ)
# μ: mean vector
# Σ: covariance matrix

mean = [0, 0]
cov = [[1, 0.5],
       [0.5, 1]]  # Correlation = 0.5

X = np.random.multivariate_normal(mean, cov, size=1000)

plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Bivariate Normal Distribution')
plt.axis('equal')
plt.grid(True)
plt.show()

# ML applications:
# - Gaussian mixture models
# - Generative models
# - Bayesian inference
```

### Maximum Likelihood Estimation (MLE)

```python
# Find parameters that maximize P(data | parameters)

def mle_normal_distribution(data):
    """
    Estimate μ and σ² for normal distribution

    Likelihood: L(μ, σ²) = Π (1/√(2πσ²)) exp(-(x_i - μ)²/(2σ²))

    Log-likelihood: log L = -n/2 log(2πσ²) - Σ(x_i - μ)²/(2σ²)

    MLE estimates:
    μ_MLE = sample mean
    σ²_MLE = sample variance
    """
    mu_mle = np.mean(data)
    sigma2_mle = np.var(data, ddof=0)  # ddof=0 for MLE (ddof=1 for unbiased)

    return mu_mle, sigma2_mle

# Generate data
true_mu, true_sigma = 5, 2
data = np.random.normal(true_mu, true_sigma, size=1000)

# Estimate parameters
mu_hat, sigma2_hat = mle_normal_distribution(data)

print(f"True parameters: μ={true_mu}, σ²={true_sigma**2}")
print(f"MLE estimates: μ={mu_hat:.3f}, σ²={sigma2_hat:.3f}")
# MLE estimates should be very close to true parameters!
```

**MLE for Linear Regression**:
```python
# Assume: y = Xw + ε, where ε ~ N(0, σ²)
# Likelihood: P(y | X, w, σ²) = Π N(y_i | x_i^T w, σ²)

# Maximizing likelihood ≡ Minimizing MSE!

# log L = -n/2 log(2πσ²) - Σ(y_i - x_i^T w)²/(2σ²)
# Maximizing log L w.r.t. w ≡ Minimizing Σ(y_i - x_i^T w)²
# This is exactly MSE!

# MLE for linear regression = Least squares solution
```

---

## 📊 Statistics

Statistics provides tools for inference, hypothesis testing, and understanding data.

### Descriptive Statistics

**Measures of Central Tendency**:
```python
data = np.array([1, 2, 3, 4, 5, 100])  # Note outlier

# Mean (sensitive to outliers)
mean = np.mean(data)  # 19.17

# Median (robust to outliers)
median = np.median(data)  # 3.5

# Mode (most frequent value)
from scipy.stats import mode
mode_result = mode(data, keepdims=True)  # No mode (all unique)

print(f"Mean: {mean:.2f}")
print(f"Median: {median:.2f}")

# For skewed distributions (with outliers), median is better
```

**Measures of Spread**:
```python
# Variance
variance = np.var(data, ddof=1)  # Sample variance (ddof=1)

# Standard deviation
std = np.std(data, ddof=1)

# Interquartile range (IQR) - robust to outliers
Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1

print(f"Variance: {variance:.2f}")
print(f"Std dev: {std:.2f}")
print(f"IQR: {IQR:.2f}")
```

**Covariance and Correlation**:
```python
# Two variables: height and weight

height = np.array([160, 165, 170, 175, 180])
weight = np.array([55, 60, 65, 70, 75])

# Covariance: How much do they vary together?
cov = np.cov(height, weight)[0, 1]  # 25.0

# Correlation: Normalized covariance (-1 to 1)
corr = np.corrcoef(height, weight)[0, 1]  # 1.0 (perfect positive correlation)

print(f"Covariance: {cov:.2f}")
print(f"Correlation: {corr:.3f}")

# Interpretation:
# corr = 1: Perfect positive correlation
# corr = 0: No linear correlation
# corr = -1: Perfect negative correlation

# ML application: Feature selection
# Remove highly correlated features (multicollinearity)
def remove_correlated_features(X, threshold=0.95):
    corr_matrix = np.corrcoef(X.T)
    to_remove = set()

    for i in range(len(corr_matrix)):
        for j in range(i+1, len(corr_matrix)):
            if abs(corr_matrix[i, j]) > threshold:
                to_remove.add(j)

    mask = np.ones(X.shape[1], dtype=bool)
    mask[list(to_remove)] = False
    return X[:, mask]
```

### Hypothesis Testing

**t-test** (Compare two means):
```python
from scipy.stats import ttest_ind

# A/B test: Did the new feature increase conversion?
control = np.random.binomial(1, 0.10, size=1000)  # 10% conversion
treatment = np.random.binomial(1, 0.12, size=1000)  # 12% conversion

# Two-sample t-test
t_stat, p_value = ttest_ind(treatment, control)

print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    print("Result: Statistically significant! ✅")
    print("The new feature improved conversion.")
else:
    print("Result: Not statistically significant ❌")
    print("Cannot conclude the new feature is better.")

# Confidence interval
from scipy.stats import t as t_dist

mean_diff = treatment.mean() - control.mean()
se = np.sqrt(treatment.var()/len(treatment) + control.var()/len(control))
ci_lower = mean_diff - 1.96 * se
ci_upper = mean_diff + 1.96 * se

print(f"\nMean difference: {mean_diff:.4f}")
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
```

**Chi-Square Test** (Categorical data):
```python
from scipy.stats import chi2_contingency

# Are gender and product preference independent?
#              Product A   Product B
# Male            30          70
# Female          50          50

observed = np.array([[30, 70],
                     [50, 50]])

chi2, p_value, dof, expected = chi2_contingency(observed)

print(f"Chi-square statistic: {chi2:.3f}")
print(f"p-value: {p_value:.4f}")
print(f"Expected frequencies:\n{expected}")

if p_value < 0.05:
    print("Gender and product preference are dependent")
else:
    print("Gender and product preference are independent")
```

---

## 🎯 Optimization

Optimization is how we train ML models - finding parameters that minimize loss.

### Convex Optimization

**Convex Function**:
```
f is convex if: f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y) for all λ ∈ [0,1]

Geometric meaning: Line segment between any two points lies above the function

Good news: Local minimum = Global minimum!
```

**Examples**:
```python
# Convex functions:
# - f(x) = x²
# - f(x) = |x|
# - f(x) = exp(x)
# - f(x) = -log(x)

# Non-convex functions:
# - f(x) = x³
# - f(x) = sin(x)
# - Neural networks (non-convex!)

# Visualize
x = np.linspace(-3, 3, 100)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Convex
axes[0].plot(x, x**2, label='x²')
axes[0].set_title('Convex Function')
axes[0].legend()
axes[0].grid(True)

# Non-convex
axes[1].plot(x, x**3 - 3*x, label='x³ - 3x')
axes[1].set_title('Non-Convex Function')
axes[1].legend()
axes[1].grid(True)

plt.show()
```

### Optimization Algorithms

**Gradient Descent** (covered earlier):
```python
# w_new = w_old - learning_rate × gradient
```

**Newton's Method** (uses second derivative):
```python
def newtons_method(f, grad_f, hess_f, x0, num_iterations=10):
    """
    Newton's method for optimization

    Uses second-order information (Hessian)
    Update: x_new = x_old - H^(-1) @ grad

    Advantages: Faster convergence (quadratic vs linear)
    Disadvantages: Expensive (need to compute and invert Hessian)
    """
    x = x0.copy()

    for i in range(num_iterations):
        grad = grad_f(x)
        hess = hess_f(x)

        # Newton step
        delta = np.linalg.solve(hess, grad)  # H^(-1) @ grad
        x = x - delta

        print(f"Iteration {i+1}: x = {x}, f(x) = {f(x):.6f}")

    return x

# Example: f(x) = x^2 + 2x + 1
f = lambda x: x**2 + 2*x + 1
grad_f = lambda x: 2*x + 2
hess_f = lambda x: np.array([[2.0]])  # Second derivative

x_min = newtons_method(f, grad_f, hess_f, x0=np.array([5.0]))
# Converges in 1 iteration! (function is quadratic)
```

**Constrained Optimization** (Lagrange Multipliers):
```python
# Minimize f(x) subject to g(x) = 0

# Example: Maximize f(x,y) = xy subject to x + y = 1

# Lagrangian: L(x, y, λ) = xy + λ(1 - x - y)

# Set derivatives to 0:
# ∂L/∂x = y - λ = 0
# ∂L/∂y = x - λ = 0
# ∂L/∂λ = 1 - x - y = 0

# Solve:
# y = λ, x = λ, x + y = 1
# 2λ = 1 → λ = 0.5
# x = y = 0.5

# Maximum: f(0.5, 0.5) = 0.25

def lagrange_example():
    from scipy.optimize import minimize

    # Objective: Minimize -xy (negative for maximization)
    def objective(vars):
        x, y = vars
        return -x * y

    # Constraint: x + y - 1 = 0
    constraint = {'type': 'eq', 'fun': lambda vars: vars[0] + vars[1] - 1}

    # Initial guess
    x0 = [0, 0]

    # Optimize
    result = minimize(objective, x0, constraints=constraint)

    print(f"Optimal solution: x={result.x[0]:.3f}, y={result.x[1]:.3f}")
    print(f"Maximum value: {-result.fun:.3f}")

lagrange_example()
# Output: x=0.500, y=0.500, maximum=0.250
```

---

## 📡 Information Theory

Information theory quantifies information, essential for understanding entropy, KL divergence, and mutual information in ML.

### Entropy

**Shannon Entropy**: Measure of uncertainty/surprise
```
H(X) = -Σ P(x) log₂ P(x)

Units: bits (if log base 2)

Interpretation: Average number of bits needed to encode X
```

**Example**:
```python
def entropy(probabilities):
    """
    Compute Shannon entropy

    H(X) = -Σ p(x) log₂ p(x)
    """
    # Remove zeros (0 log 0 = 0 by definition)
    p = np.array(probabilities)
    p = p[p > 0]

    return -np.sum(p * np.log2(p))

# Fair coin: P(H) = P(T) = 0.5
H_fair = entropy([0.5, 0.5])
print(f"Entropy of fair coin: {H_fair:.3f} bits")  # 1.000 bit

# Biased coin: P(H) = 0.9, P(T) = 0.1
H_biased = entropy([0.9, 0.1])
print(f"Entropy of biased coin: {H_biased:.3f} bits")  # 0.469 bits
# Less entropy = more predictable!

# Uniform distribution (maximum entropy)
H_uniform = entropy([0.25, 0.25, 0.25, 0.25])
print(f"Entropy of uniform (4 outcomes): {H_uniform:.3f} bits")  # 2.000 bits

# Deterministic (minimum entropy)
H_deterministic = entropy([1.0])
print(f"Entropy of deterministic: {H_deterministic:.3f} bits")  # 0.000 bits
```

**Cross-Entropy** (ML loss function):
```python
def cross_entropy(p, q):
    """
    Cross-entropy H(p, q) = -Σ p(x) log q(x)

    p: true distribution
    q: predicted distribution

    Measures: How many bits needed to encode p using q
    """
    q = np.clip(q, 1e-10, 1.0)  # Numerical stability
    return -np.sum(p * np.log(q))

# Example: 3-class classification
true_distribution = np.array([1, 0, 0])  # Class 0
predicted_distribution = np.array([0.7, 0.2, 0.1])

CE = cross_entropy(true_distribution, predicted_distribution)
print(f"Cross-Entropy: {CE:.3f}")  # 0.357

# Perfect prediction
perfect_prediction = np.array([1.0, 0.0, 0.0])
CE_perfect = cross_entropy(true_distribution, perfect_prediction)
print(f"Cross-Entropy (perfect): {CE_perfect:.3f}")  # ≈ 0.000
```

**KL Divergence** (Relative Entropy):
```python
def kl_divergence(p, q):
    """
    KL(p || q) = Σ p(x) log(p(x) / q(x))

    Measures: How different q is from p
    Properties:
    - KL(p || q) ≥ 0
    - KL(p || q) = 0 iff p = q
    - Not symmetric: KL(p || q) ≠ KL(q || p)
    """
    p = np.array(p)
    q = np.array(q)
    q = np.clip(q, 1e-10, 1.0)

    return np.sum(p * np.log(p / q))

# Example
p = [0.5, 0.3, 0.2]
q1 = [0.5, 0.3, 0.2]  # Same as p
q2 = [0.4, 0.4, 0.2]  # Different from p

print(f"KL(p || q1): {kl_divergence(p, q1):.4f}")  # 0.0000
print(f"KL(p || q2): {kl_divergence(p, q2):.4f}")  # 0.0085

# ML application: Variational inference
# Minimize KL(q || p) to approximate true posterior p with q
```

**Mutual Information**:
```python
# I(X; Y) = H(X) + H(Y) - H(X, Y)
# Measures: How much knowing Y reduces uncertainty about X

def mutual_information(X, Y):
    """
    Estimate mutual information between X and Y
    """
    from sklearn.metrics import mutual_info_score

    return mutual_info_score(X, Y)

# Example: Feature selection
# Select features with high mutual information with target

from sklearn.feature_selection import mutual_info_classif

X = np.random.rand(1000, 10)
y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Target depends on features 0 and 1

mi_scores = mutual_info_classif(X, y)

print("Mutual information scores:")
for i, score in enumerate(mi_scores):
    print(f"  Feature {i}: {score:.4f}")
# Features 0 and 1 should have highest MI with target
```

---

## 🔑 Key Takeaways

**For Linear Algebra**:
- Vectors and matrices represent data and transformations
- Matrix multiplication is the core operation in neural networks
- Eigenvalues/eigenvectors are central to PCA and spectral methods
- SVD is a swiss army knife for dimensionality reduction and matrix factorization

**For Calculus**:
- Derivatives measure rates of change
- Gradients point in direction of steepest increase
- Chain rule enables backpropagation
- Optimization is gradient descent on steroids

**For Probability**:
- Bayes' theorem is fundamental for inference
- MLE connects optimization to statistics
- Normal distribution appears everywhere
- Conditional probability models dependencies

**For Statistics**:
- Descriptive statistics summarize data
- Hypothesis testing quantifies uncertainty
- Correlation ≠ causation
- Always visualize before analyzing

**For Optimization**:
- Convex problems have unique global minima
- Neural networks are non-convex (local minima exist)
- Second-order methods converge faster but are expensive
- Constrained optimization uses Lagrange multipliers

**For Information Theory**:
- Entropy quantifies uncertainty
- Cross-entropy is the ML loss function
- KL divergence measures distribution difference
- Mutual information measures dependence

---

## 📚 Further Reading

**Books**:
1. **Deisenroth, M. P., Faisal, A. A., & Ong, C. S.** (2020). *Mathematics for Machine Learning*. Cambridge University Press. Available free at: https://mml-book.github.io/
2. **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). *Deep Learning*. MIT Press. Available at: https://www.deeplearningbook.org/
3. **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. Springer.
4. **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. Available free at: https://hastie.su.stanford.edu/ElemStatLearn/
5. **Strang, G.** (2016). *Introduction to Linear Algebra* (5th ed.). Wellesley-Cambridge Press.
6. **Boyd, S., & Vandenberghe, L.** (2004). *Convex Optimization*. Cambridge University Press. Available free at: https://web.stanford.edu/~boyd/cvxbook/

**Seminal Papers**:
1. **Rumelhart, D. E., Hinton, G. E., & Williams, R. J.** (1986). "Learning representations by back-propagating errors." *Nature*, 323(6088), 533-536.
2. **Robbins, H., & Monro, S.** (1951). "A stochastic approximation method." *The Annals of Mathematical Statistics*, 22(3), 400-407.
3. **Kingma, D. P., & Ba, J.** (2015). "Adam: A method for stochastic optimization." *ICLR 2015*. arXiv:1412.6980

**Online Resources**:
- **3Blue1Brown** (YouTube): Visual explanations of linear algebra and calculus
  - "Essence of Linear Algebra" series
  - "Essence of Calculus" series
- **Khan Academy**: Foundations of probability and statistics
- **MIT OpenCourseWare**:
  - 18.06 Linear Algebra (Gilbert Strang)
  - 18.01 Single Variable Calculus
- **Stanford CS229**: Machine Learning (Andrew Ng)
  - Linear Algebra and Calculus review notes

**Practice**:
- Implement algorithms from scratch using only NumPy
- Derive gradients by hand before using automatic differentiation
- Visualize mathematical concepts with matplotlib
- Solve problems on Project Euler and Brilliant.org
- Work through exercises in the books listed above

---

## 📖 References

Key concepts and formulations in this guide are based on:

- **Linear Algebra**: Strang (2016), Deisenroth et al. (2020, Chapter 2)
- **Calculus & Optimization**: Boyd & Vandenberghe (2004), Goodfellow et al. (2016, Chapter 4)
- **Probability Theory**: Bishop (2006, Chapter 1-2), Deisenroth et al. (2020, Chapter 6)
- **Statistics**: Hastie et al. (2009, Chapter 2), James et al. (2021)
- **Information Theory**: Cover & Thomas (2006), Goodfellow et al. (2016, Chapter 3)
- **Backpropagation**: Rumelhart et al. (1986), Goodfellow et al. (2016, Chapter 6)
- **Gradient Descent**: Robbins & Monro (1951), Bottou (2010)

---

*Master these mathematical foundations and you'll have deep understanding of how and why ML algorithms work!*
