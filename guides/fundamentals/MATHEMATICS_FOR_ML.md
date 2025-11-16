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
# dot(a, b) = ||a|| × ||b|| × cos(θ)
# where θ is the angle between vectors

# Mathematical properties:
# 1. Commutativity: a·b = b·a
# 2. Distributivity: a·(b+c) = a·b + a·c
# 3. Cauchy-Schwarz inequality: |a·b| ≤ ||a|| ||b||
# 4. When a·b = 0: vectors are orthogonal (perpendicular)
# 5. When a·b > 0: angle θ < 90° (similar direction)
# 6. When a·b < 0: angle θ > 90° (opposite direction)

# Normalized dot product (cosine similarity):
# cos(θ) = (a·b) / (||a|| ||b||) ∈ [-1, 1]

# ML application: Cosine similarity between embeddings
cosine_sim = (word_embedding_1 @ word_embedding_2) / (
    np.linalg.norm(word_embedding_1) * np.linalg.norm(word_embedding_2)
)
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
# 1. (A^T)^T = A (involution)
# 2. (AB)^T = B^T A^T (reversal rule)
# 3. (A + B)^T = A^T + B^T (linearity)
# 4. (cA)^T = cA^T for scalar c
# 5. For symmetric matrix: A^T = A

# Special matrices:
# - Symmetric: A = A^T (covariance matrices)
# - Skew-symmetric: A = -A^T (cross-product matrices)
# - Orthogonal: A^T A = I (rotation matrices)

# ML application: Backpropagation gradient computation
# For Y = XW, gradient of loss L w.r.t. W:
# ∂L/∂W = X^T @ (∂L/∂Y)
# Shape: (d_in, d_out) = (n, d_in)^T @ (n, d_out)
```

### Advanced Concepts

**Eigenvalues and Eigenvectors**:
```python
# Definition: Av = λv
# where v ≠ 0 is eigenvector, λ ∈ ℂ is eigenvalue

# Characteristic equation: det(A - λI) = 0
# For n×n matrix: n eigenvalues (counting multiplicity)

A = np.array([[4, 2],
              [1, 3]])

eigenvalues, eigenvectors = np.linalg.eig(A)

# eigenvalues: [5, 2]
# eigenvectors: [[0.89, -0.71],
#                [0.45,  0.71]]

# Geometric meaning:
# v is a direction that only gets scaled (not rotated) by transformation A
# λ is the scaling factor

# Properties:
# 1. tr(A) = Σλ_i (trace equals sum of eigenvalues)
# 2. det(A) = Πλ_i (determinant equals product of eigenvalues)
# 3. For symmetric A: all eigenvalues are real
# 4. For positive definite A: all eigenvalues > 0
# 5. Eigenvectors corresponding to distinct eigenvalues are orthogonal (for symmetric A)

# Spectral theorem (symmetric matrices):
# A = QΛQ^T where Q is orthogonal, Λ is diagonal with eigenvalues

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
# Singular Value Decomposition (SVD)
# Theorem: Any matrix A ∈ ℝ^(m×n) can be decomposed as:
# A = UΣV^T

# where:
# - U ∈ ℝ^(m×m): Left singular vectors (U^T U = I)
# - Σ ∈ ℝ^(m×n): Diagonal matrix with singular values σ_i ≥ 0
# - V ∈ ℝ^(n×n): Right singular vectors (V^T V = I)

# Relationship to eigendecomposition:
# - A^T A = V Σ^T Σ V^T  (eigendecomposition of A^T A)
# - A A^T = U Σ Σ^T U^T  (eigendecomposition of A A^T)
# - σ_i² are eigenvalues of both A^T A and A A^T

A = np.random.rand(4, 3)
U, S, VT = np.linalg.svd(A, full_matrices=False)

# U (4×3): Left singular vectors (orthonormal columns)
# S (3,): Singular values σ₁ ≥ σ₂ ≥ ... ≥ σ_r > 0, r = rank(A)
# VT (3×3): Right singular vectors (orthonormal rows)

# Properties:
# 1. Singular values are unique and non-negative
# 2. rank(A) = number of non-zero singular values
# 3. ||A||₂ = σ₁ (largest singular value)
# 4. ||A||_F = √(σ₁² + ... + σ_r²) (Frobenius norm)
# 5. cond(A) = σ₁/σ_r (condition number)

# Reconstruction:
A_reconstructed = U @ np.diag(S) @ VT
# A_reconstructed ≈ A (within numerical precision ~1e-15)

# Best rank-k approximation (Eckart-Young theorem):
# A_k = Σ_{i=1}^k σ_i u_i v_i^T minimizes ||A - A_k||₂ and ||A - A_k||_F

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

# Formal definition: ||A|| is a norm if it satisfies:
# 1. ||A|| ≥ 0, equality iff A = 0 (positive definiteness)
# 2. ||cA|| = |c| ||A|| for scalar c (absolute homogeneity)
# 3. ||A + B|| ≤ ||A|| + ||B|| (triangle inequality)
# 4. ||AB|| ≤ ||A|| ||B|| (submultiplicativity)

# Frobenius norm (element-wise L2)
# ||A||_F = √(Σ_{i,j} a_{ij}²) = √(tr(A^T A))
frobenius_norm = np.linalg.norm(A, 'fro')
# = √(1² + 2² + 3² + 4²) = √30 ≈ 5.477

# Induced p-norms: ||A||_p = max_{x≠0} ||Ax||_p / ||x||_p

# L1 norm (maximum absolute column sum)
# ||A||₁ = max_j Σ_i |a_{ij}|
l1_norm = np.linalg.norm(A, 1)  # max(|1|+|3|, |2|+|4|) = 6

# L∞ norm (maximum absolute row sum)
# ||A||_∞ = max_i Σ_j |a_{ij}|
linf_norm = np.linalg.norm(A, np.inf)  # max(|1|+|2|, |3|+|4|) = 7

# L2 norm / Spectral norm (largest singular value)
# ||A||₂ = σ_max(A) = √(λ_max(A^T A))
l2_norm = np.linalg.norm(A, 2)  # ≈ 5.465

# Nuclear norm (sum of singular values)
# ||A||_* = Σ_i σ_i
nuclear_norm = np.linalg.norm(A, 'nuc')

# ML applications:
# 1. L2 regularization (weight decay): min L(θ) + λ||θ||₂²
# 2. Frobenius norm for matrix regularization: ||W||_F²
# 3. Nuclear norm for low-rank matrix completion: ||M||_*
# 4. Spectral norm for Lipschitz constraint: ||∇f|| ≤ L
# 5. Condition number: κ(A) = ||A||₂ · ||A⁻¹||₂ = σ_max/σ_min
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

**Definition**: Rate of change of a function

**Formal definition:**
```
f'(x) = lim_{h→0} [f(x+h) - f(x)] / h

Equivalent formulation:
f'(x₀) = lim_{x→x₀} [f(x) - f(x₀)] / (x - x₀)
```

**Interpretation:**
- Geometric: Slope of tangent line to f at x
- Physical: Instantaneous rate of change
- ML: Sensitivity of loss to parameter changes

**Existence conditions:**
- f is differentiable at x if the limit exists and is finite
- Differentiable ⟹ Continuous (but not vice versa)
- Counter-example: f(x) = |x| is continuous but not differentiable at x=0

**Higher-order derivatives:**
- f''(x): Second derivative (curvature, acceleration)
- f'''(x): Third derivative (jerk)
- f^(n)(x): n-th derivative

**Applications in ML:**
- First derivative ∇L: Direction of steepest ascent
- Second derivative H (Hessian): Curvature, used in Newton's method
- Lipschitz constant L: max ||f'(x)|| bounds learning rate
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

**Formal definition:**
```
For f: ℝⁿ → ℝ, the gradient is:

∇f(x) = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]^T ∈ ℝⁿ

Example: f(x, y) = x² + y²
∇f = [∂f/∂x, ∂f/∂y]^T = [2x, 2y]^T
```

**Geometric interpretation:**
- Direction: Points in direction of steepest increase of f
- Magnitude: ||∇f|| is the rate of increase in that direction
- Orthogonality: ∇f ⟂ level curves {x : f(x) = c}

**Mathematical properties:**
1. Directional derivative: D_v f(x) = ∇f(x) · v for unit vector v
2. Maximum directional derivative: max_{||v||=1} D_v f = ||∇f||
3. At local minimum/maximum: ∇f = 0 (critical point)
4. Convex function: f(y) ≥ f(x) + ∇f(x)^T(y-x) for all x,y

**Gradient descent update:**
```
x_{k+1} = x_k - α∇f(x_k)

where:
- α: Learning rate (step size)
- -∇f: Negative gradient (direction of steepest decrease)

Convergence guarantee (for L-smooth, μ-strongly convex f):
||x_k - x*||² ≤ (1 - μ/L)^k ||x_0 - x*||²

where κ = L/μ is condition number
```

**ML interpretation:**
- Loss L(θ): Gradient ∇L tells us how to adjust parameters θ
- Backpropagation: Efficient algorithm to compute ∇L
- Batch gradient: ∇L = (1/n)Σ_i ∇L_i (average over samples)
- Stochastic gradient: ∇̂L = ∇L_i (estimate from single sample)
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

**Probability Axioms (Kolmogorov, 1933)**:
```
Formal Framework:
(Ω, F, P) is a probability space where:
- Ω: Sample space (set of all possible outcomes)
- F: σ-algebra on Ω (collection of events, closed under complements and countable unions)
- P: Probability measure satisfying:

Axiom 1 (Non-negativity): P(A) ≥ 0 for all A ∈ F

Axiom 2 (Normalization): P(Ω) = 1

Axiom 3 (Countable Additivity): For countable disjoint events A₁, A₂, ...
P(⋃ᵢ₌₁^∞ Aᵢ) = Σᵢ₌₁^∞ P(Aᵢ)

Derived Properties:
1. P(∅) = 0 (probability of impossible event)
2. P(Aᶜ) = 1 - P(A) (complement rule)
3. If A ⊆ B, then P(A) ≤ P(B) (monotonicity)
4. P(A ∪ B) = P(A) + P(B) - P(A ∩ B) (inclusion-exclusion)
5. P(A) ≤ 1 for all events A (boundedness)

Measure Theory Connection:
Probability is a normalized measure (total measure = 1)
Integration: E[X] = ∫ X dP (expectation as integral)
```

**Conditional Probability**:
```
Definition: P(A|B) = P(A ∩ B) / P(B) for P(B) > 0

Interpretation: Probability of A given B has occurred

Properties:
1. P(A|B) ≥ 0 (non-negative)
2. P(Ω|B) = 1 (normalization)
3. P(·|B) is a valid probability measure

Independence:
Events A and B are independent if and only if:
P(A ∩ B) = P(A) × P(B)

Equivalent characterizations:
- P(A|B) = P(A) (knowing B doesn't change probability of A)
- P(B|A) = P(B) (knowing A doesn't change probability of B)

Pairwise vs Mutual Independence:
- Pairwise: P(Aᵢ ∩ Aⱼ) = P(Aᵢ)P(Aⱼ) for all i ≠ j
- Mutual: P(⋂ᵢ∈S Aᵢ) = ∏ᵢ∈S P(Aᵢ) for all subsets S
- Mutual ⟹ Pairwise, but not vice versa!

Chain Rule (Law of Multiplication):
P(A₁ ∩ A₂ ∩ ... ∩ Aₙ) = P(A₁) × P(A₂|A₁) × P(A₃|A₁∩A₂) × ... × P(Aₙ|A₁∩...∩Aₙ₋₁)

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

### Convergence Theorems (Limit Laws)

These fundamental theorems justify many ML practices like averaging predictions and empirical risk minimization.

**Law of Large Numbers (LLN)**:
```
Weak Law of Large Numbers (WLLN):
Let X₁, X₂, ..., Xₙ be i.i.d. random variables with E[Xᵢ] = μ and Var(Xᵢ) = σ² < ∞

Sample mean: X̄ₙ = (1/n) Σᵢ₌₁ⁿ Xᵢ

Then: X̄ₙ →ᵖ μ as n → ∞

(Convergence in probability: lim P(|X̄ₙ - μ| > ε) = 0 for all ε > 0)

Strong Law of Large Numbers (SLLN):
X̄ₙ → μ almost surely (a.s.)

(P(lim X̄ₙ = μ) = 1)

Proof Sketch (Weak Law via Chebyshev):
By Chebyshev inequality: P(|X̄ₙ - μ| ≥ ε) ≤ Var(X̄ₙ)/ε²

Var(X̄ₙ) = Var((1/n)Σ Xᵢ) = (1/n²) × n × σ² = σ²/n

Therefore: P(|X̄ₙ - μ| ≥ ε) ≤ σ²/(nε²) → 0 as n → ∞  ✓

ML Applications:
1. Empirical Risk Minimization (ERM):
   Sample loss (1/n)Σ L(θ; xᵢ) → Expected loss E[L(θ; X)] as n → ∞

2. Monte Carlo Estimation:
   Sample average → True expectation

3. Ensemble Methods:
   Average of predictions → Expected prediction
```

**Central Limit Theorem (CLT)**:
```
Let X₁, X₂, ..., Xₙ be i.i.d. with E[Xᵢ] = μ and Var(Xᵢ) = σ² < ∞

Sample mean: X̄ₙ = (1/n) Σᵢ₌₁ⁿ Xᵢ

Standardized sum: Zₙ = √n(X̄ₙ - μ)/σ

Then: Zₙ →ᵈ N(0,1) as n → ∞

(Convergence in distribution to standard normal)

Equivalently: X̄ₙ ≈ N(μ, σ²/n) for large n

Berry-Esseen Theorem (Quantitative CLT):
Let ρ = E[|Xᵢ - μ|³] < ∞

Then: sup_x |P(Zₙ ≤ x) - Φ(x)| ≤ Cρ/(σ³√n)

where Φ is standard normal CDF, C ≈ 0.4748

Convergence Rate: O(1/√n)

Practical Rule: n ≥ 30 usually sufficient for normal approximation

ML Applications:
1. Confidence Intervals:
   X̄ₙ ± 1.96 × σ/√n gives 95% CI for μ

2. Hypothesis Testing:
   Test statistic √n(X̄ₙ - μ₀)/σ ~ N(0,1) under H₀

3. Bootstrap Distribution:
   Sampling distribution of statistics approximately normal

4. Gradient Descent Noise:
   Stochastic gradient ∇̂L ≈ N(∇L, Σ/n) for large batch size n

5. Neural Network Outputs:
   Sum of many small contributions → approximately normal
```

**Example Application**:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Demonstrate CLT with non-normal distribution
def clt_demo(n_samples, n_means=10000, distribution='uniform'):
    """
    Demonstrate Central Limit Theorem

    Even for non-normal distributions, sample means are approximately normal!
    """
    # Generate data from various distributions
    sample_means = []

    for _ in range(n_means):
        if distribution == 'uniform':
            # Uniform [0, 1]: μ=0.5, σ²=1/12
            sample = np.random.uniform(0, 1, n_samples)
            mu, sigma2 = 0.5, 1/12
        elif distribution == 'exponential':
            # Exponential(λ=1): μ=1, σ²=1
            sample = np.random.exponential(1, n_samples)
            mu, sigma2 = 1.0, 1.0
        elif distribution == 'bernoulli':
            # Bernoulli(p=0.3): μ=0.3, σ²=0.21
            sample = np.random.binomial(1, 0.3, n_samples)
            mu, sigma2 = 0.3, 0.3*0.7

        sample_means.append(np.mean(sample))

    sample_means = np.array(sample_means)

    # Theoretical normal distribution
    theoretical_mean = mu
    theoretical_std = np.sqrt(sigma2 / n_samples)

    # Plot
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(sample_means, bins=50, density=True, alpha=0.7, edgecolor='black')
    x = np.linspace(sample_means.min(), sample_means.max(), 100)
    plt.plot(x, norm.pdf(x, theoretical_mean, theoretical_std),
             'r-', linewidth=2, label=f'N({theoretical_mean:.2f}, {theoretical_std:.4f}²)')
    plt.xlabel('Sample Mean')
    plt.ylabel('Density')
    plt.title(f'CLT Demo: {distribution.capitalize()} Distribution (n={n_samples})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Q-Q plot
    plt.subplot(1, 2, 2)
    from scipy.stats import probplot
    probplot(sample_means, dist="norm", plot=plt)
    plt.title('Q-Q Plot (Normal)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Numerical verification
    empirical_mean = np.mean(sample_means)
    empirical_std = np.std(sample_means, ddof=1)

    print(f"Distribution: {distribution}")
    print(f"Sample size n = {n_samples}")
    print(f"Theoretical: μ = {theoretical_mean:.4f}, σ = {theoretical_std:.4f}")
    print(f"Empirical:   μ = {empirical_mean:.4f}, σ = {empirical_std:.4f}")
    print(f"Error: |μ_emp - μ_theory| = {abs(empirical_mean - theoretical_mean):.6f}")

# Try with different sample sizes
for n in [5, 30, 100]:
    print(f"\n{'='*60}")
    clt_demo(n_samples=n, distribution='exponential')
    # As n increases, empirical matches theoretical better!
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

### Statistical Inference Theory

**Confidence Intervals**:
```
Definition: A (1-α)100% confidence interval for parameter θ is an interval [L, U] where:

P(L ≤ θ ≤ U) = 1 - α

Common values: α = 0.05 (95% CI), α = 0.01 (99% CI)

Interpretation (Frequentist):
- If we repeat the experiment many times and construct CIs each time,
  approximately (1-α)100% of intervals will contain true θ
- NOT: "θ has 95% probability of being in this interval" (θ is fixed, not random!)

For Sample Mean (σ known):
X̄ₙ ~ N(μ, σ²/n) by CLT

95% CI: X̄ₙ ± z_{α/2} × σ/√n
where z_{0.025} = 1.96 for 95% CI

For Sample Mean (σ unknown):
Use sample std s: 95% CI: X̄ₙ ± t_{n-1,α/2} × s/√n
where t_{n-1,α/2} is t-distribution critical value with n-1 degrees of freedom

Standard Error (SE):
SE(X̄ₙ) = σ/√n (known variance)
SE(X̄ₙ) = s/√n (estimated variance)

Margin of Error: ME = z_{α/2} × SE

Precision: ∝ 1/√n (quadruple sample size → double precision)
```

**Hypothesis Testing Framework**:
```
Formal Structure:

1. Null Hypothesis H₀: Default assumption (e.g., "no effect", "μ = μ₀")
2. Alternative Hypothesis H₁: What we want to test (e.g., "effect exists", "μ ≠ μ₀")
3. Test Statistic T: Function of data that measures evidence against H₀
4. Significance Level α: Threshold for rejecting H₀ (typically 0.05 or 0.01)
5. p-value: P(observe T as extreme as observed | H₀ is true)
6. Decision Rule: Reject H₀ if p-value < α

Types of Tests:
- Two-sided: H₁: μ ≠ μ₀ (detect any difference)
- One-sided: H₁: μ > μ₀ or H₁: μ < μ₀ (directional)

Error Types:
┌─────────────┬──────────────────┬──────────────────┐
│             │ H₀ True          │ H₀ False         │
├─────────────┼──────────────────┼──────────────────┤
│ Reject H₀   │ Type I Error (α) │ Correct (Power)  │
│             │ False Positive   │ True Positive    │
├─────────────┼──────────────────┼──────────────────┤
│ Fail to     │ Correct (1-α)    │ Type II Error (β)│
│ Reject H₀   │ True Negative    │ False Negative   │
└─────────────┴──────────────────┴──────────────────┘

Definitions:
- Significance Level α = P(Type I Error) = P(Reject H₀ | H₀ true)
- Type II Error Rate β = P(Type II Error) = P(Fail to reject H₀ | H₁ true)
- Statistical Power = 1 - β = P(Reject H₀ | H₁ true)

Power Analysis:
Power depends on:
1. Sample size n (larger n → higher power)
2. Effect size δ = |μ₁ - μ₀|/σ (larger effect → higher power)
3. Significance level α (larger α → higher power, but more false positives)

For t-test with known σ:
Power = Φ(√n × δ/σ - z_{α/2}) + Φ(-√n × δ/σ - z_{α/2})

To achieve power = 0.8 for two-sided test (α = 0.05):
n ≈ 16 × (σ/δ)² (rule of thumb)

Example: To detect effect size δ = 0.5σ with 80% power:
n ≈ 16 × (σ/0.5σ)² = 64 samples per group

p-value Interpretation:
- p < 0.001: Very strong evidence against H₀
- 0.001 ≤ p < 0.01: Strong evidence against H₀
- 0.01 ≤ p < 0.05: Moderate evidence against H₀
- 0.05 ≤ p < 0.10: Weak evidence against H₀
- p ≥ 0.10: Little to no evidence against H₀

Common Misconceptions:
✗ p-value is NOT P(H₀ is true | data)
✓ p-value IS P(data as extreme | H₀ is true)

✗ p < 0.05 does NOT mean "important" or "large effect"
✓ Statistical significance ≠ Practical significance

✗ p > 0.05 does NOT prove H₀ is true
✓ Absence of evidence ≠ Evidence of absence

Multiple Testing Correction:
When testing m hypotheses:
- Bonferroni: Use α/m for each test (conservative)
- Holm-Bonferroni: Sequential procedure (less conservative)
- False Discovery Rate (FDR): Control expected proportion of false positives
  Benjamini-Hochberg: Ensures FDR ≤ q (e.g., q = 0.05)

Family-Wise Error Rate (FWER):
P(at least one Type I error) = 1 - (1-α)^m ≈ m×α for small α
Example: m=20 tests, α=0.05 → FWER ≈ 0.64 (64% chance of false positive!)
```

**Sufficient Statistics and Fisher Information**:
```
Sufficient Statistic T(X):
A statistic T(X) is sufficient for θ if and only if:
P(X | T(X), θ) = P(X | T(X))

(Data X contains no more information about θ beyond T(X))

Factorization Theorem (Neyman-Fisher):
T(X) is sufficient for θ ⟺ p(x|θ) = g(T(x), θ) × h(x)

Example: For X₁, ..., Xₙ ~ N(μ, σ²):
- T(X) = X̄ is sufficient for μ (when σ² known)
- T(X) = (X̄, S²) is sufficient for (μ, σ²)

Fisher Information:
I(θ) = E[(∂/∂θ log p(X|θ))²] = -E[∂²/∂θ² log p(X|θ)]

Measures: Information about θ contained in single observation

Properties:
1. I(θ) ≥ 0 (non-negative)
2. For n i.i.d. observations: Iₙ(θ) = n × I(θ) (additivity)
3. Under reparameterization θ → φ(θ): I_φ = I_θ × (dθ/dφ)²

Cramér-Rao Lower Bound:
For any unbiased estimator θ̂ of θ:

Var(θ̂) ≥ 1/I(θ)

Efficient Estimator: Achieves equality (minimum variance among unbiased estimators)

Example: For X ~ N(μ, σ²) with σ² known:
Fisher information: I(μ) = 1/σ²
Sample mean X̄ₙ: Var(X̄ₙ) = σ²/n = 1/(n×I(μ))
Therefore X̄ₙ is efficient! ✓

ML Application:
Fisher information appears in:
- Natural gradient descent: θ_{t+1} = θ_t - α × I(θ)⁻¹ × ∇L
- Uncertainty quantification in neural networks
- Bayesian posterior approximation (Laplace approximation)
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

#### Convergence Theory for Gradient Descent Variants

**Mathematical Framework:**

**Assumptions and Definitions:**
```
L-Lipschitz Continuous Gradient:
||∇f(x) - ∇f(y)|| ≤ L·||x - y|| for all x, y

Convexity:
f(y) ≥ f(x) + ∇f(x)^T(y - x) for all x, y

Strong Convexity (with parameter μ > 0):
f(y) ≥ f(x) + ∇f(x)^T(y - x) + (μ/2)||y - x||² for all x, y

Smoothness implies: f(y) ≤ f(x) + ∇f(x)^T(y - x) + (L/2)||y - x||²
```

**Theorem 1: Gradient Descent Convergence for Smooth Convex Functions**
```
Problem: min_{x∈ℝ^d} f(x), where f is convex and L-smooth

Algorithm: x_{k+1} = x_k - α·∇f(x_k)

Step Size: α ≤ 1/L

Convergence Rate:
f(x_k) - f(x*) ≤ (2L·||x_0 - x*||²) / (k + 4)
             = O(1/k)

Result: Sublinear convergence, need O(1/ε) iterations for ε-accuracy
```

**Proof Sketch:**
```
Key Lemma (Descent Lemma):
For α ≤ 1/L and L-smooth f:
f(x_{k+1}) ≤ f(x_k) - (α/2)||∇f(x_k)||²

Proof of Lemma:
1. By L-smoothness:
   f(x_{k+1}) ≤ f(x_k) + ∇f(x_k)^T(x_{k+1} - x_k) + (L/2)||x_{k+1} - x_k||²

2. Substitute x_{k+1} = x_k - α·∇f(x_k):
   f(x_{k+1}) ≤ f(x_k) - α·||∇f(x_k)||² + (Lα²/2)||∇f(x_k)||²
              = f(x_k) - α(1 - Lα/2)||∇f(x_k)||²

3. If α ≤ 1/L, then 1 - Lα/2 ≥ 1/2:
   f(x_{k+1}) ≤ f(x_k) - (α/2)||∇f(x_k)||²  ✓

Main Convergence Proof:
1. By convexity: f(x_k) - f(x*) ≤ ∇f(x_k)^T(x_k - x*)

2. Expand ||x_{k+1} - x*||²:
   ||x_{k+1} - x*||² = ||x_k - α·∇f(x_k) - x*||²
                     = ||x_k - x*||² - 2α·∇f(x_k)^T(x_k - x*) + α²||∇f(x_k)||²

3. Rearrange:
   2α·∇f(x_k)^T(x_k - x*) = ||x_k - x*||² - ||x_{k+1} - x*||² + α²||∇f(x_k)||²
                           ≥ ||x_k - x*||² - ||x_{k+1} - x*||²  (drop positive term)

4. Therefore:
   f(x_k) - f(x*) ≤ (||x_k - x*||² - ||x_{k+1} - x*||²) / (2α)

5. Sum from k=0 to K-1:
   Σ_{k=0}^{K-1} [f(x_k) - f(x*)] ≤ (||x_0 - x*||² - ||x_K - x*||²) / (2α)
                                   ≤ ||x_0 - x*||² / (2α)

6. Since f(x_k) is decreasing (by Descent Lemma):
   K·[f(x_K) - f(x*)] ≤ Σ_{k=0}^{K-1} [f(x_k) - f(x*)] ≤ ||x_0 - x*||² / (2α)

7. Final bound:
   f(x_K) - f(x*) ≤ ||x_0 - x*||² / (2αK)

   With α = 1/L:
   f(x_K) - f(x*) ≤ L·||x_0 - x*||² / (2K) = O(1/K)  ✓
```

**Theorem 2: Gradient Descent for Strongly Convex Functions**
```
Problem: min f(x), where f is μ-strongly convex and L-smooth

Step Size: α ≤ 2/(μ + L) (or simply α = 1/L)

Convergence Rate:
||x_k - x*||² ≤ (1 - μ/L)^k · ||x_0 - x*||²
f(x_k) - f(x*) ≤ (L/2)(1 - μ/L)^k · ||x_0 - x*||²

Result: Linear (exponential) convergence!
Need O(log(1/ε)) iterations for ε-accuracy
```

**Proof Sketch:**
```
Key Property: For μ-strongly convex and L-smooth f:
||∇f(x)||² ≥ 2μ[f(x) - f(x*)]

Proof:
1. By strong convexity at x*:
   f(x) ≥ f(x*) + ∇f(x*)^T(x - x*) + (μ/2)||x - x*||²
        = f(x*) + (μ/2)||x - x*||²  (since ∇f(x*) = 0)

2. By smoothness at x:
   f(x*) ≥ f(x) + ∇f(x)^T(x* - x) - (L/2)||x* - x||²

3. Combine:
   ||∇f(x)||² = ||∇f(x) - ∇f(x*)||² ≥ μL·||x - x*||²  (PL inequality)
              ≥ 2μ[f(x) - f(x*)]  ✓

Main Convergence:
1. Start with:
   ||x_{k+1} - x*||² = ||x_k - x*||² - 2α·∇f(x_k)^T(x_k - x*) + α²||∇f(x_k)||²

2. By strong convexity:
   ∇f(x_k)^T(x_k - x*) ≥ f(x_k) - f(x*) + (μ/2)||x_k - x*||²

3. Substitute α = 1/L:
   ||x_{k+1} - x*||² ≤ ||x_k - x*||²[1 - μ/L] + (1/L²)||∇f(x_k)||²[1 - μ/L]
                     ≤ (1 - μ/L)||x_k - x*||²

4. Iterate:
   ||x_k - x*||² ≤ (1 - μ/L)^k · ||x_0 - x*||²  ✓

Condition Number: κ = L/μ
- If κ is small (well-conditioned): Fast convergence
- If κ is large (ill-conditioned): Slow convergence
```

**Theorem 3: Stochastic Gradient Descent (SGD) Convergence**
```
Problem: min f(x) = E_{ξ}[f(x; ξ)]
         Where ξ represents random data samples

Algorithm: x_{k+1} = x_k - α_k·∇f(x_k; ξ_k)
          ∇f(x_k; ξ_k) is unbiased: E[∇f(x_k; ξ_k)] = ∇f(x_k)

Robbins-Monro Conditions (for learning rate α_k):
1. Σ_{k=1}^∞ α_k = ∞        (step sizes sum to infinity)
2. Σ_{k=1}^∞ α_k² < ∞       (step sizes squared sum is finite)

Example: α_k = α_0/√k satisfies both conditions

Convergence Result (for convex f):
E[f(x_k)] - f(x*) = O(1/√k)

Result: Slower than batch GD (O(1/k)), but much cheaper per iteration!
```

**Proof Intuition:**
```
Key Inequality:
E[||x_{k+1} - x*||²] = E[||x_k - α_k·g_k - x*||²]
                     = ||x_k - x*||² - 2α_k·∇f(x_k)^T(x_k - x*) + α_k²·E[||g_k||²]

where g_k = ∇f(x_k; ξ_k) is stochastic gradient with variance σ²

Trade-off:
- Term 1: -2α_k·∇f(x_k)^T(x_k - x*) → Progress towards optimum
- Term 2: +α_k²·σ² → Variance from stochastic gradient

As k → ∞:
- α_k → 0 makes variance term → 0 (condition 2)
- But Σα_k = ∞ ensures we reach optimum (condition 1)
```

**Theorem 4: Momentum Convergence (Nesterov Accelerated Gradient)**
```
Algorithm:
v_{k+1} = β·v_k + ∇f(x_k)
x_{k+1} = x_k - α·v_{k+1}

Convergence for Convex L-smooth:
f(x_k) - f(x*) = O(1/k²)  (compared to O(1/k) for vanilla GD!)

Convergence for Strongly Convex:
||x_k - x*|| = O((1 - √(μ/L))^k)  (improved constant)

Result: Optimal first-order method for smooth convex optimization
```

**Theorem 5: Adam Convergence (Sketch)**
```
Algorithm:
m_k = β_1·m_{k-1} + (1-β_1)·g_k         (first moment)
v_k = β_2·v_{k-1} + (1-β_2)·g_k²        (second moment)
m̂_k = m_k / (1 - β_1^k)                 (bias correction)
v̂_k = v_k / (1 - β_2^k)                 (bias correction)
x_{k+1} = x_k - α·m̂_k / (√v̂_k + ε)

Typical values: β_1 = 0.9, β_2 = 0.999, ε = 10^(-8)

Convergence (for convex case):
Regret bound: R_T = O(√T)
Average convergence: (1/T)Σ_{k=1}^T [f(x_k) - f(x*)] = O(1/√T)

Note: Adam may NOT converge for some convex problems!
Fix: AMSGrad variant with max(v_1, ..., v_k) instead of v_k
```

**Summary Table: Convergence Rates**
```
Algorithm        | Convex      | Strongly Convex    | Per-iteration Cost
-----------------|-------------|--------------------|-----------------
GD               | O(1/k)      | O(exp(-μk/L))     | O(nd)
SGD              | O(1/√k)     | O(1/k)            | O(d)
Momentum (NAG)   | O(1/k²)     | O(exp(-√μ/L·k))   | O(nd)
Adam/RMSprop     | O(1/√k)     | O(1/√k)           | O(d)
Newton           | O(1/k²)     | Quadratic         | O(nd²+d³)

Where:
- n: dataset size
- d: dimension
- k: iteration number
- μ: strong convexity parameter
- L: smoothness parameter
- κ = L/μ: condition number
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

**Constrained Optimization (Lagrange Multipliers)**:
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

### Advanced Constrained Optimization

**Karush-Kuhn-Tucker (KKT) Conditions**:
```
General Constrained Problem:
minimize f(x)
subject to:
  gᵢ(x) ≤ 0  for i = 1, ..., m  (inequality constraints)
  hⱼ(x) = 0  for j = 1, ..., p  (equality constraints)

Lagrangian:
L(x, λ, μ) = f(x) + Σᵢ λᵢgᵢ(x) + Σⱼ μⱼhⱼ(x)

where λᵢ ≥ 0 (inequality multipliers), μⱼ ∈ ℝ (equality multipliers)

KKT Necessary Conditions (for x* to be optimal):
Assume f, gᵢ, hⱼ are differentiable and constraint qualification holds

1. Stationarity:
   ∇f(x*) + Σᵢ λᵢ*∇gᵢ(x*) + Σⱼ μⱼ*∇hⱼ(x*) = 0

2. Primal Feasibility:
   gᵢ(x*) ≤ 0  for all i
   hⱼ(x*) = 0  for all j

3. Dual Feasibility:
   λᵢ* ≥ 0  for all i

4. Complementary Slackness:
   λᵢ* × gᵢ(x*) = 0  for all i
   (Either constraint is inactive (gᵢ < 0) or multiplier is positive (λᵢ > 0))

KKT Sufficient Conditions:
If f and gᵢ are convex, hⱼ are affine, and x* satisfies KKT conditions,
then x* is a global minimum!

Constraint Qualification:
Common conditions ensuring KKT conditions are necessary:
- Linear Independence Constraint Qualification (LICQ)
- Slater's Condition (for convex problems)
- Mangasarian-Fromovitz Constraint Qualification (MFCQ)

Example: Support Vector Machines (SVM)
Primal Problem:
minimize  (1/2)||w||² + C Σᵢ ξᵢ
subject to:
  yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ  for all i
  ξᵢ ≥ 0  for all i

Dual Problem (via KKT):
maximize  Σᵢ αᵢ - (1/2) Σᵢⱼ αᵢαⱼyᵢyⱼ(xᵢ·xⱼ)
subject to:
  Σᵢ αᵢyᵢ = 0
  0 ≤ αᵢ ≤ C  for all i

Complementary Slackness:
αᵢ[yᵢ(w·xᵢ + b) - 1 + ξᵢ] = 0
- αᵢ = 0 → Non-support vector (correctly classified with margin)
- 0 < αᵢ < C → Support vector on margin (ξᵢ = 0)
- αᵢ = C → Support vector inside margin or misclassified (ξᵢ > 0)
```

**Subgradient Methods (Non-smooth Optimization)**:
```
Problem: Many ML loss functions are non-differentiable
- L1 norm: |x| not differentiable at x = 0
- Hinge loss: max(0, 1 - y·f(x)) not differentiable at y·f(x) = 1
- ReLU: max(0, x) not differentiable at x = 0

Subgradient:
For convex function f, g is a subgradient at x if:
f(y) ≥ f(x) + g^T(y - x)  for all y

Subdifferential:
∂f(x) = {all subgradients of f at x}

Properties:
1. If f is differentiable at x: ∂f(x) = {∇f(x)} (singleton)
2. For L1 norm at 0: ∂|x| = [-1, 1]
3. For max(f, g): ∂max(f,g) ⊆ ∂f ∪ ∂g

Subgradient Descent:
x_{k+1} = x_k - α_k × g_k

where g_k ∈ ∂f(x_k) is any subgradient

Convergence (for convex f):
With diminishing step sizes α_k such that:
- Σ α_k = ∞ (step sizes sum to infinity)
- Σ α_k² < ∞ (squared step sizes converge)

Then: f(x_k) - f(x*) = O(1/√k)

Example: α_k = c/√k satisfies both conditions

Note: Slower than gradient descent (O(1/√k) vs O(1/k))
But works for non-smooth functions!

Proximal Gradient Method:
For f(x) = g(x) + h(x) where g is smooth, h is non-smooth convex:

x_{k+1} = prox_{α_k h}(x_k - α_k∇g(x_k))

Proximal operator:
prox_h(y) = argmin_x [h(x) + (1/2)||x - y||²]

Example: Lasso regression
f(w) = ||Xw - y||² + λ||w||₁
      └──g(w)──┘   └─h(w)─┘

Proximal operator of h(w) = λ||w||₁:
prox_h(w)ᵢ = sign(wᵢ) × max(|wᵢ| - λ, 0)  (soft thresholding)

Convergence: O(1/k) like gradient descent!
```

**Practical Optimization Tips**:
```
1. Choose Right Algorithm:
   - Smooth unconstrained → Gradient Descent / Adam
   - Non-smooth → Subgradient / Proximal Gradient
   - Constrained → Projected Gradient / Interior Point
   - Large scale → Stochastic methods (SGD, Adam)

2. Hyperparameter Tuning:
   - Learning rate: Most important! Use grid search or learning rate schedules
   - Batch size: Larger → more stable but slower, smaller → noisy but faster
   - Momentum: 0.9 or 0.99 usually good defaults

3. Convergence Diagnostics:
   - Monitor: Loss, gradient norm, parameter changes
   - Plot: Loss vs iteration (should decrease monotonically for convex)
   - Check: KKT residuals for constrained problems

4. Numerical Stability:
   - Scale features to similar ranges
   - Use numerically stable formulations (log-sum-exp trick)
   - Add small epsilon to denominators (e.g., Adam: 1e-8)

5. Initialization:
   - Random initialization to break symmetry
   - Xavier/He initialization for neural networks
   - Warm start from previous solutions

6. Regularization:
   - L2 (Ridge): Improves conditioning, smooth optimization landscape
   - L1 (Lasso): Sparse solutions, use proximal gradient
   - Early stopping: Implicit regularization
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

## 🧠 Neural Network Initialization Theory

Proper weight initialization is critical for successful neural network training. Poor initialization can lead to vanishing/exploding gradients, slow convergence, or complete training failure.

### The Initialization Problem

**Why Initialization Matters:**
```
Problem: Neural networks are highly non-convex
- Different initializations → different local minima
- Bad initialization → vanishing/exploding gradients
- Good initialization → faster convergence, better final performance

Key Insight: Initialize weights to preserve signal variance across layers
```

**Naive Approaches (Don't Do This!):**
```python
# ❌ All zeros: Symmetry problem
W = np.zeros((n_out, n_in))
# All neurons learn the same function!
# Gradient for all neurons is identical
# Network effectively has only one neuron per layer

# ❌ All same value: Same problem
W = np.ones((n_out, n_in)) * 0.5

# ❌ Too large values
W = np.random.randn(n_out, n_in) * 10
# Output variance explodes: Var(output) = (n_in × 10²) × Var(input)
# Gradients explode

# ❌ Too small values
W = np.random.randn(n_out, n_in) * 0.001
# Output variance vanishes: Var(output) ≈ 0
# Gradients vanish
```

### Xavier/Glorot Initialization (2010)

**Mathematical Foundation:**

**Goal:** Preserve variance of activations and gradients across layers

**Assumption:** Linear activation (or near-linear like tanh around 0)

**Forward Pass Analysis:**
```
Layer computation: y = W·x + b

For one neuron: y_i = Σ_{j=1}^{n_in} w_{ij} x_j

Assumptions:
1. x_j are i.i.d. with mean 0 and variance σ²_x
2. w_{ij} are i.i.d. with mean 0 and variance σ²_w
3. x and w are independent

Variance of output:
Var(y_i) = Var(Σ_j w_{ij} x_j)
         = Σ_j Var(w_{ij} x_j)           (independence)
         = Σ_j E[w_{ij}²] E[x_j²]         (independence)
         = Σ_j Var(w_{ij}) Var(x_j)       (mean 0)
         = n_in · σ²_w · σ²_x

To preserve variance (Var(y_i) = σ²_x):
n_in · σ²_w = 1
σ²_w = 1 / n_in
```

**Backward Pass Analysis:**
```
Gradient backprop: ∂L/∂x = W^T · ∂L/∂y

By similar analysis:
Var(∂L/∂x_j) = n_out · σ²_w · Var(∂L/∂y)

To preserve gradient variance:
n_out · σ²_w = 1
σ²_w = 1 / n_out
```

**Xavier/Glorot Compromise:**
```
Problem: Forward wants σ²_w = 1/n_in, backward wants σ²_w = 1/n_out

Solution: Average them!
σ²_w = 2 / (n_in + n_out)

Xavier Uniform:
W ~ U[-√(6/(n_in + n_out)), √(6/(n_in + n_out))]

Xavier Normal:
W ~ N(0, 2/(n_in + n_out))

Note: U[-a, a] has variance a²/3, so a = √(3·2/(n_in+n_out)) = √(6/(n_in+n_out))
```

**Implementation:**
```python
def xavier_uniform(n_in, n_out):
    """
    Xavier/Glorot uniform initialization

    Used for: tanh, sigmoid activations
    """
    limit = np.sqrt(6.0 / (n_in + n_out))
    return np.random.uniform(-limit, limit, size=(n_out, n_in))

def xavier_normal(n_in, n_out):
    """
    Xavier/Glorot normal initialization

    Used for: tanh, sigmoid activations
    """
    std = np.sqrt(2.0 / (n_in + n_out))
    return np.random.randn(n_out, n_in) * std
```

### He Initialization (2015)

**Motivation:** Xavier assumes linear activation, but ReLU is non-linear!

**ReLU Analysis:**
```
ReLU(x) = max(0, x)

Property: Kills half the neurons (negative values → 0)

Effect on variance:
- Input variance: σ²
- After ReLU: σ²/2 (approximately, for zero-mean input)

Derivation:
For x ~ N(0, σ²):
E[ReLU(x)] = E[x | x > 0] · P(x > 0) = (σ/√(2π)) · 0.5

Var(ReLU(x)) = E[ReLU(x)²] - E[ReLU(x)]²
             = E[x² | x > 0] · P(x > 0) - (σ/√(2π) · 0.5)²
             = σ²/2 - small term
             ≈ σ²/2

So ReLU halves the variance!
```

**He Initialization:**
```
Forward pass with ReLU:
Var(y_i) = n_in · σ²_w · σ²_x / 2  (ReLU kills half)

To preserve variance (Var(y_i) = σ²_x):
n_in · σ²_w / 2 = 1
σ²_w = 2 / n_in

He Normal (most common):
W ~ N(0, 2/n_in)

He Uniform:
W ~ U[-√(6/n_in), √(6/n_in)]
```

**Implementation:**
```python
def he_normal(n_in, n_out):
    """
    He initialization (Kaiming initialization)

    Used for: ReLU, Leaky ReLU, PReLU activations

    Reference: He et al., "Delving Deep into Rectifiers: Surpassing
    Human-Level Performance on ImageNet Classification", ICCV 2015
    """
    std = np.sqrt(2.0 / n_in)
    return np.random.randn(n_out, n_in) * std

def he_uniform(n_in, n_out):
    """He uniform initialization"""
    limit = np.sqrt(6.0 / n_in)
    return np.random.uniform(-limit, limit, size=(n_out, n_in))
```

### Comparison and Guidelines

**Initialization Summary:**
```
Activation       | Forward Preserve | Backward Preserve | Recommended
-----------------|------------------|-------------------|-------------
Linear/None      | Var = 1/n_in    | Var = 1/n_out    | Xavier
tanh             | Var = 1/n_in    | Var = 1/n_out    | Xavier
sigmoid          | Var = 1/n_in    | Var = 1/n_out    | Xavier
ReLU             | Var = 2/n_in    | Var = 2/n_out    | He
Leaky ReLU       | Var ≈ 2/n_in    | Var ≈ 2/n_out    | He
ELU              | Var ≈ 1.5/n_in  | Var ≈ 1.5/n_out  | He or Xavier
SELU             | Special         | Special          | LeCun*

*LeCun Normal: W ~ N(0, 1/n_in)
```

**Modern PyTorch/TensorFlow Defaults:**
```python
import torch.nn as nn

# Linear layer
nn.Linear(n_in, n_out)
# Default: Xavier uniform (Glorot)

# Conv2D layer
nn.Conv2d(in_channels, out_channels, kernel_size)
# Default: He (Kaiming) uniform for ReLU

# LSTM/GRU
nn.LSTM(input_size, hidden_size)
# Default: Xavier uniform (Glorot)
```

**Complete Initialization Example:**
```python
class NeuralNetwork:
    """Neural network with proper initialization"""

    def __init__(self, layers, activation='relu'):
        """
        Args:
            layers: [n_input, n_hidden1, n_hidden2, ..., n_output]
            activation: 'relu', 'tanh', 'sigmoid'
        """
        self.weights = []
        self.biases = []

        for i in range(len(layers) - 1):
            n_in, n_out = layers[i], layers[i+1]

            # Initialize weights
            if activation == 'relu':
                # He initialization
                W = np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in)
            elif activation in ['tanh', 'sigmoid']:
                # Xavier initialization
                W = np.random.randn(n_out, n_in) * np.sqrt(2.0 / (n_in + n_out))
            else:
                # Default: small random
                W = np.random.randn(n_out, n_in) * 0.01

            # Initialize biases to zero (common practice)
            b = np.zeros((n_out, 1))

            self.weights.append(W)
            self.biases.append(b)

    def forward(self, x, activation='relu'):
        """Forward pass with specified activation"""
        a = x
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            z = W @ a + b
            if activation == 'relu':
                a = np.maximum(0, z)
            elif activation == 'tanh':
                a = np.tanh(z)
            elif activation == 'sigmoid':
                a = 1 / (1 + np.exp(-z))

        # Output layer (no activation for regression, or apply softmax for classification)
        z = self.weights[-1] @ a + self.biases[-1]
        return z

# Example usage
model = NeuralNetwork([784, 512, 256, 10], activation='relu')
print(f"Layer 1 weights std: {model.weights[0].std():.4f}")
print(f"Expected std: {np.sqrt(2.0/784):.4f}")
```

### Advanced Initialization Strategies

**1. LSUV (Layer-Sequential Unit-Variance, 2016):**
```python
def lsuv_init(model, data_sample):
    """
    Initialize weights then adjust to unit variance

    1. Initialize with orthogonal matrices
    2. Forward pass with sample data
    3. Scale weights to make output variance = 1
    4. Repeat for each layer
    """
    x = data_sample

    for layer in model.layers:
        # Initialize with orthogonal matrix
        W = np.linalg.qr(np.random.randn(layer.n_out, layer.n_in))[0]
        layer.W = W

        # Forward pass
        z = layer.forward(x)

        # Adjust to unit variance
        std = z.std()
        layer.W = layer.W / std

        x = layer.activation(z)
```

**2. Fixup Initialization (2019):**
```
For very deep networks (ResNets):
- Initialize most layers with He/Xavier
- Scale residual branches by 1/√L (L = depth)
- No batch normalization needed!
```

**3. Batch Normalization Alternative:**
```
Instead of careful initialization:
- Use Batch Normalization after each layer
- BN normalizes activations to mean=0, std=1
- Makes network more robust to initialization
- Trade-off: BN adds computation and complexity
```

### Theoretical Guarantees

**Theorem (He et al., 2015):**
```
For ReLU networks with He initialization:
- Forward signal does not vanish or explode
- Backward gradient does not vanish or explode
- Enables training of networks with 30+ layers

Mathematically:
E[||y^(l)||²] = E[||x^(0)||²]  (forward)
E[||∂L/∂x^(0)||²] = E[||∂L/∂y^(L)||²]  (backward)

where l = layer index, L = total layers
```

**Condition for Gradient Flow:**
```
For stable training:

Forward: σ²_out = σ²_in  (variance preservation)
Backward: σ²_grad = constant across layers

This requires: σ²_w = O(1/n_in)

Violation leads to:
- σ²_w too large → exploding gradients
- σ²_w too small → vanishing gradients
```

### Summary and Best Practices

**Quick Reference:**
```python
import torch.nn as nn

# For ReLU/Leaky ReLU (most common):
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
nn.init.zeros_(layer.bias)

# For tanh/sigmoid:
nn.init.xavier_normal_(layer.weight)
nn.init.zeros_(layer.bias)

# For LSTM/GRU:
nn.init.orthogonal_(layer.weight_ih)
nn.init.orthogonal_(layer.weight_hh)
nn.init.zeros_(layer.bias)

# General tip: Biases usually initialized to zero
# Exception: LSTM forget gate bias can be initialized to 1
```

**Key Insights:**
```
1. **Never initialize all weights to same value** (breaks symmetry)

2. **Match initialization to activation:**
   - ReLU family → He initialization
   - tanh/sigmoid → Xavier initialization

3. **Consider network depth:**
   - Very deep networks (>50 layers): Use Fixup or normalization layers
   - Moderate depth (10-30): He/Xavier sufficient

4. **Empirical tuning:**
   - Monitor activation/gradient statistics during training
   - Activation std should stay ≈1 across layers
   - Gradient norm should not explode or vanish

5. **Modern best practice:**
   - Use He/Xavier + Batch/Layer Normalization
   - This combination is very robust
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
