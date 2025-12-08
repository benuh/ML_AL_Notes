# 🧠 Deep Learning Architectures - Complete Reference Guide

**From Fundamentals to State-of-the-Art Architectures**

> Last Updated: November 2025
> Comprehensive guide to deep learning architectures with mathematical foundations and implementations

---

## 📋 Table of Contents

1. [Neural Network Fundamentals](#neural-network-fundamentals)
2. [Convolutional Neural Networks (CNNs)](#convolutional-neural-networks)
3. [Recurrent Neural Networks (RNNs)](#recurrent-neural-networks)
4. [Attention and Transformers](#attention-and-transformers)
5. [Modern CNN Architectures](#modern-cnn-architectures)
6. [Generative Models](#generative-models)
7. [Graph Neural Networks](#graph-neural-networks)
8. [Optimization Techniques](#optimization-techniques)
9. [Regularization Methods](#regularization-methods)
10. [Advanced Training Techniques](#advanced-training-techniques)

---

## 🔢 Neural Network Fundamentals

### Multi-Layer Perceptron (MLP)

**Architecture:**
```
Input Layer → Hidden Layer(s) → Output Layer
```

**Forward Propagation (Layer-wise):**
```
For layer l ∈ {1, 2, ..., L}:

z^[l] = W^[l] a^[l-1] + b^[l]
a^[l] = g^[l](z^[l])

where a^[0] = x (input)
```

**Notation:**
- `z^[l] ∈ ℝ^(n_l)`: Pre-activation (linear combination) at layer l
- `W^[l] ∈ ℝ^(n_l × n_(l-1))`: Weight matrix for layer l
- `b^[l] ∈ ℝ^(n_l)`: Bias vector for layer l
- `a^[l] ∈ ℝ^(n_l)`: Post-activation (nonlinear transformation) at layer l
- `g^[l]: ℝ^(n_l) → ℝ^(n_l)`: Element-wise activation function
- `n_l`: Number of neurons in layer l

**Universal Approximation Theorem:**
A feedforward network with:
- Single hidden layer with finite neurons
- Non-polynomial activation function (e.g., sigmoid, ReLU)
can approximate any continuous function on compact subsets of ℝⁿ to arbitrary precision.

**Formal statement (Cybenko, 1989):**
For any continuous f: [0,1]ⁿ → ℝ and ε > 0, ∃ network with single hidden layer:
||F(x) - f(x)|| < ε for all x ∈ [0,1]ⁿ

where F(x) = Σ_{i=1}^N α_i σ(w_i^T x + b_i) for some N, α_i, w_i, b_i

**Backpropagation (Reverse-Mode Automatic Differentiation):**

**Algorithm:**
1. **Compute output layer error:**
   ```
   δ^[L] = ∇_{a^[L]} L ⊙ g'^[L](z^[L])
   ```
   For cross-entropy + softmax: δ^[L] = a^[L] - y (simplified!)

2. **Propagate error backwards (chain rule):**
   ```
   For l = L-1, L-2, ..., 1:
   δ^[l] = ((W^[l+1])^T δ^[l+1]) ⊙ g'^[l](z^[l])
   ```

3. **Compute parameter gradients:**
   ```
   ∂L/∂W^[l] = δ^[l] (a^[l-1])^T / m
   ∂L/∂b^[l] = (1/m) Σ_i δ_i^[l]
   ```
   where m is batch size

**Mathematical foundation (Chain rule):**
```
∂L/∂W^[l] = (∂L/∂z^[l]) × (∂z^[l]/∂W^[l])
          = δ^[l] × (a^[l-1])^T

where δ^[l] = ∂L/∂z^[l] is computed recursively:
δ^[l] = (∂L/∂z^[l+1]) × (∂z^[l+1]/∂a^[l]) × (∂a^[l]/∂z^[l])
      = δ^[l+1] × W^[l+1] × g'^[l](z^[l])
```

**Computational complexity:**
- Forward pass: O(Σ_l n_l × n_(l-1))
- Backward pass: O(Σ_l n_l × n_(l-1)) (same as forward!)
- Memory: O(Σ_l n_l) to store activations

**Key insight:** Backpropagation computes all gradients in one backward pass,
avoiding naive application of chain rule which would be O(L × Σ_l n_l × n_(l-1)).

### Implementation from Scratch

```python
import numpy as np

class MLP:
    """Multi-Layer Perceptron from scratch."""

    def __init__(self, layer_sizes, activation='relu'):
        """
        Args:
            layer_sizes: List of layer sizes [input_dim, hidden1, hidden2, ..., output_dim]
            activation: 'relu', 'sigmoid', or 'tanh'
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.activation = activation

        # Initialize weights and biases
        self.weights = []
        self.biases = []

        for i in range(self.num_layers):
            # He initialization for ReLU, Xavier for others
            if activation == 'relu':
                scale = np.sqrt(2.0 / layer_sizes[i])
            else:
                scale = np.sqrt(1.0 / layer_sizes[i])

            W = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * scale
            b = np.zeros((layer_sizes[i+1], 1))

            self.weights.append(W)
            self.biases.append(b)

    def activate(self, Z, derivative=False):
        """Activation function."""
        if self.activation == 'relu':
            if derivative:
                return (Z > 0).astype(float)
            return np.maximum(0, Z)

        elif self.activation == 'sigmoid':
            sig = 1 / (1 + np.exp(-Z))
            if derivative:
                return sig * (1 - sig)
            return sig

        elif self.activation == 'tanh':
            if derivative:
                return 1 - np.tanh(Z)**2
            return np.tanh(Z)

    def forward(self, X):
        """Forward propagation."""
        self.cache = {'A0': X}
        A = X

        for i in range(self.num_layers):
            Z = self.weights[i] @ A + self.biases[i]

            # Use activation for hidden layers, softmax for output
            if i < self.num_layers - 1:
                A = self.activate(Z)
            else:
                # Softmax for output layer
                A = self.softmax(Z)

            self.cache[f'Z{i+1}'] = Z
            self.cache[f'A{i+1}'] = A

        return A

    def softmax(self, Z):
        """Stable softmax."""
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    def backward(self, Y):
        """Backpropagation."""
        m = Y.shape[1]  # Number of samples

        # Output layer gradient (cross-entropy + softmax)
        dZ = self.cache[f'A{self.num_layers}'] - Y

        self.gradients = {}

        # Backpropagate through layers
        for i in reversed(range(self.num_layers)):
            # Gradients
            self.gradients[f'dW{i}'] = (1/m) * (dZ @ self.cache[f'A{i}'].T)
            self.gradients[f'db{i}'] = (1/m) * np.sum(dZ, axis=1, keepdims=True)

            if i > 0:
                # Propagate to previous layer
                dA = self.weights[i].T @ dZ
                dZ = dA * self.activate(self.cache[f'Z{i}'], derivative=True)

    def update_parameters(self, learning_rate):
        """Update weights and biases."""
        for i in range(self.num_layers):
            self.weights[i] -= learning_rate * self.gradients[f'dW{i}']
            self.biases[i] -= learning_rate * self.gradients[f'db{i}']

    def compute_loss(self, Y):
        """Cross-entropy loss."""
        m = Y.shape[1]
        A_final = self.cache[f'A{self.num_layers}']

        # Clip to prevent log(0)
        A_final = np.clip(A_final, 1e-10, 1 - 1e-10)

        loss = -np.sum(Y * np.log(A_final)) / m
        return loss

    def train(self, X, Y, epochs=1000, learning_rate=0.01, verbose=True):
        """Train the network."""
        losses = []

        for epoch in range(epochs):
            # Forward pass
            _ = self.forward(X)

            # Compute loss
            loss = self.compute_loss(Y)
            losses.append(loss)

            # Backward pass
            self.backward(Y)

            # Update parameters
            self.update_parameters(learning_rate)

            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

        return losses

    def predict(self, X):
        """Make predictions."""
        A = self.forward(X)
        return np.argmax(A, axis=0)

# Example usage
if __name__ == "__main__":
    # Generate dummy data
    np.random.seed(42)
    X = np.random.randn(10, 100)  # 10 features, 100 samples
    Y = np.zeros((3, 100))  # 3 classes
    Y[np.random.randint(0, 3, 100), np.arange(100)] = 1  # One-hot

    # Create and train network
    mlp = MLP([10, 20, 10, 3], activation='relu')
    losses = mlp.train(X, Y, epochs=500, learning_rate=0.1)

    # Predict
    predictions = mlp.predict(X)
    accuracy = np.mean(predictions == np.argmax(Y, axis=0))
    print(f"\nTraining Accuracy: {accuracy:.2%}")
```

### Theoretical Foundations of Neural Networks

Before discussing activation functions, we establish the rigorous mathematical theory underlying neural networks.

#### Weight Initialization Theory

Proper initialization is critical for training deep networks. Random initialization that is too large or too small leads to gradient explosion or vanishing.

**Problem Setup:**
Consider a layer with n_in inputs and n_out outputs:
z = Wx + b, where W ∈ ℝ^(n_out × n_in)

Assume:
- Inputs x_i are i.i.d. with E[x_i] = 0, Var[x_i] = σ²_x
- Weights W_ij are i.i.d. with E[W_ij] = 0, Var[W_ij] = σ²_W
- Inputs and weights are independent

**Theorem 1 (Variance Propagation):**
The variance of pre-activation z_i is:

Var[z_i] = n_in · σ²_W · σ²_x

Proof:
z_i = Σ_{j=1}^{n_in} W_ij x_j

Var[z_i] = Var[Σ_j W_ij x_j]
         = Σ_j Var[W_ij x_j]    (independence)
         = Σ_j E[W²_ij]E[x²_j]  (independence, zero mean)
         = Σ_j σ²_W · σ²_x
         = n_in · σ²_W · σ²_x ∎

**Corollary 1.1 (Variance Preservation):**
To maintain Var[z_i] = Var[x_j], we need:

n_in · σ²_W = 1  ⟹  σ²_W = 1/n_in

This is **Xavier/Glorot initialization** for linear activations.

**Xavier/Glorot Initialization (2010):**

For forward pass, want: Var[z^[l]] = Var[z^[l-1]]
For backward pass, want: Var[∂L/∂z^[l]] = Var[∂L/∂z^[l+1]]

Forward: requires σ²_W = 1/n_in
Backward: requires σ²_W = 1/n_out

**Compromise:** Use harmonic mean:

σ²_W = 2/(n_in + n_out)

Implementation:
```python
W ~ Uniform(-√(6/(n_in + n_out)), √(6/(n_in + n_out)))
# Or equivalently:
W ~ Normal(0, √(2/(n_in + n_out)))
```

**Theorem 2 (He Initialization for ReLU):**
For ReLU activation σ(z) = max(0, z), approximately half of neurons are inactive (output 0).
This halves the variance of activations.

To compensate, use:
σ²_W = 2/n_in

This maintains variance through ReLU layers.

Proof sketch:
For ReLU(z) where z ~ N(0, σ²):
- P(ReLU(z) = 0) = 0.5
- E[ReLU(z)] = σ/√(2π)
- Var[ReLU(z)] = σ²(1 - 1/π)/2 ≈ 0.5σ²

To maintain Var[a^[l]] = Var[a^[l-1]], need:
n_in · σ²_W · 0.5 = 1  ⟹  σ²_W = 2/n_in ∎

**He Initialization (2015):**
```python
W ~ Normal(0, √(2/n_in))
```

**Theorem 3 (Orthogonal Initialization):**
For recurrent networks, initializing weight matrices to be orthogonal preserves gradient norms during backpropagation.

An orthogonal matrix Q satisfies Q^T Q = I, thus ||Qv|| = ||v|| (preserves norms).

For RNN: h_t = tanh(W_h h_{t-1} + W_x x_t)

If W_h is orthogonal and |tanh'(z)| ≤ 1:
||∂L/∂h_{t-1}|| ≈ ||∂L/∂h_t||

This mitigates vanishing/exploding gradients in RNNs.

Implementation via QR decomposition:
```python
W = np.random.randn(n, n)
Q, R = np.linalg.qr(W)
W_init = Q  # Orthogonal matrix
```

**Summary of Initialization Methods:**
```
Method             | Variance      | Use Case
-------------------|---------------|-------------------
Xavier/Glorot      | 2/(n_in+n_out)| Linear, tanh, sigmoid
He                 | 2/n_in        | ReLU, Leaky ReLU, ELU
Orthogonal         | 1             | RNNs, LSTMs, GRUs
LeCun              | 1/n_in        | SELU
```

#### Universal Approximation Theorem (Rigorous Treatment)

**Theorem 4 (Universal Approximation - Cybenko 1989, Hornik 1991):**
Let φ: ℝ → ℝ be a non-constant, bounded, continuous function (e.g., sigmoid).
Let I_n = [0,1]^n be the n-dimensional unit hypercube.
Let C(I_n) be the space of continuous functions on I_n.

Then for any f ∈ C(I_n) and ε > 0, there exists a network with one hidden layer:

F(x) = Σ_{i=1}^N α_i φ(w_i^T x + b_i)

such that:
||F - f||_∞ = sup_{x∈I_n} |F(x) - f(x)| < ε

**Proof outline:**
1. Show that functions of form φ(w^T x + b) are **discriminatory**:
   If Σ_i α_i φ(w_i^T x + b_i) = 0 for all x, then all α_i = 0

2. By Riesz representation theorem, discriminatory functions are dense in C(I_n)

3. Therefore, finite linear combinations can approximate any continuous function

Full proof uses measure theory and functional analysis. ∎

**Corollary 4.1 (ReLU Networks):**
The theorem extends to ReLU networks despite ReLU being unbounded.
A 2-layer ReLU network can approximate any continuous function on compact domains.

**Theorem 5 (Approximation Rate - Barron 1993):**
For functions f: ℝ^d → ℝ with bounded Fourier magnitude C_f:

min_{N neurons} ||F_N - f||²_L2 ≤ C_f² / N

This shows approximation error decreases as O(1/N) with network width N,
independent of input dimension d!

**Key insight:** Neural networks overcome the curse of dimensionality for
functions with certain smoothness properties (bounded Fourier spectrum).

#### Expressivity: Depth vs Width

**Theorem 6 (Expressive Power of Depth):**
Some functions require exponentially many neurons in a shallow network
but can be represented with polynomially many neurons in a deep network.

**Example:** Parity function f(x₁,...,x_d) = x₁ ⊕ x₂ ⊕ ... ⊕ x_d
- Shallow network: requires Ω(2^d) neurons
- Deep network (d layers): requires O(d) neurons

**Theorem 7 (Width vs Depth Trade-off - Telgarsky 2016):**
For any depth k, there exists a function f_k that:
- Can be computed by a network with depth k and width O(k³)
- Requires width 2^Ω(k) if depth is k-1

This establishes an exponential separation between depth k and k-1.

**Practical implications:**
- Depth provides exponentially more expressivity than width
- Deep networks can represent hierarchical compositional structure efficiently
- Shallow networks require exponentially more neurons for same expressivity

#### Gradient Flow and Vanishing/Exploding Gradients

**Gradient Propagation Analysis:**

For a network with L layers, the gradient w.r.t. first layer weights is:

∂L/∂W^[1] = ∂L/∂z^[L] · ∂z^[L]/∂z^[L-1] · ... · ∂z^[2]/∂z^[1] · ∂z^[1]/∂W^[1]

Each term ∂z^[l]/∂z^[l-1] = W^[l] · diag(g'(z^[l-1])) where g is activation.

**Theorem 8 (Gradient Norm Bound):**
The gradient norm is bounded by:

||∂L/∂W^[1]|| ≤ ||∂L/∂z^[L]|| · Π_{l=2}^L ||W^[l]|| · ||diag(g'(z^[l-1]))||

For sigmoid activation: g'(z) = σ(z)(1-σ(z)) ≤ 0.25
For tanh: g'(z) = 1 - tanh²(z) ≤ 1

**Vanishing Gradient Criterion:**
If ||W^[l]|| · ||g'(z^[l])|| < 1 for all layers, then:

||∂L/∂W^[1]|| ≤ ||∂L/∂z^[L]|| · (λ)^{L-1}

where λ < 1, causing exponential decay as L grows.

**Example (sigmoid):**
With ||W^[l]|| = 1 and ||g'(z)|| = 0.25:
After 10 layers: gradient scaled by (0.25)^9 ≈ 4 × 10^{-6}
After 20 layers: gradient scaled by (0.25)^{19} ≈ 10^{-12}

**ReLU Solution:**
For ReLU, g'(z) ∈ {0, 1}, so no gradient saturation (when z > 0).
With proper initialization (He), ||W^[l]|| ≈ 1 and gradient flow is stable.

**Theorem 9 (ResNet Gradient Flow):**
For ResNet with skip connections h_{l+1} = h_l + F(h_l):

∂L/∂h_l = ∂L/∂h_{l+1} · (I + ∂F/∂h_l)

By chain rule:
∂L/∂h_1 = ∂L/∂h_L · Π_{l=1}^{L-1} (I + ∂F/∂h_l)

The identity term I ensures at least some gradient flows directly,
preventing complete vanishing even if ∂F/∂h_l is small.

**Mathematical insight:**
Π(I + A_l) ≥ I  (gradient cannot vanish completely)

vs. standard network: Π A_l can approach 0

This explains why ResNets enable training 100+ layer networks.

#### Neural Tangent Kernel (NTK) - Modern Theory

For overparameterized networks (width → ∞), neural network training can be analyzed via kernel methods.

**Definition (Neural Tangent Kernel):**
For network f(x; θ) with parameters θ, the NTK is:

K(x, x') = ⟨∇_θ f(x; θ), ∇_θ f(x'; θ)⟩

**Theorem 10 (NTK for Infinite Width - Jacot et al. 2018):**
As width m → ∞ with proper initialization, the NTK:
1. Converges to a deterministic limit K*
2. Remains constant during training (lazy training regime)
3. Training dynamics become linear in function space

**Gradient descent dynamics:**
df/dt = -∇_θ f · ∂L/∂f
      → -K · ∂L/∂f  (as width → ∞)

This is equivalent to kernel ridge regression with kernel K!

**Corollary 10.1 (Global Convergence):**
In the NTK regime, gradient descent on overparameterized networks
converges to global minimum at rate:

||f_t - f*||² ≤ e^{-λ_min(K)·t} ||f_0 - f*||²

where λ_min(K) is the minimum eigenvalue of the NTK.

**Practical implications:**
- Explains why overparameterized networks don't overfit
- Provides theoretical justification for wide networks
- Shows connection between neural networks and kernel methods
- Lazy training regime: features barely change during training

**Limitation:** Real networks (finite width) often operate in "feature learning regime"
where features evolve significantly, going beyond NTK predictions.

### Activation Functions Deep Dive

#### **1. ReLU (Rectified Linear Unit)**

**Formula:**
```
f(x) = max(0, x) = {x  if x > 0
                    {0  if x ≤ 0

f'(x) = {1  if x > 0
        {0  if x ≤ 0
        {undefined at x = 0 (use subgradient ∈ [0,1])
```

**Mathematical properties:**
- Non-saturating for x > 0 (f'(x) = 1)
- Piecewise linear (locally linear everywhere except origin)
- Non-differentiable at x = 0 (use subgradient in practice)
- Convex function
- Range: [0, ∞)

**Advantages:**
- O(1) computational complexity (simple threshold)
- Mitigates vanishing gradient (constant gradient for x > 0)
- Induces sparsity: ~50% of neurons typically inactive
- Biological plausibility: similar to neural firing rates

**Disadvantages:**
- **Dying ReLU problem:** If z^[l] ≤ 0 for all samples, then:
  - Gradient ∂L/∂W^[l] = 0 forever
  - Neuron becomes permanently inactive
  - Can affect 10-40% of neurons in practice
  - Caused by: large learning rates, poor initialization, or data shifts
- Not zero-centered: outputs always ≥ 0
  - Gradient updates always same sign for weights in same layer
  - Can slow down optimization (zig-zagging)
- Unbounded output: can lead to exploding activations

**When to use:**
- Default choice for hidden layers in most architectures
- Works well with He initialization: W ~ N(0, 2/n_in)
- Combine with batch normalization to reduce dying ReLU

#### **2. Leaky ReLU**

**Formula:**
```
f(x) = max(αx, x) where α = 0.01 typically
f'(x) = 1 if x > 0 else α
```

**Advantage:** Solves dying ReLU problem

#### **3. ELU (Exponential Linear Unit)**

**Formula:**
```
f(x) = x if x > 0 else α(e^x - 1)
f'(x) = 1 if x > 0 else f(x) + α
```

**Advantages:**
- Smooth everywhere
- Negative values push mean activation closer to zero
- Better gradient flow

**Disadvantage:** More expensive to compute

#### **4. GELU (Gaussian Error Linear Unit)**

**Exact formula:**
```
f(x) = x · Φ(x) = x · P(X ≤ x) where X ~ N(0,1)

where Φ(x) = (1/√(2π)) ∫_{-∞}^x e^(-t²/2) dt
```

**Fast approximation (tanh-based):**
```
f(x) ≈ 0.5x(1 + tanh[√(2/π)(x + 0.044715x³)])
```

**Faster approximation (sigmoid-based):**
```
f(x) ≈ x · σ(1.702x) where σ(z) = 1/(1 + e^(-z))
```

**Derivative:**
```
f'(x) = Φ(x) + x · φ(x)

where φ(x) = (1/√(2π)) e^(-x²/2) (standard normal PDF)
```

**Properties:**
- Smooth, differentiable everywhere (unlike ReLU)
- Non-monotonic: slight negative values allowed for x < 0
- Stochastic interpretation: Expected transformation of x scaled by dropout
  - GELU(x) ≈ E[x · 1_{X > -x}] where X ~ N(0,1)
- Asymptotic behavior:
  - x → +∞: f(x) → x (linear)
  - x → -∞: f(x) → 0 (ReLU-like)

**Used in:**
- BERT, GPT-2/3, modern transformers
- Shows improved performance over ReLU in language models
- Particularly effective in deep networks (20+ layers)

#### **5. Swish / SiLU**

**Formula:**
```
f(x) = x · σ(x) = x / (1 + e^(-x))
```

**Advantages:**
- Smooth, non-monotonic
- Self-gated
- Performs well in deep networks

**Used in:** EfficientNet, modern architectures

### Implementation of All Activations

```python
import numpy as np
import matplotlib.pyplot as plt

class Activations:
    """Collection of activation functions."""

    @staticmethod
    def relu(x, derivative=False):
        if derivative:
            return (x > 0).astype(float)
        return np.maximum(0, x)

    @staticmethod
    def leaky_relu(x, alpha=0.01, derivative=False):
        if derivative:
            return np.where(x > 0, 1, alpha)
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def elu(x, alpha=1.0, derivative=False):
        if derivative:
            return np.where(x > 0, 1, alpha * np.exp(x))
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    @staticmethod
    def gelu(x, derivative=False):
        """GELU approximation."""
        cdf = 0.5 * (1.0 + np.tanh(
            np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)
        ))
        if derivative:
            # Numerical derivative
            h = 1e-5
            return (Activations.gelu(x + h) - Activations.gelu(x - h)) / (2 * h)
        return x * cdf

    @staticmethod
    def swish(x, derivative=False):
        """Swish / SiLU."""
        sigmoid = 1 / (1 + np.exp(-x))
        if derivative:
            return sigmoid + x * sigmoid * (1 - sigmoid)
        return x * sigmoid

    @staticmethod
    def sigmoid(x, derivative=False):
        sig = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        if derivative:
            return sig * (1 - sig)
        return sig

    @staticmethod
    def tanh(x, derivative=False):
        if derivative:
            return 1 - np.tanh(x)**2
        return np.tanh(x)

# Visualization
def plot_activations():
    """Plot all activation functions."""
    x = np.linspace(-5, 5, 1000)

    activations = {
        'ReLU': Activations.relu,
        'Leaky ReLU': Activations.leaky_relu,
        'ELU': Activations.elu,
        'GELU': Activations.gelu,
        'Swish': Activations.swish,
        'Sigmoid': Activations.sigmoid,
        'Tanh': Activations.tanh
    }

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, (name, func) in enumerate(activations.items()):
        ax = axes[idx]
        y = func(x)
        dy = func(x, derivative=True)

        ax.plot(x, y, label=f'{name}', linewidth=2)
        ax.plot(x, dy, '--', label=f"{name}'", alpha=0.7)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.set_title(name)
        ax.legend()
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')

    axes[-1].axis('off')  # Hide last subplot

    plt.tight_layout()
    plt.savefig('activation_functions.png', dpi=150, bbox_inches='tight')
    plt.show()

# plot_activations()
```

---

## 🖼️ Convolutional Neural Networks (CNNs)

### Convolution Operation

**2D Convolution:**
```
(I * K)[i, j] = Σ_m Σ_n I[i+m, j+n] · K[m, n]
```

**Parameters:**
- **Input:** H × W × C_in
- **Kernel:** K × K × C_in × C_out
- **Output:** H' × W' × C_out

**Output dimensions:**
```
H' = (H + 2P - K) / S + 1
W' = (W + 2P - K) / S + 1
```

Where:
- P: Padding
- S: Stride
- K: Kernel size

### CNN Components

#### **1. Convolutional Layer**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    """Convolutional layer with batch norm and activation."""

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, use_bn=True, activation='relu'):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_bn  # No bias if using batch norm
        )

        self.bn = nn.BatchNorm2d(out_channels) if use_bn else None

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
```

#### **2. Pooling Layers**

**Max Pooling:**
- Takes maximum value in each region
- Provides translation invariance
- Reduces spatial dimensions

**Average Pooling:**
- Takes average value
- Smoother than max pooling
- Used in global average pooling

```python
# Max pooling
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

# Average pooling
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

# Global average pooling
global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
```

#### **3. Complete CNN Architecture**

```python
class SimpleCNN(nn.Module):
    """Simple CNN for image classification."""

    def __init__(self, num_classes=10):
        super().__init__()

        # Convolutional layers
        self.conv1 = ConvLayer(3, 32, kernel_size=3, padding=1)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, padding=1)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, padding=1)
        self.conv4 = ConvLayer(128, 256, kernel_size=3, padding=1)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.pool(x)

        # Block 2
        x = self.conv2(x)
        x = self.pool(x)

        # Block 3
        x = self.conv3(x)
        x = self.pool(x)

        # Block 4
        x = self.conv4(x)

        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Classifier
        x = self.fc(x)

        return x

# Example
model = SimpleCNN(num_classes=10)
x = torch.randn(4, 3, 224, 224)  # Batch of 4 RGB images
output = model(x)
print(f"Output shape: {output.shape}")  # (4, 10)
```

### Receptive Field Calculation

**Receptive field:** Region in input that affects a particular feature

**Formula:**
```
r_out = r_in + (k - 1) * ∏(previous strides)
```

**Example calculation:**
```python
def calculate_receptive_field(layers):
    """
    Calculate receptive field.

    layers: List of (kernel_size, stride) tuples
    """
    receptive_field = 1
    stride_product = 1

    for k, s in layers:
        receptive_field += (k - 1) * stride_product
        stride_product *= s

    return receptive_field

# Example: VGG-style network
layers = [
    (3, 1),  # Conv1
    (3, 1),  # Conv2
    (2, 2),  # Pool
    (3, 1),  # Conv3
    (3, 1),  # Conv4
    (2, 2),  # Pool
]

rf = calculate_receptive_field(layers)
print(f"Receptive field: {rf}x{rf}")  # 14x14
```

---

## 🏗️ Modern CNN Architectures

### 1. VGG (Visual Geometry Group)

**Key Ideas:**
- Deep networks with small (3×3) filters
- Stacking small filters instead of large ones
- Uniform architecture (easy to understand)

**Architecture:**
```
VGG-16:
Block 1: Conv3-64, Conv3-64, MaxPool
Block 2: Conv3-128, Conv3-128, MaxPool
Block 3: Conv3-256, Conv3-256, Conv3-256, MaxPool
Block 4: Conv3-512, Conv3-512, Conv3-512, MaxPool
Block 5: Conv3-512, Conv3-512, Conv3-512, MaxPool
FC: 4096, 4096, 1000
```

**Advantages:**
- Simple, uniform architecture
- Strong performance
- Good for feature extraction

**Disadvantages:**
- 138M parameters (very large)
- Slow to train
- Memory intensive

```python
class VGG16(nn.Module):
    """VGG-16 architecture."""

    def __init__(self, num_classes=1000):
        super().__init__()

        # Feature extractor
        self.features = nn.Sequential(
            # Block 1
            ConvLayer(3, 64, 3, 1, 1),
            ConvLayer(64, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            # Block 2
            ConvLayer(64, 128, 3, 1, 1),
            ConvLayer(128, 128, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            # Block 3
            ConvLayer(128, 256, 3, 1, 1),
            ConvLayer(256, 256, 3, 1, 1),
            ConvLayer(256, 256, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            # Block 4
            ConvLayer(256, 512, 3, 1, 1),
            ConvLayer(512, 512, 3, 1, 1),
            ConvLayer(512, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            # Block 5
            ConvLayer(512, 512, 3, 1, 1),
            ConvLayer(512, 512, 3, 1, 1),
            ConvLayer(512, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
```

### 2. ResNet (Residual Networks)

**Key Innovation:** Skip connections (residual connections)

**Residual Block:**
```
y = F(x, {Wi}) + x
```

Where F is the residual function to be learned.

#### Rigorous Theory of Residual Networks

**Theorem 1 (Gradient Flow in ResNets - He et al., 2016):**

For a ResNet with L layers using residual blocks y^(l) = F(x^(l), W^(l)) + x^(l):

The gradient w.r.t. input of layer l is:

∂L/∂x^(l) = ∂L/∂x^(L) · ∏_{i=l}^{L-1} (I + ∂F^(i)/∂x^(i))

**Key Property:** Can be rewritten as:

∂L/∂x^(l) = ∂L/∂x^(L) · (I + ∑_{i=l}^{L-1} ∂F^(i)/∂x^(i) + higher order terms)

**Interpretation:**
- **Direct path:** ∂L/∂x^(L) flows directly to any layer (identity term I)
- **Residual path:** Additional signal from ∂F^(i)/∂x^(i) terms
- **Never vanishes:** Gradient always has magnitude ≥ ||∂L/∂x^(L)||

**Proof:**
Forward pass for L layers:
x^(l+1) = F(x^(l), W^(l)) + x^(l)

Backward pass by chain rule:
∂L/∂x^(l) = ∂L/∂x^(l+1) · ∂x^(l+1)/∂x^(l)
          = ∂L/∂x^(l+1) · (∂F/∂x^(l) + I)

Applying recursively from l to L:
∂L/∂x^(l) = ∂L/∂x^(L) · ∏_{i=l}^{L-1} (I + ∂F^(i)/∂x^(i))

For product of (I + A_i) terms:
∏(I + A_i) = I + ∑A_i + ∑_{i<j} A_iA_j + ... (by multinomial expansion) ∎

**Comparison with plain networks:**
- **Plain:** ∂L/∂x^(l) = ∂L/∂x^(L) · ∏ ∂F^(i)/∂x^(i)
  - If ||∂F^(i)/∂x^(i)|| < 1: exponential decay → vanishing gradients
  - If ||∂F^(i)/∂x^(i)|| > 1: exponential growth → exploding gradients

- **ResNet:** ∂L/∂x^(l) always contains ∂L/∂x^(L) term (no multiplicative decay!)

**Theorem 2 (Identity Mapping Property):**

For residual block with ReLU: h^(l+1) = ReLU(F(h^(l)) + h^(l))

When F = 0 (zero initialization or early training):
h^(l+1) = ReLU(h^(l)) = h^(l) (if h^(l) ≥ 0)

**Forward and backward propagation simplify to identity:**
- Forward: h^(L) = h^(0) (skip all layers!)
- Backward: ∂L/∂h^(0) = ∂L/∂h^(L) (perfect gradient flow)

**Initialization insight:** Start near identity mapping → easier optimization landscape.

**Theorem 3 (Optimization Landscape - Li et al., 2018):**

ResNets have **smoother loss landscapes** than plain networks:

For ResNet loss L_res(W) and plain network loss L_plain(W):

max_{||Δ||=ε} |L_res(W + Δ) - L_res(W)| ≤ O(ε)
max_{||Δ||=ε} |L_plain(W + Δ) - L_plain(W)| ≤ O(L·ε)

where L is network depth.

**Proof sketch:**
Loss surface smoothness measured by Lipschitz constant of gradient:

||∇L(W_1) - ∇L(W_2)|| ≤ β||W_1 - W_2||

For ResNet: β_res = O(1) (independent of depth!)
For plain: β_plain = O(L) (linear in depth)

Skip connections prevent gradient compounding across layers. ∎

**Empirical consequence:**
- ResNets: Can use larger learning rates (10× higher)
- ResNets: More stable training dynamics
- ResNets: Better convergence even with random initialization

**Theorem 4 (Expressiveness of ResNets - Veit et al., 2016):**

An L-layer ResNet is an **ensemble of 2^L paths** of varying lengths:

Total paths from input to output: 2^L

Path length distribution:
- Paths of length k: (L choose k)
- Expected path length: L/2
- Most paths are short (ensemble of shallow networks!)

**Proof:**
Each residual block offers 2 choices:
1. Use residual function F (length +1)
2. Skip via identity (length +0)

With L blocks: 2^L possible combinations. ∎

**Interpretation:**
- ResNets don't behave like very deep networks
- More like ensemble of O(L) networks with depths [0, 1, ..., L]
- Explains why deeper ResNets train easier than expected

**Theorem 5 (Capacity and Depth - Ba & Caruana, 2014):**

For function approximation with ε error:

**Shallow network:** Requires width w = O(ε^(-d)) (exponential in dimension d)
**Deep network:** Requires depth L = O(log(1/ε)) and width w = O(d)

**ResNet advantage:** Can increase depth L with:
- Negligible gradient degradation (Theorem 1)
- Improved optimization (Theorem 3)
- Exponentially larger function class (Theorem 4)

**Effective depth scaling:**
Training cost: O(L·w²) per iteration
ResNet allows L → ∞ while maintaining stable gradients!

**Theorem 6 (Skip Connection Variants):**

Different skip connection patterns have different properties:

**(a) Post-activation:** y = F(ReLU(x)) + x
- Gradient: ∂L/∂x = ∂L/∂y · (∂F/∂(ReLU(x))·𝟙[x>0] + I)
- Problem: Identity path can be blocked by ReLU

**(b) Pre-activation (ResNet v2):** y = F(x) + x, then ReLU(y)
- Gradient: ∂L/∂x = ∂L/∂ReLU(y) · 𝟙[y>0] · (∂F/∂x + I)
- Better: Identity path always active before ReLU
- Empirical improvement: 0.5-1% accuracy gain

**(c) Dense connections (DenseNet):** y = [x, F1(x), F2(x), ...]
- Gradient: Direct paths from all previous layers
- Parameter efficiency: Better feature reuse
- Cost: O(L²) feature maps

**Optimal design (Identity Mappings paper):**
```
x -> BN -> ReLU -> Conv -> BN -> ReLU -> Conv --(+)--> out
 |                                              |
 +----------------------------------------------+
```
This achieves cleanest gradient flow.

**Why it works (Summary):**
1. **Gradient flow:** Direct path for gradients (no vanishing)
2. **Identity mapping:** Easy optimization starting point
3. **Smooth landscape:** Loss surface has smaller curvature
4. **Ensemble effect:** Acts like exponentially many shallow networks
5. **Capacity:** Enables deep networks with stable training

**Architecture variants:**
- ResNet-18, 34: Basic blocks
- ResNet-50, 101, 152: Bottleneck blocks

```python
class BasicBlock(nn.Module):
    """ResNet basic block (used in ResNet-18, 34)."""

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()

        self.conv1 = ConvLayer(in_channels, out_channels, 3, stride, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # Residual connection
        out = self.relu(out)

        return out

class BottleneckBlock(nn.Module):
    """ResNet bottleneck block (used in ResNet-50, 101, 152)."""

    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()

        # Bottleneck: 1x1 -> 3x3 -> 1x1
        self.conv1 = ConvLayer(in_channels, out_channels, 1, 1, 0)
        self.conv2 = ConvLayer(out_channels, out_channels, 3, stride, 1)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    """ResNet architecture."""

    def __init__(self, block, layers, num_classes=1000):
        """
        Args:
            block: BasicBlock or BottleneckBlock
            layers: List of number of blocks per stage [3, 4, 6, 3] for ResNet-50
            num_classes: Number of output classes
        """
        super().__init__()

        self.in_channels = 64

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        downsample = None

        # Need downsampling if stride != 1 or channels change
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          1, stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# Create ResNet-50
def resnet50(num_classes=1000):
    return ResNet(BottleneckBlock, [3, 4, 6, 3], num_classes)

# Create ResNet-18
def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

# Example
model = resnet50()
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### 3. Inception / GoogLeNet

**Key Idea:** Multiple filter sizes in parallel

**Inception Module:**
```
Input
├── 1x1 conv
├── 1x1 conv → 3x3 conv
├── 1x1 conv → 5x5 conv
└── 3x3 pool → 1x1 conv
    ↓
Concatenate
```

**Advantages:**
- Multi-scale feature extraction
- Parameter efficient (1x1 convs for dimensionality reduction)
- Good performance with fewer parameters than VGG

```python
class InceptionModule(nn.Module):
    """Inception module (Inception v1)."""

    def __init__(self, in_channels, ch1x1, ch3x3_reduce, ch3x3,
                 ch5x5_reduce, ch5x5, pool_proj):
        super().__init__()

        # 1x1 branch
        self.branch1 = ConvLayer(in_channels, ch1x1, kernel_size=1)

        # 1x1 -> 3x3 branch
        self.branch2 = nn.Sequential(
            ConvLayer(in_channels, ch3x3_reduce, kernel_size=1),
            ConvLayer(ch3x3_reduce, ch3x3, kernel_size=3, padding=1)
        )

        # 1x1 -> 5x5 branch
        self.branch3 = nn.Sequential(
            ConvLayer(in_channels, ch5x5_reduce, kernel_size=1),
            ConvLayer(ch5x5_reduce, ch5x5, kernel_size=5, padding=2)
        )

        # Pool -> 1x1 branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvLayer(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        # Concatenate along channel dimension
        return torch.cat([branch1, branch2, branch3, branch4], 1)
```

### 4. EfficientNet

**Key Innovation:** Compound scaling (depth, width, resolution)

**Scaling formula:**
```
depth: d = α^φ
width: w = β^φ
resolution: r = γ^φ

subject to: α · β² · γ² ≈ 2
            α ≥ 1, β ≥ 1, γ ≥ 1
```

**MBConv Block (Mobile Inverted Bottleneck):**
```
Input
  ↓ 1x1 expand
  ↓ Depthwise 3x3 or 5x5
  ↓ SE (Squeeze-Excitation)
  ↓ 1x1 project
  ↓ Skip connection (if same size)
Output
```

```python
class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""

    def __init__(self, in_channels, reduction=4):
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.SiLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        # Squeeze
        y = self.squeeze(x).view(b, c)

        # Excitation
        y = self.excitation(y).view(b, c, 1, 1)

        # Scale
        return x * y.expand_as(x)

class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution."""

    def __init__(self, in_channels, out_channels, expand_ratio=6,
                 kernel_size=3, stride=1, se_ratio=0.25):
        super().__init__()

        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)

        hidden_dim = in_channels * expand_ratio

        layers = []

        # Expansion
        if expand_ratio != 1:
            layers.append(ConvLayer(in_channels, hidden_dim, kernel_size=1))

        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride,
                      padding=kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
        ])

        # SE
        if se_ratio > 0:
            layers.append(SEBlock(hidden_dim, reduction=int(1/se_ratio)))

        # Projection
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)
```

### Architecture Comparison

| Architecture | Year | Params | Top-1 Acc | Key Innovation |
|--------------|------|--------|-----------|----------------|
| AlexNet | 2012 | 61M | 63.3% | GPU training, ReLU, Dropout |
| VGG-16 | 2014 | 138M | 74.4% | Small filters, deep networks |
| ResNet-50 | 2015 | 25M | 77.1% | Skip connections |
| Inception-v3 | 2015 | 24M | 78.8% | Multi-scale, efficient |
| DenseNet-121 | 2017 | 8M | 75.0% | Dense connections |
| EfficientNet-B0 | 2019 | 5.3M | 77.1% | Compound scaling |
| EfficientNetV2-S | 2021 | 24M | 84.9% | Training efficiency |

---

## 📊 Optimization Techniques

### Gradient Descent Variants

#### **1. SGD with Momentum**

**Problem with vanilla SGD:** Oscillates in narrow valleys

**Solution:** Add momentum term

**Update rule:**
```
v_t = βv_{t-1} + ∇L(θ_{t-1})
θ_t = θ_{t-1} - αv_t
```

Where β is momentum coefficient (typically 0.9)

#### **2. RMSprop**

**Adaptive learning rate per parameter:**
```
s_t = βs_{t-1} + (1-β)(∇L)²
θ_t = θ_{t-1} - α · ∇L / (√s_t + ε)
```

#### **3. Adam (Adaptive Moment Estimation)**

**Combines momentum and RMSprop:**
```
m_t = β₁m_{t-1} + (1-β₁)∇L        # First moment (mean)
v_t = β₂v_{t-1} + (1-β₂)(∇L)²      # Second moment (variance)

m̂_t = m_t / (1 - β₁^t)              # Bias correction
v̂_t = v_t / (1 - β₂^t)

θ_t = θ_{t-1} - α · m̂_t / (√v̂_t + ε)
```

**Default hyperparameters:**
- β₁ = 0.9
- β₂ = 0.999
- ε = 1e-8
- α = 0.001

#### **4. AdamW (Adam with Weight Decay)**

**Fixes weight decay in Adam:**
```
θ_t = θ_{t-1} - α · (m̂_t / (√v̂_t + ε) + λθ_{t-1})
```

Where λ is weight decay coefficient.

#### **Optimizer Comparison**

```python
import torch.optim as optim

# SGD with momentum
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# RMSprop
optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99)

# Adam
optimizer = optim.Adam(model.parameters(), lr=0.001,
                       betas=(0.9, 0.999), eps=1e-8)

# AdamW (better weight decay)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

### Learning Rate Schedules

#### **1. Step Decay**
```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
# Multiply LR by 0.1 every 30 epochs
```

#### **2. Cosine Annealing**
```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
# Cosine decay over 100 epochs
```

#### **3. One Cycle Policy**
```python
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.1, steps_per_epoch=len(train_loader), epochs=10
)
# Used in FastAI, very effective
```

#### **4. Warmup + Cosine**
```python
from torch.optim.lr_scheduler import LambdaLR
import math

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)

# Usage
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000,
    num_training_steps=10000
)
```

---

## 🛡️ Regularization Methods

### 1. Dropout

**Idea:** Randomly drop neurons during training

**Effect:** Prevents co-adaptation, ensemble effect

```python
# Dropout layer
dropout = nn.Dropout(p=0.5)  # Drop 50% of neurons

# During training
x = dropout(x)  # Some neurons set to 0

# During inference
# Automatically scales activations (no dropout applied)
```

**Variants:**
- **Spatial Dropout:** Drop entire feature maps (for CNNs)
- **DropConnect:** Drop connections instead of neurons

```python
# Spatial Dropout
spatial_dropout = nn.Dropout2d(p=0.2)
```

### 2. Batch Normalization

**Problem:** Internal covariate shift

**Solution:** Normalize activations per mini-batch

**Formula:**
```
μ = (1/m) Σ x_i
σ² = (1/m) Σ (x_i - μ)²
x̂ = (x - μ) / √(σ² + ε)
y = γx̂ + β
```

Where γ and β are learnable parameters.

#### Rigorous Theory of Batch Normalization

**Theorem 7 (Smoothing Effect on Loss Landscape - Santurkar et al., 2018):**

Batch Normalization makes the optimization landscape significantly smoother:

For loss L(w) without BN and L_BN(w) with BN:

||∇²L_BN(w)|| ≤ ||∇²L(w)|| / O(√m)

where m is batch size.

**Proof sketch:**
BN introduces normalization: h̃ = γ(h - μ_B)/σ_B + β

Second derivative of BN:
∂²h̃/∂w² involves terms (∂μ_B/∂w)², (∂σ_B/∂w)² which are O(1/m)

This reduces Hessian eigenvalues → smoother landscape. ∎

**Empirical consequence:**
- **Learning rate tolerance:** Can use 10-100× larger learning rates
- **Lipschitz constant:** β-smoothness constant reduced by √m
- **Convergence speed:** Fewer iterations to reach same loss

**Theorem 8 (Gradient Predictiveness - Santurkar et al., 2018):**

Define gradient predictiveness as correlation between gradients at consecutive steps:

ρ_t = cos(∇L(w_t), ∇L(w_{t-1}))

**Without BN:** ρ_t → 0 as training proceeds (gradient directions become random)
**With BN:** ρ_t ≈ 0.9 maintained throughout training

**Quantitative bound:**
|∇L_BN(w + Δw) - ∇L_BN(w)| ≤ β_BN·||Δw||

where β_BN = O(L/√m) for L-layer network
and β_plain = O(L²) without BN

Reduction factor: √m·L (e.g., for m=32, L=50: 280× smoother!)

**Theorem 9 (Reparametrization Effect):**

BN effectively reparametrizes the network, decoupling scale and direction of weights:

For layer output h = BN(Wx):
- Weight scale ||W|| doesn't affect forward pass (normalized out!)
- Only weight direction matters: W/||W||

**Gradient decomposition:**
∇_W L = ∇_W L_BN = (I - ww^T/||w||²) · ∇_h L · x^T

The term (I - ww^T/||w||²) projects gradient perpendicular to w.

**Consequence:** Weight updates only change direction, not magnitude:
- Larger initial weights don't cause larger gradients
- Gradient norm bounded: ||∇_W L|| ≤ ||∇_h L|| · ||x|| (independent of ||W||!)

**Theorem 10 (Regularization Effect):**

BN acts as implicit regularizer with effective regularization strength:

λ_eff ≈ 1 / (2√m · σ_W)

where σ_W is standard deviation of weights.

**Proof:**
Batch statistics μ_B, σ_B introduce noise (different per batch).
This noise acts like dropout with rate ≈ 1/m.

Expected perturbation: E[(h_i - BN(h_i))²] ≈ σ²_h / m

This is equivalent to L2 penalty λ_eff||w||² on weights. ∎

**Empirical validation:**
- Larger batch → less regularization (λ_eff ∝ 1/√m)
- m = 32: Strong regularization
- m = 1024: Weak regularization
- May need explicit regularization for large batches

**Theorem 11 (Moving Average Statistics at Test Time):**

During training, BN uses batch statistics μ_B, σ_B.
At test time, uses moving averages μ̂, σ̂:

μ̂_t = α·μ̂_{t-1} + (1-α)·μ_B,t
σ̂²_t = α·σ̂²_{t-1} + (1-α)·σ²_B,t

Typical: α = 0.9 (exponential moving average)

**Convergence of moving average:**
After T batches with IID data:
E[||μ̂_T - E[μ_B]||²] ≤ σ²_μ / (T(1-α))

For α = 0.9: requires T ≈ 10/(1-α) = 100 batches for convergence.

**Train-test discrepancy bound:**
|BN_train(h) - BN_test(h)| ≤ γ·|σ_B - σ̂|/σ_B

For well-estimated σ̂ (after enough batches): discrepancy ≈ 1-2%

**Theorem 12 (BN Variants Comparison):**

Different normalization schemes have different properties:

**(a) Batch Norm (BN):** Normalize over batch dimension
- Statistics: μ = E_batch[h], σ² = Var_batch[h]
- Pros: Reduces internal covariate shift, smooth landscape
- Cons: Batch size dependent, not suitable for small batches (<4)

**(b) Layer Norm (LN):** Normalize over feature dimension
- Statistics: μ = E_features[h], σ² = Var_features[h]
- Pros: Independent of batch size, works for RNNs/Transformers
- Cons: Less smoothing effect than BN for CNNs

**(c) Instance Norm (IN):** Normalize each sample independently
- Statistics: μ = E_spatial[h], σ² = Var_spatial[h] per sample
- Pros: Style normalization (used in style transfer)
- Cons: Loses batch-level information

**(d) Group Norm (GN):** Normalize within groups of channels
- Statistics: Divide C channels into G groups, normalize within groups
- Pros: Batch size independent, better than LN for CNNs
- Cons: Requires tuning number of groups G

**Smoothness comparison (Hessian spectral norm):**
||∇²L||_2: BN < GN < LN < IN < No Norm

BN provides strongest smoothing (by factor √m).

**Theorem 13 (Optimal Batch Size for BN):**

There's a trade-off in choosing batch size m:

**Small batch (m < 16):**
- Strong regularization (good!)
- Poor statistics estimation (bad!)
- High variance in μ_B, σ_B

**Large batch (m > 256):**
- Accurate statistics (good!)
- Weak regularization (may need explicit regularization)
- Larger memory requirement

**Optimal range:** m ∈ [32, 128] for most tasks

**Quantitative guideline:**
Choose m such that: √m ≈ λ_desired^{-1}
where λ_desired is target regularization strength.

For moderate regularization: m = 32-64 works well.

**Benefits (Summary):**
1. **Faster training:** Smoother loss landscape (Theorem 7)
2. **Higher learning rates:** Improved gradient predictiveness (Theorem 8)
3. **Less sensitive to initialization:** Scale reparametrization (Theorem 9)
4. **Acts as regularization:** Batch noise as dropout (Theorem 10)
5. **Stable convergence:** Bounded Hessian eigenvalues

```python
# Batch normalization
bn = nn.BatchNorm2d(num_features)

# Layer normalization (for sequences)
ln = nn.LayerNorm(normalized_shape)

# Group normalization
gn = nn.GroupNorm(num_groups=32, num_channels=64)
```

### 3. Data Augmentation

```python
from torchvision import transforms

# Standard augmentation for images
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Advanced augmentations
from torchvision.transforms import AutoAugment, RandAugment

# AutoAugment
auto_augment = AutoAugment()

# RandAugment
rand_augment = RandAugment()
```

### 4. Label Smoothing

**Idea:** Prevent overconfident predictions

**Formula:**
```
y_smooth = (1 - ε)y + ε/K
```

Where:
- ε: smoothing parameter (e.g., 0.1)
- K: number of classes

```python
class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy with label smoothing."""

    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_pred = F.log_softmax(pred, dim=-1)

        # Smooth labels
        with torch.no_grad():
            true_dist = torch.zeros_like(log_pred)
            true_dist.fill_(self.epsilon / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.epsilon)

        loss = -torch.sum(true_dist * log_pred, dim=-1).mean()
        return loss
```

### 5. Mixup

**Idea:** Train on linear combinations of samples

**Formula:**
```
x̃ = λx_i + (1-λ)x_j
ỹ = λy_i + (1-λ)y_j
```

Where λ ~ Beta(α, α), typically α = 0.2

```python
def mixup_data(x, y, alpha=0.2):
    """Apply mixup to batch."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Usage in training loop
inputs, targets_a, targets_b, lam = mixup_data(inputs, targets)
outputs = model(inputs)
loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
```

---

## 🎯 Training Best Practices

### Complete Training Loop

```python
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping (optional)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = running_loss / total
    val_acc = correct / total

    return val_loss, val_acc

def train(model, train_loader, val_loader, epochs=100, lr=0.001, device='cuda'):
    """Complete training pipeline."""
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_val_acc = 0.0

    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

        # Print progress
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"LR: {scheduler.get_last_lr()[0]:.6f}\n")

    return history
```

---

## 📚 Key References

### Foundational Papers
- **ImageNet Classification with Deep CNNs** - Krizhevsky et al. (2012) - AlexNet
- **Very Deep Convolutional Networks** - Simonyan & Zisserman (2014) - VGG
- **Deep Residual Learning** - He et al. (2015) - ResNet
- **Batch Normalization** - Ioffe & Szegedy (2015)
- **Adam: A Method for Stochastic Optimization** - Kingma & Ba (2014)

### Modern Architectures
- **EfficientNet** - Tan & Le (2019)
- **Vision Transformer** - Dosovitskiy et al. (2020)
- **Swin Transformer** - Liu et al. (2021)

---

## 🎯 Interview Questions

1. **Why do we use ReLU instead of sigmoid in hidden layers?**
   - Mitigates vanishing gradient
   - Computationally efficient
   - Sparse activation

2. **Explain the vanishing gradient problem and how ResNets solve it.**
   - Deep networks: gradients become very small
   - ResNets: Skip connections provide gradient highway
   - Identity mapping is easy to learn

3. **What is batch normalization and why does it help?**
   - Normalizes activations per mini-batch
   - Reduces internal covariate shift
   - Enables higher learning rates
   - Acts as regularization

4. **Compare SGD with momentum vs Adam optimizer.**
   - SGD+momentum: Simple, generalizes well, requires tuning
   - Adam: Adaptive per-parameter learning rates, less tuning, might overfit

5. **How does dropout prevent overfitting?**
   - Prevents co-adaptation of neurons
   - Ensemble effect (averaging many sub-networks)
   - Forces redundant representations

---

**Next:** See [MODERN_ML_AI_TECHNIQUES_2024_2025.md](./MODERN_ML_AI_TECHNIQUES_2024_2025.md) for cutting-edge architectures!

---

*Last Updated: October 2025*
*For hands-on practice: See notebooks 06 (Deep Learning) and 10 (Computer Vision)*
