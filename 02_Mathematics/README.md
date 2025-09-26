# Module 02: Mathematics for Machine Learning

## 🎯 Learning Objectives
By the end of this module, you will:
- Master essential linear algebra concepts for ML
- Understand calculus concepts used in optimization
- Apply probability and statistics in ML contexts
- Implement mathematical concepts in code

## 📚 Essential Math Areas

### 2.1 Linear Algebra - The Foundation of ML

#### Why Linear Algebra Matters
*Source: "Mathematics for Machine Learning" - Deisenroth, Faisal, Ong [Chapter 2]*
> "Linear algebra is the branch of mathematics concerning linear equations and their representations through matrices and vector spaces. In machine learning, we work with datasets that can be represented as matrices..."

#### Vectors
**Definition**: A vector is an ordered list of numbers.

```python
import numpy as np
import matplotlib.pyplot as plt

# Vector representation
v1 = np.array([3, 4])  # 2D vector
v2 = np.array([1, 2, 3])  # 3D vector

# Vector visualization
plt.figure(figsize=(8, 6))
plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='blue', width=0.005)
plt.xlim(-1, 5)
plt.ylim(-1, 5)
plt.grid(True)
plt.title('Vector [3, 4] Visualization')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
```

#### Vector Operations
```python
# Vector addition
v1 = np.array([1, 2])
v2 = np.array([3, 1])
addition = v1 + v2  # [4, 3]

# Scalar multiplication
scalar = 2
multiplication = scalar * v1  # [2, 4]

# Dot product (key for ML!)
dot_product = np.dot(v1, v2)  # 1*3 + 2*1 = 5
# Physical meaning: measure of similarity between vectors
```

*Source: "Linear Algebra and Its Applications" - David Lay [Section 1.3]*

#### Matrices - Data Containers
**Definition**: A matrix is a rectangular array of numbers arranged in rows and columns.

```python
# Dataset as matrix: each row = sample, each column = feature
data_matrix = np.array([
    [5.1, 3.5, 1.4, 0.2],  # Sample 1: sepal_length, sepal_width, petal_length, petal_width
    [4.9, 3.0, 1.4, 0.2],  # Sample 2
    [6.2, 3.4, 5.4, 2.3],  # Sample 3
])
print(f"Dataset shape: {data_matrix.shape}")  # (3, 4) = 3 samples, 4 features
```

#### Matrix Operations Critical for ML
```python
# Matrix multiplication (fundamental for neural networks)
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
matrix_mult = np.dot(A, B)

# Transpose (flipping rows and columns)
A_transpose = A.T

# Inverse (solving linear equations)
A_inverse = np.linalg.inv(A)
print(f"A * A_inverse = \n{np.dot(A, A_inverse)}")  # Should be identity matrix
```

*Source: "Introduction to Linear Algebra" - Gilbert Strang [Chapter 2]*

### 2.2 Calculus - The Engine of Optimization

#### Why Calculus in ML?
*Source: "Pattern Recognition and Machine Learning" - Christopher Bishop [Section 3.1]*
> "The goal of many machine learning algorithms is to find parameter values that minimize (or maximize) some objective function. This optimization process relies heavily on calculus, particularly derivatives."

#### Derivatives - Rate of Change
```python
import numpy as np
import matplotlib.pyplot as plt

# Simple function: f(x) = x^2
def f(x):
    return x**2

# Derivative: f'(x) = 2x
def f_derivative(x):
    return 2*x

# Visualization
x = np.linspace(-3, 3, 100)
y = f(x)
y_derivative = f_derivative(x)

plt.figure(figsize=(12, 5))

# Original function
plt.subplot(1, 2, 1)
plt.plot(x, y, 'b-', label='f(x) = x²')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Original Function')
plt.grid(True)
plt.legend()

# Derivative
plt.subplot(1, 2, 2)
plt.plot(x, y_derivative, 'r-', label="f'(x) = 2x")
plt.xlabel('x')
plt.ylabel("f'(x)")
plt.title('Derivative (Rate of Change)')
plt.grid(True)
plt.legend()

plt.tight_layout()
```

#### Gradient Descent - The Heart of ML Training
*Source: "Deep Learning" - Goodfellow, Bengio, Courville [Chapter 4]*

```python
# Simple gradient descent implementation
def gradient_descent_demo():
    """
    Minimize f(x) = (x-3)^2 using gradient descent
    Derivative: f'(x) = 2(x-3)
    """
    x = 0  # Starting point
    learning_rate = 0.1
    history = [x]

    for i in range(20):
        # Calculate gradient (derivative)
        gradient = 2 * (x - 3)

        # Update x in opposite direction of gradient
        x = x - learning_rate * gradient
        history.append(x)

        print(f"Step {i+1}: x = {x:.3f}, f(x) = {(x-3)**2:.3f}")

    return history

# Run gradient descent
history = gradient_descent_demo()
```

### 2.3 Probability & Statistics - Handling Uncertainty

#### Probability Distributions
*Source: "All of Statistics" - Larry Wasserman [Chapter 2]*

```python
import scipy.stats as stats

# Normal (Gaussian) Distribution - most important in ML
mu, sigma = 0, 1  # mean and standard deviation
x = np.linspace(-4, 4, 100)
normal_pdf = stats.norm.pdf(x, mu, sigma)

plt.figure(figsize=(10, 6))
plt.plot(x, normal_pdf, 'b-', label=f'Normal(μ={mu}, σ={sigma})')
plt.fill_between(x, normal_pdf, alpha=0.3)
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title('Normal Distribution - Foundation of Many ML Algorithms')
plt.grid(True)
plt.legend()

# Why normal distribution matters in ML:
# 1. Central Limit Theorem
# 2. Error assumptions in regression
# 3. Initialization of neural network weights
```

#### Bayes' Theorem - The Foundation of Probabilistic ML
*Source: "Pattern Recognition and Machine Learning" - Christopher Bishop [Chapter 1]*

```python
# Bayes' Theorem: P(A|B) = P(B|A) * P(A) / P(B)
# Example: Medical diagnosis

def bayes_medical_example():
    """
    Example: Disease testing with Bayes' theorem
    """
    # Prior probabilities
    P_disease = 0.01  # 1% of population has disease
    P_no_disease = 0.99

    # Likelihood (test accuracy)
    P_positive_given_disease = 0.95  # Test correctly identifies 95% of sick people
    P_positive_given_no_disease = 0.05  # Test incorrectly identifies 5% of healthy people

    # Total probability of positive test
    P_positive = (P_positive_given_disease * P_disease +
                  P_positive_given_no_disease * P_no_disease)

    # Posterior: Probability of disease given positive test
    P_disease_given_positive = (P_positive_given_disease * P_disease) / P_positive

    print(f"P(Disease | Positive Test) = {P_disease_given_positive:.3f}")
    print(f"Even with 95% accurate test, only {P_disease_given_positive:.1%} chance of having disease!")

    return P_disease_given_positive

bayes_medical_example()
```

## 🎮 Interactive Math Playground

### Linear Algebra Playground
```python
def interactive_matrix_operations():
    """
    Interactive demonstration of matrix operations
    """
    print("🎮 Matrix Operations Playground")
    print("=" * 40)

    # Create sample matrices
    A = np.random.randint(1, 10, (3, 3))
    B = np.random.randint(1, 10, (3, 3))

    print(f"Matrix A:\n{A}\n")
    print(f"Matrix B:\n{B}\n")

    # Operations
    operations = {
        "Addition": A + B,
        "Element-wise multiplication": A * B,
        "Matrix multiplication": np.dot(A, B),
        "A transpose": A.T,
        "Determinant of A": np.linalg.det(A)
    }

    for name, result in operations.items():
        print(f"{name}:")
        if isinstance(result, np.ndarray):
            print(f"{result}\n")
        else:
            print(f"{result:.3f}\n")

# Run the playground
interactive_matrix_operations()
```

## 📊 Visual Concept Map

```python
def create_math_concept_map():
    """
    Visual representation of how math concepts connect in ML
    """
    import networkx as nx

    # Create graph
    G = nx.Graph()

    # Add nodes (math concepts)
    concepts = [
        "Linear Algebra", "Calculus", "Statistics", "Probability",
        "Vectors", "Matrices", "Derivatives", "Gradients",
        "Distributions", "Bayes Theorem", "Optimization", "ML Algorithms"
    ]

    G.add_nodes_from(concepts)

    # Add edges (relationships)
    relationships = [
        ("Linear Algebra", "Vectors"),
        ("Linear Algebra", "Matrices"),
        ("Calculus", "Derivatives"),
        ("Calculus", "Gradients"),
        ("Statistics", "Distributions"),
        ("Probability", "Bayes Theorem"),
        ("Gradients", "Optimization"),
        ("Matrices", "ML Algorithms"),
        ("Optimization", "ML Algorithms"),
        ("Bayes Theorem", "ML Algorithms")
    ]

    G.add_edges_from(relationships)

    # Visualize
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=3, iterations=50)
    nx.draw(G, pos, with_labels=True, node_color='lightblue',
            node_size=2000, font_size=8, font_weight='bold')
    plt.title("Mathematical Foundations of Machine Learning")
    plt.axis('off')

create_math_concept_map()
```

## 📖 Sources & Free Resources

### 📚 **Primary Textbooks** (with free PDF availability):

1. **"Mathematics for Machine Learning"** - Deisenroth, Faisal, Ong
   - **Free PDF**: [https://mml-book.github.io/](https://mml-book.github.io/)
   - **Specific References**:
     - Chapter 2: Linear Algebra (vectors, matrices, eigenvalues)
     - Chapter 5: Vector Calculus (gradients, optimization)

2. **"Introduction to Statistical Learning"** - James, Witten, Hastie, Tibshirani
   - **Free PDF**: [https://www.statlearning.com/](https://www.statlearning.com/)
   - **Specific References**:
     - Chapter 2: Statistical Learning (bias-variance tradeoff)

3. **"The Elements of Statistical Learning"** - Hastie, Tibshirani, Friedman
   - **Free PDF**: [https://web.stanford.edu/~hastie/ElemStatLearn/](https://web.stanford.edu/~hastie/ElemStatLearn/)
   - **Specific References**:
     - Chapter 3: Linear Methods (mathematical foundations)

4. **"Linear Algebra and Its Applications"** - David Lay
   - **University Libraries**: Often available through institutional access
   - **Specific References**: Sections 1.1-1.9 (vector operations, matrix algebra)

### 🌐 **Online Courses** (Free):
- **Khan Academy Linear Algebra**: [https://www.khanacademy.org/math/linear-algebra](https://www.khanacademy.org/math/linear-algebra)
- **MIT 18.06 Linear Algebra**: [https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)
- **3Blue1Brown Essence of Linear Algebra**: [https://www.3blue1brown.com/topics/linear-algebra](https://www.3blue1brown.com/topics/linear-algebra)

### 📑 **Key Papers**:
- "A Tutorial on Principal Component Analysis" - Jonathon Shlens (2014) *[Explains eigenvalues in ML context]*
- "Gradient-Based Learning Applied to Document Recognition" - LeCun et al. (1998) *[Backpropagation and gradients]*

## ✅ Practice Problems

### Problem 1: Vector Operations
```python
# Given vectors u = [2, 3] and v = [1, 4]
# Calculate:
# a) u + v
# b) 3u - 2v
# c) dot product u·v
# d) angle between u and v

# Your solution here:
```

### Problem 2: Matrix Multiplication
```python
# Given matrices:
A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[1, 4], [2, 5], [3, 6]])

# Calculate A @ B and explain the dimensions
# Your solution here:
```

### Problem 3: Gradient Calculation
```python
# For function f(x) = 3x² + 2x + 1
# a) Find the derivative f'(x)
# b) Calculate f'(2)
# c) Implement gradient descent to find minimum

# Your solution here:
```

## 🚀 Next Steps

Congratulations! You now have the mathematical foundation for machine learning.

**Next Module**: [03_Statistics](../03_Statistics/README.md) - Dive deeper into statistical concepts crucial for understanding data and model evaluation.

---
*Estimated completion time: 4-6 hours*
*Prerequisites: High school algebra and basic calculus*