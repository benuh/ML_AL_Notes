# Algorithm Complexity Reference: Complete Guide

## Table of Contents
1. [Introduction to Complexity Analysis](#introduction)
2. [Classical Machine Learning](#classical-ml)
3. [Deep Learning](#deep-learning)
4. [Optimization Algorithms](#optimization)
5. [Ensemble Methods](#ensemble-methods)
6. [Dimensionality Reduction](#dimensionality-reduction)
7. [Clustering Algorithms](#clustering)
8. [Natural Language Processing](#nlp)
9. [Computer Vision](#computer-vision)
10. [Recommendation Systems](#recommendation-systems)
11. [Reinforcement Learning](#reinforcement-learning)
12. [Complexity Comparison Tables](#comparison-tables)

---

## Introduction to Complexity Analysis

### Notation and Definitions

**Time Complexity:**
```
O(f(n)): Upper bound (worst-case)
Ω(f(n)): Lower bound (best-case)
Θ(f(n)): Tight bound (average-case)

Common complexities (best to worst):
O(1)         - Constant
O(log n)     - Logarithmic
O(n)         - Linear
O(n log n)   - Linearithmic
O(n²)        - Quadratic
O(n³)        - Cubic
O(2^n)       - Exponential
O(n!)        - Factorial
```

**Space Complexity:**
```
Memory required beyond input size

In-place: O(1) additional space
Out-of-place: O(n) or more additional space
```

**Variables Used:**
```
n: Number of samples/data points
d: Number of features/dimensions
k: Number of clusters/classes/components
m: Number of users (RecSys)
p: Number of items (RecSys)
L: Number of layers (Deep Learning)
h: Hidden dimension
T: Number of iterations/epochs
```

---

## Classical Machine Learning

### Linear Models

**Linear Regression (Ordinary Least Squares)**
```
Problem: min ||Xw - y||²
         w

Closed-form solution: w = (X^T X)^(-1) X^T y

Training Complexity:
- Matrix multiplication X^T X: O(nd²)
- Matrix inversion: O(d³)
- Matrix multiplication X^T y: O(nd)
- Total: O(nd² + d³)

Prediction: O(nd)
Space: O(nd) for data + O(d²) for X^T X

Variants:
- Ridge Regression: Same as OLS
- Lasso Regression (coordinate descent): O(T·nd) where T = iterations
- Elastic Net: O(T·nd)
```

**Logistic Regression**
```
Training (Gradient Descent):
- Per iteration: O(nd)
- Total: O(T·nd) where T = iterations

Training (Newton-Raphson):
- Per iteration: O(nd² + d³)
  - Gradient: O(nd)
  - Hessian: O(nd²)
  - Inversion: O(d³)
- Total: O(T·(nd² + d³))
- Converges faster: T typically 10-20

Prediction: O(nd)
Space: O(nd + d)
```

### Support Vector Machines

**Linear SVM**
```
Training (SMO algorithm):
- Time: O(n² · d) to O(n³ · d)
- Depends on support vectors (s ≈ 0.1n typically)
- Practical: O(s² · d) ≈ O(0.01n² · d)

Prediction: O(s · d) where s = number of support vectors
Space: O(s · d)

Linear SVM (LibLinear for large-scale):
- Training: O(T·nd) - similar to logistic regression
- Prediction: O(d)
- Scales to millions of samples
```

**Kernel SVM**
```
Training:
- Time: O(n² · d) to O(n³)
- Kernel matrix computation: O(n² · d)
- QP solver: O(n³) worst-case, O(n²) average

Prediction:
- Time: O(s · d) per sample
- s = support vectors (typically 20-80% of n)

Space: O(n² + sd) for kernel matrix and support vectors

Popular Kernels:
- RBF: k(x,y) = exp(-γ||x-y||²) - O(d) per computation
- Polynomial: k(x,y) = (γx^T y + r)^p - O(d) per computation
```

### Decision Trees

**CART (Classification and Regression Trees)**
```
Training:
- At each node: Find best split
  - For each feature: Sort values O(n log n)
  - Evaluate split: O(n)
  - Total per node: O(d · n log n)

- Tree depth: O(log n) balanced, O(n) worst-case
- Number of nodes: O(n) worst-case

Total Training:
- Balanced tree: O(n · d · log n)
- Worst-case (degenerate): O(n² · d)

Prediction: O(log n) balanced, O(n) worst-case
Space: O(n) for tree structure
```

**Optimized Decision Trees (Histogram-based)**
```
Used in: XGBoost, LightGBM, CatBoost

Training:
- Histogram construction: O(nd)
- Split finding: O(k · d) where k = number of bins
- Total per level: O(nd + kd)
- Full tree: O(L · (nd + kd)) where L = depth

Typical: k = 255, L = 6-10
Complexity: O(nd) dominated by data scan

Prediction: O(L) = O(log n) typically
Space: O(nd) for histograms
```

### Naive Bayes

**Multinomial Naive Bayes**
```
Training:
- Count occurrences: O(nd)
- Compute probabilities: O(kd) where k = classes
- Total: O(nd + kd) ≈ O(nd)

Prediction:
- Per sample: O(kd)
- Batch: O(nkd)

Space: O(kd) for probability tables

Variants:
- Gaussian NB: Same complexity
- Bernoulli NB: Same complexity
```

### K-Nearest Neighbors (KNN)

**Brute Force KNN**
```
Training: O(1) - just store data
Prediction:
- Compute distances: O(nd)
- Find k nearest: O(n log k) with heap
- Total per query: O(nd + n log k)

Space: O(nd)

Major bottleneck: Prediction is expensive!
```

**KD-Tree KNN**
```
Training: O(dn log n) - build tree
Prediction:
- Best-case: O(k + log n)
- Average: O(k + d log n)
- Worst-case: O(kn) - when d is large (curse of dimensionality)

Space: O(nd)

Effective when: d < 20
Degrades when: d > 20 (use brute force or approximate methods)
```

**Ball Tree KNN**
```
Training: O(dn log n)
Prediction:
- Better than KD-tree for high dimensions
- Average: O(k + d log n)

Space: O(nd)

Effective when: d < 50
```

**Approximate KNN (LSH, Annoy, FAISS)**
```
Locality-Sensitive Hashing (LSH):
- Training: O(nd · L) where L = hash tables
- Prediction: O(L · d + k log k)

Annoy (Spotify):
- Training: O(nd log n)
- Prediction: O(d log n)

FAISS (Facebook):
- Training: O(nd log n)
- Prediction: O(d log k)

Trade-off: ~95-99% accuracy with 10-100x speedup
```

---

## Deep Learning

### Feed-Forward Neural Networks

**Fully Connected Layer**
```
Forward Pass:
y = Wx + b
- Matrix multiplication: O(n_out · n_in · batch_size)
- Bias addition: O(n_out · batch_size)
- Total: O(n_out · n_in · batch_size)

Backward Pass:
- Gradient w.r.t. input: O(n_out · n_in · batch_size)
- Gradient w.r.t. weights: O(n_out · n_in · batch_size)
- Total: O(n_out · n_in · batch_size)

Parameters: n_out · n_in + n_out
Memory: O(n_out · n_in + batch_size · (n_in + n_out))
```

**Multi-Layer Perceptron (MLP)**
```
Network: [d_in, h_1, h_2, ..., h_L, d_out]

Forward Pass (per sample):
Total FLOPs: Σ_l (h_l · h_{l-1}) ≈ O(L · h²) if all layers ≈ h

Backward Pass:
Same as forward: O(L · h²)

Training (one epoch):
Time: O(T · n · L · h²) where T = epochs
Space: O(L · h² + batch_size · h) for weights + activations

Parameters: Σ_l (h_l · h_{l-1}) ≈ O(L · h²)
```

### Convolutional Neural Networks

**Convolutional Layer**
```
Input: (H, W, C_in)
Output: (H', W', C_out)
Kernel: (K, K, C_in, C_out)

Forward Pass:
FLOPs per output pixel: K² · C_in · C_out
Total output pixels: H' · W'
Total FLOPs: H' · W' · K² · C_in · C_out

For batch: Multiply by batch_size

Backward Pass: Same complexity

Parameters: K² · C_in · C_out + C_out (bias)
Memory: O(H · W · C_in + K² · C_in · C_out)

Example (VGG Conv):
H=224, W=224, K=3, C_in=64, C_out=128
FLOPs = 224 · 224 · 9 · 64 · 128 ≈ 3.7 billion
```

**Pooling Layer**
```
Max/Average Pooling (K × K):

Forward: O(H · W · C · K²)
Backward: O(H · W · C)

Parameters: 0
Memory: O(H · W · C)
```

**Batch Normalization**
```
For feature map (batch_size, H, W, C):

Forward:
- Compute mean/variance: O(batch_size · H · W · C)
- Normalize: O(batch_size · H · W · C)
- Scale/shift: O(batch_size · H · W · C)
- Total: O(batch_size · H · W · C)

Backward: O(batch_size · H · W · C)

Parameters: 2C (γ, β)
Memory: O(batch_size · H · W · C + C)
```

### Recurrent Neural Networks

**Vanilla RNN**
```
Single timestep: h_t = tanh(W_hh h_{t-1} + W_xh x_t + b)

Forward (per timestep):
- W_hh h_{t-1}: O(h²)
- W_xh x_t: O(d · h)
- Total per step: O(h² + dh)

Sequence length T:
- Forward: O(T · (h² + dh))
- Backward (BPTT): O(T · (h² + dh))

Parameters: h² + dh + h
Memory: O(T · h + h² + dh) - store all hidden states for BPTT
```

**LSTM**
```
Four gates: forget, input, cell, output

Per timestep:
- Gate computations: 4 · O(h² + dh)
- Element-wise operations: O(h)
- Total: O(4(h² + dh))

Sequence length T:
- Forward: O(T · 4(h² + dh))
- Backward: O(T · 4(h² + dh))

Parameters: 4(h² + dh + h) ≈ 4h² + 4dh
Memory: O(T · h + h² + dh)

GRU: Slightly cheaper - 3 gates instead of 4
```

**Bidirectional RNN/LSTM**
```
Forward and backward passes:
Time: 2 · O(T · (h² + dh))
Parameters: 2 · (h² + dh)
Memory: 2 · O(T · h)
```

### Transformer Architecture

**Self-Attention**
```
Input: (batch_size, seq_len, d_model)
Query, Key, Value: (batch_size, seq_len, d_k)

Attention(Q, K, V) = softmax(QK^T / √d_k) V

Complexity:
- QK^T: O(seq_len² · d_k)
- Softmax: O(seq_len²)
- Multiply with V: O(seq_len² · d_v)
- Total: O(seq_len² · d_model)

Space: O(seq_len²) for attention matrix

This is the bottleneck for long sequences!
```

**Multi-Head Attention**
```
h heads, each with d_k = d_v = d_model / h

Total complexity:
Time: O(h · seq_len² · d_k) = O(seq_len² · d_model)
Space: O(h · seq_len²)

Parameters: 4 · d_model² (Q, K, V, O projections)
```

**Feed-Forward Network (in Transformer)**
```
FFN(x) = ReLU(xW_1 + b_1)W_2 + b_2

Per position:
- First layer: O(d_model · d_ff)
- Second layer: O(d_ff · d_model)
- Total per position: O(d_model · d_ff)

For sequence:
Total: O(seq_len · d_model · d_ff)

Typically: d_ff = 4 · d_model
So: O(seq_len · 4 · d_model²)

Parameters: 2 · d_model · d_ff ≈ 8 · d_model²
```

**Full Transformer Block**
```
One encoder block:
- Multi-head attention: O(seq_len² · d_model)
- FFN: O(seq_len · d_model · d_ff)
- LayerNorm: O(seq_len · d_model)

Total: O(seq_len² · d_model + seq_len · d_model · d_ff)

For short sequences (seq_len < d_ff):
  Dominated by FFN: O(seq_len · d_model²)

For long sequences (seq_len > d_ff):
  Dominated by attention: O(seq_len² · d_model)

L layers:
Total: O(L · (seq_len² · d_model + seq_len · d_model²))
```

**Efficient Attention Variants**

```
Method              | Time            | Space
--------------------|-----------------|--------
Standard Attention  | O(n² · d)       | O(n²)
Sparse Attention    | O(n · √n · d)   | O(n√n)
Linformer          | O(n · k · d)    | O(nk), k<<n
Performer          | O(n · d²)       | O(nd)
Flash Attention    | O(n² · d)       | O(n)    ← Same time, less memory!

where n = seq_len, d = d_model
```

---

## Optimization Algorithms

### Gradient Descent Variants

**Batch Gradient Descent**
```
Per iteration:
- Gradient computation: O(nd)
- Parameter update: O(d)
- Total: O(nd)

T iterations: O(T · nd)

Convergence:
- Convex: O(1/ε) iterations for ε-accuracy
- Strongly convex: O(log(1/ε)) iterations
```

**Stochastic Gradient Descent (SGD)**
```
Per iteration (one sample):
- Gradient: O(d)
- Update: O(d)
- Total: O(d)

One epoch (n samples): O(nd)
T epochs: O(T · nd)

Convergence:
- Convex: O(1/√T) after T iterations
- Slower than batch but much cheaper per iteration
```

**Mini-Batch SGD**
```
Batch size: b

Per iteration:
- Gradient: O(bd)
- Update: O(d)
- Total: O(bd)

One epoch (n/b batches): O(nd)
T epochs: O(T · nd)

Sweet spot: b = 32-256 (GPU efficiency)
```

**Momentum / Nesterov Accelerated Gradient**
```
Per iteration: O(nd) - same as SGD
Extra memory: O(d) for velocity

Convergence:
- Convex: O(1/T²) (better than vanilla GD!)
- Strongly convex: O(exp(-√(μ/L) · T))
```

**Adam / RMSprop / Adagrad**
```
Per iteration: O(nd)
Extra memory: O(2d) for first and second moments

Same time complexity as SGD, but:
- Better convergence in practice
- More memory overhead
```

**Second-Order Methods (Newton, L-BFGS)**
```
Newton's Method:
- Gradient: O(nd)
- Hessian: O(nd²)
- Inversion: O(d³)
- Total: O(nd² + d³)

Practical when: d < 10,000

L-BFGS (limited memory):
- Per iteration: O(nd + md) where m ≈ 10-20
- Memory: O(md) instead of O(d²)

Convergence: O(log(1/ε)) - quadratic (very fast!)
```

---

## Ensemble Methods

### Bagging (Bootstrap Aggregating)

**Random Forest**
```
Training:
- B trees, each on n samples with d features
- Per tree: O(n · d · log n)
- Total: O(B · n · d · log n)

Typical: B = 100-500 trees

Prediction:
- Per sample: O(B · log n)
- Batch of m samples: O(m · B · log n)

Space: O(B · n) for tree structures

Feature subsampling (√d features per split):
Training: O(B · n · √d · log n)
```

**Bagging Ensemble**
```
M base models, each with complexity C_train, C_pred

Training:
- With parallel: O(C_train) - all models train in parallel
- Sequential: O(M · C_train)

Prediction:
- Per sample: O(M · C_pred)

Space: O(M · model_size)
```

### Boosting

**AdaBoost**
```
T weak learners (typically decision stumps)

Training:
- Per iteration: Train weak learner O(nd) + update weights O(n)
- Total: O(T · nd)

Typical: T = 50-500

Prediction: O(T · d_tree) ≈ O(T) for stumps

Space: O(Tn)
```

**Gradient Boosting (GBM)**
```
T trees, each of depth L

Training:
- Per tree: O(n · d · L)
- Compute residuals: O(n)
- Total: O(T · n · d · L)

Prediction: O(T · L)

Space: O(T · n)

Typical: T = 100-500, L = 3-8
```

**XGBoost**
```
Optimized gradient boosting:

Training:
- Histogram-based split finding: O(k · d) per split
- Cache-aware access
- Parallel tree construction
- Total: O(T · n · d) dominated by data scan

Approximate algorithm (for very large data):
- Quantile sketching: O(nd log n)
- Split finding on quantiles: O(k · d)

Prediction: O(T · L)

Space: O(nd) for data + O(k · d) for histograms

Speedup over vanilla GBM: 10-100x
```

**LightGBM**
```
Leaf-wise growth (vs level-wise in XGBoost):

Training:
- Histogram: O(nd)
- Leaf-wise: Grows best leaf first
- GOSS (sampling): O(a · n · d) where a < 1
- EFB (feature bundling): O(nd') where d' < d

Total: O(T · a · n · d')

Speedup over XGBoost: 2-10x
Lower memory: 50% reduction typical
```

**CatBoost**
```
Ordered boosting + categorical feature handling:

Training:
- Ordered TS: O(T · n · d · log n)
- Symmetric trees: O(T · n · d)

Prediction: O(T · L)

Better quality on categorical features
Slightly slower training than LightGBM
```

---

## Dimensionality Reduction

### Principal Component Analysis (PCA)

**Exact PCA (SVD-based)**
```
Input: X ∈ ℝ^(n × d)

Methods:
1. Covariance matrix eigendecomposition:
   - Compute X^T X: O(nd²)
   - Eigendecomposition: O(d³)
   - Total: O(nd² + d³)

2. SVD of X directly:
   - Time: O(min(nd², n²d))
   - More numerically stable

Projection (to k components): O(ndk)

Space: O(nd + d²)

Practical: Use when d < 10,000
```

**Randomized PCA**
```
Approximate top k components:

Time: O(ndk + dk²)
- Much faster when k << min(n, d)

Space: O(nk + dk)

Accuracy: Very good approximation (>99% explained variance)

Used in: scikit-learn for large matrices
```

**Incremental PCA**
```
For data that doesn't fit in memory:

Per batch of size b:
- Update: O(bd² + d³)

Total for n samples:
Time: O((n/b) · (bd² + d³)) = O(nd² + (n/b)d³)

Space: O(bd + d²) - only need one batch in memory
```

### t-SNE

```
Input: n samples in d dimensions → 2D/3D

Time Complexity:
- Pairwise distances: O(n² d)
- P matrix (affinities): O(n²)
- Gradient descent (T iterations): O(T · n²)
- Total: O(n² d + T · n²)

With Barnes-Hut approximation:
- Time: O(n log n) per iteration
- Total: O(T · n log n)
- Typical: T = 1000 iterations

Space: O(n²) exact, O(n) with Barnes-Hut

Practical limit: n < 10,000 (exact), n < 100,000 (Barnes-Hut)
```

### UMAP

```
Faster alternative to t-SNE:

Time:
- Construct k-NN graph: O(n^1.14) approximate
- Optimize embedding: O(T · n)
- Total: O(n^1.14 + T · n)

Typical: T = 200-500 iterations

Space: O(n · k) where k = n_neighbors ≈ 15

Speedup over t-SNE: 10-100x
Scales to: millions of points
```

### Autoencoders

**Vanilla Autoencoder**
```
Encoder: d → h → ... → bottleneck (k dimensions)
Decoder: k → h → ... → d

Per sample:
- Forward: O(L · h² + dk)
- Backward: O(L · h² + dk)

Training (n samples, T epochs):
Time: O(T · n · L · h²)

Space: O(L · h² + batch_size · h)

Parameters: O(L · h²)
```

---

## Clustering Algorithms

### K-Means

```
n samples, d features, k clusters, T iterations

Per iteration:
1. Assign to nearest centroid: O(nkd)
2. Update centroids: O(nd)
Total per iteration: O(nkd)

Total: O(T · nkd)

Space: O(nd + kd)

Typical: T = 10-100 iterations

Variants:
- K-Means++: O(nkd log k) initialization (better results)
- Mini-batch K-Means: O(T · b · kd) where b = batch_size
```

### DBSCAN

```
Density-based clustering:

Time:
- With spatial index (KD-tree/Ball-tree): O(n log n)
- Without index: O(n²)

Space: O(n)

Parameters: ε (radius), minPts (minimum points)

Advantages:
- No need to specify k
- Finds arbitrary-shaped clusters
- Robust to outliers

Disadvantages:
- Sensitive to parameters
- Struggles with varying densities
```

### Hierarchical Clustering

**Agglomerative Clustering**
```
Bottom-up approach:

Time:
- Naive: O(n³)
- With heap: O(n² log n)

Space: O(n²) for distance matrix

Linkage methods:
- Single linkage: O(n²)
- Complete/average: O(n² log n)
- Ward: O(n² log n)

Output: Dendrogram with n-1 merges

Practical limit: n < 10,000
```

### Gaussian Mixture Models (GMM)

```
Expectation-Maximization algorithm:

Per iteration:
- E-step (assign responsibilities): O(nkd)
- M-step (update parameters): O(nkd)
Total per iteration: O(nkd)

T iterations: O(T · nkd)

Space: O(nd + kd²) for data and covariance matrices

Full covariance: kd² parameters (expensive!)
Diagonal covariance: kd parameters (cheaper)

Typical: T = 100-200 iterations
```

---

## Natural Language Processing

### Word Embeddings

**Word2Vec (Skip-gram)**
```
Vocabulary size: V
Embedding dimension: d
Training samples: n

Per sample (one word context pair):
- Forward: O(d)
- Negative sampling (k negatives): O(k · d)
- Backward: O(k · d)

Total per sample: O(k · d)
Full training: O(n · k · d)

Hierarchical softmax alternative:
- Per sample: O(d · log V)
- Better for large vocabularies

Parameters: V · d
Memory: O(V · d)

Typical: V = 10,000-100,000, d = 100-300, k = 5-20
```

**GloVe**
```
Co-occurrence matrix: V × V

Training:
- Build co-occurrence: O(corpus_size)
- Matrix factorization: O(T · nnz(C) · d)
  where nnz = non-zero entries

Space: O(V²) for co-occurrence matrix

Sparse matrix optimization:
- Only store non-zero entries
- Typical nnz << V²
```

**Transformer Language Models**

```
GPT-2 (1.5B parameters):
- Layers: L = 48
- Hidden: d = 1600
- Heads: h = 25
- Sequence length: n = 1024

Forward pass (one token):
Per layer: O(n² d + n d²)
All layers: O(L · (n² d + n d²))

For GPT-2: O(48 · (1024² · 1600 + 1024 · 1600²))
         ≈ 80 billion FLOPs

Training (autoregressive):
Per sequence: n times more expensive
One batch (b sequences): O(b · n · L · (n² d + n d²))

Memory: O(b · n · L · d) for activations
```

**BERT**
```
BERT-base:
- L = 12, d = 768, h = 12, n = 512

Forward (bi-directional):
Per layer: O(n² d + n d²)
All layers: O(L · (n² d + n d²))

Training (MLM + NSP):
- Mask random tokens: O(n)
- Forward/backward: O(L · (n² d + n d²))

Fine-tuning:
- Usually cheaper: 2-4 epochs vs 40+ for pre-training

Parameters:
- Embeddings: V · d ≈ 30K · 768 ≈ 23M
- Encoder: L · (4d² + 12d²) ≈ 85M  (attention + FFN)
- Total: 110M parameters
```

---

## Computer Vision

### CNN Architectures

**LeNet-5 (1998)**
```
~60K parameters
Forward: ~0.3M FLOPs
Memory: ~0.5MB
```

**AlexNet (2012)**
```
~60M parameters
Forward: ~720M FLOPs
Memory: ~240MB (for 224×224 input)

Breakdown per layer:
- Conv1 (11×11, stride 4): 105M FLOPs
- Conv2-5: 448M FLOPs
- FC layers: 167M FLOPs
```

**VGG-16 (2014)**
```
~138M parameters
Forward: ~15.5B FLOPs
Memory: ~528MB

Complexity: Mostly in FC layers (parameters), Conv layers (FLOPs)

Per Conv block (3×3 conv):
FLOPs = H · W · K² · C_in · C_out
Example: 224×224, K=3, C=512
       = 224 · 224 · 9 · 512 · 512
       ≈ 118B FLOPs for one layer!
```

**ResNet-50 (2015)**
```
~25M parameters (much less than VGG!)
Forward: ~4B FLOPs
Memory: ~200MB

Residual block:
- 1×1 conv (reduce): H·W·C_in·C_mid
- 3×3 conv: H·W·9·C_mid·C_mid
- 1×1 conv (expand): H·W·C_mid·C_out
- Skip connection: free (addition)

Total: ~50 residual blocks
Depth: 50 layers
```

**EfficientNet-B7 (2019)**
```
~66M parameters
Forward: ~37B FLOPs
Top-1 accuracy: 84.4% (ImageNet)

Scales depth, width, resolution together:
- Depth: L = α^φ
- Width: W = β^φ
- Resolution: R = γ^φ

Constraint: α·β²·γ² ≈ 2^φ

Better accuracy per FLOP than ResNet
```

### Object Detection

**R-CNN**
```
Per image:
- Selective search: ~2000 proposals
- Extract features per proposal: 2000 · O(CNN)
- SVM classification: 2000 · O(d · k)

Total: O(2000 · CNN_forward)

Very slow: ~47 seconds per image (GPU)
```

**Fast R-CNN**
```
Per image:
- Single CNN forward: O(CNN)
- RoI pooling: 2000 · O(k²) where k = pool size
- FC layers per RoI: 2000 · O(d²)

Total: O(CNN + 2000 · d²)

Speedup: ~0.3 seconds per image (GPU)
```

**Faster R-CNN**
```
Per image:
- Backbone CNN: O(H·W·d²)
- RPN: O(H·W·k·a) where k = anchors per position
- RoI head: O(num_proposals · d²)

Total: O(CNN + num_proposals · d²)

Typical: num_proposals = 300
Speed: ~0.2 seconds per image (GPU)
```

**YOLO v3**
```
Single-shot detector:

Per image:
- Backbone (Darknet-53): ~65B FLOPs
- Detection heads: ~17B FLOPs
- Total: ~82B FLOPs

Predictions: 10,647 bounding boxes
NMS post-processing: O(n² · k) where n = boxes, k = classes

Speed: ~30ms per image (real-time!)
```

**EfficientDet**
```
Scalable object detection:

EfficientDet-D7:
- Backbone: EfficientNet-B7 (37B FLOPs)
- BiFPN: ~100B FLOPs
- Total: ~137B FLOPs

Better accuracy per FLOP than YOLO/RetinaNet
```

### Image Segmentation

**FCN (Fully Convolutional Networks)**
```
Per pixel prediction:

Forward: O(H · W · d²) - similar to classification CNN
Output: H × W × num_classes

Upsampling (transposed conv):
- Same complexity as downsampling

Total: O(H · W · d²)
```

**U-Net**
```
Encoder-decoder with skip connections:

Encoder: O(H · W · d²)
Decoder: O(H · W · d²)
Skip connections: negligible

Total: O(H · W · d²)

Parameters: Fewer than VGG due to fewer channels
```

**Mask R-CNN**
```
Faster R-CNN + Mask branch:

Per image:
- Backbone + RPN: O(CNN)
- RoI head: O(num_proposals · d²)
- Mask branch: O(num_proposals · k² · d)
  where k × k = mask resolution (28×28 typical)

Total: O(CNN + num_proposals · (d² + k² · d))

Speed: ~200ms per image (GPU)
```

---

## Recommendation Systems

### Collaborative Filtering

**User-User CF**
```
m users, n items

Training (compute similarities):
- Pairwise similarity: O(m² · n)
- Space: O(m²) for similarity matrix

Prediction (for one user):
- Find k nearest users: O(m log k) with heap
- Weighted average: O(k · n)

Total prediction: O(m log k + kn)

Bottleneck: m² similarity computation
Practical limit: m < 100,000
```

**Item-Item CF**
```
Training:
- Item similarities: O(n² · m)
- Sparse matrices: O(n² · m̄) where m̄ = avg items/user << m

Prediction (for one user):
- User has r_u rated items
- Compute for all items: O(r_u · n)
- More efficient when r_u << m

Scales better than user-user when m >> n
```

### Matrix Factorization

**SVD (Exact)**
```
Rating matrix: R ∈ ℝ^(m × n)

Computation:
- Full SVD: O(min(mn², m²n))
- Truncated SVD (k factors): O(mnk)

Space: O(mn + mk + nk)

For large matrices: Not practical (dense computation)
```

**Alternating Least Squares (ALS)**
```
Factors: P ∈ ℝ^(m × k), Q ∈ ℝ^(n × k)

Per iteration:
- Fix Q, solve for P: O(m · n̄_u · k²)
  where n̄_u = avg ratings per user
- Fix P, solve for Q: O(n · m̄_i · k²)
  where m̄_i = avg ratings per item

Total per iteration: O((m · n̄_u + n · m̄_i) · k²)

For sparse matrix (nnz = number of ratings):
Per iteration: O(nnz · k²)

Typical: k = 50-200, 10-20 iterations
Complexity: O(20 · nnz · k²)

Space: O(mk + nk + nnz)

Parallelizable: Each user/item independent
```

**SGD for Matrix Factorization**
```
Per rating update:
- Prediction: O(k)
- Gradient: O(k)
- Update: O(k)

One epoch (nnz ratings): O(nnz · k)

Typical: 20-100 epochs
Total: O(100 · nnz · k)

Faster than ALS for very large datasets
Memory efficient: O(mk + nk)
```

### Neural Collaborative Filtering

**NCF (Neural Collaborative Filtering)**
```
Architecture: Embedding → MLP → Output

Forward (per user-item pair):
- Embeddings: O(k)
- MLP: O(L · h²) where L = layers, h = hidden size
- Total: O(k + L · h²)

Training (n interactions, T epochs):
Time: O(T · n · (k + L · h²))

Negative sampling (r negatives per positive):
Time: O(T · n · r · (k + L · h²))

Parameters: 2mk + 2nk + L·h² (embeddings + MLP)
```

**Two-Tower Model**
```
Separate networks for user and item:

Training:
- User tower: O(d_u → h)
- Item tower: O(d_i → h)
- Similarity: O(h)
- Per sample: O(d_u · h + d_i · h)

Batch (b samples): O(b · (d_u · h + d_i · h))

Inference:
- Pre-compute item embeddings: O(n · d_i · h)
- Compute user embedding: O(d_u · h)
- Similarity search: O(h) with approximate NN (FAISS)

Scales to billions of items!
```

---

## Reinforcement Learning

### Value-Based Methods

**Q-Learning (Tabular)**
```
States: S, Actions: A

Space: O(|S| · |A|) for Q-table

Per update:
- Lookup: O(1)
- Bellman update: O(1)
- Total: O(1)

Episode of length T: O(T)

Only works for discrete, small state spaces
Practical limit: |S| · |A| < 10^6
```

**Deep Q-Network (DQN)**
```
Q-network: Neural network with |A| outputs

Per step:
- Forward: O(L · h²) for network
- Target computation: O(L · h²)
- Backward: O(L · h²)

Experience replay:
- Sample batch: O(b)
- Train on batch: O(b · L · h²)

Total per step: O(L · h²)

Memory: O(replay_buffer_size · state_dim)
Typical buffer: 1M transitions
```

**Dueling DQN / Double DQN**
```
Same complexity as DQN: O(L · h²)

Slightly more parameters:
- Value stream: h → h → 1
- Advantage stream: h → h → |A|
- Additional: ~h² parameters
```

### Policy-Based Methods

**REINFORCE (Policy Gradient)**
```
Policy network: π_θ(a|s)

Per episode of length T:
- Forward (T steps): O(T · L · h²)
- Backward (update): O(T · L · h²)

Total: O(T · L · h²)

High variance: Needs many episodes for stable learning
```

**Actor-Critic (A2C)**
```
Actor network: π_θ(a|s)
Critic network: V_φ(s)

Per step:
- Actor forward: O(L_π · h²)
- Critic forward: O(L_V · h²)
- TD error computation: O(1)
- Both backwards: O(L_π · h² + L_V · h²)

Total: O((L_π + L_V) · h²)

More sample efficient than REINFORCE
```

**PPO (Proximal Policy Optimization)**
```
Per mini-batch update:
- Compute advantages: O(T)
- K epochs on batch:
  - Actor forward/backward: O(K · b · L_π · h²)
  - Critic forward/backward: O(K · b · L_V · h²)

Typical: K = 4-10 epochs, b = 64-256

Total per update: O(K · b · (L_π + L_V) · h²)

More stable than vanilla policy gradient
State-of-the-art for many tasks
```

**SAC (Soft Actor-Critic)**
```
Components:
- Actor: π_θ(a|s)
- Twin critics: Q_φ1(s,a), Q_φ2(s,a)

Per update:
- Sample from replay: O(b)
- Actor forward: O(b · L_π · h²)
- Both critics forward: O(2 · b · L_Q · h²)
- Backward all: O(b · (L_π + 2L_Q) · h²)

Total: O(b · (L_π + 2L_Q) · h²)

Off-policy: More sample efficient than on-policy (PPO)
```

### Model-Based Methods

**Dyna-Q**
```
Combines model-free (Q-learning) + model

Per step:
- Q-learning update: O(1)
- Learn model: O(1)
- Planning (n simulated steps): O(n)

Total: O(n)

Typical: n = 5-50 simulated steps per real step

More sample efficient than model-free alone
```

**AlphaZero**
```
Components:
- Policy/Value network: f_θ(s) → (π, v)
- MCTS planning

Per move:
- MCTS (N simulations): O(N · L_max · network_forward)
  where L_max = max depth of simulation
- Network forward: O(L · h²)
- Total planning: O(N · L_max · L · h²)

Typical: N = 800 simulations, L_max = 10-20

Training:
- Self-play: expensive
- Network update: O(b · L · h²)

Computational intensive: requires TPUs/GPUs
```

---

## Comparison Tables

### Classical ML Algorithms

```
Algorithm          | Training         | Prediction    | Space     | Notes
-------------------|------------------|---------------|-----------|------------------------
Linear Regression  | O(nd² + d³)     | O(d)         | O(nd)     | Fast, interpretable
Logistic Reg (GD)  | O(T·nd)         | O(d)         | O(nd)     | Scalable
SVM (Linear)       | O(T·nd)         | O(d)         | O(d)      | Large-scale friendly
SVM (Kernel)       | O(n² ·d)        | O(s·d)       | O(n²)     | Small to medium data
Decision Tree      | O(n·d·log n)    | O(log n)     | O(n)      | Fast, non-linear
Random Forest      | O(B·n·d·log n)  | O(B·log n)   | O(Bn)     | Best general-purpose
XGBoost            | O(T·nd)         | O(T·L)       | O(nd)     | Best performance
KNN                | O(1)            | O(nd)        | O(nd)     | Lazy learning
Naive Bayes        | O(nd)           | O(kd)        | O(kd)     | Very fast, simple
K-Means            | O(T·nkd)        | O(kd)        | O(nd)     | Fast clustering

where: n=samples, d=features, k=classes/clusters, B=trees, T=iterations, s=support vectors
```

### Deep Learning Models

```
Model               | Parameters    | FLOPs/image  | Memory    | Accuracy (ImageNet)
--------------------|---------------|--------------|-----------|--------------------
AlexNet             | 60M           | 720M         | 240MB     | 57.1%
VGG-16              | 138M          | 15.5B        | 528MB     | 71.3%
ResNet-50           | 25M           | 4.1B         | 200MB     | 76.1%
ResNet-152          | 60M           | 11.3B        | 440MB     | 77.6%
Inception-v3        | 24M           | 5.7B         | 189MB     | 77.5%
EfficientNet-B0     | 5.3M          | 0.39B        | 20MB      | 77.1%
EfficientNet-B7     | 66M           | 37B          | 256MB     | 84.4%
ViT-B/16            | 86M           | 17.6B        | 330MB     | 77.9%
```

### NLP Models

```
Model               | Parameters    | Training      | Inference    | Use Case
--------------------|---------------|---------------|--------------|------------------
Word2Vec            | V·d ≈10M      | Hours         | <1ms         | Embeddings
GloVe               | V·d ≈10M      | Hours         | <1ms         | Embeddings
LSTM (small)        | 10M           | Hours         | ~10ms        | Sequences
BERT-base           | 110M          | Days (TPU)    | ~20ms        | Understanding
BERT-large          | 340M          | Days (TPU)    | ~50ms        | Understanding
GPT-2               | 1.5B          | Weeks         | ~100ms       | Generation
GPT-3               | 175B          | Months        | ~1s          | Few-shot learning
T5-11B              | 11B           | Weeks         | ~500ms       | Seq2seq

Times approximate for single sequence (512 tokens)
```

### Training Time Estimates

```
Dataset Size    | Model Type       | Hardware      | Time
----------------|------------------|---------------|------------------
1K samples      | Linear           | CPU           | Seconds
10K samples     | Random Forest    | CPU           | Minutes
100K samples    | XGBoost          | CPU/GPU       | Minutes to Hours
1M samples      | Deep NN          | GPU           | Hours
10M samples     | ResNet-50        | Multi-GPU     | Days
100M samples    | BERT-large       | TPU           | Weeks
Billions        | GPT-3            | Supercomputer | Months
```

---

## References

### Complexity Analysis

1. **Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C.** (2009). *Introduction to algorithms* (3rd ed.). MIT Press.
   - Fundamental algorithms and complexity theory

2. **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). *Deep Learning*. MIT Press.
   - Deep learning computational costs

### Classical ML

3. **Bishop, C. M.** (2006). *Pattern recognition and machine learning*. Springer.
   - Complexity of classical algorithms

4. **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The elements of statistical learning* (2nd ed.). Springer.
   - Statistical learning computational aspects

### Deep Learning

5. **He, K., Zhang, X., Ren, S., & Sun, J.** (2015). "Delving deep into rectifiers: Surpassing human-level performance on imagenet classification." *ICCV*.
   - ResNet complexity analysis

6. **Vaswani, A., et al.** (2017). "Attention is all you need." *NeurIPS*.
   - Transformer complexity

7. **Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C.** (2022). "FlashAttention: Fast and memory-efficient exact attention with IO-awareness." *NeurIPS*.
   - Efficient attention mechanisms

### Ensemble Methods

8. **Chen, T., & Guestrin, C.** (2016). "XGBoost: A scalable tree boosting system." *KDD*.
   - XGBoost complexity and optimizations

9. **Ke, G., et al.** (2017). "LightGBM: A highly efficient gradient boosting decision tree." *NeurIPS*.
   - LightGBM algorithmic improvements

### Recommendation Systems

10. **Koren, Y., Bell, R., & Volinsky, C.** (2009). "Matrix factorization techniques for recommender systems." *Computer*, 42(8), 30-37.
    - Matrix factorization complexity

11. **Covington, P., Adams, J., & Sargin, E.** (2016). "Deep neural networks for YouTube recommendations." *RecSys*.
    - Large-scale recommendation system complexity

### Reinforcement Learning

12. **Mnih, V., et al.** (2015). "Human-level control through deep reinforcement learning." *Nature*, 518(7540), 529-533.
    - DQN computational requirements

13. **Schulman, J., et al.** (2017). "Proximal policy optimization algorithms." *arXiv:1707.06347*.
    - PPO complexity

---

*This comprehensive reference provides Big-O analysis and practical complexity estimates for modern machine learning algorithms. Use it to make informed decisions about algorithm selection based on computational constraints.*
