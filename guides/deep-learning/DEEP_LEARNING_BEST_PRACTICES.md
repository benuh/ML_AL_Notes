# Deep Learning Best Practices and Tips

**Practical Guide for Training Deep Neural Networks**

> Battle-tested tips, common pitfalls, and production best practices from real-world deep learning projects.

**Last Updated:** October 2025

---

## Table of Contents

1. [Getting Started Right](#1-getting-started-right)
2. [Data Preparation](#2-data-preparation)
3. [Model Architecture](#3-model-architecture)
4. [Training Loop](#4-training-loop)
5. [Debugging and Troubleshooting](#5-debugging-and-troubleshooting)
6. [Performance Optimization](#6-performance-optimization)
7. [Experiment Tracking](#7-experiment-tracking)
8. [Production Deployment](#8-production-deployment)
9. [Hardware Considerations](#9-hardware-considerations)
10. [Common Pitfalls and Solutions](#10-common-pitfalls-and-solutions)

---

## 1. Getting Started Right

### 1.1 Start Simple, Add Complexity Gradually

**The Golden Rule**: Always start with the simplest possible model and incrementally add complexity.

```python
# ❌ DON'T: Start with complex architecture
class ComplexModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TransformerEncoder(...)
        self.attention = MultiHeadAttention(...)
        self.decoder = ComplexDecoder(...)
        # ... 100 more lines

# ✅ DO: Start with simple baseline
class SimpleBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Train simple model first, then gradually add:
# 1. More layers
# 2. Residual connections
# 3. Attention mechanisms
# 4. Advanced features
```

**Why?**
- Quickly identify data/setup issues
- Establish baseline performance
- Understand if complex model is needed
- Easier debugging

### 1.2 Overfit a Single Batch First

**Critical Step**: Before training on full dataset, ensure your model can overfit a single batch.

```python
def test_overfitting_single_batch(model, criterion, num_iterations=1000):
    """
    Test if model can overfit a single batch
    If this fails, there's a fundamental problem!
    """
    # Get single batch
    data, targets = next(iter(train_loader))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    for i in range(num_iterations):
        optimizer.zero_grad()

        outputs = model(data)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            # Compute accuracy
            preds = outputs.argmax(dim=1)
            acc = (preds == targets).float().mean()
            print(f'Iteration {i}: Loss = {loss.item():.4f}, Acc = {acc.item():.4f}')

    # Final accuracy should be 100% (or very close)
    final_acc = (outputs.argmax(dim=1) == targets).float().mean()

    if final_acc < 0.95:
        print("⚠️  WARNING: Model cannot overfit single batch!")
        print("Check: model architecture, loss function, learning rate")
        return False

    print("✅ Model can overfit single batch - proceed with full training")
    return True

# Usage
if not test_overfitting_single_batch(model, criterion):
    # Fix issues before proceeding!
    pass
```

**If this fails, check:**
- Model has enough capacity
- Loss function is correct
- Learning rate is not too low
- No bugs in forward/backward pass

### 1.3 Establish Strong Baselines

**Always compare against:**

```python
class Baselines:
    """Collection of baseline models for comparison"""

    @staticmethod
    def random_baseline(num_classes):
        """Random prediction baseline"""
        def predict(x):
            return torch.randint(0, num_classes, (x.size(0),))
        return predict

    @staticmethod
    def majority_class_baseline(train_labels):
        """Always predict most frequent class"""
        majority_class = train_labels.mode().values.item()

        def predict(x):
            return torch.full((x.size(0),), majority_class)
        return predict

    @staticmethod
    def simple_linear_baseline(input_dim, output_dim):
        """Simple linear model"""
        return nn.Linear(input_dim, output_dim)

    @staticmethod
    def shallow_network_baseline(input_dim, output_dim):
        """Simple 2-layer network"""
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

# Establish baselines
baselines_results = {
    'random': evaluate(Baselines.random_baseline(num_classes), test_loader),
    'majority': evaluate(Baselines.majority_class_baseline(train_labels), test_loader),
    'linear': train_and_evaluate(Baselines.simple_linear_baseline(...), train_loader, test_loader),
    'shallow': train_and_evaluate(Baselines.shallow_network_baseline(...), train_loader, test_loader),
}

print("Baseline Results:")
for name, acc in baselines_results.items():
    print(f"{name}: {acc:.2%}")

# Your model must beat these!
```

---

## 2. Data Preparation

### 2.1 Understand Your Data First

**Before any training, do thorough EDA:**

```python
def analyze_dataset(dataset):
    """Comprehensive dataset analysis"""

    print("=" * 50)
    print("DATASET ANALYSIS")
    print("=" * 50)

    # 1. Basic statistics
    print(f"\nDataset size: {len(dataset)}")

    # 2. Label distribution
    labels = [label for _, label in dataset]
    from collections import Counter
    label_dist = Counter(labels)

    print("\nLabel distribution:")
    for label, count in sorted(label_dist.items()):
        print(f"  Class {label}: {count} ({count/len(labels):.1%})")

    # Check for imbalance
    max_count = max(label_dist.values())
    min_count = min(label_dist.values())
    imbalance_ratio = max_count / min_count

    if imbalance_ratio > 3:
        print(f"\n⚠️  WARNING: Class imbalance detected! Ratio: {imbalance_ratio:.1f}:1")
        print("Consider: class weights, oversampling, or focal loss")

    # 3. Data statistics
    # Get sample batch
    loader = DataLoader(dataset, batch_size=1000, shuffle=True)
    data, _ = next(iter(loader))

    print(f"\nData statistics:")
    print(f"  Shape: {data.shape}")
    print(f"  Mean: {data.mean():.4f}")
    print(f"  Std: {data.std():.4f}")
    print(f"  Min: {data.min():.4f}")
    print(f"  Max: {data.max():.4f}")

    # 4. Check for NaN/Inf
    if torch.isnan(data).any():
        print("\n⚠️  WARNING: NaN values detected in data!")

    if torch.isinf(data).any():
        print("\n⚠️  WARNING: Inf values detected in data!")

    # 5. Visualize samples
    print("\nVisualizing samples...")
    visualize_samples(dataset, num_samples=16)

    print("=" * 50)

# Run before training!
analyze_dataset(train_dataset)
```

### 2.2 Proper Data Normalization

**Critical for convergence:**

```python
class DataNormalizer:
    """Compute and apply normalization statistics"""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, train_loader):
        """
        Compute normalization statistics from training data
        ONLY use training data for statistics!
        """
        print("Computing normalization statistics...")

        mean = 0.
        std = 0.
        total_samples = 0

        for data, _ in train_loader:
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)

            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            total_samples += batch_samples

        mean /= total_samples
        std /= total_samples

        self.mean = mean
        self.std = std

        print(f"Mean: {mean}")
        print(f"Std: {std}")

        return self

    def transform(self, data):
        """Apply normalization"""
        if self.mean is None or self.std is None:
            raise ValueError("Must fit normalizer first!")

        return (data - self.mean.view(1, -1, 1, 1)) / self.std.view(1, -1, 1, 1)

# Usage
normalizer = DataNormalizer()
normalizer.fit(train_loader)

# Apply to all splits
train_dataset.transform = normalizer.transform
val_dataset.transform = normalizer.transform
test_dataset.transform = normalizer.transform

# ❌ DON'T: Fit normalizer on test data
# ❌ DON'T: Use different statistics for train/val/test
```

### 2.3 Data Augmentation Best Practices

```python
class SmartAugmentation:
    """
    Intelligent data augmentation
    Start weak, increase strength gradually
    """

    def __init__(self, stage='weak'):
        """
        stage: 'weak', 'medium', 'strong'
        """
        self.stage = stage
        self.transforms = self._get_transforms()

    def _get_transforms(self):
        """Progressive augmentation difficulty"""

        if self.stage == 'weak':
            # Start with weak augmentation
            return transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
            ])

        elif self.stage == 'medium':
            # Add color jittering
            return transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(32, padding=4),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
            ])

        elif self.stage == 'strong':
            # Add advanced augmentations
            return transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(32, padding=4),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.5),
            ])

    def __call__(self, img):
        return self.transforms(img)

# Progressive training
# 1. Train with weak augmentation
train_dataset.transform = SmartAugmentation('weak')
train(model, num_epochs=50)

# 2. Fine-tune with medium augmentation
train_dataset.transform = SmartAugmentation('medium')
train(model, num_epochs=30)

# 3. Fine-tune with strong augmentation
train_dataset.transform = SmartAugmentation('strong')
train(model, num_epochs=20)
```

**Augmentation Tips:**
- Test augmentations visually first
- Don't use augmentation that changes semantics (e.g., vertical flip for digits)
- More augmentation = need more training time
- Validation/test data should NOT be augmented

### 2.4 Handle Class Imbalance

```python
def handle_class_imbalance(train_dataset, strategy='weighted_loss'):
    """
    Multiple strategies for class imbalance
    """

    # Compute class weights
    labels = [label for _, label in train_dataset]
    from collections import Counter
    class_counts = Counter(labels)
    num_classes = len(class_counts)

    if strategy == 'weighted_loss':
        # Strategy 1: Weighted loss function
        # weight = 1 / count (inversely proportional)
        class_weights = torch.tensor([
            1.0 / class_counts[i] for i in range(num_classes)
        ])

        # Normalize
        class_weights = class_weights / class_weights.sum() * num_classes

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        return criterion

    elif strategy == 'oversampling':
        # Strategy 2: Oversample minority classes
        from torch.utils.data import WeightedRandomSampler

        # Compute sample weights (higher weight = more likely to sample)
        sample_weights = [1.0 / class_counts[label] for _, label in train_dataset]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )

        train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
        return train_loader

    elif strategy == 'focal_loss':
        # Strategy 3: Use focal loss
        class FocalLoss(nn.Module):
            def __init__(self, gamma=2.0, alpha=None):
                super().__init__()
                self.gamma = gamma
                self.alpha = alpha

            def forward(self, inputs, targets):
                ce_loss = F.cross_entropy(inputs, targets, reduction='none')
                pt = torch.exp(-ce_loss)
                focal_loss = ((1 - pt) ** self.gamma) * ce_loss

                if self.alpha is not None:
                    focal_loss = self.alpha[targets] * focal_loss

                return focal_loss.mean()

        return FocalLoss(gamma=2.0)

# Choose strategy
if max_imbalance_ratio > 3:
    criterion = handle_class_imbalance(train_dataset, strategy='weighted_loss')
```

---

## 3. Model Architecture

### 3.1 Weight Initialization Matters

**Rigorous Theory of Weight Initialization:**

```
Problem: Deep networks suffer from vanishing/exploding gradients without proper initialization.

Theorem 1 (Variance Propagation - Glorot & Bengio 2010):
Consider L-layer network with weights W^(ℓ) and activations z^(ℓ) = W^(ℓ)h^(ℓ-1).

Forward pass variance:
Var[z^(ℓ)] = n_(ℓ-1) · Var[W^(ℓ)] · Var[h^(ℓ-1)]

where n_(ℓ-1) = number of input units to layer ℓ.

Backward pass gradient variance:
Var[∂L/∂h^(ℓ)] = n_ℓ · Var[W^(ℓ)] · Var[∂L/∂z^(ℓ)]

where n_ℓ = number of output units from layer ℓ.

Ideal condition (preserve variance):
Forward: n_(ℓ-1) · Var[W^(ℓ)] = 1
Backward: n_ℓ · Var[W^(ℓ)] = 1

Xavier/Glorot Initialization:
Compromise between forward and backward:

Var[W^(ℓ)] = 2/(n_(ℓ-1) + n_ℓ)

For uniform distribution: W ~ U[-a, a]
a = sqrt(6/(n_in + n_out))

For normal distribution: W ~ N(0, σ²)
σ = sqrt(2/(n_in + n_out))
```

**Theorem 2 (He Initialization for ReLU - He et al. 2015):**

```
Problem: ReLU activations have different variance properties.

For ReLU activation: h = max(0, z)
E[h] = E[z]/2  (half of inputs zeroed)
Var[h] = Var[z]/2  (variance halved)

Modified forward variance:
Var[z^(ℓ)] = n_(ℓ-1) · Var[W^(ℓ)] · (Var[h^(ℓ-1)]/2)

To preserve variance through ReLU layers:
n_(ℓ-1) · Var[W^(ℓ)] · (1/2) = 1

He Initialization:
Var[W^(ℓ)] = 2/n_(ℓ-1)  (fan-in)

For normal distribution: W ~ N(0, 2/n_in)
For uniform distribution: W ~ U[-sqrt(6/n_in), sqrt(6/n_in)]

Empirical result: 30% faster convergence than Xavier for ReLU networks!
```

**Theorem 3 (Orthogonal Initialization for RNNs - Saxe et al. 2014):**

```
Problem: Recurrent connections amplify/diminish gradients exponentially.

For RNN: h_t = tanh(W_hh h_(t-1) + W_xh x_t)

Gradient through time steps τ:
∂L/∂h_(t-τ) ∝ (W_hh^T)^τ ∂L/∂h_t

Spectral analysis:
||∂L/∂h_(t-τ)|| ≈ λ_max(W_hh)^τ ||∂L/∂h_t||

where λ_max is largest singular value of W_hh.

Vanishing gradients: λ_max < 1 ⇒ ||∂L/∂h_(t-τ)|| → 0 exponentially
Exploding gradients: λ_max > 1 ⇒ ||∂L/∂h_(t-τ)|| → ∞ exponentially

Orthogonal Initialization:
Initialize W_hh as orthogonal matrix: W_hh W_hh^T = I

Property: All singular values = 1
⇒ λ_max = 1 ⇒ Gradient norm preserved!

Practical: Use QR decomposition or SVD to construct orthogonal matrix.
```

**Theorem 4 (LSUV - Layer-Sequential Unit-Variance Initialization):**

```
Algorithm:
1. Initialize with standard method (Xavier/He)
2. For each layer ℓ = 1, ..., L:
   a. Pass mini-batch through network up to layer ℓ
   b. Compute Var[z^(ℓ)]
   c. Rescale: W^(ℓ) ← W^(ℓ) / sqrt(Var[z^(ℓ)])
   d. Ensures Var[z^(ℓ)] = 1 exactly

Convergence guarantee (Mishkin & Matas 2016):
After LSUV, for all layers ℓ:
|Var[z^(ℓ)] - 1| < ε

where ε ≈ 0.01 (1% error) after 5-10 iterations.

Benefit: Provably stable gradients from initialization!
Cost: Requires one forward pass through training batch.
```

**Comparison Table:**

```
| Method | Activation | Var[W] | Best For | Convergence Speed |
|--------|------------|--------|----------|-------------------|
| Xavier | Tanh/Sigmoid | 2/(n_in+n_out) | Symmetric activations | Baseline |
| He | ReLU/LeakyReLU | 2/n_in | ReLU networks | +30% |
| Orthogonal | Any | λ=1 (orthogonal) | RNNs/LSTMs | +50% (RNN) |
| LSUV | Any | Adaptive | Deep networks (>20 layers) | +40% |

Empirical Results (ImageNet):
- Random init: diverges
- Xavier: 75% top-1 after 90 epochs
- He: 76.5% top-1 after 90 epochs (+1.5%)
- LSUV: 76.8% top-1 after 90 epochs (+1.8%)
```

```python
def initialize_weights(model):
    """
    Proper weight initialization prevents vanishing/exploding gradients
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # He initialization for ReLU activations
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            # He initialization for ReLU
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            # BatchNorm: gamma=1, beta=0
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LSTM):
            # LSTM initialization
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0)

# Apply initialization
model = MyModel()
initialize_weights(model)

# Check initial gradients
def check_initial_gradients(model, sample_batch):
    """Verify gradients are reasonable after initialization"""
    data, targets = sample_batch
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    optimizer.zero_grad()
    outputs = model(data)
    loss = F.cross_entropy(outputs, targets)
    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"{name}: grad_norm = {grad_norm:.6f}")

            if grad_norm < 1e-7:
                print(f"  ⚠️  Very small gradient - might have vanishing gradient")
            elif grad_norm > 100:
                print(f"  ⚠️  Very large gradient - might have exploding gradient")

check_initial_gradients(model, sample_batch)
```

### 3.2 Batch Normalization Best Practices

```python
class BatchNormRules:
    """Best practices for BatchNorm"""

    # ✅ DO: Place BN after Conv/Linear, before activation
    correct_order = nn.Sequential(
        nn.Conv2d(3, 64, 3),
        nn.BatchNorm2d(64),
        nn.ReLU()
    )

    # ❌ DON'T: Place BN after activation
    wrong_order = nn.Sequential(
        nn.Conv2d(3, 64, 3),
        nn.ReLU(),
        nn.BatchNorm2d(64)  # Wrong!
    )

    # ⚠️  Special case: With residual connections
    residual_block = lambda in_ch, out_ch: nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        # No activation here - added after residual addition
    )

# BatchNorm requirements
def check_batchnorm_requirements(batch_size):
    """
    BatchNorm needs sufficient batch size for good statistics
    """
    if batch_size < 16:
        print("⚠️  WARNING: Batch size too small for BatchNorm")
        print("Consider: GroupNorm or LayerNorm instead")
        return False

    if batch_size < 32:
        print("⚠️  Batch size < 32 may hurt BatchNorm performance")
        print("Consider increasing batch size or using GroupNorm")

    return True

# Alternative normalization for small batches
def get_normalization_layer(num_channels, batch_size):
    """Choose normalization based on batch size"""

    if batch_size >= 32:
        # Use BatchNorm
        return nn.BatchNorm2d(num_channels)

    elif batch_size >= 8:
        # Use GroupNorm (groups=8 typically)
        num_groups = min(8, num_channels)
        return nn.GroupNorm(num_groups, num_channels)

    else:
        # Use LayerNorm for very small batches
        return nn.GroupNorm(1, num_channels)  # Equivalent to LayerNorm
```

### 3.3 Dropout Usage

```python
class DropoutBestPractices:
    """When and how to use dropout"""

    # ✅ DO: Use dropout in fully connected layers
    good_fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 10)
    )

    # ⚠️  CAREFUL: Dropout in CNNs
    # Use lower dropout rate, or use after pooling
    good_cnn = nn.Sequential(
        nn.Conv2d(3, 64, 3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout2d(0.1),  # Spatial dropout, lower rate
    )

    # ✅ DO: Progressive dropout rates
    progressive_dropout = nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.3),   # Lower rate early
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),   # Higher rate later
        nn.Linear(256, 10)
    )

# Dropout rate tuning
def tune_dropout_rate(model_fn, rates=[0.1, 0.3, 0.5, 0.7]):
    """
    Find optimal dropout rate
    """
    results = {}

    for rate in rates:
        print(f"\nTesting dropout rate: {rate}")
        model = model_fn(dropout_rate=rate)
        val_acc = train_and_validate(model, train_loader, val_loader, num_epochs=20)
        results[rate] = val_acc

    best_rate = max(results, key=results.get)
    print(f"\nBest dropout rate: {best_rate} (val_acc: {results[best_rate]:.2%})")

    return best_rate

# Remember: model.eval() disables dropout!
model.eval()  # Dropout disabled for inference
with torch.no_grad():
    predictions = model(test_data)
```

---

## 4. Training Loop

### 4.1 Learning Rate is Most Important

**Rigorous Theory of Learning Rate Schedules:**

```
Theorem 5 (Learning Rate and Convergence - Robbins & Monro 1951):
For stochastic gradient descent to converge, learning rate α_t must satisfy:

Σ_{t=1}^∞ α_t = ∞  (sufficient steps)
Σ_{t=1}^∞ α_t² < ∞  (decreasing noise)

Classic schedule: α_t = α_0 / (1 + γt)

Guarantees convergence to local minimum for non-convex functions.
```

**Theorem 6 (Optimal Constant Learning Rate - Polyak & Juditsky 1992):**

```
For strongly convex loss with condition number κ = L/μ:

Optimal constant learning rate:
α* = 2/(μ + L) = 2/(L(1 + 1/κ))

Convergence rate:
||x_t - x*||² ≤ ((κ-1)/(κ+1))^t ||x_0 - x*||²

Practical implication:
- κ = 10: Need ~22 iterations for 10× error reduction
- κ = 100: Need ~220 iterations for 10× error reduction
- κ = 1000: Need ~2200 iterations (very slow!)

Solution: Adaptive methods or learning rate schedules
```

**Theorem 7 (Cyclical Learning Rates - Smith 2017):**

```
CLR oscillates between bounds [α_min, α_max]:

α_t = α_min + (α_max - α_min) · triangle_wave(t)

Theoretical justification:
1. Escape saddle points: High LR helps jump out of poor local minima
2. Refine solution: Low LR fine-tunes near good minimum
3. Faster convergence: Average across multiple basins

Empirical result (ResNet-56 CIFAR-10):
- Fixed LR: 93.1% accuracy after 350 epochs
- CLR: 93.1% accuracy after 150 epochs (2.3× faster!)
- Bonus: Better generalization (+0.2% test accuracy)

Optimal cycle length:
c = 2-8 epochs (typical: 4-6 epochs)
```

**Theorem 8 (1cycle Policy - Smith & Topin 2019):**

```
Super-convergence phenomenon:
Train with very high learning rate for brief period → much faster convergence!

1cycle schedule:
Phase 1 (0-45% iterations): α increases linearly from α_min to α_max
Phase 2 (45-90% iterations): α decreases linearly from α_max to α_min
Phase 3 (90-100% iterations): α decreases to α_min/10 (final refinement)

Momentum schedule (inverse):
Phase 1: β decreases from 0.95 to 0.85
Phase 2: β increases from 0.85 to 0.95
Phase 3: β = 0.95

Key insight: α_max can be 10× higher than traditional!

Convergence bound (Smith & Le 2018):
With 1cycle and batch size B:

Iterations to ε-accuracy: O(√(κ/B) · log(1/ε))

vs Standard SGD: O(κ/B · log(1/ε))

Speedup: O(√κ) for ill-conditioned problems!

Example:
κ = 1000: Super-convergence is ~30× faster than standard SGD
```

**Theorem 9 (Cosine Annealing - Loshchilov & Hutter 2017):**

```
Cosine schedule:
α_t = α_min + (1/2)(α_max - α_min)(1 + cos(πt/T))

where T = total iterations.

Properties:
1. Smooth decay (no sudden drops)
2. Accelerates near end: dα/dt|_(t=T) = -∞
3. SGDR (Warm Restarts): Reset schedule periodically

SGDR with warm restarts:
After T_i iterations, reset α ← α_max

T_i = T_0 · T_mult^i

Example: T_0 = 10, T_mult = 2
Restarts at: 10, 30, 70, 150, ...  epochs

Theoretical advantage (Loshchilov & Hutter):
Multiple restarts → explore multiple basins → better generalization

Empirical: +0.5-1% test accuracy vs monotonic decay
```

**Theorem 10 (Linear Warmup - Goyal et al. 2017):**

```
Problem: Large batch training with high initial LR causes divergence.

Warmup schedule:
For t ≤ t_warmup:
α_t = t/t_warmup · α_target

Theoretical justification:
Large batch ⇒ Low gradient variance ⇒ Can use large LR
But: Variance high initially (random weights)

Warmup duration:
t_warmup ≈ 5-10 epochs typically

For very large batch (B > 8K):
t_warmup = O(√B) epochs

Convergence guarantee (Ma & Yarats 2019):
With warmup + high LR for batch size B:

E[||∇L(x_t)||²] ≤ O(1/(t·√B))

vs without warmup: divergence when B > critical threshold!

Practical: Warmup essential for batch size > 1024
```

**LR Schedule Comparison:**

```
| Schedule | Convergence Rate | Total Iterations | Generalization | Complexity |
|----------|------------------|------------------|----------------|------------|
| Constant | O((κ-1)/(κ+1))^t | O(κ log(1/ε)) | Baseline | Low |
| Step Decay | O(1/√t) | O(1/ε²) | +0% | Low |
| Exponential | O(exp(-μt)) | O(log(1/ε)) | -0.5% (overfit) | Low |
| Polynomial | O(1/t) | O(1/ε) | +0.2% | Low |
| Cosine | O(1/t) | O(1/ε) | +0.5% | Low |
| Cyclical | O(1/√t) | O(1/ε²) | +0.3% | Medium |
| 1cycle | O(1/(√t·√κ)) | O(√κ/ε) | +0.5% | Medium |
| SGDR | O(1/t) | O(1/ε) | +1.0% | High |

Recommendation by problem:
- Convex: Polynomial or Exponential
- Non-convex (κ < 100): 1cycle
- Non-convex (κ > 100): SGDR with cosine
- Large batch (B > 1K): Warmup + any of above
```

```python
class LearningRateManager:
    """Comprehensive LR management"""

    @staticmethod
    def find_optimal_lr(model, train_loader, criterion, init_lr=1e-8, final_lr=10):
        """
        LR Range Test (Leslie Smith's method)
        Plot loss vs LR, pick LR where loss decreases fastest
        """
        num_batches = len(train_loader)
        mult = (final_lr / init_lr) ** (1 / num_batches)

        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=mult)

        lrs = []
        losses = []

        model.train()

        for batch_idx, (data, targets) in enumerate(train_loader):
            optimizer.zero_grad()

            outputs = model(data)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
            scheduler.step()

            lrs.append(optimizer.param_groups[0]['lr'])
            losses.append(loss.item())

            # Stop if loss explodes
            if loss.item() > 4 * min(losses):
                break

        # Plot
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(lrs, losses)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('LR Range Test')
        plt.grid(True)
        plt.show()

        # Suggest LR (where loss decreases fastest)
        # Typically pick 10x before minimum
        min_idx = losses.index(min(losses))
        suggested_lr = lrs[max(0, min_idx - len(lrs) // 10)]

        print(f"Suggested LR: {suggested_lr:.2e}")

        return suggested_lr, lrs, losses

    @staticmethod
    def get_lr_scheduler(optimizer, strategy='onecycle', **kwargs):
        """
        Choose LR scheduler strategy
        """
        if strategy == 'onecycle':
            # Best for most cases
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=kwargs.get('max_lr', 0.1),
                epochs=kwargs.get('epochs', 100),
                steps_per_epoch=kwargs.get('steps_per_epoch', 100)
            )

        elif strategy == 'cosine':
            # Good for long training
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=kwargs.get('epochs', 100),
                eta_min=kwargs.get('min_lr', 1e-6)
            )

        elif strategy == 'reduce_on_plateau':
            # Conservative, safe choice
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.1,
                patience=10,
                verbose=True
            )

        elif strategy == 'step':
            # Simple step decay
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1
            )

# Usage
lr_manager = LearningRateManager()

# 1. Find optimal LR
optimal_lr, lrs, losses = lr_manager.find_optimal_lr(model, train_loader, criterion)

# 2. Setup optimizer with found LR
optimizer = torch.optim.Adam(model.parameters(), lr=optimal_lr)

# 3. Setup scheduler
scheduler = lr_manager.get_lr_scheduler(
    optimizer,
    strategy='onecycle',
    max_lr=optimal_lr,
    epochs=100,
    steps_per_epoch=len(train_loader)
)
```

### 4.2 Proper Training Loop

```python
class Trainer:
    """
    Production-ready training loop with all best practices
    """

    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler,
                 device='cuda', mixed_precision=True, gradient_clip=1.0):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.mixed_precision = mixed_precision
        self.gradient_clip = gradient_clip

        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None

        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.val_accs = []
        self.best_val_acc = 0.0

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()

        running_loss = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, (data, targets) in enumerate(self.train_loader):
            data, targets = data.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            # Mixed precision training
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(data)
                    loss = self.criterion(outputs, targets)

                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.gradient_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)

                loss.backward()

                # Gradient clipping
                if self.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

                self.optimizer.step()

            # Update scheduler (if per-batch)
            if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()

            running_loss += loss.item()

            # Progress
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch} [{batch_idx}/{num_batches}] Loss: {loss.item():.4f}')

        avg_loss = running_loss / num_batches
        self.train_losses.append(avg_loss)

        return avg_loss

    def validate(self, epoch):
        """Validate model"""
        self.model.eval()

        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device)

                outputs = self.model(data)
                loss = self.criterion(outputs, targets)

                val_loss += loss.item()

                # Compute accuracy
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        avg_val_loss = val_loss / len(self.val_loader)
        val_acc = correct / total

        self.val_losses.append(avg_val_loss)
        self.val_accs.append(val_acc)

        print(f'Epoch {epoch} Validation: Loss = {avg_val_loss:.4f}, Acc = {val_acc:.4f}')

        # Save best model
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'best_model.pth')
            print(f'✅ Saved best model (val_acc: {val_acc:.4f})')

        return avg_val_loss, val_acc

    def train(self, num_epochs):
        """Full training loop"""
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Mixed precision: {self.mixed_precision}")
        print(f"Gradient clipping: {self.gradient_clip}")
        print()

        for epoch in range(1, num_epochs + 1):
            print(f"{'=' * 60}")
            print(f"EPOCH {epoch}/{num_epochs}")
            print(f"{'=' * 60}")

            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss, val_acc = self.validate(epoch)

            # Update scheduler (if per-epoch)
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            elif not isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()

            # Current LR
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Current LR: {current_lr:.2e}")
            print()

        print(f"Training complete! Best val_acc: {self.best_val_acc:.4f}")

        # Plot training curves
        self.plot_curves()

    def plot_curves(self):
        """Plot training/validation curves"""
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        # Accuracy curve
        ax2.plot(self.val_accs, label='Val Accuracy')
        ax2.axhline(y=self.best_val_acc, color='r', linestyle='--',
                    label=f'Best ({self.best_val_acc:.4f})')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=150)
        plt.show()

# Usage
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    device='cuda',
    mixed_precision=True,
    gradient_clip=1.0
)

trainer.train(num_epochs=100)
```

---

## 5. Debugging and Troubleshooting

### 5.1 Loss Not Decreasing

```python
class LossDebugger:
    """Systematic debugging when loss doesn't decrease"""

    @staticmethod
    def diagnose(model, train_loader, criterion, optimizer):
        """
        Step-by-step diagnosis
        """
        print("=" * 60)
        print("LOSS DEBUGGING")
        print("=" * 60)

        # Get sample batch
        data, targets = next(iter(train_loader))

        # 1. Check if model outputs make sense
        print("\n1. Checking model outputs...")
        model.eval()
        with torch.no_grad():
            outputs = model(data[:4])
            print(f"Output shape: {outputs.shape}")
            print(f"Output sample:\n{outputs[0]}")
            print(f"Output range: [{outputs.min():.4f}, {outputs.max():.4f}]")

            if torch.isnan(outputs).any():
                print("❌ NaN in outputs!")
                return "NaN in outputs - check model/data"

        # 2. Check loss computation
        print("\n2. Checking loss computation...")
        loss = criterion(outputs, targets[:4])
        print(f"Loss: {loss.item():.4f}")

        if torch.isnan(loss):
            print("❌ NaN loss!")
            return "NaN loss - check criterion"

        # 3. Check gradients
        print("\n3. Checking gradients...")
        model.train()
        optimizer.zero_grad()

        outputs = model(data[:4])
        loss = criterion(outputs, targets[:4])
        loss.backward()

        grad_norms = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms[name] = grad_norm

                if grad_norm < 1e-7:
                    print(f"⚠️  Very small gradient in {name}: {grad_norm:.2e}")
                elif grad_norm > 100:
                    print(f"⚠️  Very large gradient in {name}: {grad_norm:.2e}")

        # 4. Check learning rate
        print("\n4. Checking learning rate...")
        lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {lr:.2e}")

        if lr < 1e-5:
            print("⚠️  Learning rate might be too small")
        elif lr > 1e-1:
            print("⚠️  Learning rate might be too large")

        # 5. Test optimization step
        print("\n5. Testing optimization step...")
        initial_loss = loss.item()

        optimizer.step()

        outputs = model(data[:4])
        new_loss = criterion(outputs, targets[:4]).item()

        print(f"Loss before: {initial_loss:.4f}")
        print(f"Loss after:  {new_loss:.4f}")
        print(f"Change:      {new_loss - initial_loss:.4f}")

        if new_loss >= initial_loss:
            print("⚠️  Loss not decreasing after optimization step!")

        print("\n" + "=" * 60)

        return grad_norms

# Run diagnostics
LossDebugger.diagnose(model, train_loader, criterion, optimizer)
```

**Common Solutions:**

```python
# Problem 1: Loss oscillates wildly
# Solution: Reduce learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Instead of 1e-2

# Problem 2: Loss stuck at random baseline (e.g., log(num_classes))
# Solution: Check model has enough capacity, or learning rate too low
model = LargerModel()  # More parameters
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Higher LR

# Problem 3: Loss becomes NaN
# Solution: Gradient clipping + lower learning rate
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Problem 4: Loss decreases then increases
# Solution: Overfitting - add regularization
model.dropout = 0.5  # Increase dropout
criterion = nn.CrossEntropyLoss(weight_decay=1e-4)  # Add weight decay

# Problem 5: Validation loss increases while training loss decreases
# Solution: Overfitting - early stopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1)
```

### 5.2 Model Performance Plateaus

```python
def diagnose_plateau(train_losses, val_losses, val_accs, patience=10):
    """
    Diagnose why model performance plateaued
    """
    print("Analyzing performance plateau...")

    # Check if truly plateaued
    recent_val_accs = val_accs[-patience:]
    std = np.std(recent_val_accs)

    if std < 0.001:  # Very little improvement
        print(f"✅ Confirmed plateau (std: {std:.6f})")

        # 1. Check train-val gap
        train_acc_estimate = 1 - train_losses[-1]  # Rough estimate
        val_acc = val_accs[-1]
        gap = abs(train_acc_estimate - val_acc)

        if gap > 0.1:
            print("\n📊 Large train-val gap detected")
            print("Likely cause: Overfitting")
            print("Solutions:")
            print("  - Add dropout/regularization")
            print("  - Increase data augmentation")
            print("  - Collect more data")
            print("  - Reduce model complexity")

        else:
            print("\n📊 Small train-val gap")
            print("Likely cause: Underfitting / Model capacity limit")
            print("Solutions:")
            print("  - Increase model capacity")
            print("  - Train longer")
            print("  - Adjust learning rate")
            print("  - Check if data quality is limiting factor")

        # 2. Check loss curves
        recent_train_loss_change = train_losses[-1] - train_losses[-patience]
        recent_val_loss_change = val_losses[-1] - val_losses[-patience]

        print(f"\nRecent train loss change: {recent_train_loss_change:.6f}")
        print(f"Recent val loss change: {recent_val_loss_change:.6f}")

        if recent_train_loss_change < -0.01:  # Still decreasing
            print("Train loss still decreasing - continue training")
        else:
            print("Train loss plateaued - may need architecture changes")

    else:
        print(f"Performance still improving (std: {std:.6f})")
        print("Continue training!")

# Run diagnosis
diagnose_plateau(trainer.train_losses, trainer.val_losses, trainer.val_accs)
```

---

## 6. Performance Optimization

### 6.1 Speed Up Training

```python
class PerformanceOptimizer:
    """
    Optimize training speed without sacrificing accuracy
    """

    @staticmethod
    def enable_cudnn_benchmark():
        """
        Enable cuDNN auto-tuner
        ~10-20% speedup for fixed input sizes
        """
        torch.backends.cudnn.benchmark = True
        print("✅ Enabled cuDNN benchmark")

    @staticmethod
    def pin_memory_in_dataloader(train_loader):
        """
        Pin memory for faster CPU-GPU transfer
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            pin_memory=True,  # Enable this!
            num_workers=4
        )
        print("✅ Enabled pin_memory")
        return train_loader

    @staticmethod
    def optimize_num_workers(dataset, batch_size):
        """
        Find optimal number of DataLoader workers
        """
        import time

        print("Finding optimal num_workers...")

        best_time = float('inf')
        best_workers = 0

        for num_workers in [0, 2, 4, 8, 12, 16]:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True
            )

            start = time.time()

            # Time loading 100 batches
            for i, (data, _) in enumerate(loader):
                if i >= 100:
                    break

            elapsed = time.time() - start

            print(f"num_workers={num_workers}: {elapsed:.2f}s")

            if elapsed < best_time:
                best_time = elapsed
                best_workers = num_workers

        print(f"\n✅ Optimal num_workers: {best_workers}")
        return best_workers

    @staticmethod
    def use_mixed_precision():
        """
        Mixed precision training (2-3x speedup, 50% memory)
        """
        scaler = torch.cuda.amp.GradScaler()

        for data, targets in train_loader:
            optimizer.zero_grad()

            # Autocast for mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(data)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        print("✅ Using mixed precision training")

    @staticmethod
    def gradient_accumulation(accumulation_steps=4):
        """
        Simulate larger batch size
        Useful when GPU memory is limited
        """
        optimizer.zero_grad()

        for i, (data, targets) in enumerate(train_loader):
            outputs = model(data)
            loss = criterion(outputs, targets) / accumulation_steps

            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        print(f"✅ Using gradient accumulation (effective batch size: "
              f"{batch_size * accumulation_steps})")

# Apply optimizations
optimizer = PerformanceOptimizer()

optimizer.enable_cudnn_benchmark()
optimal_workers = optimizer.optimize_num_workers(train_dataset, batch_size=32)
train_loader = optimizer.pin_memory_in_dataloader(train_loader)
```

### 6.2 Reduce Memory Usage

```python
class MemoryOptimizer:
    """
    Reduce GPU memory usage
    """

    @staticmethod
    def use_gradient_checkpointing(model):
        """
        Trade compute for memory
        Reduces memory by ~50% for deep models
        """
        from torch.utils.checkpoint import checkpoint

        class CheckpointedModel(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                # Checkpoint every 2 layers
                for i, layer in enumerate(self.model.children()):
                    if i % 2 == 0:
                        x = checkpoint(layer, x)
                    else:
                        x = layer(x)
                return x

        return CheckpointedModel(model)

    @staticmethod
    def reduce_batch_size_with_accumulation(original_batch_size, gpu_limit):
        """
        Maintain effective batch size while reducing memory
        """
        new_batch_size = gpu_limit
        accumulation_steps = original_batch_size // new_batch_size

        print(f"Original batch size: {original_batch_size}")
        print(f"New batch size: {new_batch_size}")
        print(f"Accumulation steps: {accumulation_steps}")

        return new_batch_size, accumulation_steps

    @staticmethod
    def optimize_model_for_inference(model):
        """
        Reduce memory for inference
        """
        model.eval()

        # Fuse Conv+BN+ReLU
        model_fused = torch.quantization.fuse_modules(
            model,
            [['conv', 'bn', 'relu']]
        )

        # Half precision (FP16)
        model_half = model.half()

        print("✅ Model optimized for inference")

        return model_half

# Example: Check memory usage
def check_memory_usage():
    """Monitor GPU memory"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3

        print(f"GPU Memory: {allocated:.2f} GB allocated, {cached:.2f} GB cached")

# Monitor during training
for epoch in range(num_epochs):
    for data, targets in train_loader:
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            check_memory_usage()
```

---

## 7. Experiment Tracking

### 7.1 Organize Experiments

```python
import hashlib
import json
from pathlib import Path

class ExperimentTracker:
    """
    Track experiments systematically
    """

    def __init__(self, base_dir='experiments'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

    def create_experiment(self, config):
        """
        Create new experiment with unique ID
        """
        # Generate unique ID from config
        config_str = json.dumps(config, sort_keys=True)
        exp_id = hashlib.md5(config_str.encode()).hexdigest()[:8]

        # Create experiment directory
        exp_dir = self.base_dir / exp_id
        exp_dir.mkdir(exist_ok=True)

        # Save config
        with open(exp_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Created experiment: {exp_id}")
        print(f"Directory: {exp_dir}")

        return exp_id, exp_dir

    def log_metrics(self, exp_id, metrics, step):
        """Log metrics for experiment"""
        exp_dir = self.base_dir / exp_id

        # Append to metrics file
        with open(exp_dir / 'metrics.jsonl', 'a') as f:
            log_entry = {'step': step, **metrics}
            f.write(json.dumps(log_entry) + '\n')

    def save_checkpoint(self, exp_id, model, optimizer, epoch, metrics):
        """Save model checkpoint"""
        exp_dir = self.base_dir / exp_id

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }

        torch.save(checkpoint, exp_dir / f'checkpoint_epoch_{epoch}.pth')

        # Also save as latest
        torch.save(checkpoint, exp_dir / 'latest.pth')

    def load_experiment(self, exp_id):
        """Load experiment"""
        exp_dir = self.base_dir / exp_id

        # Load config
        with open(exp_dir / 'config.json', 'r') as f:
            config = json.load(f)

        # Load metrics
        metrics = []
        with open(exp_dir / 'metrics.jsonl', 'r') as f:
            for line in f:
                metrics.append(json.loads(line))

        return config, metrics

    def compare_experiments(self, exp_ids, metric='val_acc'):
        """Compare multiple experiments"""
        import pandas as pd

        results = []

        for exp_id in exp_ids:
            config, metrics = self.load_experiment(exp_id)

            # Get best metric
            best = max([m[metric] for m in metrics if metric in m])

            results.append({
                'exp_id': exp_id,
                'best_' + metric: best,
                **config
            })

        df = pd.DataFrame(results)
        print(df.sort_values('best_' + metric, ascending=False))

        return df

# Usage
tracker = ExperimentTracker()

# Create experiment
config = {
    'model': 'ResNet18',
    'lr': 0.001,
    'batch_size': 32,
    'optimizer': 'Adam',
    'augmentation': 'standard'
}

exp_id, exp_dir = tracker.create_experiment(config)

# Training loop
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader)
    val_loss, val_acc = validate(model, val_loader)

    # Log metrics
    tracker.log_metrics(exp_id, {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_acc': val_acc
    }, step=epoch)

    # Save checkpoint
    if val_acc > best_val_acc:
        tracker.save_checkpoint(exp_id, model, optimizer, epoch, {
            'val_acc': val_acc,
            'val_loss': val_loss
        })
```

### 7.2 Use Weights & Biases

```python
import wandb

class WandbTracker:
    """
    Integration with Weights & Biases
    """

    def __init__(self, project_name, config):
        """Initialize wandb"""
        wandb.init(project=project_name, config=config)

        # Log model architecture
        wandb.watch(model)

    def log_metrics(self, metrics, step=None):
        """Log metrics"""
        wandb.log(metrics, step=step)

    def log_images(self, images, caption):
        """Log images"""
        wandb.log({caption: [wandb.Image(img) for img in images]})

    def log_confusion_matrix(self, y_true, y_pred, class_names):
        """Log confusion matrix"""
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true,
                preds=y_pred,
                class_names=class_names
            )
        })

    def finish(self):
        """Finish wandb run"""
        wandb.finish()

# Usage
config = {
    'model': 'ResNet18',
    'lr': 0.001,
    'batch_size': 32,
}

tracker = WandbTracker('my-project', config)

for epoch in range(num_epochs):
    train_loss = train_epoch()
    val_loss, val_acc = validate()

    tracker.log_metrics({
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'epoch': epoch
    })

tracker.finish()
```

---

## 8. Production Deployment

### 8.1 Model Export

```python
class ModelExporter:
    """Export models for production"""

    @staticmethod
    def export_torchscript(model, example_input, save_path='model.pt'):
        """
        Export to TorchScript (recommended for production)
        """
        model.eval()

        # Trace model
        traced_model = torch.jit.trace(model, example_input)

        # Save
        traced_model.save(save_path)

        print(f"✅ Exported TorchScript model to {save_path}")

        # Test loading
        loaded_model = torch.jit.load(save_path)
        with torch.no_grad():
            output_original = model(example_input)
            output_loaded = loaded_model(example_input)

            if torch.allclose(output_original, output_loaded):
                print("✅ Model outputs match")
            else:
                print("⚠️  Model outputs don't match!")

        return traced_model

    @staticmethod
    def export_onnx(model, example_input, save_path='model.onnx'):
        """
        Export to ONNX (for cross-framework compatibility)
        """
        model.eval()

        torch.onnx.export(
            model,
            example_input,
            save_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

        print(f"✅ Exported ONNX model to {save_path}")

        # Verify
        import onnx
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model is valid")

    @staticmethod
    def quantize_model(model, calibration_loader):
        """
        Quantize model for faster inference (4x smaller, 2-4x faster)
        """
        model.eval()

        # Post-training quantization
        model_quantized = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )

        print("✅ Model quantized")

        # Compare sizes
        torch.save(model.state_dict(), 'model_fp32.pth')
        torch.save(model_quantized.state_dict(), 'model_int8.pth')

        import os
        size_fp32 = os.path.getsize('model_fp32.pth') / 1e6
        size_int8 = os.path.getsize('model_int8.pth') / 1e6

        print(f"FP32 size: {size_fp32:.2f} MB")
        print(f"INT8 size: {size_int8:.2f} MB")
        print(f"Compression: {size_fp32/size_int8:.2f}x")

        return model_quantized

# Export model
exporter = ModelExporter()

example_input = torch.randn(1, 3, 224, 224)

# TorchScript (recommended)
traced_model = exporter.export_torchscript(model, example_input)

# ONNX (for deployment to other frameworks)
exporter.export_onnx(model, example_input)

# Quantization (for mobile/edge devices)
model_quantized = exporter.quantize_model(model, calibration_loader)
```

---

## 9. Hardware Considerations

### 9.1 GPU Utilization

```python
def monitor_gpu_utilization():
    """
    Monitor GPU utilization during training
    """
    import subprocess
    import time
    from threading import Thread

    def log_gpu_stats():
        while True:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                 '--format=csv,noheader,nounits'],
                stdout=subprocess.PIPE
            )

            output = result.stdout.decode('utf-8').strip()
            gpu_util, mem_used = output.split(', ')

            print(f"GPU Util: {gpu_util}%, Memory: {mem_used} MB")

            time.sleep(5)  # Log every 5 seconds

    # Start monitoring in background
    monitor_thread = Thread(target=log_gpu_stats, daemon=True)
    monitor_thread.start()

# Start monitoring
monitor_gpu_utilization()

# If GPU utilization < 80%, bottleneck is likely:
# - Data loading (increase num_workers)
# - CPU preprocessing (optimize transforms)
# - Batch size too small (increase if memory allows)
```

---

## 10. Common Pitfalls and Solutions

### 10.1 Quick Reference

```python
# ❌ PITFALL 1: Not setting model.eval() during evaluation
model.train()  # Training mode - dropout/batchnorm active
predictions = model(test_data)  # WRONG! Dropout is active

# ✅ SOLUTION:
model.eval()  # Evaluation mode - dropout disabled
with torch.no_grad():  # Don't track gradients
    predictions = model(test_data)

# ❌ PITFALL 2: Using softmax before CrossEntropyLoss
output = F.softmax(logits, dim=1)  # Don't do this!
loss = nn.CrossEntropyLoss()(output, targets)  # WRONG!

# ✅ SOLUTION: CrossEntropyLoss includes softmax
loss = nn.CrossEntropyLoss()(logits, targets)  # Correct!

# ❌ PITFALL 3: Forgetting to zero gradients
for data, targets in train_loader:
    outputs = model(data)
    loss = criterion(outputs, targets)
    loss.backward()  # Gradients accumulate!
    optimizer.step()  # WRONG ORDER!

# ✅ SOLUTION:
for data, targets in train_loader:
    optimizer.zero_grad()  # Clear gradients first
    outputs = model(data)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

# ❌ PITFALL 4: Data leakage (normalizing before split)
# Normalize all data
normalized_data = (data - data.mean()) / data.std()
# Then split
train, test = split(normalized_data)  # WRONG!

# ✅ SOLUTION: Split first, then normalize using only train statistics
train, test = split(data)
mean, std = train.mean(), train.std()
train_normalized = (train - mean) / std
test_normalized = (test - mean) / std  # Use train statistics!

# ❌ PITFALL 5: Too high learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Too high!
# Loss will oscillate or explode

# ✅ SOLUTION: Use LR finder or start with conservative LR
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Safe default

# ❌ PITFALL 6: Not using data augmentation
transform = transforms.ToTensor()  # Only this!

# ✅ SOLUTION: Add augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
])

# ❌ PITFALL 7: Batch size too small with BatchNorm
train_loader = DataLoader(dataset, batch_size=4)  # Too small!
model = MyModelWithBatchNorm()  # Won't work well

# ✅ SOLUTION: Use larger batch size or different normalization
train_loader = DataLoader(dataset, batch_size=32)  # Better
# Or use GroupNorm instead of BatchNorm

# ❌ PITFALL 8: Not shuffling training data
train_loader = DataLoader(dataset, batch_size=32, shuffle=False)  # WRONG!

# ✅ SOLUTION:
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)  # Correct!

# ❌ PITFALL 9: Not using learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# LR stays constant forever

# ✅ SOLUTION: Use scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
# Or OneCycleLR, CosineAnnealing, etc.

# ❌ PITFALL 10: Training too long without validation
for epoch in range(1000):  # Train forever without checking
    train(model)

# ✅ SOLUTION: Regular validation with early stopping
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(1000):
    train_loss = train(model)
    val_loss = validate(model)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        save_checkpoint(model)
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping!")
        break
```

---

## Summary Checklist

**Before Training:**
- [ ] Analyze dataset (label distribution, statistics)
- [ ] Visualize data samples
- [ ] Establish strong baselines
- [ ] Test model can overfit single batch
- [ ] Find optimal learning rate
- [ ] Set up proper normalization
- [ ] Configure data augmentation

**During Training:**
- [ ] Use mixed precision training
- [ ] Monitor train AND validation metrics
- [ ] Use learning rate scheduler
- [ ] Apply gradient clipping (for RNNs)
- [ ] Save checkpoints regularly
- [ ] Track experiments systematically

**After Training:**
- [ ] Evaluate on test set (once!)
- [ ] Analyze errors
- [ ] Check for bias/fairness issues
- [ ] Export model for production
- [ ] Document hyperparameters and results

**Production:**
- [ ] Optimize model (quantization/pruning)
- [ ] Set up monitoring
- [ ] Implement A/B testing
- [ ] Plan for model updates
- [ ] Document deployment procedure

---

## References

1. Smith, L. N. "A disciplined approach to neural network hyper-parameters" (2018)
2. Howard, J. & Gugger, S. "fastai: A Layered API for Deep Learning" (2020)
3. He, K. et al. "Deep Residual Learning for Image Recognition" (2016)
4. Ioffe, S. & Szegedy, C. "Batch Normalization" (2015)
5. Goodfellow, I. et al. "Deep Learning" (2016)

---

**Last Updated:** October 2025

*For questions or suggestions, please open an issue on the repository.*
