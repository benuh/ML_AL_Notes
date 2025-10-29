# 🔧 Debugging Machine Learning Models - Complete Guide

**Systematic Approach to Diagnosing and Fixing ML Problems**

> Last Updated: October 2025
> A practical, step-by-step guide to debugging ML models

---

## 📋 Table of Contents

1. [The Debugging Mindset](#the-debugging-mindset)
2. [Data Problems](#data-problems)
3. [Model Problems](#model-problems)
4. [Training Problems](#training-problems)
5. [Performance Problems](#performance-problems)
6. [Deployment Problems](#deployment-problems)
7. [Debugging Tools](#debugging-tools)
8. [Debugging Checklist](#debugging-checklist)

---

## 🧠 The Debugging Mindset

### Systematic Debugging Process

```
1. Reproduce the problem
2. Simplify (use minimal example)
3. Isolate (binary search for the issue)
4. Form hypothesis
5. Test hypothesis
6. Fix and verify
```

### Common Debugging Anti-Patterns

❌ **Don't:**
- Change multiple things at once
- Skip data validation
- Ignore warnings
- Assume the data is correct
- Jump to conclusions

✅ **Do:**
- Make one change at a time
- Document what you try
- Check data first, always
- Use version control
- Write tests

---

## 📊 Data Problems

### Problem 1: Model Won't Learn (Loss Not Decreasing)

**Symptoms:**
- Loss stays constant or increases
- Accuracy doesn't improve
- Model predicts same class for everything

**Debug Steps:**

#### Step 1: Check Data Loading

```python
import numpy as np
import matplotlib.pyplot as plt

def debug_data_loading(dataloader):
    """Verify data is loading correctly."""

    # Get first batch
    batch = next(iter(dataloader))
    X, y = batch

    print(f"Batch shape: {X.shape}")
    print(f"Label shape: {y.shape}")
    print(f"Data type: {X.dtype}")
    print(f"Label type: {y.dtype}")
    print(f"Data range: [{X.min():.4f}, {X.max():.4f}]")
    print(f"Unique labels: {np.unique(y)}")

    # Visualize samples
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flat):
        if i < len(X):
            # Handle different image formats
            img = X[i]
            if img.shape[0] in [1, 3]:  # Channel first
                img = img.transpose(1, 2, 0)
            if img.shape[-1] == 1:  # Grayscale
                img = img.squeeze()
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)
            ax.set_title(f'Label: {y[i]}')
            ax.axis('off')
    plt.tight_layout()
    plt.show()

    return X, y

# Usage
# X, y = debug_data_loading(train_loader)
```

#### Step 2: Check Label Distribution

```python
def check_label_distribution(labels):
    """Check for class imbalance."""
    from collections import Counter

    counts = Counter(labels)
    print(f"Total samples: {len(labels)}")
    print(f"Number of classes: {len(counts)}")
    print(f"\nClass distribution:")

    for label, count in sorted(counts.items()):
        percentage = 100 * count / len(labels)
        print(f"  Class {label}: {count:5d} ({percentage:5.2f}%)")

    # Visualize
    plt.figure(figsize=(10, 5))
    plt.bar(counts.keys(), counts.values())
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.show()

    # Check for severe imbalance
    min_count = min(counts.values())
    max_count = max(counts.values())
    imbalance_ratio = max_count / min_count

    if imbalance_ratio > 10:
        print(f"\n⚠️  Severe class imbalance detected (ratio: {imbalance_ratio:.1f})")
        print("Consider:")
        print("  - Class weights")
        print("  - Oversampling minority class")
        print("  - Undersampling majority class")
        print("  - Different evaluation metrics (F1, AUC-ROC)")

# Usage
# check_label_distribution(y_train)
```

#### Step 3: Verify Data Normalization

```python
def check_normalization(data):
    """Check if data is properly normalized."""

    mean = data.mean()
    std = data.std()
    min_val = data.min()
    max_val = data.max()

    print(f"Mean: {mean:.4f}")
    print(f"Std:  {std:.4f}")
    print(f"Min:  {min_val:.4f}")
    print(f"Max:  {max_val:.4f}")

    # Check for common issues
    if abs(mean) > 1.0:
        print("\n⚠️  Data not centered (mean should be ~0)")
        print("Apply: (data - mean) / std")

    if std < 0.1 or std > 10:
        print(f"\n⚠️  Unusual standard deviation: {std:.4f}")
        print("Consider normalizing to [0, 1] or [-1, 1]")

    if min_val < -10 or max_val > 10:
        print(f"\n⚠️  Large value range: [{min_val:.2f}, {max_val:.2f}]")
        print("Consider scaling data")

    # Visualize distribution
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.hist(data.flatten(), bins=50, edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Data Distribution')

    plt.subplot(1, 2, 2)
    plt.boxplot(data.flatten())
    plt.ylabel('Value')
    plt.title('Boxplot')

    plt.tight_layout()
    plt.show()

# Usage
# check_normalization(X_train)
```

### Problem 2: Data Leakage

**Symptoms:**
- Perfect training accuracy but poor test accuracy
- Unrealistically high validation scores
- Model performs worse in production

**Debug Steps:**

```python
def detect_data_leakage(X_train, X_test, y_train=None, y_test=None):
    """Detect potential data leakage."""

    print("Checking for data leakage...\n")

    # 1. Check for duplicate samples
    train_hashes = set([hash(x.tobytes()) for x in X_train])
    test_hashes = set([hash(x.tobytes()) for x in X_test])

    overlap = train_hashes.intersection(test_hashes)
    if overlap:
        print(f"⚠️  Found {len(overlap)} duplicate samples in train and test!")
    else:
        print("✓ No duplicate samples found")

    # 2. Check for information leakage in features
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    if y_train is not None and y_test is not None:
        # Train simple model
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, clf.predict(X_train))
        test_acc = accuracy_score(y_test, clf.predict(X_test))

        print(f"\nSimple model performance:")
        print(f"  Train accuracy: {train_acc:.4f}")
        print(f"  Test accuracy:  {test_acc:.4f}")

        if train_acc > 0.99 and test_acc < 0.7:
            print("\n⚠️  Suspicious performance gap suggests possible leakage")
            print("Check:")
            print("  - Features calculated using test data")
            print("  - Target encoding without proper CV")
            print("  - Scaling fit on entire dataset")

        # 3. Check feature importance for suspicious features
        importances = clf.feature_importances_
        top_features = np.argsort(importances)[-5:]

        print(f"\nTop 5 most important features: {top_features}")
        if any(importances > 0.8):
            print("⚠️  Single feature has very high importance (>0.8)")
            print("This could indicate:")
            print("  - Feature derived from target")
            print("  - Identifier leaked as feature")

# Usage
# detect_data_leakage(X_train, X_test, y_train, y_test)
```

### Problem 3: Wrong Data Types or Shapes

```python
def validate_data_shapes(model, X_train, y_train, X_test=None):
    """Validate all data shapes are correct."""

    print("Validating data shapes...\n")

    # Check training data
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    if X_test is not None:
        print(f"X_test shape:  {X_test.shape}")

        # Check feature count matches
        if X_train.shape[1] != X_test.shape[1]:
            print(f"\n❌ Feature mismatch!")
            print(f"   Train: {X_train.shape[1]} features")
            print(f"   Test:  {X_test.shape[1]} features")

    # Try a forward pass
    try:
        import torch
        if isinstance(X_train, torch.Tensor):
            with torch.no_grad():
                sample = X_train[:1]  # Single sample
                output = model(sample)
                print(f"\n✓ Forward pass successful")
                print(f"  Input shape:  {sample.shape}")
                print(f"  Output shape: {output.shape}")
    except Exception as e:
        print(f"\n❌ Forward pass failed: {str(e)}")
        print("Check:")
        print("  - Input dimensions match model expects")
        print("  - Batch dimension is first")
        print("  - Channel ordering (CHW vs HWC)")

# Usage
# validate_data_shapes(model, X_train, y_train, X_test)
```

---

## 🧮 Model Problems

### Problem 4: Vanishing or Exploding Gradients

**Symptoms:**
- Gradients become very small (vanishing) or very large (exploding)
- Training becomes unstable
- NaN or Inf in losses

**Debug Steps:**

```python
def check_gradients(model, loss):
    """Check for gradient problems."""

    print("Gradient Statistics:\n")

    total_norm = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

            grad_mean = param.grad.data.mean()
            grad_std = param.grad.data.std()
            grad_max = param.grad.data.abs().max()

            print(f"{name:40s}")
            print(f"  Mean: {grad_mean:10.6f}")
            print(f"  Std:  {grad_std:10.6f}")
            print(f"  Max:  {grad_max:10.6f}")

            # Check for problems
            if grad_max < 1e-7:
                print(f"  ⚠️  Vanishing gradients detected")
            elif grad_max > 100:
                print(f"  ⚠️  Exploding gradients detected")
            elif torch.isnan(param.grad).any():
                print(f"  ❌ NaN gradients detected")
            elif torch.isinf(param.grad).any():
                print(f"  ❌ Inf gradients detected")

    total_norm = total_norm ** 0.5
    print(f"\nTotal gradient norm: {total_norm:.6f}")

    if total_norm < 1e-7:
        print("⚠️  Vanishing gradient problem")
        print("Solutions:")
        print("  - Use ReLU instead of sigmoid/tanh")
        print("  - Add skip connections (ResNet)")
        print("  - Use batch normalization")
        print("  - Reduce network depth")
        print("  - Check learning rate")

    elif total_norm > 100:
        print("⚠️  Exploding gradient problem")
        print("Solutions:")
        print("  - Add gradient clipping")
        print("  - Reduce learning rate")
        print("  - Use batch normalization")
        print("  - Check weight initialization")

# Usage in training loop
# loss.backward()
# check_gradients(model, loss)
# optimizer.step()
```

### Problem 5: Poor Weight Initialization

```python
def check_weight_initialization(model):
    """Check if weights are properly initialized."""

    print("Weight Initialization Check:\n")

    for name, param in model.named_parameters():
        if 'weight' in name:
            data = param.data

            mean = data.mean().item()
            std = data.std().item()
            min_val = data.min().item()
            max_val = data.max().item()

            print(f"{name:40s}")
            print(f"  Mean: {mean:8.4f}")
            print(f"  Std:  {std:8.4f}")
            print(f"  Range: [{min_val:8.4f}, {max_val:8.4f}]")

            # Check for problems
            if abs(mean) > 0.1:
                print(f"  ⚠️  Mean not close to 0")

            if std < 0.01 or std > 1.0:
                print(f"  ⚠️  Unusual standard deviation")

            if data.abs().max() > 10:
                print(f"  ⚠️  Large weight values")

            print()

# Proper initialization
def initialize_weights(model):
    """Properly initialize model weights."""
    for name, param in model.named_parameters():
        if 'weight' in name:
            if 'conv' in name or 'linear' in name:
                # He initialization for ReLU
                torch.nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            else:
                # Xavier for other layers
                torch.nn.init.xavier_normal_(param)
        elif 'bias' in name:
            torch.nn.init.zeros_(param)

# Usage
# check_weight_initialization(model)
# initialize_weights(model)
```

---

## 🏋️ Training Problems

### Problem 6: Overfitting

**Symptoms:**
- Training accuracy much higher than validation
- Loss decreasing on train but increasing on validation
- Model memorizes training data

**Debug and Fix:**

```python
def diagnose_overfitting(train_losses, val_losses, train_accs, val_accs):
    """Visualize and diagnose overfitting."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)

    # Plot losses
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot accuracies
    ax2.plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Calculate overfitting metrics
    final_train_acc = train_accs[-1]
    final_val_acc = val_accs[-1]
    gap = final_train_acc - final_val_acc

    print(f"\nOverfitting Analysis:")
    print(f"  Final train accuracy: {final_train_acc:.4f}")
    print(f"  Final val accuracy:   {final_val_acc:.4f}")
    print(f"  Gap:                  {gap:.4f}")

    if gap > 0.1:
        print(f"\n⚠️  Significant overfitting detected (gap > 0.1)")
        print("\nSolutions (in order of priority):")
        print("  1. Get more training data")
        print("  2. Add data augmentation")
        print("  3. Add dropout (start with 0.5)")
        print("  4. Add L2 regularization (weight_decay=0.01)")
        print("  5. Reduce model complexity")
        print("  6. Use early stopping")
    elif gap > 0.05:
        print(f"\n⚠️  Moderate overfitting (gap > 0.05)")
        print("\nConsider:")
        print("  - Data augmentation")
        print("  - Small dropout (0.2-0.3)")
        print("  - Early stopping")
    else:
        print("\n✓ No significant overfitting")

# Usage
# diagnose_overfitting(train_losses, val_losses, train_accs, val_accs)
```

### Problem 7: Underfitting

```python
def diagnose_underfitting(train_acc, val_acc, expected_acc=0.7):
    """Check if model is underfitting."""

    print("Underfitting Analysis:\n")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Val accuracy:   {val_acc:.4f}")
    print(f"Expected:       {expected_acc:.4f}")

    if train_acc < expected_acc:
        print("\n⚠️  Model is underfitting")
        print("\nSolutions:")
        print("  1. Increase model capacity:")
        print("     - Add more layers")
        print("     - Add more neurons per layer")
        print("     - Use more complex architecture")
        print("  2. Train longer:")
        print("     - More epochs")
        print("     - Check if loss still decreasing")
        print("  3. Reduce regularization:")
        print("     - Lower dropout rate")
        print("     - Lower weight decay")
        print("  4. Check learning rate:")
        print("     - Might be too low")
        print("  5. Check data quality:")
        print("     - Ensure features are informative")
        print("     - Check for label errors")
    else:
        print("\n✓ Model capacity seems adequate")

# Usage
# diagnose_underfitting(train_acc, val_acc, expected_acc=0.85)
```

---

## 📈 Performance Problems

### Problem 8: Slow Training

```python
import time

def profile_training_step(model, dataloader, device='cuda'):
    """Profile training step to find bottlenecks."""

    model.train()
    model.to(device)

    times = {
        'data_loading': [],
        'forward': [],
        'backward': [],
        'optimizer': [],
        'total': []
    }

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for i, (inputs, targets) in enumerate(dataloader):
        if i >= 10:  # Profile first 10 batches
            break

        start_total = time.time()

        # Data loading time
        start = time.time()
        inputs, targets = inputs.to(device), targets.to(device)
        times['data_loading'].append(time.time() - start)

        # Forward pass
        start = time.time()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        times['forward'].append(time.time() - start)

        # Backward pass
        start = time.time()
        optimizer.zero_grad()
        loss.backward()
        times['backward'].append(time.time() - start)

        # Optimizer step
        start = time.time()
        optimizer.step()
        times['optimizer'].append(time.time() - start)

        times['total'].append(time.time() - start_total)

    # Print results
    print("Training Step Profiling (average over 10 batches):\n")
    for key, values in times.items():
        avg_time = np.mean(values) * 1000  # Convert to ms
        percentage = 100 * avg_time / np.mean(times['total']) / 10
        print(f"{key:20s}: {avg_time:8.2f} ms ({percentage:5.1f}%)")

    # Recommendations
    data_pct = 100 * np.mean(times['data_loading']) / np.mean(times['total'])
    if data_pct > 30:
        print(f"\n⚠️  Data loading is {data_pct:.1f}% of training time")
        print("Solutions:")
        print("  - Increase num_workers in DataLoader")
        print("  - Use pin_memory=True")
        print("  - Preload data to RAM/SSD")
        print("  - Use data prefetching")

# Usage
# profile_training_step(model, train_loader, device='cuda')
```

---

## 🔍 Debugging Tools

### Tool 1: Gradient Visualization

```python
def visualize_gradients(model, named_parameters=None):
    """Visualize gradient flow through network."""

    gradients = []
    layers = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients.append(param.grad.abs().mean().item())
            layers.append(name.replace('.weight', '').replace('.bias', ''))

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(gradients)), gradients)
    plt.xticks(range(len(gradients)), layers, rotation=90)
    plt.xlabel('Layer')
    plt.ylabel('Average Absolute Gradient')
    plt.title('Gradient Flow')
    plt.tight_layout()
    plt.show()

# Usage after backward pass
# loss.backward()
# visualize_gradients(model)
```

### Tool 2: Activation Visualization

```python
def visualize_activations(model, input_data):
    """Visualize activations through network."""

    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.ReLU)):
            hook = module.register_forward_hook(hook_fn(name))
            hooks.append(hook)

    # Forward pass
    with torch.no_grad():
        _ = model(input_data)

    # Visualize
    fig, axes = plt.subplots(len(activations), 1, figsize=(12, 3*len(activations)))
    if len(activations) == 1:
        axes = [axes]

    for ax, (name, activation) in zip(axes, activations.items()):
        # Plot activation statistics
        act_flat = activation.cpu().numpy().flatten()
        ax.hist(act_flat, bins=50, edgecolor='black')
        ax.set_title(f'{name} - Mean: {act_flat.mean():.4f}, Std: {act_flat.std():.4f}')
        ax.set_xlabel('Activation Value')
        ax.set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    # Remove hooks
    for hook in hooks:
        hook.remove()

# Usage
# visualize_activations(model, sample_input)
```

---

## ✅ Complete Debugging Checklist

### Before Training

- [ ] Data loaded correctly
- [ ] Shapes are correct
- [ ] Labels are correct
- [ ] Data is normalized
- [ ] No data leakage
- [ ] Train/val/test split is correct
- [ ] Class distribution checked
- [ ] Model architecture makes sense
- [ ] Weights initialized properly

### During Training

- [ ] Loss is decreasing
- [ ] Gradients are in reasonable range
- [ ] No NaN or Inf values
- [ ] Validation performance tracked
- [ ] Overfitting monitored
- [ ] Learning rate is appropriate
- [ ] Training is not too slow

### After Training

- [ ] Model performance meets expectations
- [ ] No overfitting
- [ ] Test set performance evaluated
- [ ] Confusion matrix checked
- [ ] Edge cases tested
- [ ] Model saved correctly

---

## 🎯 Quick Reference

### Loss Not Decreasing

1. Check data (labels, normalization, leakage)
2. Check learning rate (try 1e-3, 1e-4)
3. Check model (can it overfit small batch?)
4. Check gradients (vanishing/exploding?)

### Overfitting

1. Get more data
2. Add data augmentation
3. Add dropout (0.5)
4. Add weight decay (0.01)
5. Reduce model size
6. Early stopping

### Poor Test Performance

1. Check for data leakage
2. Check train/test distribution match
3. Verify preprocessing same for train/test
4. Check for overfitting
5. Try ensemble methods

### Training Too Slow

1. Increase batch size
2. Use GPU if available
3. Increase DataLoader workers
4. Use mixed precision
5. Profile to find bottleneck

---

## 📚 References

- *Practical Deep Learning for Coders* - fast.ai
- *Deep Learning* - Goodfellow et al.
- *A Recipe for Training Neural Networks* - Andrej Karpathy

---

**Happy Debugging!** 🔧

Remember: Most ML problems are data problems. Always check your data first!
