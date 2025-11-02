# 🔧 ML Debugging Strategies: A Systematic Approach

**Models not working as expected? Debug systematically, not randomly.** This guide provides a structured approach to identifying and fixing ML problems.

---

## 📋 Table of Contents

- [The Debugging Mindset](#the-debugging-mindset)
- [Quick Diagnostic Checklist](#quick-diagnostic-checklist)
- [Data Issues](#data-issues)
- [Model Architecture Issues](#model-architecture-issues)
- [Training Issues](#training-issues)
- [Evaluation Issues](#evaluation-issues)
- [Production Issues](#production-issues)
- [Debugging Tools](#debugging-tools)
- [Case Studies](#case-studies)

---

## The Debugging Mindset

### Scientific Method for ML Debugging

```
1. Observe → What's the symptom?
2. Hypothesize → What could cause this?
3. Test → Run controlled experiment
4. Analyze → Did it work? Why/why not?
5. Iterate → Refine hypothesis and repeat
```

### Common Anti-Patterns

❌ **Random changes**: "Let me try changing learning rate to 0.1... 0.01... 0.001..."
✅ **Systematic approach**: "Loss isn't decreasing. Let me verify: 1) data loading correctly, 2) loss function correct, 3) gradients flowing"

❌ **Multiple changes at once**: Changed optimizer, learning rate, batch size, and architecture
✅ **One variable at a time**: Changed only learning rate, keeping everything else constant

❌ **No logging**: "Why did it work yesterday but not today?"
✅ **Comprehensive logging**: Track all hyperparameters, data versions, random seeds

---

## Quick Diagnostic Checklist

### When Model Performance is Poor

**Run through this checklist first:**

```python
# 1. Sanity check: Can model overfit single batch?
single_batch = next(iter(train_loader))
for i in range(100):
    loss = train_step(model, single_batch)
    if i % 10 == 0:
        print(f"Iteration {i}: Loss = {loss:.4f}")

# If loss doesn't decrease to near zero → Model/training issue
# If loss decreases → Data/generalization issue
```

**Checklist:**

- [ ] **Data loading correctly?**
  - Print first batch, visually inspect
  - Check shapes: `print(X.shape, y.shape)`
  - Verify labels match inputs

- [ ] **Data preprocessed correctly?**
  - Check value ranges (should be normalized?)
  - Check for NaN/Inf: `torch.isnan(X).any()`
  - Verify train/test split order (no shuffling before split?)

- [ ] **Loss function makes sense?**
  - Classification: CrossEntropyLoss expects logits, not probabilities
  - Regression: MSE vs MAE vs Huber?
  - Check loss value range (too high? too low?)

- [ ] **Optimizer configured correctly?**
  - Learning rate reasonable? (0.001 is common starting point)
  - Optimizer seeing all parameters? `len(list(optimizer.param_groups))`
  - Gradients being computed? `param.grad is not None`

- [ ] **Model architecture reasonable?**
  - Output dimension matches number of classes?
  - Activation functions appropriate?
  - Skip connections for deep networks?

---

## Data Issues

### Issue 1: Data Leakage

**Symptom:** Near-perfect training accuracy, terrible test accuracy

**Diagnosis:**
```python
# Check if test data was seen during preprocessing
print(f"Train mean: {X_train.mean()}")
print(f"Test mean: {X_test.mean()}")
# If identical → likely leakage (scaler fit on combined data)

# Check for target leakage
print(train_df.columns)
# Any column that contains future information?
# Example: 'days_since_cancellation' in churn prediction
```

**Fix:**
```python
# ❌ Wrong: Fit on all data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test = train_test_split(X_scaled)

# ✅ Correct: Fit only on train
X_train, X_test = train_test_split(X)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Only transform
```

### Issue 2: Class Imbalance

**Symptom:** High accuracy but poor recall on minority class

**Diagnosis:**
```python
# Check class distribution
print(np.bincount(y_train))
# Output: [9900, 100] → 99:1 imbalance!

# Check what model predicts
predictions = model.predict(X_test)
print(np.bincount(predictions))
# Output: [1000, 0] → Model predicts only majority class!
```

**Fix:**
```python
# Option 1: Class weights
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
model = RandomForestClassifier(class_weight='balanced')

# Option 2: Resampling
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Option 3: Different metric
from sklearn.metrics import f1_score, recall_score
# Don't use accuracy - use F1 or recall
```

### Issue 3: Data Distribution Shift

**Symptom:** Good validation performance, poor production performance

**Diagnosis:**
```python
# Compare distributions
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Training data
axes[0].hist(X_train[:, feature_idx], bins=50)
axes[0].set_title('Training Data Distribution')

# Production data
axes[1].hist(X_prod[:, feature_idx], bins=50)
axes[1].set_title('Production Data Distribution')

plt.show()
# If distributions are very different → distribution shift
```

**Fix:**
```python
# 1. Retrain on recent data
X_train_recent = X_train[dates > '2024-01-01']

# 2. Domain adaptation techniques
# 3. Monitor and alert on drift
from scipy.stats import ks_2samp

def detect_drift(X_train, X_prod, threshold=0.05):
    for i in range(X_train.shape[1]):
        statistic, p_value = ks_2samp(X_train[:, i], X_prod[:, i])
        if p_value < threshold:
            print(f"Drift detected in feature {i}: p={p_value:.4f}")
```

### Issue 4: Incorrect Data Types

**Symptom:** Model treats categorical features as continuous

**Diagnosis:**
```python
# Check data types
print(df.dtypes)
# zip_code    int64    ← Should be categorical!
# age         int64    ← Correct (continuous)
# income      float64  ← Correct

# Check if model is treating categorical as continuous
print(df['zip_code'].unique())
# [94102, 94103, ..., 10001, 10002]
# Model thinks 94102 > 10001 (wrong!)
```

**Fix:**
```python
# ✅ Correct: Encode categoricals
categorical_cols = ['zip_code', 'country', 'product_category']
numerical_cols = ['age', 'income', 'credit_score']

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

X_processed = preprocessor.fit_transform(X_train)
```

---

## Model Architecture Issues

### Issue 5: Vanishing/Exploding Gradients

**Symptom:** Loss becomes NaN or doesn't decrease in deep networks

**Diagnosis:**
```python
# Check gradient magnitudes
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.abs().mean().item():.6f}")

# Gradients close to 0 → Vanishing
# Gradients > 100 → Exploding
# Loss = NaN → Exploding
```

**Fix:**
```python
# For vanishing gradients:
# 1. Use ReLU instead of sigmoid/tanh
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),  # Instead of nn.Sigmoid()
    nn.Linear(256, 10)
)

# 2. Add residual connections (ResNet-style)
class ResidualBlock(nn.Module):
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = x + residual  # Skip connection
        return x

# 3. Batch normalization
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# For exploding gradients:
# 1. Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 2. Lower learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Instead of 1e-2
```

### Issue 6: Wrong Output Dimension

**Symptom:** Shape mismatch errors during training

**Diagnosis:**
```python
# Print shapes at each layer
x = torch.randn(32, 3, 224, 224)  # Batch of 32 images
print(f"Input: {x.shape}")

x = conv1(x)
print(f"After conv1: {x.shape}")

x = pool1(x)
print(f"After pool1: {x.shape}")

x = x.view(x.size(0), -1)  # Flatten
print(f"After flatten: {x.shape}")

x = fc1(x)
print(f"After fc1: {x.shape}")
# RuntimeError: size mismatch → Check dimensions
```

**Fix:**
```python
# ✅ Calculate correct dimensions
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Calculate flattened size
        # Input: 224x224 → After pool: 112x112
        self.fc1 = nn.Linear(64 * 112 * 112, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Or use adaptive pooling (easier)
class CNN_Adaptive(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((7, 7))  # Always output 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_classes)
```

### Issue 7: Model Too Simple/Complex

**Symptom:** Underfitting (high train loss) or overfitting (train loss << test loss)

**Diagnosis:**
```python
# Plot learning curves
train_losses = []
val_losses = []

for epoch in range(100):
    train_loss = train_epoch(model, train_loader)
    val_loss = evaluate(model, val_loader)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Validation')
plt.legend()
plt.show()

# Train and val both high → Underfitting (too simple)
# Train low, val high → Overfitting (too complex)
```

**Fix:**
```python
# For underfitting (model too simple):
# 1. Add capacity
model = nn.Sequential(
    nn.Linear(784, 512),  # Increased from 256
    nn.ReLU(),
    nn.Linear(512, 256),  # Added layer
    nn.ReLU(),
    nn.Linear(256, 10)
)

# 2. Add features
# Add polynomial features, interactions, etc.

# For overfitting (model too complex):
# 1. Regularization
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.5),  # Add dropout
    nn.Linear(256, 10)
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4  # L2 regularization
)

# 2. Early stopping
best_val_loss = float('inf')
patience = 10
counter = 0

for epoch in range(1000):
    train_loss = train_epoch(model, train_loader)
    val_loss = evaluate(model, val_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        save_checkpoint(model)
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

# 3. More data
# Collect more samples or use data augmentation
```

---

## Training Issues

### Issue 8: Learning Rate Too High/Low

**Symptom:** Loss oscillates wildly (too high) or barely decreases (too low)

**Diagnosis:**
```python
# Learning rate too high:
# Loss: 0.5 → 0.3 → 0.8 → 0.2 → 1.5 → NaN

# Learning rate too low:
# Loss: 2.3 → 2.29 → 2.288 → 2.287 (too slow)

# Find optimal learning rate
lrs = []
losses = []

for lr in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
    model = create_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss = train_n_steps(model, optimizer, n_steps=100)
    lrs.append(lr)
    losses.append(loss)

plt.semilogx(lrs, losses)
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.show()
# Choose LR where loss decreases fastest
```

**Fix:**
```python
# Option 1: Use learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5
)

for epoch in range(100):
    train_loss = train_epoch(model, train_loader)
    val_loss = evaluate(model, val_loader)
    scheduler.step(val_loss)  # Reduce LR if val_loss plateaus

# Option 2: Warmup + cosine annealing
from torch.optim.lr_scheduler import CosineAnnealingLR

def warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return epoch / warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

scheduler = warmup_cosine_scheduler(optimizer, warmup_epochs=5, total_epochs=100)

# Option 3: Use adaptive optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Usually works well
```

### Issue 9: Batch Size Issues

**Symptom:** Out of memory or noisy gradients

**Diagnosis:**
```python
# Too large batch size → OOM
# RuntimeError: CUDA out of memory

# Too small batch size → Noisy gradients
# Loss: 0.5 → 0.3 → 0.7 → 0.2 → 0.6 (very noisy)

# Find maximum batch size
batch_size = 32
while True:
    try:
        batch = torch.randn(batch_size, 3, 224, 224).cuda()
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
        print(f"Batch size {batch_size} works")
        batch_size *= 2
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print(f"Max batch size: {batch_size // 2}")
            break
        else:
            raise e
```

**Fix:**
```python
# Option 1: Reduce batch size
train_loader = DataLoader(dataset, batch_size=16)  # Instead of 64

# Option 2: Gradient accumulation (simulate large batch)
accumulation_steps = 4
optimizer.zero_grad()

for i, (inputs, targets) in enumerate(train_loader):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss = loss / accumulation_steps  # Normalize
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# Option 3: Mixed precision training (reduce memory)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for inputs, targets in train_loader:
    optimizer.zero_grad()

    with autocast():  # Use FP16
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Issue 10: Wrong Loss Function

**Symptom:** Model trains but makes nonsensical predictions

**Diagnosis:**
```python
# Common mistake: Using BCELoss with logits
output = model(x)  # Logits (unbounded)
loss = nn.BCELoss()(output, target)  # Expects probabilities [0,1]!

# Check loss function requirements
print(f"Output range: [{output.min():.2f}, {output.max():.2f}]")
# Output: [-3.2, 5.1] → Not probabilities!
```

**Fix:**
```python
# Binary classification:
# ❌ Wrong
output = model(x)
loss = nn.BCELoss()(output, target)

# ✅ Correct: Use BCEWithLogitsLoss (includes sigmoid)
output = model(x)  # Logits
loss = nn.BCEWithLogitsLoss()(output, target)

# Or manually apply sigmoid
output = model(x)
prob = torch.sigmoid(output)
loss = nn.BCELoss()(prob, target)

# Multi-class classification:
# ❌ Wrong
output = F.softmax(model(x), dim=1)
loss = nn.CrossEntropyLoss()(output, target)

# ✅ Correct: CrossEntropyLoss expects logits
output = model(x)  # Logits (no softmax!)
loss = nn.CrossEntropyLoss()(output, target)

# Regression:
# MSE: Sensitive to outliers
loss = nn.MSELoss()(predictions, targets)

# Huber: Robust to outliers
loss = nn.SmoothL1Loss()(predictions, targets)

# MAE: Even more robust
loss = nn.L1Loss()(predictions, targets)
```

---

## Evaluation Issues

### Issue 11: Wrong Metric for Problem

**Symptom:** High accuracy but model is useless

**Example:**
```python
# Fraud detection: 99% non-fraud, 1% fraud
accuracy = 99%  # Model predicts "not fraud" for everything!

# Check confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)
print(cm)
# [[9900    0]
#  [ 100    0]]
# Caught 0 fraud cases!
```

**Fix:**
```python
# Use appropriate metrics
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    roc_auc_score
)

# For imbalanced classification
print(classification_report(y_true, y_pred))

# Optimize for recall (catch all fraud)
# or precision (avoid false alarms)
# or F1 (balance both)

# For ranking problems
auc = roc_auc_score(y_true, y_scores)

# For regression
from sklearn.metrics import mean_absolute_error, r2_score

mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
```

### Issue 12: Evaluation on Training Data

**Symptom:** Unrealistically good performance

**Diagnosis:**
```python
# Accidentally evaluating on training data
model.fit(X_train, y_train)
score = model.score(X_train, y_train)  # Should be X_test!
print(f"Accuracy: {score:.2%}")  # 99.9% (too good to be true)
```

**Fix:**
```python
# ✅ Always evaluate on held-out test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model.fit(X_train, y_train)
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Train: {train_score:.2%}, Test: {test_score:.2%}")

# Use cross-validation for more robust estimate
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"CV: {scores.mean():.2%} ± {scores.std():.2%}")
```

---

## Production Issues

### Issue 13: Model Degradation Over Time

**Symptom:** Good offline metrics, declining production performance

**Diagnosis:**
```python
# Monitor prediction distribution over time
import matplotlib.pyplot as plt

def monitor_predictions(predictions_log):
    dates = predictions_log['date']
    avg_pred = predictions_log.groupby('date')['prediction'].mean()

    plt.figure(figsize=(12, 4))
    plt.plot(dates.unique(), avg_pred)
    plt.xlabel('Date')
    plt.ylabel('Average Prediction')
    plt.title('Prediction Drift Over Time')
    plt.show()

    # If trend changes → Model degradation

# Monitor feature distributions
def detect_feature_drift(X_train, X_prod, feature_names):
    from scipy.stats import ks_2samp

    for i, name in enumerate(feature_names):
        stat, p_value = ks_2samp(X_train[:, i], X_prod[:, i])
        if p_value < 0.05:
            print(f"⚠️  Drift in {name}: p={p_value:.4f}")
```

**Fix:**
```python
# 1. Retrain regularly
if days_since_training > 30:
    X_recent = get_recent_data(days=90)
    model.fit(X_recent, y_recent)
    deploy_model(model)

# 2. Online learning
from river import linear_model

model = linear_model.LogisticRegression()

for x, y in stream:
    y_pred = model.predict_one(x)
    model.learn_one(x, y)  # Update with new data

# 3. Ensemble with recent model
final_pred = 0.7 * old_model.predict(X) + 0.3 * new_model.predict(X)
```

### Issue 14: Inference Latency

**Symptom:** Model too slow for production requirements

**Diagnosis:**
```python
import time

# Measure inference time
times = []
for _ in range(100):
    start = time.time()
    prediction = model.predict(X_sample)
    elapsed = time.time() - start
    times.append(elapsed)

print(f"Mean: {np.mean(times)*1000:.2f}ms")
print(f"P95: {np.percentile(times, 95)*1000:.2f}ms")
print(f"P99: {np.percentile(times, 99)*1000:.2f}ms")

# If P99 > requirement → Too slow
```

**Fix:**
```python
# 1. Model quantization
import torch.quantization

model_fp32 = MyModel()
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,
    {torch.nn.Linear},
    dtype=torch.qint8
)
# 2-4x speedup, minimal accuracy loss

# 2. Knowledge distillation (train smaller model)
teacher_model = LargeModel()  # Accurate but slow
student_model = SmallModel()  # Fast but less accurate

# Train student to mimic teacher
for x, y in data:
    teacher_output = teacher_model(x).detach()
    student_output = student_model(x)

    # Match both true labels and teacher predictions
    loss = (
        0.5 * CrossEntropyLoss(student_output, y) +
        0.5 * KLDivLoss(student_output, teacher_output)
    )
    loss.backward()

# 3. Batch predictions
# Instead of: [predict(x1), predict(x2), ...]
# Do: predict([x1, x2, ...])  # Much faster!

# 4. Caching
from functools import lru_cache

@lru_cache(maxsize=10000)
def predict_cached(input_hash):
    return model.predict(input_hash)
```

---

## Debugging Tools

### Essential Debugging Tools

**1. TensorBoard (PyTorch/TensorFlow)**
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment1')

for epoch in range(100):
    # Log scalars
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)

    # Log histograms (check weight distributions)
    for name, param in model.named_parameters():
        writer.add_histogram(name, param, epoch)
        if param.grad is not None:
            writer.add_histogram(f'{name}.grad', param.grad, epoch)

    # Log images
    writer.add_images('predictions', prediction_images, epoch)

writer.close()

# View: tensorboard --logdir=runs
```

**2. Model Profiling**
```python
# PyTorch profiler
import torch.profiler as profiler

with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True
) as prof:
    model(inputs)

print(prof.key_averages().table(sort_by="cuda_time_total"))
# Shows which operations are slowest
```

**3. Gradient Checking**
```python
# Check if gradients are computed correctly
def gradient_check(model, X, y, epsilon=1e-7):
    model.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"⚠️  No gradient for {name}")
            continue

        # Check gradient magnitude
        grad_norm = param.grad.norm().item()
        if grad_norm == 0:
            print(f"⚠️  Zero gradient for {name}")
        elif grad_norm > 100:
            print(f"⚠️  Exploding gradient for {name}: {grad_norm:.2f}")
        elif grad_norm < 1e-7:
            print(f"⚠️  Vanishing gradient for {name}: {grad_norm:.2e}")

gradient_check(model, X_batch, y_batch)
```

**4. Data Inspection**
```python
# Visualize what model sees
import matplotlib.pyplot as plt

def visualize_batch(images, labels, predictions=None):
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].permute(1, 2, 0))
        title = f"True: {labels[i]}"
        if predictions is not None:
            title += f"\nPred: {predictions[i]}"
        ax.set_title(title)
        ax.axis('off')
    plt.show()

# Check if data looks correct
batch = next(iter(train_loader))
visualize_batch(batch[0], batch[1])
```

---

## Case Studies

### Case Study 1: Image Classifier with 50% Accuracy

**Symptoms:**
- Binary image classifier stuck at 50% accuracy
- Loss decreases but accuracy doesn't improve

**Debugging steps:**

```python
# 1. Check data loading
batch = next(iter(train_loader))
images, labels = batch
print(f"Images: {images.shape}, min={images.min()}, max={images.max()}")
print(f"Labels: {labels.unique()}")

# Found issue: Images not normalized (range 0-255 instead of 0-1)
```

**Fix:**
```python
# Add normalization
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts to [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])
```

### Case Study 2: NLP Model with NaN Loss

**Symptoms:**
- Text classification model
- Loss becomes NaN after a few batches

**Debugging steps:**

```python
# 1. Check for overflow in embeddings
for batch in train_loader:
    embeddings = model.embed(batch)
    print(f"Embedding range: [{embeddings.min():.2f}, {embeddings.max():.2f}]")
    if torch.isnan(embeddings).any():
        print("NaN in embeddings!")
        break

# 2. Found issue: Learning rate too high (0.1)
# Gradients exploding → NaN
```

**Fix:**
```python
# 1. Reduce learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Instead of 0.1

# 2. Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Case Study 3: Great Training, Poor Test Performance

**Symptoms:**
- Training accuracy: 98%
- Test accuracy: 65%
- Clear overfitting

**Debugging steps:**

```python
# 1. Check model complexity
print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
# Output: 50M parameters for 10K training samples!

# 2. Check regularization
print(f"Dropout: {model.dropout.p}")  # 0.0 - no dropout!
print(f"Weight decay: {optimizer.param_groups[0]['weight_decay']}")  # 0.0
```

**Fix:**
```python
# 1. Add dropout
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Dropout(0.5),  # Added
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.5),  # Added
    nn.Linear(256, 10)
)

# 2. Add weight decay
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4  # L2 regularization
)

# 3. Data augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

# Result: Test accuracy improved to 89%
```

---

## Quick Reference: Debugging Flowchart

```
Model not working?
│
├─ Can't overfit single batch?
│  ├─ Loss stays high → Check loss function, optimizer
│  ├─ Loss = NaN → Check learning rate, gradients
│  └─ Gradients = 0 → Check model architecture, backward()
│
├─ Overfits single batch but not full dataset?
│  ├─ Training loss high → Model too simple, add capacity
│  └─ Training loss low, test high → Overfitting
│     ├─ Add regularization (dropout, weight decay)
│     ├─ Add more data
│     └─ Early stopping
│
└─ Works in training but not production?
   ├─ Check data distribution shift
   ├─ Monitor prediction drift
   └─ Retrain on recent data
```

---

## Key Takeaways

**Debugging principles:**
1. **Start simple** - Overfit single batch first
2. **Change one thing at a time** - Isolate variables
3. **Log everything** - Data, hyperparameters, metrics
4. **Visualize** - Plot learning curves, inspect data
5. **Check assumptions** - Is data correct? Is loss function appropriate?

**Most common issues:**
- Data problems (70% of ML bugs)
- Wrong hyperparameters (20%)
- Model architecture (10%)

**When stuck:**
- Take a break, come back fresh
- Explain problem to rubber duck
- Ask for help (Stack Overflow, forums)
- Simplify: Remove components until it works, add back one at a time

---

**Related guides:**
- [Debugging ML Models](./DEBUGGING_ML_MODELS.md) - Production debugging
- [ML Engineering Best Practices](./ML_ENGINEERING_BEST_PRACTICES.md) - Prevent bugs
- [Common ML Mistakes](../../resources/COMMON_ML_MISTAKES.md) - Avoid pitfalls

---

*Last Updated: October 30, 2025*
