# 🔧 ML/AI Troubleshooting Guide

## Complete Guide to Debugging ML Models and Code

This comprehensive guide helps you diagnose and fix common issues in machine learning projects, from data problems to model training issues to deployment challenges.

---

## 📋 Table of Contents

1. [Model Not Learning](#model-not-learning)
2. [Overfitting Issues](#overfitting-issues)
3. [Underfitting Issues](#underfitting-issues)
4. [Training Instability](#training-instability)
5. [Memory Issues](#memory-issues)
6. [Slow Training](#slow-training)
7. [Data Problems](#data-problems)
8. [Deployment Issues](#deployment-issues)

---

## 🧠 Model Not Learning

### Symptom: Loss not decreasing or staying constant

**Potential Causes & Solutions:**

#### 1. Learning Rate Too High
```python
# Problem: Loss oscillates or increases
# Training loss: 2.5 → 3.1 → 2.8 → 4.2 → NaN

# Solution: Reduce learning rate
# Try learning rates: 0.1, 0.01, 0.001, 0.0001

optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Start small

# Or use learning rate finder
def find_optimal_lr(model, train_loader, init_lr=1e-7, final_lr=1, num_iterations=100):
    """Find optimal learning rate"""
    import math

    lrs = []
    losses = []

    optimizer = optim.SGD(model.parameters(), lr=init_lr)
    lr_mult = (final_lr / init_lr) ** (1 / num_iterations)

    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx >= num_iterations:
            break

        # Training step
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Log
        lrs.append(optimizer.param_groups[0]['lr'])
        losses.append(loss.item())

        # Increase learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_mult

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.show()

    # Optimal LR is usually before minimum (where slope is steepest)
    return lrs[losses.index(min(losses))] / 10
```

#### 2. Learning Rate Too Low
```python
# Problem: Loss decreases very slowly
# Epoch 1: loss=2.5, Epoch 10: loss=2.48, Epoch 100: loss=2.3

# Solution: Increase learning rate or use adaptive optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam adapts LR automatically

# Or use learning rate scheduling
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,  # Reduce LR by half
    patience=5,  # After 5 epochs without improvement
    verbose=True
)

for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader, optimizer)
    val_loss = validate(model, val_loader)

    scheduler.step(val_loss)  # Adjust LR based on validation loss
```

#### 3. Incorrect Loss Function
```python
# Problem: Using wrong loss for task

# Binary classification with CrossEntropyLoss (WRONG)
criterion = nn.CrossEntropyLoss()  # Expects multiple classes!

# Solution: Use appropriate loss
criterion = nn.BCEWithLogitsLoss()  # For binary classification

# Common mistakes:
# - Using MSE for classification → Use CrossEntropy
# - Using CrossEntropy for regression → Use MSE/MAE
# - Forgetting to apply softmax → Use nn.CrossEntropyLoss (has softmax built-in)

# Correct usage:
# Binary classification
model = nn.Sequential(
    nn.Linear(10, 1),
    # No sigmoid! BCEWithLogitsLoss includes it
)
criterion = nn.BCEWithLogitsLoss()

# Multi-class classification
model = nn.Sequential(
    nn.Linear(10, 5),
    # No softmax! CrossEntropyLoss includes it
)
criterion = nn.CrossEntropyLoss()

# Regression
model = nn.Sequential(
    nn.Linear(10, 1),
)
criterion = nn.MSELoss()
```

#### 4. Dead ReLU Problem
```python
# Problem: Neurons output 0 for all inputs (dead ReLU)
# Symptom: Many neurons have 0 gradient, weights don't update

# Check for dead ReLUs
def check_dead_relus(model, dataloader):
    """Check percentage of dead ReLU neurons"""
    activations = []

    def hook(module, input, output):
        activations.append(output.detach())

    # Register hooks
    handles = []
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            handles.append(module.register_forward_hook(hook))

    # Forward pass
    model.eval()
    with torch.no_grad():
        for data, _ in dataloader:
            _ = model(data)
            break

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Analyze activations
    for i, activation in enumerate(activations):
        dead_percentage = (activation == 0).float().mean().item()
        print(f"ReLU layer {i}: {dead_percentage:.1%} dead neurons")
        if dead_percentage > 0.5:
            print(f"  ⚠️ Warning: >50% dead neurons!")

# Solutions:

# 1. Use Leaky ReLU
model = nn.Sequential(
    nn.Linear(10, 128),
    nn.LeakyReLU(0.01),  # negative_slope=0.01
    nn.Linear(128, 10)
)

# 2. Lower learning rate
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 3. Better weight initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

model.apply(init_weights)
```

#### 5. Vanishing/Exploding Gradients
```python
# Check gradient magnitudes
def check_gradients(model):
    """Monitor gradient norms"""
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    print(f"Gradient norm: {total_norm:.4f}")

    if total_norm < 1e-7:
        print("⚠️ Warning: Vanishing gradients!")
    elif total_norm > 100:
        print("⚠️ Warning: Exploding gradients!")

# Solutions:

# 1. Gradient clipping (for exploding gradients)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 2. Batch normalization
model = nn.Sequential(
    nn.Linear(10, 128),
    nn.BatchNorm1d(128),  # Stabilizes training
    nn.ReLU(),
    nn.Linear(128, 10)
)

# 3. Residual connections (skip connections)
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.layers(x)  # Skip connection!

# 4. Better initialization
for m in model.modules():
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
```

#### 6. Data Not Normalized
```python
# Problem: Features have vastly different scales
# Age: [0-100], Income: [0-1000000], Binary: [0-1]

# Check feature scales
print("Feature statistics:")
print(X_train.mean(axis=0))
print(X_train.std(axis=0))

# If std varies by >100x, normalize!

# Solution: Standardize features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)  # Use same scaler!

# For images: Normalize to [0, 1] or [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),  # [0, 255] → [0, 1]
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
])

# For neural networks: Always normalize!
```

---

## 📉 Overfitting Issues

### Symptom: Train accuracy >> Validation accuracy

**Example:**
```
Epoch 10: Train Acc=99%, Val Acc=65%
Epoch 20: Train Acc=100%, Val Acc=60%
```

**Solutions:**

#### 1. Get More Data
```python
# Best solution: Collect more training data
# Rule of thumb: 10x parameters as training samples

# If can't get more data, use data augmentation

from torchvision import transforms

# For images
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# For text (using nlpaug)
import nlpaug.augmenter.word as naw

aug = naw.SynonymAug(aug_src='wordnet')
augmented_text = aug.augment(original_text)

# For tabular data (SMOTE)
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

#### 2. Add Regularization
```python
# L2 regularization (weight decay)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# L1 regularization (manual)
l1_lambda = 0.001
l1_norm = sum(p.abs().sum() for p in model.parameters())
loss = loss + l1_lambda * l1_norm

# Dropout
model = nn.Sequential(
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Dropout(0.5),  # Drop 50% of neurons during training
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.3),  # 30% dropout
    nn.Linear(128, 10)
)

# Dropout for CNNs (spatial dropout)
model = nn.Sequential(
    nn.Conv2d(3, 64, 3),
    nn.ReLU(),
    nn.Dropout2d(0.25),  # Drop entire feature maps
    nn.Conv2d(64, 128, 3),
    nn.ReLU(),
    nn.Dropout2d(0.25)
)
```

#### 3. Early Stopping
```python
class EarlyStopping:
    """Stop training when validation loss stops improving"""

    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Usage
early_stopping = EarlyStopping(patience=10)

for epoch in range(1000):
    train_loss = train_epoch(model, train_loader)
    val_loss = validate(model, val_loader)

    early_stopping(val_loss)

    if early_stopping.early_stop:
        print(f"Early stopping at epoch {epoch}")
        break
```

#### 4. Reduce Model Complexity
```python
# Too complex (overfits)
model = nn.Sequential(
    nn.Linear(10, 1024),
    nn.ReLU(),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 2)
)
# 10×1024 + 1024×512 + 512×256 + 256×2 = 665,600 parameters!

# Simpler (better generalization)
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(32, 2)
)
# 10×64 + 64×32 + 32×2 = 2,752 parameters

# Rule of thumb: Start simple, add complexity only if underfitting
```

#### 5. Ensemble Methods
```python
from sklearn.ensemble import VotingClassifier

# Combine multiple models to reduce overfitting
model1 = LogisticRegression()
model2 = RandomForestClassifier()
model3 = GradientBoostingClassifier()

ensemble = VotingClassifier(
    estimators=[('lr', model1), ('rf', model2), ('gb', model3)],
    voting='soft'  # Average probabilities
)

ensemble.fit(X_train, y_train)
```

---

## 📈 Underfitting Issues

### Symptom: Both train and validation accuracy low

**Example:**
```
Epoch 50: Train Acc=60%, Val Acc=58%
Epoch 100: Train Acc=62%, Val Acc=60%
```

**Solutions:**

#### 1. Increase Model Capacity
```python
# Too simple (underfits)
model = nn.Sequential(
    nn.Linear(100, 10)
)

# Add more layers and neurons
model = nn.Sequential(
    nn.Linear(100, 256),
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.BatchNorm1d(128),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)
```

#### 2. Train Longer
```python
# Increase number of epochs
# Check if loss is still decreasing

history = {'train_loss': [], 'val_loss': []}

for epoch in range(200):  # Instead of 50
    train_loss = train_epoch(model, train_loader)
    val_loss = validate(model, val_loader)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)

# Plot to see if still improving
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Validation')
plt.legend()
plt.show()
```

#### 3. Add More Features
```python
# Feature engineering
df['age_squared'] = df['age'] ** 2
df['income_log'] = np.log(df['income'] + 1)
df['age_income_interaction'] = df['age'] * df['income']

# Polynomial features
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Feature crosses
df['feature_cross'] = df['feature1'] * df['feature2']
```

#### 4. Reduce Regularization
```python
# Too much regularization causes underfitting

# Before (too much)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-2)
dropout_rate = 0.7

# After (reduced)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
dropout_rate = 0.2
```

#### 5. Use Better Model Architecture
```python
# For images: Use CNNs instead of fully connected
# Before (underfits images)
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# After (better for images)
model = nn.Sequential(
    nn.Conv2d(1, 32, 3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(64 * 5 * 5, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# For sequences: Use RNNs/Transformers instead of feed-forward
# For tabular: Try Gradient Boosting (XGBoost, LightGBM)
```

---

## ⚠️ Training Instability

### Symptom: Loss spikes or NaN values

#### 1. Loss Becomes NaN
```python
# Causes:
# - Learning rate too high
# - Numerical instability
# - Gradient explosion

# Solutions:

# 1. Reduce learning rate
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# 2. Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 3. Check for NaN in data
assert not torch.isnan(data).any(), "NaN in input data!"
assert not torch.isinf(data).any(), "Inf in input data!"

# 4. Use numerically stable operations
# Instead of: log(x)
# Use: log(x + epsilon)
epsilon = 1e-8
loss = -torch.log(predictions + epsilon)

# 5. Mixed precision training (can help)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in train_loader:
    optimizer.zero_grad()

    with autocast():
        output = model(data)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

#### 2. Loss Oscillates
```python
# Symptom: Loss goes up and down erratically
# Epoch 1: 2.5, Epoch 2: 1.8, Epoch 3: 3.2, Epoch 4: 2.1

# Solutions:

# 1. Reduce learning rate
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 2. Increase batch size (more stable gradients)
train_loader = DataLoader(train_dataset, batch_size=128)  # Instead of 32

# 3. Use learning rate scheduling
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(100):
    train_epoch(model, train_loader, optimizer)
    scheduler.step()

# 4. Add batch normalization
model = nn.Sequential(
    nn.Linear(10, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
```

---

## 💾 Memory Issues

### Symptom: CUDA out of memory / RAM exhausted

#### 1. GPU Out of Memory
```python
# Error: RuntimeError: CUDA out of memory

# Solutions:

# 1. Reduce batch size
train_loader = DataLoader(dataset, batch_size=16)  # Instead of 64

# 2. Use gradient accumulation (simulate larger batch)
accumulation_steps = 4  # Effective batch size = 16 × 4 = 64

optimizer.zero_grad()
for i, (data, target) in enumerate(train_loader):
    output = model(data)
    loss = criterion(output, target)
    loss = loss / accumulation_steps  # Normalize loss

    loss.backward()  # Accumulate gradients

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 3. Use mixed precision training (half memory)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in train_loader:
    with autocast():  # Use FP16 instead of FP32
        output = model(data)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

# 4. Clear cache
torch.cuda.empty_cache()

# 5. Delete unused variables
del large_tensor
torch.cuda.empty_cache()

# 6. Use gradient checkpointing (trade compute for memory)
from torch.utils.checkpoint import checkpoint

def forward_with_checkpointing(model, x):
    return checkpoint(model, x)
```

#### 2. CPU Memory Issues
```python
# Problem: Dataset too large for RAM

# Solutions:

# 1. Load data in chunks (pandas)
chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    # Process chunk
    process(chunk)

# 2. Use Dask for out-of-core computation
import dask.dataframe as dd

df = dd.read_csv('large_file_*.csv')
result = df.groupby('column').mean().compute()

# 3. Use HDF5 for efficient storage
import h5py

with h5py.File('data.h5', 'r') as f:
    # Load only what you need
    subset = f['dataset'][1000:2000]

# 4. Custom Dataset with lazy loading
class LazyDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Load data only when needed
        data = load_file(self.file_list[idx])
        return data
```

---

## 🐌 Slow Training

### Solutions to Speed Up Training

#### 1. Use GPU
```python
# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")

# Move model and data to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

for data, target in train_loader:
    data, target = data.to(device), target.to(device)
    # Train...
```

#### 2. Optimize DataLoader
```python
# Slow
train_loader = DataLoader(dataset, batch_size=32, num_workers=0)

# Fast
train_loader = DataLoader(
    dataset,
    batch_size=64,  # Larger batch (if memory allows)
    num_workers=4,  # Parallel data loading
    pin_memory=True,  # Faster GPU transfer
    persistent_workers=True  # Keep workers alive
)
```

#### 3. Mixed Precision Training
```python
# 2x speedup with minimal accuracy loss

from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in train_loader:
    optimizer.zero_grad()

    with autocast():  # FP16 operations
        output = model(data)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

#### 4. Compile Model (PyTorch 2.0+)
```python
# Significant speedup with torch.compile

model = torch.compile(model)  # Optimize for your hardware

# Then train as usual
for data, target in train_loader:
    output = model(data)
    # ...
```

#### 5. Profile Code to Find Bottlenecks
```python
import cProfile
import pstats

def train_epoch(model, loader, optimizer):
    for data, target in loader:
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Profile
cProfile.run('train_epoch(model, train_loader, optimizer)', 'profile_stats')

# Analyze
p = pstats.Stats('profile_stats')
p.sort_stats('cumulative').print_stats(10)

# Or use PyTorch profiler
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    train_epoch(model, train_loader, optimizer)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

---

## 📊 Data Problems

### 1. Class Imbalance
```python
# Problem: 99% class 0, 1% class 1

# Solutions:

# 1. Use class weights
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = torch.FloatTensor(class_weights).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

# 2. Oversample minority class
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 3. Undersample majority class
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

# 4. Use appropriate metrics (not accuracy!)
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# Accuracy is misleading for imbalanced data!
# Use F1, precision, recall, or AUC instead
```

### 2. Data Leakage
```python
# Problem: Information from test set leaks into training

# Common mistakes:

# ❌ WRONG: Scaling before splitting
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Uses ALL data!
X_train, X_test = train_test_split(X_scaled, test_size=0.2)

# ✅ CORRECT: Scale after splitting
X_train, X_test = train_test_split(X, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on train only
X_test_scaled = scaler.transform(X_test)  # Transform test using train stats

# ❌ WRONG: Feature selection using all data
selector = SelectKBest(k=10)
X_selected = selector.fit_transform(X, y)  # Leak!

# ✅ CORRECT: Feature selection on train only
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
selector = SelectKBest(k=10)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
```

### 3. Missing Values
```python
# Check for missing values
print(df.isnull().sum())

# Visualize missing data
import missingno as msno
msno.matrix(df)
plt.show()

# Strategies:

# 1. Remove rows with missing values (if <5% missing)
df_clean = df.dropna()

# 2. Impute with simple statistics
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')  # or 'mean', 'most_frequent'
X_imputed = imputer.fit_transform(X)

# 3. Impute with KNN
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)

# 4. Create missing indicator feature
df['age_missing'] = df['age'].isnull().astype(int)
df['age'] = df['age'].fillna(df['age'].median())
```

---

## 🚀 Deployment Issues

### 1. Model Works in Development but Fails in Production
```python
# Common issues:

# Issue 1: Different Python/library versions
# Solution: Use requirements.txt with exact versions
# requirements.txt:
# torch==2.0.0
# numpy==1.24.0
# scikit-learn==1.2.2

# Issue 2: Missing model.eval()
# Training mode vs eval mode differences
model.train()  # Dropout ON, BatchNorm uses batch stats
model.eval()   # Dropout OFF, BatchNorm uses running stats

# Always use eval() for inference!
model.eval()
with torch.no_grad():
    predictions = model(input_data)

# Issue 3: Not saving preprocessing
# Save scaler/encoder with model
import joblib

model_data = {
    'model': model.state_dict(),
    'scaler': scaler,
    'encoder': encoder,
    'feature_names': feature_names
}
joblib.dump(model_data, 'model_package.pkl')

# Load everything
model_data = joblib.load('model_package.pkl')
model.load_state_dict(model_data['model'])
scaler = model_data['scaler']
```

### 2. Slow Inference
```python
# Solutions:

# 1. Use model.eval() and torch.no_grad()
model.eval()
with torch.no_grad():  # Disables gradient computation
    predictions = model(data)

# 2. Batch predictions
def predict_batch(model, data_list, batch_size=32):
    """Batch predictions for efficiency"""
    predictions = []

    for i in range(0, len(data_list), batch_size):
        batch = data_list[i:i+batch_size]
        with torch.no_grad():
            batch_preds = model(batch)
        predictions.append(batch_preds)

    return torch.cat(predictions)

# 3. Model quantization (INT8)
import torch.quantization

model_quantized = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},  # Quantize Linear layers
    dtype=torch.qint8
)
# 4x smaller, 2-4x faster inference

# 4. ONNX export for optimized inference
import torch.onnx

dummy_input = torch.randn(1, 10)
torch.onnx.export(model, dummy_input, "model.onnx")

# Use ONNX Runtime for faster inference
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
inputs = {session.get_inputs()[0].name: input_data.numpy()}
outputs = session.run(None, inputs)
```

### 3. Model Drift
```python
# Problem: Model performance degrades over time

# Solutions:

# 1. Monitor predictions
from scipy.stats import ks_2samp

def detect_data_drift(reference_data, new_data, threshold=0.05):
    """Detect distribution shift using KS test"""
    drifted_features = []

    for i in range(reference_data.shape[1]):
        statistic, p_value = ks_2samp(
            reference_data[:, i],
            new_data[:, i]
        )

        if p_value < threshold:
            drifted_features.append(i)
            print(f"⚠️ Drift detected in feature {i}: p={p_value:.4f}")

    return drifted_features

# 2. Retrain regularly
# Set up automated retraining pipeline

# 3. Monitor model metrics in production
def log_prediction_quality(predictions, actuals):
    """Track model performance over time"""
    from sklearn.metrics import accuracy_score

    accuracy = accuracy_score(actuals, predictions)
    timestamp = pd.Timestamp.now()

    # Log to database/monitoring service
    log_metric(timestamp, 'accuracy', accuracy)

    if accuracy < 0.8:  # Threshold
        send_alert("Model performance degraded!")
        trigger_retraining()
```

---

## 🎯 Quick Debugging Checklist

When your model isn't working, go through this checklist:

**Data**
- [ ] Check for NaN/Inf values
- [ ] Verify data types are correct
- [ ] Check class distribution (imbalance?)
- [ ] Visualize a few samples
- [ ] Check train/val/test split is correct

**Model**
- [ ] Print model architecture
- [ ] Count parameters (too many = overfit, too few = underfit)
- [ ] Check output shape matches target
- [ ] Verify loss function is appropriate

**Training**
- [ ] Monitor train AND validation loss
- [ ] Check gradients aren't NaN/exploding/vanishing
- [ ] Verify learning rate is reasonable
- [ ] Try different optimizers
- [ ] Check data augmentation isn't too aggressive

**Debugging**
- [ ] Overfit on small subset (should reach ~100%)
- [ ] Check gradient flow
- [ ] Visualize predictions
- [ ] Profile for bottlenecks

---

*When in doubt, start simple and add complexity gradually!*
