# 📝 ML/AI Cheat Sheets

## Quick reference guides for practitioners

---

## 📚 Table of Contents

1. [Model Selection](#model-selection)
2. [Hyperparameter Tuning](#hyperparameters)
3. [Loss Functions](#loss-functions)
4. [Evaluation Metrics](#metrics)
5. [Common Architectures](#architectures)
6. [Training Tips](#training-tips)
7. [Debugging Guide](#debugging)
8. [Production Deployment](#deployment)

---

<a id='model-selection'></a>
## 1. 🎯 Model Selection Cheat Sheet

### Decision Tree

```
START HERE: What type of data?
    │
    ├─→ TABULAR/STRUCTURED
    │   │
    │   ├─→ Small dataset (<10K rows)
    │   │   └─→ Logistic Regression / Decision Tree
    │   │
    │   ├─→ Medium dataset (10K-1M rows)
    │   │   ├─→ Classification: Random Forest, XGBoost
    │   │   └─→ Regression: Random Forest, LightGBM
    │   │
    │   └─→ Large dataset (>1M rows)
    │       └─→ XGBoost, LightGBM, Deep Learning
    │
    ├─→ IMAGES
    │   ├─→ Classification: ResNet, EfficientNet
    │   ├─→ Object Detection: YOLO, Faster R-CNN
    │   └─→ Segmentation: U-Net, Mask R-CNN
    │
    ├─→ TEXT
    │   ├─→ Classification: BERT, RoBERTa
    │   ├─→ Generation: GPT, T5
    │   └─→ Translation: mT5, NLLB
    │
    ├─→ TIME SERIES
    │   ├─→ Forecasting: ARIMA, Prophet, LSTM
    │   ├─→ Anomaly: Isolation Forest, Autoencoder
    │   └─→ Classification: LSTM, Transformer
    │
    └─→ SEQUENCES (general)
        └─→ Transformer, LSTM, GRU
```

### Quick Selection Guide

| Task | Dataset Size | Recommended | Why |
|------|-------------|-------------|-----|
| **Binary Classification** | <1K | Logistic Regression | Simple, interpretable |
| | 1K-100K | Random Forest | Robust, no tuning |
| | >100K | XGBoost | Best performance |
| **Multi-class** | <10K | SVM, Random Forest | Handle multiple classes well |
| | >10K | XGBoost, Neural Net | Scale better |
| **Regression** | <10K | Linear Regression | Baseline |
| | >10K | XGBoost | Usually best |
| **Image** | Any | ResNet/EfficientNet | Transfer learning |
| **Text** | <10K | TF-IDF + SVM | Classical works |
| | >10K | BERT fine-tuned | State-of-the-art |
| **Time Series** | <1000 points | ARIMA | Statistical |
| | >1000 points | LSTM/Transformer | Capture patterns |

---

<a id='hyperparameters'></a>
## 2. ⚙️ Hyperparameter Tuning Guide

### Neural Networks

```python
# Starting point (usually works)
config = {
    'learning_rate': 0.001,      # Most important!
    'batch_size': 32,            # 16/32/64 for most tasks
    'optimizer': 'Adam',         # Default choice
    'epochs': 50,                # With early stopping
    'dropout': 0.2,              # Light regularization
}

# If underfitting (low train accuracy):
config = {
    'learning_rate': 0.0001,     # Try smaller
    'batch_size': 16,            # Smaller batches
    'model_size': 'larger',      # More parameters
    'dropout': 0.1,              # Less dropout
}

# If overfitting (train >> test accuracy):
config = {
    'learning_rate': 0.0005,     # Slightly smaller
    'dropout': 0.5,              # More dropout
    'weight_decay': 0.01,        # L2 regularization
    'data_augmentation': True,   # More data
}
```

### XGBoost

```python
# Good defaults
params = {
    'max_depth': 6,              # Tree depth
    'learning_rate': 0.1,        # Step size
    'n_estimators': 100,         # Number of trees
    'subsample': 0.8,            # Row sampling
    'colsample_bytree': 0.8,     # Column sampling
}

# For better performance (slower):
params = {
    'max_depth': 8,
    'learning_rate': 0.05,       # Slower, more trees
    'n_estimators': 500,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
}

# For speed (less accuracy):
params = {
    'max_depth': 4,
    'learning_rate': 0.3,        # Faster convergence
    'n_estimators': 50,
    'subsample': 0.9,
}
```

### Learning Rate Finder

```python
# Find optimal learning rate
learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

for lr in learning_rates:
    model = create_model()
    optimizer = Adam(lr=lr)
    train(model, optimizer, epochs=5)
    log(lr, validation_loss)

# Plot: Loss vs LR
# Optimal: Point where loss decreases fastest
```

---

<a id='loss-functions'></a>
## 3. 📉 Loss Functions Cheat Sheet

### Regression

| Loss | Formula | When to Use | Pros | Cons |
|------|---------|-------------|------|------|
| **MSE** | (y - ŷ)² | Default | Smooth gradient | Sensitive to outliers |
| **MAE** | \|y - ŷ\| | Robust | Outlier resistant | Non-smooth at 0 |
| **Huber** | Hybrid MSE/MAE | Best of both | Robust + smooth | Extra hyperparameter |
| **Log-Cosh** | log(cosh(y - ŷ)) | Alternative | Smooth + robust | Complex |

### Classification

| Loss | Formula | When to Use | Notes |
|------|---------|-------------|-------|
| **Binary Cross-Entropy** | -[y log(ŷ) + (1-y) log(1-ŷ)] | Binary class | Requires sigmoid output |
| **Categorical Cross-Entropy** | -Σ y_i log(ŷ_i) | Multi-class | Requires softmax output |
| **Sparse Categorical** | Same as above | Integer labels | No one-hot encoding needed |
| **Focal Loss** | -(1-ŷ)^γ log(ŷ) | Imbalanced | Focuses on hard examples |

### Ranking/Similarity

| Loss | Use Case | Description |
|------|----------|-------------|
| **Triplet Loss** | Face recognition | Anchor, positive, negative |
| **Contrastive Loss** | Siamese networks | Similar vs dissimilar pairs |
| **Cosine Embedding** | Embeddings | Cosine similarity |

### Code Examples

```python
import torch
import torch.nn as nn

# Regression
mse_loss = nn.MSELoss()
mae_loss = nn.L1Loss()
huber_loss = nn.SmoothL1Loss()

# Classification
bce_loss = nn.BCELoss()                    # Binary
ce_loss = nn.CrossEntropyLoss()            # Multi-class
focal_loss = FocalLoss(gamma=2, alpha=0.25)  # Imbalanced

# Custom: Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss
```

---

<a id='metrics'></a>
## 4. 📊 Evaluation Metrics

### Classification Metrics

```
                  PREDICTED
                  Pos    Neg
        ┌─────────────────────┐
  Pos   │   TP  │   FN       │
ACTUAL  ├───────┼────────────┤
  Neg   │   FP  │   TN       │
        └─────────────────────┘

Accuracy  = (TP + TN) / Total
Precision = TP / (TP + FP)     "Of predicted positives, % correct"
Recall    = TP / (TP + FN)     "Of actual positives, % found"
F1-Score  = 2 × P×R / (P+R)    "Harmonic mean of P and R"
```

### When to Use Each Metric

| Metric | Use When | Example |
|--------|----------|---------|
| **Accuracy** | Balanced classes | General classification |
| **Precision** | False positives costly | Spam detection (don't mark real emails as spam) |
| **Recall** | False negatives costly | Disease detection (don't miss sick patients) |
| **F1-Score** | Need balance | Most real-world scenarios |
| **AUC-ROC** | Compare models | Model selection |
| **PR-AUC** | Imbalanced data | Fraud detection (1% fraud) |

### Regression Metrics

| Metric | Formula | Interpretation | Good Value |
|--------|---------|----------------|------------|
| **MAE** | Σ\|y - ŷ\|/n | Average error | Lower is better |
| **RMSE** | √(Σ(y - ŷ)²/n) | Penalizes large errors | Lower is better |
| **R²** | 1 - SS_res/SS_tot | Variance explained | Closer to 1 |
| **MAPE** | Σ\|y - ŷ\|/y/n | % error | Lower is better |

### Code Examples

```python
from sklearn.metrics import *

# Classification
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_pred_proba)

# Detailed report
print(classification_report(y_true, y_pred))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Regression
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)
```

---

<a id='architectures'></a>
## 5. 🏗️ Common Architectures

### CNN Architectures

```python
# Basic CNN
nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(64 * 8 * 8, 10)
)

# ResNet Block (Skip Connection)
class ResBlock(nn.Module):
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual  # Skip connection
        return F.relu(out)
```

### Architecture Quick Reference

| Architecture | Parameters | Best For | Key Feature |
|--------------|-----------|----------|-------------|
| **LeNet** | 60K | MNIST digits | Historical |
| **AlexNet** | 60M | ImageNet 2012 | First deep CNN |
| **VGG16** | 138M | Transfer learning | Simple, deep |
| **ResNet50** | 25M | Most vision tasks | Skip connections |
| **EfficientNet** | 5-66M | Efficiency | Compound scaling |
| **Vision Transformer** | 86M | Large datasets | Attention-based |

### Transformer

```python
# Basic Transformer Encoder
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, x):
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x
```

---

<a id='training-tips'></a>
## 6. 💡 Training Tips

### Training Checklist

```
BEFORE TRAINING
  □ Normalize/standardize inputs
  □ Shuffle training data
  □ Split: 80% train, 10% val, 10% test
  □ Set random seeds (reproducibility)
  □ Check data leakage

DURING TRAINING
  □ Monitor train AND validation loss
  □ Use early stopping
  □ Save best model (by val loss)
  □ Log metrics every epoch
  □ Visualize training curves

AFTER TRAINING
  □ Evaluate on test set (once!)
  □ Check for overfitting/underfitting
  □ Analyze errors (confusion matrix)
  □ Test edge cases
```

### Common Issues & Solutions

| Problem | Symptom | Solution |
|---------|---------|----------|
| **Overfitting** | Train loss ↓, Val loss ↑ | More data, dropout, regularization |
| **Underfitting** | Both losses high | Bigger model, more features, less regularization |
| **Vanishing gradient** | Early layers not learning | ReLU, skip connections, batch norm |
| **Exploding gradient** | Loss becomes NaN | Gradient clipping, lower learning rate |
| **Slow convergence** | Loss decreases slowly | Higher learning rate, better optimizer (Adam) |
| **Class imbalance** | Ignores minority class | Class weights, oversampling, focal loss |

### Learning Rate Schedules

```python
# Constant (default)
lr = 0.001

# Step decay
lr = 0.001 * (0.1 ** (epoch // 30))

# Exponential decay
lr = 0.001 * (0.95 ** epoch)

# Cosine annealing
lr = 0.001 * 0.5 * (1 + cos(π * epoch / max_epochs))

# PyTorch
from torch.optim.lr_scheduler import *

scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
scheduler = CosineAnnealingLR(optimizer, T_max=100)
scheduler = ReduceLROnPlateau(optimizer, patience=10)
```

---

<a id='debugging'></a>
## 7. 🐛 Debugging Neural Networks

### Debugging Checklist

```
1. START SIMPLE
   □ Train on 1 batch (should overfit perfectly)
   □ If can't overfit → Bug in model/loss
   □ Use tiny dataset first

2. CHECK DATA
   □ Visualize samples
   □ Check labels are correct
   □ Verify normalization
   □ Look for duplicates

3. CHECK MODEL
   □ Print model summary
   □ Verify output shape
   □ Check parameter count
   □ Test forward pass

4. CHECK TRAINING
   □ Monitor gradients (not too large/small)
   □ Check learning rate
   □ Verify loss decreases
   □ Compare to baseline

5. COMMON BUGS
   □ Wrong loss function
   □ Forgot to zero gradients
   □ Data leakage
   □ Incorrect input shape
   □ Missing activation functions
```

### Debug Code

```python
# 1. Overfit single batch
batch = next(iter(train_loader))
for epoch in range(1000):
    loss = model(batch)
    # Should go to ~0

# 2. Check shapes
print(f"Input: {x.shape}")
print(f"Output: {model(x).shape}")
print(f"Target: {y.shape}")

# 3. Check gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")
    # Too large (>10): Exploding
    # Too small (<0.001): Vanishing

# 4. Verify loss
loss = criterion(output, target)
print(f"Loss: {loss.item()}")
# Check if reasonable

# 5. Compare to baseline
# Random model: ~ln(num_classes) for cross-entropy
# Your model should be much better
```

---

<a id='deployment'></a>
## 8. 🚀 Production Deployment

### Deployment Checklist

```
PRE-DEPLOYMENT
  □ Model accuracy meets threshold
  □ Latency < SLA (e.g., 100ms)
  □ Memory usage acceptable
  □ Tested on edge cases
  □ Security audit passed
  □ Documentation complete
  □ Monitoring setup
  □ Rollback plan ready

OPTIMIZATION
  □ Quantize model (FP32 → INT8)
  □ Prune unnecessary weights
  □ Use optimized runtime (ONNX, TensorRT)
  □ Batch requests
  □ Cache common queries

MONITORING
  □ Track latency (p50, p95, p99)
  □ Monitor prediction distribution
  □ Log errors and exceptions
  □ A/B test new models
  □ Collect user feedback
```

### Model Serving

```python
# FastAPI serving
from fastapi import FastAPI
import torch

app = FastAPI()
model = torch.load('model.pt')
model.eval()

@app.post("/predict")
async def predict(data: InputData):
    with torch.no_grad():
        output = model(data.to_tensor())
    return {"prediction": output.tolist()}

# Run: uvicorn app:app --host 0.0.0.0 --port 8000
```

### Optimization

```python
# 1. Quantization (4x smaller, 2-4x faster)
import torch.quantization as quant

model_quantized = quant.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# 2. ONNX export (framework-agnostic)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=11
)

# 3. TorchScript (optimized execution)
scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")
```

---

## 🎯 Quick Command Reference

### PyTorch

```python
# Create model
model = MyModel()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Loss
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(batch.x)
        loss = criterion(outputs, batch.y)
        loss.backward()
        optimizer.step()

# Save/Load
torch.save(model.state_dict(), 'model.pth')
model.load_state_dict(torch.load('model.pth'))
```

### Scikit-learn

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
from sklearn.metrics import accuracy_score
print(f"Accuracy: {accuracy_score(y_test, predictions)}")
```

---

## 📚 Additional Resources

- **[ML/AI Glossary](ML_AI_GLOSSARY.md)** - Comprehensive terminology
- **[LLM Fundamentals](llm_fundamentals/)** - Language models deep dive
- **[Neural Networks](neural_networks/)** - From scratch tutorials
- **[RAG Tutorial](rag_tutorial/)** - Retrieval augmented generation

**Print this and keep it handy! 📋**
