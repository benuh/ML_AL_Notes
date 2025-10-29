# ML/AI Quick Reference & Cheat Sheets

**Fast-Access Reference for Machine Learning and AI**

> One-page summaries, decision trees, and code snippets for ML practitioners

---

## Table of Contents

1. [Algorithm Selection](#1-algorithm-selection)
2. [Common Formulas](#2-common-formulas)
3. [Code Snippets](#3-code-snippets)
4. [Evaluation Metrics](#4-evaluation-metrics)
5. [Hyperparameter Ranges](#5-hyperparameter-ranges)
6. [Data Processing](#6-data-processing)
7. [Deep Learning](#7-deep-learning)
8. [Common Pitfalls](#8-common-pitfalls)
9. [Interview Cheat Sheet](#9-interview-cheat-sheet)
10. [Quick Debugging Guide](#10-quick-debugging-guide)

---

## 1. Algorithm Selection

### Decision Tree: Which Algorithm to Use?

```
START
│
├─ Classification or Regression?
│  │
│  ├─ CLASSIFICATION
│  │  │
│  │  ├─ Linear separable? → Logistic Regression
│  │  ├─ Need interpretability? → Decision Tree
│  │  ├─ High accuracy needed? → Random Forest / XGBoost
│  │  ├─ Many features (>10K)? → Linear SVM
│  │  ├─ Images? → CNN
│  │  ├─ Text? → BERT / Transformers
│  │  └─ Tabular data? → XGBoost (default choice)
│  │
│  └─ REGRESSION
│     │
│     ├─ Linear relationship? → Linear Regression
│     ├─ Outliers present? → Huber / RANSAC
│     ├─ Need interpretability? → Ridge with L2
│     ├─ Feature selection? → Lasso with L1
│     ├─ Non-linear? → Random Forest / XGBoost
│     └─ Time series? → ARIMA / LSTM

├─ Clustering?
│  │
│  ├─ Know # clusters? → K-Means
│  ├─ Hierarchical structure? → Agglomerative
│  ├─ Arbitrary shapes? → DBSCAN
│  └─ High-dimensional? → HDBSCAN

└─ Dimensionality Reduction?
   │
   ├─ Linear relationships? → PCA
   ├─ Visualization? → t-SNE / UMAP
   └─ Non-linear? → Kernel PCA / Autoencoders
```

### Algorithm Comparison Matrix

| Algorithm | Speed | Accuracy | Interpretability | Handles Missing | Handles Categorical | Overfitting Risk |
|-----------|-------|----------|------------------|-----------------|---------------------|------------------|
| **Linear Regression** | ⚡⚡⚡ | ⭐⭐ | ✅ High | ❌ | ❌ | Low |
| **Logistic Regression** | ⚡⚡⚡ | ⭐⭐⭐ | ✅ High | ❌ | ❌ | Low |
| **Decision Tree** | ⚡⚡ | ⭐⭐ | ✅ High | ✅ | ✅ | High |
| **Random Forest** | ⚡ | ⭐⭐⭐⭐ | ⚠️ Medium | ✅ | ✅ | Medium |
| **XGBoost** | ⚡⚡ | ⭐⭐⭐⭐⭐ | ⚠️ Medium | ✅ | ⚠️ | Medium |
| **SVM** | ⚡ | ⭐⭐⭐⭐ | ❌ Low | ❌ | ❌ | Low-Medium |
| **K-NN** | ⚡⚡⚡ (pred slow) | ⭐⭐⭐ | ✅ High | ❌ | ❌ | High |
| **Neural Network** | ⚡ | ⭐⭐⭐⭐⭐ | ❌ Low | ❌ | ⚠️ | High |
| **Naive Bayes** | ⚡⚡⚡ | ⭐⭐ | ✅ High | ✅ | ✅ | Low |

**Legend:**
- Speed: ⚡⚡⚡ (Fast) → ⚡ (Slow)
- Accuracy: ⭐ (Low) → ⭐⭐⭐⭐⭐ (High)
- ✅ Yes / ⚠️ Partial / ❌ No

---

## 2. Common Formulas

### Loss Functions

```python
# Mean Squared Error (Regression)
MSE = (1/n) * Σ(y_true - y_pred)²

# Cross-Entropy (Classification)
CE = -Σ y_true * log(y_pred)

# Hinge Loss (SVM)
L = max(0, 1 - y_true * y_pred)

# Huber Loss (Robust Regression)
L = { 0.5 * (y_true - y_pred)²           if |error| ≤ δ
    { δ * (|error| - 0.5 * δ)            otherwise
```

### Regularization

```python
# L1 Regularization (Lasso)
L = Loss + λ * Σ|w_i|

# L2 Regularization (Ridge)
L = Loss + λ * Σw_i²

# Elastic Net
L = Loss + λ₁ * Σ|w_i| + λ₂ * Σw_i²
```

### Gradient Descent

```python
# Vanilla Gradient Descent
θ_new = θ_old - α * ∇L(θ)

# Momentum
v_t = β * v_{t-1} + α * ∇L(θ)
θ_new = θ_old - v_t

# Adam (Adaptive Moment Estimation)
m_t = β₁ * m_{t-1} + (1-β₁) * ∇L(θ)
v_t = β₂ * v_{t-1} + (1-β₂) * (∇L(θ))²
θ_new = θ_old - α * m_t / (√v_t + ε)
```

### Evaluation Metrics

```python
# Precision & Recall
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1-Score = 2 * (Precision * Recall) / (Precision + Recall)

# R² Score (Regression)
R² = 1 - (Σ(y_true - y_pred)² / Σ(y_true - ȳ)²)

# AUC-ROC
Area under ROC curve (TPR vs FPR)

# Information Gain (Decision Trees)
IG(S, A) = H(S) - Σ(|S_v|/|S|) * H(S_v)
where H(S) = -Σ p_i * log₂(p_i)  # Entropy
```

### Probability & Statistics

```python
# Bayes' Theorem
P(A|B) = P(B|A) * P(A) / P(B)

# Variance & Standard Deviation
Var(X) = E[(X - μ)²]
Std(X) = √Var(X)

# Covariance
Cov(X,Y) = E[(X - μ_x)(Y - μ_y)]

# Correlation
ρ(X,Y) = Cov(X,Y) / (σ_x * σ_y)
```

---

## 3. Code Snippets

### Data Loading & Preprocessing

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('data.csv')

# Quick EDA
df.info()
df.describe()
df.isnull().sum()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Quick Model Training

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.3f}")
```

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score

# 5-fold CV
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.3f}")
```

### Feature Engineering

```python
# One-hot encoding
df_encoded = pd.get_dummies(df, columns=['category'], drop_first=True)

# Target encoding (with cross-validation to prevent leakage)
from category_encoders import TargetEncoder
encoder = TargetEncoder()
df['category_encoded'] = encoder.fit_transform(df['category'], df['target'])

# Date features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Interaction features
df['feature_interaction'] = df['feature1'] * df['feature2']

# Binning
df['age_bin'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 100],
                        labels=['child', 'young', 'middle', 'senior'])
```

### Handling Missing Values

```python
# Simple imputation
from sklearn.impute import SimpleImputer

# Mean/median for numerical
imputer_num = SimpleImputer(strategy='median')
df[num_cols] = imputer_num.fit_transform(df[num_cols])

# Most frequent for categorical
imputer_cat = SimpleImputer(strategy='most_frequent')
df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

# Advanced: Iterative imputation
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(random_state=42)
df_imputed = imputer.fit_transform(df)
```

### Model Interpretability (SHAP)

```python
import shap

# TreeSHAP (for tree models)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Force plot for single prediction
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])
```

### Deep Learning (PyTorch)

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define model
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize
model = SimpleNN(input_size=10, hidden_size=64, output_size=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
```

---

## 4. Evaluation Metrics

### Classification Metrics Cheat Sheet

| Metric | Formula | When to Use | Range |
|--------|---------|-------------|-------|
| **Accuracy** | (TP + TN) / Total | Balanced classes | [0, 1] |
| **Precision** | TP / (TP + FP) | Cost of FP high | [0, 1] |
| **Recall** | TP / (TP + FN) | Cost of FN high | [0, 1] |
| **F1-Score** | 2 × P × R / (P + R) | Balance P & R | [0, 1] |
| **AUC-ROC** | Area under ROC | Overall performance | [0, 1] |
| **AUC-PR** | Area under PR curve | Imbalanced data | [0, 1] |
| **Log Loss** | -Σ y log(ŷ) | Probability quality | [0, ∞) |
| **Matthews CC** | Complex formula | Imbalanced data | [-1, 1] |

### Regression Metrics Cheat Sheet

| Metric | Formula | When to Use | Best Value |
|--------|---------|-------------|------------|
| **MAE** | (1/n) Σ\|y - ŷ\| | Outliers present | 0 |
| **MSE** | (1/n) Σ(y - ŷ)² | Penalize large errors | 0 |
| **RMSE** | √MSE | Same unit as target | 0 |
| **R²** | 1 - SS_res/SS_tot | % variance explained | 1 |
| **MAPE** | (100/n) Σ\|(y-ŷ)/y\| | % error | 0 |
| **Huber** | Piecewise MSE/MAE | Robust to outliers | 0 |

### Metric Selection Guide

```
Classification:
├─ Balanced classes? → Accuracy, F1-Score
├─ Imbalanced classes? → AUC-PR, F1-Score
├─ False positives costly? → Precision
├─ False negatives costly? → Recall
└─ Probability predictions? → Log Loss

Regression:
├─ Outliers present? → MAE, Huber
├─ Large errors very bad? → MSE, RMSE
├─ Percentage error? → MAPE
└─ Variance explained? → R²

Ranking:
├─ Order matters? → NDCG
├─ Top-k performance? → Precision@k, Recall@k
└─ Overall ranking? → MAP (Mean Average Precision)
```

---

## 5. Hyperparameter Ranges

### Random Forest

```python
{
    'n_estimators': [100, 200, 500, 1000],      # More trees → better (but slower)
    'max_depth': [10, 20, 30, None],            # None = unlimited
    'min_samples_split': [2, 5, 10],            # Higher → less overfitting
    'min_samples_leaf': [1, 2, 4],              # Higher → smoother
    'max_features': ['sqrt', 'log2', None],     # sqrt for classification
    'bootstrap': [True, False],                 # Usually True
    'criterion': ['gini', 'entropy']            # Gini slightly faster
}
```

### XGBoost

```python
{
    'n_estimators': [100, 200, 500],            # Start with 100
    'learning_rate': [0.01, 0.05, 0.1, 0.3],    # Lower = better (but slower)
    'max_depth': [3, 5, 7, 9],                  # 3-7 usually sufficient
    'min_child_weight': [1, 3, 5],              # Higher → less overfitting
    'subsample': [0.6, 0.8, 1.0],               # 0.8 often best
    'colsample_bytree': [0.6, 0.8, 1.0],        # 0.8 often best
    'gamma': [0, 0.1, 0.2, 0.3],                # 0 = no regularization
    'reg_alpha': [0, 0.01, 0.1, 1],             # L1 regularization
    'reg_lambda': [1, 1.5, 2]                   # L2 regularization
}
```

### Neural Networks

```python
{
    'hidden_layers': [(64,), (128,), (64, 32)], # Architecture
    'activation': ['relu', 'elu', 'gelu'],      # ReLU standard
    'optimizer': ['adam', 'adamw', 'sgd'],      # Adam good default
    'learning_rate': [0.001, 0.0001],           # 0.001 for Adam
    'batch_size': [32, 64, 128],                # Larger = faster but less stable
    'dropout': [0.0, 0.2, 0.5],                 # 0.2-0.5 for regularization
    'weight_decay': [0, 1e-5, 1e-4]             # L2 regularization
}
```

### Support Vector Machine (SVM)

```python
{
    'C': [0.1, 1, 10, 100],                     # Regularization (lower = more)
    'kernel': ['linear', 'rbf', 'poly'],        # RBF good default
    'gamma': ['scale', 'auto', 0.001, 0.01],    # RBF kernel parameter
    'degree': [2, 3, 4]                         # For poly kernel only
}
```

### K-Nearest Neighbors (KNN)

```python
{
    'n_neighbors': [3, 5, 7, 9, 15],            # Odd numbers for classification
    'weights': ['uniform', 'distance'],         # Distance often better
    'metric': ['euclidean', 'manhattan'],       # Euclidean standard
    'p': [1, 2]                                 # 1=Manhattan, 2=Euclidean
}
```

---

## 6. Data Processing

### Missing Data Strategies

| Pattern | Strategy | Code |
|---------|----------|------|
| **< 5% missing** | Delete rows | `df.dropna()` |
| **5-15% missing** | Mean/median impute | `SimpleImputer(strategy='median')` |
| **15-40% missing** | Model-based impute | `IterativeImputer()` |
| **> 40% missing** | Create indicator | `df['col_missing'] = df['col'].isna()` |
| **MCAR** | Any strategy | Based on % missing |
| **MAR** | Model-based | `IterativeImputer()` |
| **MNAR** | Domain knowledge | Custom logic |

### Encoding Strategies

| Feature Type | # Unique | Strategy | Code |
|--------------|----------|----------|------|
| **Binary** | 2 | Label encode | `df['col'].map({'A': 0, 'B': 1})` |
| **Ordinal** | Any | Label encode | `OrdinalEncoder()` |
| **Nominal** | < 10 | One-hot | `pd.get_dummies()` |
| **Nominal** | 10-50 | Target encode | `TargetEncoder()` |
| **Nominal** | > 50 | Frequency encode | `df['col'].map(df['col'].value_counts())` |
| **High cardinality** | > 100 | Embeddings | `nn.Embedding()` |

### Scaling Methods

| Scaler | Formula | When to Use | Outlier Sensitive? |
|--------|---------|-------------|-------------------|
| **StandardScaler** | (x - μ) / σ | Most algorithms | Yes |
| **MinMaxScaler** | (x - min) / (max - min) | Bounded range [0,1] | Yes |
| **RobustScaler** | (x - median) / IQR | Outliers present | No |
| **Normalizer** | x / ‖x‖ | Distance-based | N/A |
| **Log Transform** | log(x + 1) | Right-skewed data | N/A |

### Feature Selection Methods

```python
# 1. Filter Methods (Fast)
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=10)

# 2. Wrapper Methods (Accurate but slow)
from sklearn.feature_selection import RFE
rfe = RFE(estimator=model, n_features_to_select=10)

# 3. Embedded Methods (Fast and accurate)
# - L1 regularization (Lasso)
# - Tree-based feature importance
importances = model.feature_importances_

# 4. Variance Threshold (Remove low variance)
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)
```

---

## 7. Deep Learning

### Activation Functions

| Function | Formula | Range | Use Case | Dying Neuron? |
|----------|---------|-------|----------|---------------|
| **ReLU** | max(0, x) | [0, ∞) | Hidden layers | Yes |
| **Leaky ReLU** | max(0.01x, x) | (-∞, ∞) | Fix dying ReLU | No |
| **ELU** | x if x>0 else α(e^x-1) | (-α, ∞) | Better than ReLU | No |
| **GELU** | x·Φ(x) | (-∞, ∞) | Transformers | No |
| **Sigmoid** | 1/(1+e^(-x)) | (0, 1) | Binary output | Yes |
| **Tanh** | (e^x-e^(-x))/(e^x+e^(-x)) | (-1, 1) | Hidden (rare) | Yes |
| **Softmax** | e^x_i / Σe^x_j | [0, 1] (sum=1) | Multi-class output | No |

### Loss Functions

| Task | Loss Function | When to Use |
|------|---------------|-------------|
| **Binary Classification** | Binary Cross-Entropy | Probabilities needed |
| **Multi-class** | Categorical Cross-Entropy | Mutually exclusive classes |
| **Multi-label** | Binary Cross-Entropy | Multiple labels possible |
| **Regression** | MSE | Standard regression |
| **Regression (outliers)** | MAE or Huber | Robust to outliers |
| **Imbalanced** | Focal Loss | Rare class important |
| **Ranking** | Hinge Loss or Triplet Loss | Order matters |

### Optimizers

| Optimizer | Learning Rate | When to Use | Pros | Cons |
|-----------|---------------|-------------|------|------|
| **SGD** | 0.01 - 0.1 | Simple problems | Stable | Slow convergence |
| **SGD + Momentum** | 0.01 - 0.1 | Most problems | Faster than SGD | Tuning needed |
| **Adam** | 0.001 | Default choice | Adaptive LR, fast | Can overfit |
| **AdamW** | 0.001 | Large models | Better regularization | Slightly slower |
| **RMSprop** | 0.001 | RNNs | Good for RNNs | Less common now |

### Learning Rate Schedules

```python
# 1. Step Decay
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# 2. Exponential Decay
scheduler = ExponentialLR(optimizer, gamma=0.95)

# 3. Cosine Annealing
scheduler = CosineAnnealingLR(optimizer, T_max=100)

# 4. ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# 5. OneCycleLR (recommended)
scheduler = OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(train_loader), epochs=10)
```

### Regularization Techniques

| Technique | How | When | Code |
|-----------|-----|------|------|
| **Dropout** | Randomly drop neurons | Overfitting | `nn.Dropout(0.5)` |
| **L2 (Weight Decay)** | Penalize large weights | Always | `weight_decay=1e-4` |
| **L1** | Sparse weights | Feature selection | Custom loss |
| **Batch Norm** | Normalize activations | Deep networks | `nn.BatchNorm1d()` |
| **Data Augmentation** | Synthetic data | Images, text | `transforms.RandomCrop()` |
| **Early Stopping** | Stop when val loss increases | Always | Monitor validation |

### CNN Architectures (Image Classification)

```
Input Size → Architecture Choice

32x32 (CIFAR-10):
├─ Simple CNN (few layers)
└─ ResNet-18 (transfer learning)

224x224 (ImageNet):
├─ ResNet-50 (good balance)
├─ EfficientNet-B0 (efficient)
└─ Vision Transformer (ViT) (state-of-the-art)

Custom:
├─ Small dataset (<10K) → Transfer learning
├─ Large dataset (>100K) → Train from scratch
└─ Limited compute → MobileNet, EfficientNet
```

---

## 8. Common Pitfalls

### Top 10 ML Mistakes

| # | Mistake | Solution |
|---|---------|----------|
| 1 | **Data leakage** | Fit preprocessing on train only |
| 2 | **Not scaling features** | Use StandardScaler for distance-based algorithms |
| 3 | **Ignoring class imbalance** | Use stratified split, class weights, resampling |
| 4 | **Using wrong metric** | Match metric to business goal |
| 5 | **Overfitting** | Use regularization, more data, simpler model |
| 6 | **Not checking for outliers** | EDA first, robust scaling |
| 7 | **Improper train-test split** | Stratify, shuffle (except time series) |
| 8 | **Hyperparameter tuning on test set** | Use validation set or cross-validation |
| 9 | **Ignoring missing values** | Explicit handling strategy |
| 10 | **Not validating assumptions** | Check linearity, normality, etc. |

### Data Leakage Examples

```python
# ❌ WRONG: Scaling before split
X_scaled = scaler.fit_transform(X)
X_train, X_test = train_test_split(X_scaled)

# ✅ CORRECT: Split then scale
X_train, X_test = train_test_split(X)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ❌ WRONG: Target encoding without CV
df['cat_encoded'] = df.groupby('category')['target'].transform('mean')

# ✅ CORRECT: Target encoding with CV
from category_encoders import TargetEncoder
encoder = TargetEncoder()
df['cat_encoded'] = encoder.fit_transform(df['category'], df['target'])
```

### Class Imbalance Solutions

```python
# 1. Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2
)

# 2. Class weights
from sklearn.utils.class_weight import compute_class_weight
weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
model.fit(X, y, sample_weight=weights)

# 3. Oversampling (SMOTE)
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 4. Undersampling
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

# 5. Threshold tuning
# Find optimal threshold instead of default 0.5
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
```

---

## 9. Interview Cheat Sheet

### Must-Know Concepts (1-Minute Explanations)

**Bias-Variance Tradeoff**
- **Bias:** Error from wrong assumptions (underfitting)
- **Variance:** Error from sensitivity to training data (overfitting)
- **Goal:** Balance both for minimal total error
- **Example:** Linear model = high bias, deep tree = high variance

**Overfitting vs Underfitting**
- **Overfitting:** Model memorizes training data (low train error, high test error)
  - *Fix:* More data, regularization, simpler model
- **Underfitting:** Model too simple (high train error, high test error)
  - *Fix:* More features, complex model, less regularization

**Cross-Validation**
- Split data into k folds
- Train on k-1 folds, validate on 1
- Repeat k times, average results
- **Why?** More reliable performance estimate

**Precision vs Recall**
- **Precision:** Of predicted positives, how many are correct? (↓ false positives)
- **Recall:** Of actual positives, how many did we find? (↓ false negatives)
- **Trade-off:** Increasing one often decreases the other

**Regularization (L1 vs L2)**
- **L1 (Lasso):** Σ|w| → Sparse weights (feature selection)
- **L2 (Ridge):** Σw² → Small weights (all features)
- **Elastic Net:** Combination of both

**Random Forest vs XGBoost**
- **Random Forest:** Parallel trees, averaging, bagging
- **XGBoost:** Sequential trees, boosting, correcting errors
- **XGBoost usually more accurate**, Random Forest faster to train

### Quick Answers to Common Questions

**Q: How do you handle missing data?**
```
1. Check pattern (MCAR/MAR/MNAR)
2. If <5%: Drop rows
3. If 5-40%: Impute (median for numerical, mode for categorical)
4. If >40%: Create missing indicator
5. Advanced: Iterative imputation
```

**Q: How do you detect overfitting?**
```
1. Compare train vs validation loss
2. Learning curves (train/val error vs dataset size)
3. If train error << val error → overfitting
Solutions: More data, regularization, simpler model, dropout
```

**Q: How do you choose the right algorithm?**
```
1. Understand problem (classification/regression/clustering)
2. Check data size (<10K: simple models, >100K: complex ok)
3. Interpretability needed? (Linear/Tree vs Neural Network)
4. Feature types (tabular: XGBoost, images: CNN, text: Transformers)
5. Start simple, increase complexity if needed
```

**Q: How do you handle imbalanced data?**
```
1. Use appropriate metrics (AUC-ROC, F1, Precision-Recall)
2. Stratified sampling
3. Class weights
4. Resampling (SMOTE for oversampling, random for undersampling)
5. Threshold tuning
6. Ensemble methods
```

**Q: Explain backpropagation**
```
1. Forward pass: Calculate output and loss
2. Backward pass: Compute gradients using chain rule
3. Update weights: w_new = w_old - learning_rate * gradient
4. Repeat for all layers (hence "back" propagation)
```

### Algorithm Comparison Table (Interview)

| Question | Answer |
|----------|--------|
| Bagging vs Boosting? | Bagging: parallel (RF), reduces variance. Boosting: sequential (XGBoost), reduces bias |
| Generative vs Discriminative? | Generative models P(x,y), Discriminative models P(y\|x) |
| Parametric vs Non-parametric? | Parametric: fixed # params (linear). Non-parametric: grows with data (KNN, trees) |
| Online vs Batch learning? | Online: incremental updates. Batch: all data at once |
| Supervised vs Unsupervised? | Supervised: labeled data. Unsupervised: unlabeled (clustering, dim reduction) |

---

## 10. Quick Debugging Guide

### Model Not Learning (Loss Not Decreasing)

```
Checklist:
□ Check learning rate (try 0.001 for Adam, 0.01 for SGD)
□ Verify loss function matches task (BCE for binary, CE for multi-class)
□ Check data preprocessing (scaled? normalized?)
□ Verify labels are correct format (0-indexed for classification)
□ Try simpler model first (sanity check)
□ Check for NaN/Inf in data
□ Overfit single batch (should reach ~0 loss)
□ Verify optimizer is updating (print gradients)
```

### Model Overfitting

```
Solutions (in order of preference):
1. Get more data (best solution)
2. Data augmentation (images, text)
3. Regularization (L2, dropout 0.2-0.5)
4. Simplify model (fewer layers, smaller hidden size)
5. Early stopping (patience=5-10 epochs)
6. Reduce training epochs
7. Increase batch size
8. Batch normalization
```

### Low Test Accuracy

```
Debug steps:
1. Check train accuracy first
   ├─ Low train accuracy → underfitting
   │  └─ Solutions: Complex model, more features, less regularization
   └─ High train accuracy → overfitting
      └─ Solutions: See overfitting section

2. Check data distribution
   ├─ Train/test from same distribution?
   ├─ Class balance similar?
   └─ Feature ranges similar?

3. Check for data leakage
   ├─ Did you fit scaler on all data?
   └─ Did you use future information?

4. Try cross-validation
   └─ Gives more reliable estimate
```

### Slow Training

```
Speed-up techniques:
□ Reduce batch size (more updates per epoch)
□ Use GPU (10-100x faster for deep learning)
□ Mixed precision training (fp16)
□ Reduce model size
□ Use efficient data loading (num_workers > 0)
□ Profile code (find bottlenecks)
□ Use DataLoader with pinned memory
□ Optimize data preprocessing
```

### GPU Out of Memory

```
Solutions:
□ Reduce batch size (most effective)
□ Use gradient accumulation (simulate large batch)
□ Reduce model size
□ Use gradient checkpointing
□ Clear cache: torch.cuda.empty_cache()
□ Use mixed precision (fp16 uses less memory)
□ Reduce image resolution
□ Use smaller model (DistilBERT vs BERT)
```

---

## Quick Reference Cards

### Card 1: Train-Test-Val Split

```python
# Standard split (60-20-20)
from sklearn.model_selection import train_test_split

# First split: train + (val+test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

# Second split: val + test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Result: 60% train, 20% val, 20% test
```

### Card 2: Pipeline Template

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(random_state=42))
])

# Fit and predict
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# No data leakage - scaler only sees training data!
```

### Card 3: Quick Model Comparison

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC()
}

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    results[name] = scores.mean()

# Sort by performance
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
for name, score in sorted_results:
    print(f"{name}: {score:.3f}")
```

### Card 4: Feature Importance

```python
import numpy as np
import matplotlib.pyplot as plt

# Get feature importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot top 10
plt.figure(figsize=(10, 6))
plt.title("Top 10 Feature Importances")
plt.bar(range(10), importances[indices[:10]])
plt.xticks(range(10), feature_names[indices[:10]], rotation=45)
plt.tight_layout()
plt.show()
```

### Card 5: Confusion Matrix

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
disp.plot()
plt.title('Confusion Matrix')
plt.show()

# Interpretation
print(f"True Negatives: {cm[0,0]}")
print(f"False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}")
print(f"True Positives: {cm[1,1]}")
```

---

## Time Complexity Cheat Sheet

| Algorithm | Training | Prediction | Space |
|-----------|----------|------------|-------|
| **Linear Regression** | O(n·p²) | O(p) | O(p) |
| **Logistic Regression** | O(n·p·i) | O(p) | O(p) |
| **Decision Tree** | O(n·p·log n) | O(log n) | O(n) |
| **Random Forest** | O(k·n·p·log n) | O(k·log n) | O(k·n) |
| **SVM** | O(n²·p) to O(n³·p) | O(s·p) | O(s·p) |
| **K-NN** | O(1) | O(n·p) | O(n·p) |
| **K-Means** | O(n·k·p·i) | O(k·p) | O(k·p) |
| **Neural Network** | O(n·w·i) | O(w) | O(w) |

**Legend:**
- n = samples, p = features, k = trees/clusters, i = iterations, w = weights, s = support vectors

---

## Production Checklist

```
Pre-Deployment:
□ Model performance meets requirements
□ Model is versioned and reproducible
□ Preprocessing pipeline saved
□ API endpoint defined and tested
□ Latency < SLA (e.g., <100ms)
□ Error handling implemented
□ Logging configured
□ Monitoring dashboard ready
□ Alerts configured
□ Rollback plan documented

Post-Deployment:
□ A/B test running
□ Monitoring metrics (accuracy, latency, errors)
□ Data drift detection enabled
□ Feedback loop established
□ Model retraining schedule defined
```

---

## Useful Commands

```bash
# Check GPU
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Install common libraries
pip install numpy pandas scikit-learn matplotlib seaborn
pip install torch torchvision
pip install xgboost lightgbm catboost
pip install shap lime

# Jupyter extensions
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user

# Model size
du -sh model.pkl

# Memory profiler
python -m memory_profiler script.py
```

---

## Key Takeaways

1. **Start Simple**: Begin with logistic regression or random forest, then increase complexity
2. **EDA First**: Always explore data before modeling
3. **Avoid Data Leakage**: Fit preprocessing on training data only
4. **Right Metric**: Match metric to business goal (not always accuracy)
5. **Cross-Validate**: Never trust a single train-test split
6. **Regularize**: Almost always use some regularization
7. **Monitor Everything**: In production, track data drift and model performance
8. **Document**: Version data, code, models, and experiments
9. **Explain**: Use SHAP/LIME for model interpretability
10. **Iterate**: ML is iterative - don't expect perfect first try

---

**Document Version:** 1.0
**Last Updated:** October 21, 2025
**Maintained by:** Benjamin Hu

**Quick Reference for:**
- Interview preparation
- Quick decision making
- Production ML
- Debugging
- Algorithm selection

**Complementary Resources:**
- Full guides in main repository
- Jupyter notebooks for hands-on practice
- Production case studies for real-world examples
