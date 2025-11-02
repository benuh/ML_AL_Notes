# 🎯 Model Selection Guide: Choosing the Right Algorithm

**"All models are wrong, but some are useful."** - George Box

This guide helps you choose the right ML algorithm for your problem based on data characteristics, problem type, and constraints.

---

## 📋 Table of Contents

- [Quick Decision Tree](#quick-decision-tree)
- [Problem Type Classification](#problem-type-classification)
- [Regression Models](#regression-models)
- [Classification Models](#classification-models)
- [Clustering Models](#clustering-models)
- [Model Comparison Framework](#model-comparison-framework)
- [Practical Considerations](#practical-considerations)
- [Algorithm Cheat Sheet](#algorithm-cheat-sheet)

---

## Quick Decision Tree

```
What's your problem?
│
├─ Predict continuous value (price, temperature, etc.)
│  └─ REGRESSION
│     ├─ Linear relationship? → Linear Regression
│     ├─ Need interpretability? → Decision Tree Regressor
│     ├─ Complex patterns? → Random Forest / XGBoost
│     └─ Very large dataset? → SGD Regressor
│
├─ Predict category (spam/not spam, cat/dog/bird)
│  └─ CLASSIFICATION
│     ├─ Linear separable? → Logistic Regression / SVM (linear)
│     ├─ Need interpretability? → Decision Tree
│     ├─ Complex boundaries? → Random Forest / XGBoost
│     ├─ Image/text/sequence? → Neural Networks
│     └─ Very few samples? → Naive Bayes
│
├─ Find groups in data (customer segments)
│  └─ CLUSTERING
│     ├─ Know number of clusters? → K-Means
│     ├─ Density-based groups? → DBSCAN
│     └─ Hierarchical structure? → Agglomerative Clustering
│
└─ Reduce dimensions (visualization, compression)
   └─ DIMENSIONALITY REDUCTION
      ├─ Linear projection? → PCA
      ├─ Non-linear? → t-SNE, UMAP
      └─ Feature selection? → SelectKBest, RFE
```

---

## Problem Type Classification

### Supervised Learning

**You have labeled data (input → output pairs)**

**Regression:**
- Predict house prices
- Forecast sales
- Estimate delivery time
- Predict temperature

**Classification:**
- Binary: Spam detection, fraud detection, churn prediction
- Multi-class: Digit recognition (0-9), species classification
- Multi-label: Tag articles (sports, politics, both)

### Unsupervised Learning

**You have unlabeled data (find patterns)**

**Clustering:**
- Customer segmentation
- Anomaly detection
- Document grouping

**Dimensionality Reduction:**
- Data visualization
- Feature extraction
- Noise reduction

### Semi-Supervised Learning

**You have mostly unlabeled data, few labels**
- Use clustering + few labels
- Self-training algorithms
- Co-training

---

## Regression Models

### Linear Regression

**When to use:**
- Linear relationship between features and target
- Need interpretability (coefficients show feature impact)
- Fast training required
- Baseline model

**Pros:**
- Fast to train
- Interpretable
- Works well with small data
- No hyperparameters to tune

**Cons:**
- Assumes linear relationship
- Sensitive to outliers
- Can't model complex patterns
- Assumes features are independent

**Example:**
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

# Interpret coefficients
for feature, coef in zip(feature_names, model.coef_):
    print(f"{feature}: {coef:.2f}")
```

**Best for:**
- Initial baseline
- Problems with clear linear trends
- When you need to explain predictions

### Ridge Regression (L2)

**When to use:**
- Features are correlated (multicollinearity)
- Need regularization to prevent overfitting
- All features are relevant

**Key hyperparameter:** alpha (regularization strength)

```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
```

### Lasso Regression (L1)

**When to use:**
- Feature selection needed (sets coefficients to 0)
- Sparse models preferred
- Many irrelevant features

```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=0.1)
model.fit(X_train, y_train)

# Features with non-zero coefficients are selected
selected_features = np.where(model.coef_ != 0)[0]
```

### Decision Tree Regressor

**When to use:**
- Non-linear relationships
- Need interpretability (can visualize tree)
- Mixed feature types (numerical + categorical)
- No need for feature scaling

**Pros:**
- Easy to interpret
- Handles non-linearity
- No feature scaling needed
- Captures interactions automatically

**Cons:**
- Prone to overfitting
- High variance (unstable)
- Biased to majority class

```python
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(max_depth=5, min_samples_split=20)
model.fit(X_train, y_train)
```

### Random Forest Regressor

**When to use:**
- Don't know which algorithm to use (great default)
- Need good out-of-box performance
- Mixed feature types
- Want feature importance

**Pros:**
- Reduces overfitting vs single tree
- Robust to outliers
- Handles missing values
- Feature importance built-in

**Cons:**
- Less interpretable than linear
- Slower than linear models
- Large memory footprint

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    random_state=42
)
model.fit(X_train, y_train)

# Feature importance
importances = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
```

### XGBoost / LightGBM / CatBoost

**When to use:**
- Need state-of-the-art performance
- Kaggle competitions
- Tabular data
- Can tune hyperparameters

**XGBoost:**
```python
import xgboost as xgb

model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)
```

**LightGBM (faster for large datasets):**
```python
import lightgbm as lgb

model = lgb.LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)
```

**CatBoost (handles categorical features):**
```python
from catboost import CatBoostRegressor

model = CatBoostRegressor(
    iterations=100,
    learning_rate=0.1,
    depth=5,
    random_state=42,
    verbose=False
)
model.fit(X_train, y_train, cat_features=['city', 'category'])
```

### Support Vector Regression (SVR)

**When to use:**
- Small to medium datasets
- High-dimensional data
- Need robust predictions

**Avoid:**
- Very large datasets (slow)
- Need interpretability

```python
from sklearn.svm import SVR

model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
model.fit(X_train, y_train)
```

### Neural Networks (MLP Regressor)

**When to use:**
- Very large datasets
- Complex non-linear patterns
- Images, text, sequences

**Avoid:**
- Small datasets (<1000 samples)
- Need quick training
- Need interpretability

```python
from sklearn.neural_network import MLPRegressor

model = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    max_iter=500,
    random_state=42
)
model.fit(X_train, y_train)
```

---

## Classification Models

### Logistic Regression

**When to use:**
- Binary classification
- Need probability estimates
- Linear decision boundary
- Baseline model

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Get probabilities
probabilities = model.predict_proba(X_test)
```

### Decision Tree Classifier

**When to use:**
- Need interpretability
- Non-linear boundaries
- Mixed feature types

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=5, min_samples_split=20)
model.fit(X_train, y_train)
```

### Random Forest Classifier

**When to use:**
- Default choice for tabular data
- Need robust performance
- Want feature importance

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    class_weight='balanced',  # For imbalanced data
    random_state=42
)
model.fit(X_train, y_train)
```

### Gradient Boosting (XGBoost, LightGBM, CatBoost)

**When to use:**
- Need best performance
- Willing to tune hyperparameters
- Tabular data

```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),  # For imbalance
    random_state=42
)
model.fit(X_train, y_train)
```

### Support Vector Machines (SVM)

**When to use:**
- High-dimensional data (text classification)
- Clear margin of separation
- Small to medium datasets

**Kernels:**
- Linear: For linearly separable data
- RBF: For non-linear boundaries
- Polynomial: For polynomial boundaries

```python
from sklearn.svm import SVC

# Linear kernel
model = SVC(kernel='linear', C=1.0, probability=True)

# RBF kernel
model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)

model.fit(X_train, y_train)
```

### Naive Bayes

**When to use:**
- Text classification
- Very small datasets
- Need fast training
- Features are independent

**Variants:**
- **GaussianNB**: Continuous features
- **MultinomialNB**: Count data (text)
- **BernoulliNB**: Binary features

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

# For continuous features
model = GaussianNB()

# For text (word counts)
model = MultinomialNB(alpha=1.0)

model.fit(X_train, y_train)
```

### K-Nearest Neighbors (KNN)

**When to use:**
- Small datasets
- Non-parametric (no assumptions)
- Simple baseline

**Avoid:**
- Large datasets (slow)
- High-dimensional data (curse of dimensionality)

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance',  # Weight by distance
    metric='euclidean'
)
model.fit(X_train, y_train)
```

### Neural Networks

**When to use:**
- Very large datasets (>100K samples)
- Images, text, sequences
- Complex patterns

**Avoid:**
- Small datasets
- Need interpretability
- Limited compute

```python
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    max_iter=500,
    random_state=42
)
model.fit(X_train, y_train)
```

---

## Clustering Models

### K-Means

**When to use:**
- Know number of clusters
- Spherical clusters
- Large datasets (scales well)

```python
from sklearn.cluster import KMeans

model = KMeans(n_clusters=3, random_state=42)
labels = model.fit_predict(X)

# Evaluate (if you have ground truth)
from sklearn.metrics import silhouette_score
score = silhouette_score(X, labels)
```

### DBSCAN

**When to use:**
- Don't know number of clusters
- Arbitrary cluster shapes
- Want to find outliers

```python
from sklearn.cluster import DBSCAN

model = DBSCAN(eps=0.5, min_samples=5)
labels = model.fit_predict(X)

# -1 indicates outliers
n_outliers = (labels == -1).sum()
```

### Hierarchical Clustering

**When to use:**
- Want dendrogram
- Hierarchical structure
- Small to medium datasets

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Agglomerative clustering
model = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = model.fit_predict(X)

# Create dendrogram
linkage_matrix = linkage(X, method='ward')
dendrogram(linkage_matrix)
```

---

## Model Comparison Framework

### Step 1: Start Simple

```python
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(max_depth=5),
    'Random Forest': RandomForestRegressor(n_estimators=100)
}

results = {}

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    results[name] = {
        'mean': scores.mean(),
        'std': scores.std()
    }
    print(f"{name}: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

### Step 2: Select Best Performers

```python
# Continue with top 2-3 models
# Tune hyperparameters for those
```

### Step 3: Evaluate on Test Set

```python
# Train final model
best_model = RandomForestRegressor(n_estimators=100, max_depth=10)
best_model.fit(X_train, y_train)

# Evaluate on test set (ONCE)
test_score = best_model.score(X_test, y_test)
print(f"Test R²: {test_score:.3f}")
```

### Comparison Metrics

**Regression:**
- R² (coefficient of determination)
- MSE / RMSE
- MAE
- MAPE (mean absolute percentage error)

**Classification:**
- Accuracy (be careful with imbalanced data!)
- Precision / Recall / F1
- ROC-AUC
- PR-AUC (for imbalanced data)
- Confusion matrix

```python
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

---

## Practical Considerations

### Data Size

**Small (<1K samples):**
- Simple models (Linear, Logistic)
- K-Nearest Neighbors
- Naive Bayes
- Avoid: Deep learning

**Medium (1K-100K):**
- Random Forest
- Gradient Boosting
- SVM
- Shallow neural networks

**Large (>100K):**
- Gradient Boosting (LightGBM, XGBoost)
- SGD-based linear models
- Deep learning
- Avoid: KNN, full-batch methods

### Feature Count

**Low dimensional (<10 features):**
- Any algorithm works
- Start with Linear/Logistic

**Medium (10-100 features):**
- Tree-based models
- Linear models with regularization
- SVM

**High dimensional (>100 features):**
- Linear models with L1 regularization
- Gradient boosting
- Feature selection first
- Avoid: KNN (curse of dimensionality)

### Interpretability Requirements

**High interpretability needed:**
1. Linear/Logistic Regression (coefficients)
2. Decision Trees (rules)
3. Naive Bayes (probabilities)

**Medium interpretability:**
1. Random Forest (feature importance)
2. Gradient Boosting (SHAP values)

**Low interpretability (black box):**
1. Deep Neural Networks
2. SVM with RBF kernel

### Training Time

**Fast training needed:**
1. Linear/Logistic Regression
2. Naive Bayes
3. Decision Tree (single)

**Medium:**
1. Random Forest
2. SVM (small data)

**Slow:**
1. Gradient Boosting (many iterations)
2. Deep Neural Networks
3. SVM (large data)

### Prediction Speed

**Real-time (<1ms):**
- Linear/Logistic Regression
- Single Decision Tree
- Naive Bayes

**Fast (<10ms):**
- Random Forest (few trees)
- Small neural networks

**Slower:**
- Large ensembles
- Deep neural networks

---

## Algorithm Cheat Sheet

### Regression

| Algorithm | Best For | Pros | Cons | Training Time |
|-----------|----------|------|------|---------------|
| Linear Regression | Baseline, interpretability | Fast, interpretable | Assumes linearity | Very fast |
| Ridge/Lasso | Regularization, feature selection | Prevents overfitting | Still assumes linearity | Very fast |
| Decision Tree | Non-linear, interpretability | Easy to interpret | Overfits easily | Fast |
| Random Forest | Default choice | Robust, accurate | Less interpretable | Medium |
| XGBoost/LightGBM | Best performance | State-of-the-art | Needs tuning | Medium-Slow |
| SVR | Small data, high-dim | Robust | Slow on large data | Medium-Slow |
| Neural Networks | Very large data | Flexible | Needs lots of data | Slow |

### Classification

| Algorithm | Best For | Pros | Cons | Training Time |
|-----------|----------|------|------|---------------|
| Logistic Regression | Baseline, probabilities | Fast, interpretable | Linear boundary | Very fast |
| Decision Tree | Interpretability | Easy to understand | Overfits | Fast |
| Random Forest | Default choice | Robust, accurate | Black box | Medium |
| XGBoost/LightGBM | Best performance | State-of-the-art | Needs tuning | Medium-Slow |
| SVM | High-dim (text) | Effective in high-dim | Slow on large data | Medium-Slow |
| Naive Bayes | Text, small data | Very fast | Independence assumption | Very fast |
| KNN | Simple baseline | No training | Slow prediction | Fast (training) |
| Neural Networks | Images, text | Very flexible | Needs lots of data | Slow |

### Quick Selection Guide

**Start with:**
1. Linear/Logistic Regression (baseline)
2. Random Forest (good default)
3. Gradient Boosting (if RF works well)

**For specific cases:**
- **Text:** Naive Bayes, SVM, Transformers
- **Images:** CNNs (ResNet, EfficientNet)
- **Time series:** ARIMA, LSTM, Prophet
- **Tabular:** XGBoost, LightGBM, CatBoost
- **Small data:** Linear models, Naive Bayes
- **Need interpretability:** Linear, Decision Tree
- **Need speed:** Linear, Naive Bayes

---

## Complete Example

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Load data
X, y = load_data()

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features (for linear models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Try multiple models
models = {
    'Linear Regression': (LinearRegression(), X_train_scaled, X_test_scaled),
    'Ridge': (Ridge(alpha=1.0), X_train_scaled, X_test_scaled),
    'Decision Tree': (DecisionTreeRegressor(max_depth=10), X_train, X_test),
    'Random Forest': (RandomForestRegressor(n_estimators=100), X_train, X_test),
    'XGBoost': (xgb.XGBRegressor(n_estimators=100), X_train, X_test)
}

results = {}

for name, (model, X_tr, X_te) in models.items():
    # Cross-validation
    cv_scores = cross_val_score(model, X_tr, y_train, cv=5, scoring='r2')

    # Train on full training set
    model.fit(X_tr, y_train)

    # Test score
    test_score = model.score(X_te, y_test)

    results[name] = {
        'CV mean': cv_scores.mean(),
        'CV std': cv_scores.std(),
        'Test R²': test_score
    }

    print(f"{name}:")
    print(f"  CV: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    print(f"  Test: {test_score:.3f}\n")

# Select best model
best_model_name = max(results, key=lambda x: results[x]['Test R²'])
print(f"Best model: {best_model_name}")
```

---

## Key Takeaways

**Model selection strategy:**
1. **Start simple** (Linear/Logistic baseline)
2. **Try tree-based** (Random Forest as default)
3. **Optimize best** (Tune hyperparameters)
4. **Evaluate properly** (Cross-validation + test set)

**Rules of thumb:**
- Tabular data → Gradient Boosting
- Images → CNNs
- Text → Transformers (BERT, GPT)
- Time series → ARIMA, LSTM, Prophet
- Small data → Simple models
- Need interpretability → Linear/Tree
- Need speed → Linear/Naive Bayes

**Remember:**
- No single best algorithm for all problems
- Start simple, increase complexity if needed
- More data > better algorithm
- Feature engineering > algorithm choice
- Validate with cross-validation, evaluate once on test set

---

**Related guides:**
- [Hyperparameter Tuning](./HYPERPARAMETER_TUNING_GUIDE.md) - Optimize your chosen model
- [Feature Engineering](./FEATURE_ENGINEERING_GUIDE.md) - Improve performance
- [Common ML Mistakes](../../resources/COMMON_ML_MISTAKES.md) - Avoid pitfalls

---

*Last Updated: October 30, 2025*
