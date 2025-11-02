# 📊 Model Evaluation Metrics Guide

**"Not everything that counts can be counted, and not everything that can be counted counts."** - Albert Einstein

Choosing the right evaluation metric is crucial. This guide helps you select and interpret metrics for your ML models.

---

## 📋 Table of Contents

- [Why Metrics Matter](#why-metrics-matter)
- [Classification Metrics](#classification-metrics)
- [Regression Metrics](#regression-metrics)
- [Ranking Metrics](#ranking-metrics)
- [Clustering Metrics](#clustering-metrics)
- [Custom Metrics](#custom-metrics)
- [Metric Selection Guide](#metric-selection-guide)

---

## Why Metrics Matter

### Same Model, Different Metrics

```python
# Fraud detection model
y_true = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # 10% fraud
y_pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Predicts "no fraud" always

accuracy = accuracy_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

print(f"Accuracy: {accuracy:.0%}")  # 90% - Looks great!
print(f"Recall: {recall:.0%}")      # 0% - Catches zero fraud!

# Model is useless but accuracy looks good!
```

**Key lesson:** **Choose metrics that align with business goals.**

---

## Classification Metrics

### Binary Classification

#### Confusion Matrix

**Foundation of all classification metrics:**

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, y_pred)

# Visualize
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()

"""
                 Predicted
                 No   Yes
Actual  No      TN    FP
        Yes     FN    TP
"""
```

**Definitions:**
- **TP (True Positive):** Correctly predicted positive
- **TN (True Negative):** Correctly predicted negative
- **FP (False Positive):** Incorrectly predicted positive (Type I error)
- **FN (False Negative):** Incorrectly predicted negative (Type II error)

#### Accuracy

**Formula:** (TP + TN) / (TP + TN + FP + FN)

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.2%}")
```

**When to use:**
- ✅ Balanced classes
- ✅ All errors equally costly

**When NOT to use:**
- ❌ Imbalanced classes
- ❌ Different error costs
- ❌ Rare events (fraud, disease)

**Example where accuracy misleads:**
```python
# 99% healthy, 1% diseased
y_true = [0] * 99 + [1]
y_pred = [0] * 100  # Predict "healthy" always

accuracy = accuracy_score(y_true, y_pred)  # 99%!
# But we miss all diseases!
```

#### Precision

**Formula:** TP / (TP + FP)

**Interpretation:** Of all positive predictions, how many are correct?

```python
from sklearn.metrics import precision_score

precision = precision_score(y_true, y_pred)
print(f"Precision: {precision:.2%}")
```

**When to use:**
- Cost of False Positives is high
- Examples:
  - Spam detection (don't want real emails in spam)
  - Medical screening (don't want unnecessary treatment)

**Example:**
```python
# Email spam filter
# Precision = 95% means: Of 100 emails marked as spam, 95 are actually spam
# 5 legitimate emails wrongly marked as spam (bad!)
```

#### Recall (Sensitivity, True Positive Rate)

**Formula:** TP / (TP + FN)

**Interpretation:** Of all actual positives, how many did we catch?

```python
from sklearn.metrics import recall_score

recall = recall_score(y_true, y_pred)
print(f"Recall: {recall:.2%}")
```

**When to use:**
- Cost of False Negatives is high
- Examples:
  - Disease detection (can't miss diseases)
  - Fraud detection (can't miss fraud)
  - Security threats

**Example:**
```python
# Cancer screening
# Recall = 98% means: Of 100 cancer cases, we detected 98
# We missed 2 cases (very bad!)
```

#### F1-Score

**Formula:** 2 × (Precision × Recall) / (Precision + Recall)

**Harmonic mean of precision and recall**

```python
from sklearn.metrics import f1_score

f1 = f1_score(y_true, y_pred)
print(f"F1-Score: {f1:.2%}")
```

**When to use:**
- Need balance between precision and recall
- Imbalanced classes
- No clear preference for precision vs recall

**Interpretation:**
- F1 = 1.0: Perfect
- F1 = 0.0: Worst
- F1 close to 0: Either precision or recall is very low

#### F-Beta Score

**Generalization of F1:** Allows weighting precision vs recall

**Formula:** (1 + β²) × (Precision × Recall) / (β² × Precision + Recall)

```python
from sklearn.metrics import fbeta_score

# β = 0.5: Favor precision (2x more weight)
f_half = fbeta_score(y_true, y_pred, beta=0.5)

# β = 2: Favor recall (2x more weight)
f_two = fbeta_score(y_true, y_pred, beta=2.0)

print(f"F0.5: {f_half:.2%} (precision-focused)")
print(f"F2.0: {f_two:.2%} (recall-focused)")
```

**β values:**
- β < 1: Precision more important
- β = 1: F1-score (balanced)
- β > 1: Recall more important

#### ROC-AUC (Receiver Operating Characteristic - Area Under Curve)

**Measures ability to distinguish between classes**

```python
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Need probability scores (not binary predictions)
y_scores = model.predict_proba(X_test)[:, 1]

# Calculate AUC
auc = roc_auc_score(y_true, y_scores)
print(f"ROC-AUC: {auc:.3f}")

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Model (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

**Interpretation:**
- AUC = 1.0: Perfect classifier
- AUC = 0.5: Random guessing
- AUC < 0.5: Worse than random (predictions inverted)

**When to use:**
- ✅ Need threshold-independent metric
- ✅ Comparing multiple models
- ✅ Balanced or moderately imbalanced data

**When NOT to use:**
- ❌ Highly imbalanced data (use PR-AUC instead)
- ❌ Care about performance at specific threshold

#### Precision-Recall AUC

**Better than ROC-AUC for imbalanced data**

```python
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

y_scores = model.predict_proba(X_test)[:, 1]

# Calculate PR curve
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
pr_auc = auc(recall, precision)

print(f"PR-AUC: {pr_auc:.3f}")

# Plot
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'Model (AUC = {pr_auc:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()
```

**When to use:**
- ✅ Highly imbalanced data
- ✅ Positive class is rare but important

#### Log Loss (Cross-Entropy Loss)

**Measures quality of probability predictions**

```python
from sklearn.metrics import log_loss

y_proba = model.predict_proba(X_test)
logloss = log_loss(y_true, y_proba)
print(f"Log Loss: {logloss:.3f}")
```

**Interpretation:**
- Lower is better
- 0 = perfect predictions
- Penalizes confident wrong predictions heavily

**When to use:**
- ✅ Need well-calibrated probabilities
- ✅ Predictions used for decision-making
- ✅ Kaggle competitions (common metric)

### Multi-Class Classification

#### Macro vs Micro vs Weighted Averaging

```python
from sklearn.metrics import precision_recall_fscore_support

# Macro: Average of per-class metrics (treats all classes equally)
precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
    y_true, y_pred, average='macro'
)

# Micro: Aggregate TP, FP, FN across all classes
precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
    y_true, y_pred, average='micro'
)

# Weighted: Weighted average by class frequency
precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
    y_true, y_pred, average='weighted'
)

print(f"Macro F1: {f1_macro:.3f}")
print(f"Micro F1: {f1_micro:.3f}")
print(f"Weighted F1: {f1_weighted:.3f}")
```

**When to use:**
- **Macro:** All classes equally important (rare classes matter)
- **Micro:** Larger classes more important
- **Weighted:** Balance by actual class distribution

#### Classification Report

**Comprehensive overview:**

```python
from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1', 'Class 2']))
```

Output:
```
              precision    recall  f1-score   support

     Class 0       0.85      0.90      0.87       100
     Class 1       0.78      0.75      0.76        80
     Class 2       0.92      0.88      0.90       120

    accuracy                           0.85       300
   macro avg       0.85      0.84      0.84       300
weighted avg       0.86      0.85      0.85       300
```

---

## Regression Metrics

### Mean Absolute Error (MAE)

**Formula:** (1/n) × Σ|y_true - y_pred|

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: {mae:.2f}")
```

**Interpretation:**
- Average absolute difference between predictions and actual values
- Same units as target variable
- Easy to interpret

**When to use:**
- ✅ Want interpretable metric
- ✅ Outliers should not dominate
- ✅ All errors equally important

**Example:**
```python
# House price prediction
mae = 25000  # Predictions off by $25,000 on average
```

### Mean Squared Error (MSE)

**Formula:** (1/n) × Σ(y_true - y_pred)²

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_true, y_pred)
print(f"MSE: {mse:.2f}")
```

**Properties:**
- Penalizes large errors more (squared)
- Not in same units as target
- Always positive

**When to use:**
- ✅ Large errors are especially bad
- ✅ Optimizing for typical ML objective

**When NOT to use:**
- ❌ Want interpretable metric (use MAE or RMSE)

### Root Mean Squared Error (RMSE)

**Formula:** sqrt(MSE)

```python
from sklearn.metrics import mean_squared_error

rmse = mean_squared_error(y_true, y_pred, squared=False)
# or
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print(f"RMSE: {rmse:.2f}")
```

**Properties:**
- Same units as target variable
- Penalizes large errors
- More interpretable than MSE

**When to use:**
- ✅ Same as MSE but want interpretability
- ✅ Common standard metric

### Mean Absolute Percentage Error (MAPE)

**Formula:** (100/n) × Σ|y_true - y_pred| / |y_true|

```python
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(y_true, y_pred)
print(f"MAPE: {mape:.2f}%")
```

**Interpretation:**
- Percentage error (easy to understand)
- Scale-independent

**When to use:**
- ✅ Want scale-independent metric
- ✅ Comparing across different scales

**When NOT to use:**
- ❌ Target has zeros (division by zero!)
- ❌ Target has very small values (unstable)

### R² Score (Coefficient of Determination)

**Formula:** 1 - (SS_res / SS_tot)

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_true, y_pred)
print(f"R²: {r2:.3f}")
```

**Interpretation:**
- R² = 1.0: Perfect predictions
- R² = 0.0: Model as good as predicting mean
- R² < 0: Model worse than predicting mean

**Example:**
```python
r2 = 0.85  # Model explains 85% of variance
```

**When to use:**
- ✅ Standard regression metric
- ✅ Want to know variance explained

**When NOT to use:**
- ❌ Target is not normally distributed
- ❌ Non-linear relationships

### Huber Loss

**Combines MSE and MAE (robust to outliers)**

```python
from sklearn.metrics import mean_squared_error

def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    is_small_error = np.abs(error) <= delta
    squared_loss = 0.5 * error**2
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    return np.mean(np.where(is_small_error, squared_loss, linear_loss))

loss = huber_loss(y_true, y_pred, delta=1.0)
```

**When to use:**
- ✅ Data has outliers
- ✅ Want to penalize large errors less than MSE

---

## Ranking Metrics

### Mean Average Precision (MAP)

**Used in information retrieval, recommendation systems**

```python
def average_precision(y_true, y_scores):
    """Calculate average precision for single query"""
    # Sort by scores descending
    order = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[order]

    # Calculate precision at each relevant position
    precisions = []
    num_hits = 0

    for i, label in enumerate(y_true_sorted):
        if label == 1:
            num_hits += 1
            precisions.append(num_hits / (i + 1))

    if not precisions:
        return 0.0

    return np.mean(precisions)

# Mean over all queries
def mean_average_precision(y_true_list, y_scores_list):
    return np.mean([average_precision(y_t, y_s)
                    for y_t, y_s in zip(y_true_list, y_scores_list)])
```

### Normalized Discounted Cumulative Gain (NDCG)

**Accounts for position of relevant items**

```python
from sklearn.metrics import ndcg_score

# Relevance scores (0-5 scale, e.g., movie ratings)
y_true = [[3, 2, 3, 0, 1, 2]]
y_scores = [[0.9, 0.8, 0.7, 0.6, 0.5, 0.4]]

ndcg = ndcg_score(y_true, y_scores, k=5)  # Top 5
print(f"NDCG@5: {ndcg:.3f}")
```

**Interpretation:**
- NDCG = 1.0: Perfect ranking
- Higher is better
- Penalizes relevant items ranked low

---

## Clustering Metrics

### Silhouette Score

**Measures how similar points are to their own cluster vs other clusters**

```python
from sklearn.metrics import silhouette_score

labels = kmeans.labels_
score = silhouette_score(X, labels)
print(f"Silhouette Score: {score:.3f}")
```

**Interpretation:**
- Score ranges from -1 to 1
- 1: Clusters well separated
- 0: Overlapping clusters
- -1: Wrong clusters

### Davies-Bouldin Index

**Lower is better (measures cluster separation)**

```python
from sklearn.metrics import davies_bouldin_score

score = davies_bouldin_score(X, labels)
print(f"Davies-Bouldin Index: {score:.3f}")
```

### Calinski-Harabasz Index

**Higher is better (ratio of between-cluster to within-cluster variance)**

```python
from sklearn.metrics import calinski_harabasz_score

score = calinski_harabasz_score(X, labels)
print(f"Calinski-Harabasz Index: {score:.1f}")
```

---

## Custom Metrics

### Business-Specific Metrics

```python
def business_metric(y_true, y_pred):
    """
    Custom metric for credit card fraud detection

    Costs:
    - False Positive (block legitimate transaction): $10 customer service
    - False Negative (miss fraud): $500 average fraud loss
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    fp_cost = fp * 10    # $10 per false alarm
    fn_cost = fn * 500   # $500 per missed fraud

    total_cost = fp_cost + fn_cost
    return total_cost

# Use in cross-validation
from sklearn.metrics import make_scorer

custom_scorer = make_scorer(business_metric, greater_is_better=False)
scores = cross_val_score(model, X, y, scoring=custom_scorer, cv=5)
```

### Asymmetric Error Costs

```python
def asymmetric_mse(y_true, y_pred, overpredict_penalty=1.0, underpredict_penalty=2.0):
    """
    MSE with different costs for over vs under prediction

    Example: Inventory forecasting
    - Overpredict: Holding costs
    - Underpredict: Lost sales (more expensive)
    """
    errors = y_true - y_pred
    overpredict = errors < 0
    underpredict = errors > 0

    cost = (
        (errors[overpredict] ** 2).sum() * overpredict_penalty +
        (errors[underpredict] ** 2).sum() * underpredict_penalty
    )

    return cost / len(y_true)
```

---

## Metric Selection Guide

### Decision Tree

```
What type of problem?
│
├─ Classification
│  │
│  ├─ Binary
│  │  ├─ Balanced classes? → Accuracy, F1
│  │  ├─ Imbalanced?
│  │  │  ├─ False Positives expensive? → Precision
│  │  │  ├─ False Negatives expensive? → Recall
│  │  │  └─ Both matter? → F1, PR-AUC
│  │  ├─ Need probabilities? → Log Loss, Brier Score
│  │  └─ Threshold-independent? → ROC-AUC (balanced), PR-AUC (imbalanced)
│  │
│  └─ Multi-class
│     ├─ All classes equally important? → Macro F1
│     ├─ Larger classes more important? → Weighted F1
│     └─ Need probabilities? → Log Loss
│
├─ Regression
│  ├─ Want interpretable metric? → MAE, RMSE
│  ├─ Large errors very bad? → MSE, RMSE
│  ├─ Robust to outliers? → MAE, Huber Loss
│  ├─ Scale-independent? → MAPE, R²
│  └─ Variance explained? → R²
│
├─ Ranking
│  ├─ Position matters? → NDCG
│  ├─ Just order? → MAP
│  └─ Binary relevance? → Precision@K, Recall@K
│
└─ Clustering
   ├─ Don't have ground truth? → Silhouette, Davies-Bouldin
   └─ Have ground truth? → Adjusted Rand Index, NMI
```

### Common Use Cases

| Use Case | Primary Metric | Secondary Metrics |
|----------|----------------|-------------------|
| Spam Detection | Precision | Recall, F1 |
| Fraud Detection | Recall | Precision@90%Recall, PR-AUC |
| Medical Diagnosis | Recall | Specificity, F1 |
| Credit Scoring | ROC-AUC | Log Loss, F1 |
| Recommendation | NDCG@K, MAP | Precision@K, Recall@K |
| House Price Prediction | RMSE | MAE, R² |
| Demand Forecasting | MAPE | RMSE, MAE |
| Customer Churn | PR-AUC | Recall, Precision |

---

## Complete Example

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *

# Generate imbalanced data (10% positive)
X, y = make_classification(
    n_samples=1000,
    weights=[0.9, 0.1],
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("=== CLASSIFICATION METRICS ===\n")

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print()

# Basic metrics
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.3f}")
print()

# Probability-based metrics
print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.3f}")
print(f"Log Loss:  {log_loss(y_test, y_proba):.3f}")
print()

# Detailed report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Custom threshold analysis
print("\n=== THRESHOLD ANALYSIS ===\n")
for threshold in [0.3, 0.5, 0.7]:
    y_pred_thresh = (y_proba >= threshold).astype(int)
    precision = precision_score(y_test, y_pred_thresh)
    recall = recall_score(y_test, y_pred_thresh)
    f1 = f1_score(y_test, y_pred_thresh)

    print(f"Threshold {threshold:.1f}:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1-Score:  {f1:.3f}")
    print()
```

---

## Key Takeaways

**Metric selection rules:**
1. **Align with business goals** (what matters most?)
2. **Consider class balance** (imbalanced → F1, PR-AUC)
3. **Account for error costs** (FP vs FN costs)
4. **Think about deployment** (probabilities vs binary)

**Common mistakes:**
- Using accuracy on imbalanced data
- Not considering error costs
- Optimizing wrong metric
- Forgetting to check multiple metrics

**Best practices:**
- Report multiple metrics
- Always show confusion matrix for classification
- Use cross-validation for robust estimates
- Document why you chose specific metrics

**Remember:** The best metric depends on your problem, not on what's common!

---

**Related guides:**
- [Model Selection Guide](./MODEL_SELECTION_GUIDE.md) - Choose the right algorithm
- [Hyperparameter Tuning](./HYPERPARAMETER_TUNING_GUIDE.md) - Optimize for the right metric
- [Common ML Mistakes](../../resources/COMMON_ML_MISTAKES.md) - Avoid metric pitfalls

---

*Last Updated: October 30, 2025*
