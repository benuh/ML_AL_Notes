# Handling Imbalanced Data Guide

A comprehensive practical guide to dealing with class imbalance in classification problems.

## Table of Contents
- [Understanding Class Imbalance](#understanding-class-imbalance)
- [Why Imbalance is a Problem](#why-imbalance-is-a-problem)
- [Evaluation Metrics for Imbalanced Data](#evaluation-metrics-for-imbalanced-data)
- [Resampling Techniques](#resampling-techniques)
- [Algorithm-Level Approaches](#algorithm-level-approaches)
- [Cost-Sensitive Learning](#cost-sensitive-learning)
- [Ensemble Methods](#ensemble-methods)
- [Anomaly Detection Approaches](#anomaly-detection-approaches)
- [Best Practices](#best-practices)
- [Complete Examples](#complete-examples)

---

## Understanding Class Imbalance

**Class imbalance** occurs when classes are not represented equally in the dataset.

### Example: Credit Card Fraud Detection

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate imbalanced dataset
X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    weights=[0.99, 0.01],  # 99% class 0, 1% class 1
    random_state=42
)

# Check distribution
unique, counts = np.unique(y, return_counts=True)
print("Class Distribution:")
for label, count in zip(unique, counts):
    print(f"  Class {label}: {count} ({count/len(y)*100:.1f}%)")

# Visualize
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.bar(['Negative (0)', 'Positive (1)'], counts)
plt.ylabel('Count')
plt.title('Class Distribution')
plt.grid(True, alpha=0.3, axis='y')

plt.subplot(1, 2, 2)
plt.pie(counts, labels=['Negative (0)', 'Positive (1)'],
        autopct='%1.1f%%', startangle=90)
plt.title('Class Proportion')

plt.tight_layout()
plt.show()
```

### Degrees of Imbalance

```python
def analyze_imbalance(y):
    """
    Analyze the degree of class imbalance.
    """
    unique, counts = np.unique(y, return_counts=True)
    majority_class = unique[np.argmax(counts)]
    minority_class = unique[np.argmin(counts)]

    majority_count = counts.max()
    minority_count = counts.min()

    imbalance_ratio = majority_count / minority_count

    print(f"Majority class ({majority_class}): {majority_count} samples")
    print(f"Minority class ({minority_class}): {minority_count} samples")
    print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")

    if imbalance_ratio < 2:
        severity = "Balanced"
    elif imbalance_ratio < 10:
        severity = "Mild Imbalance"
    elif imbalance_ratio < 100:
        severity = "Moderate Imbalance"
    else:
        severity = "Severe Imbalance"

    print(f"Severity: {severity}")

    return imbalance_ratio

imbalance_ratio = analyze_imbalance(y)
```

**Output:**
```
Majority class (0): 9900 samples
Minority class (1): 100 samples
Imbalance ratio: 99.00:1
Severity: Moderate Imbalance
```

---

## Why Imbalance is a Problem

### The Accuracy Paradox

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.dummy import DummyClassifier

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Naive model: always predict majority class
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
y_pred_dummy = dummy.predict(X_test)

print("Naive Model (Always Predict Negative):")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_dummy):.4f}")
print(f"  True Positives detected: {np.sum((y_pred_dummy == 1) & (y_test == 1))}")
print(f"  Total Positives in test: {np.sum(y_test == 1)}")

# Standard logistic regression
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nLogistic Regression (Default):")
print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"  True Positives detected: {np.sum((y_pred == 1) & (y_test == 1))}")

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, y_p, title in zip(axes, [y_pred_dummy, y_pred],
                          ['Naive Model', 'Logistic Regression']):
    cm = confusion_matrix(y_test, y_p)
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred Neg', 'Pred Pos'])
    ax.set_yticklabels(['True Neg', 'True Pos'])
    ax.set_title(title)

    # Annotate
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, cm[i, j],
                          ha="center", va="center", color="black", fontsize=16)

    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()
```

**Problem:** Accuracy is misleading! A model that predicts everything as majority class can have high accuracy but is useless for detecting minority class.

---

## Evaluation Metrics for Imbalanced Data

### 1. Confusion Matrix Analysis

```python
def analyze_confusion_matrix(y_true, y_pred):
    """
    Detailed confusion matrix analysis.
    """
    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()

    print("Confusion Matrix:")
    print(f"  True Negatives (TN): {tn}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    print(f"  True Positives (TP): {tp}")

    return tn, fp, fn, tp

tn, fp, fn, tp = analyze_confusion_matrix(y_test, y_pred)
```

### 2. Precision, Recall, F1-Score

```python
from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_metrics(y_true, y_pred):
    """
    Calculate key metrics for imbalanced data.
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"Precision: {precision:.4f}")
    print(f"  Of all positive predictions, {precision*100:.1f}% were correct")
    print(f"  Formula: TP / (TP + FP)")

    print(f"\nRecall: {recall:.4f}")
    print(f"  Of all actual positives, {recall*100:.1f}% were detected")
    print(f"  Formula: TP / (TP + FN)")

    print(f"\nF1-Score: {f1:.4f}")
    print(f"  Harmonic mean of precision and recall")
    print(f"  Formula: 2 × (Precision × Recall) / (Precision + Recall)")

    return precision, recall, f1

calculate_metrics(y_test, y_pred)
```

### 3. ROC-AUC and PR-AUC

```python
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def plot_roc_and_pr_curves(y_true, y_scores):
    """
    Plot ROC and Precision-Recall curves.
    """
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ROC Curve
    axes[0].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Precision-Recall Curve
    axes[1].plot(recall, precision, linewidth=2, label=f'PR (AUC = {pr_auc:.3f})')
    baseline = np.sum(y_true) / len(y_true)
    axes[1].axhline(y=baseline, color='k', linestyle='--', linewidth=1,
                   label=f'Baseline ({baseline:.3f})')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"\nNote: For imbalanced data, PR-AUC is more informative than ROC-AUC!")

# Get prediction probabilities
y_scores = model.predict_proba(X_test)[:, 1]
plot_roc_and_pr_curves(y_test, y_scores)
```

**Key Insight:** PR-AUC is more informative than ROC-AUC for imbalanced datasets!

### 4. Matthews Correlation Coefficient (MCC)

```python
from sklearn.metrics import matthews_corrcoef

def calculate_mcc(y_true, y_pred):
    """
    Calculate Matthews Correlation Coefficient.

    MCC ranges from -1 to +1:
    +1 = perfect prediction
    0 = random prediction
    -1 = total disagreement
    """
    mcc = matthews_corrcoef(y_true, y_pred)

    print(f"Matthews Correlation Coefficient: {mcc:.4f}")
    print(f"\nMCC considers all four confusion matrix values")
    print(f"Good for imbalanced datasets")

    if mcc > 0.7:
        quality = "Excellent"
    elif mcc > 0.5:
        quality = "Good"
    elif mcc > 0.3:
        quality = "Fair"
    else:
        quality = "Poor"

    print(f"Prediction quality: {quality}")

    return mcc

calculate_mcc(y_test, y_pred)
```

---

## Resampling Techniques

### 1. Random Oversampling

**Randomly duplicate minority class samples.**

```python
from imblearn.over_sampling import RandomOverSampler

# Before resampling
print("Before Resampling:")
analyze_imbalance(y_train)

# Random oversampling
ros = RandomOverSampler(random_state=42)
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)

print("\nAfter Random Oversampling:")
analyze_imbalance(y_train_ros)

# Train model on resampled data
model_ros = LogisticRegression(random_state=42)
model_ros.fit(X_train_ros, y_train_ros)
y_pred_ros = model_ros.predict(X_test)

print("\nResults on Test Set:")
print(classification_report(y_test, y_pred_ros))
```

**Advantages:**
- ✅ Simple to implement
- ✅ No information loss

**Disadvantages:**
- ❌ Overfitting risk (exact duplicates)
- ❌ Increased training time

### 2. Random Undersampling

**Randomly remove majority class samples.**

```python
from imblearn.under_sampling import RandomUnderSampler

# Random undersampling
rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

print("After Random Undersampling:")
analyze_imbalance(y_train_rus)

# Train model
model_rus = LogisticRegression(random_state=42)
model_rus.fit(X_train_rus, y_train_rus)
y_pred_rus = model_rus.predict(X_test)

print("\nResults on Test Set:")
print(classification_report(y_test, y_pred_rus))
```

**Advantages:**
- ✅ Fast training (smaller dataset)
- ✅ Reduces overfitting

**Disadvantages:**
- ❌ Information loss
- ❌ May discard useful majority samples

### 3. SMOTE (Synthetic Minority Over-sampling Technique)

**Create synthetic samples by interpolating between minority class samples.**

**How SMOTE works (algorithm):**
1. For each minority class sample x:
   - Find k nearest minority class neighbors (default k=5)
   - Randomly select one neighbor x_neighbor
   - Generate synthetic sample: x_new = x + λ × (x_neighbor - x), where λ ~ Uniform(0, 1)
   - This creates point along line segment between x and x_neighbor

**Example:**
```
Original point: [2.0, 3.0]
Neighbor:       [4.0, 5.0]
Random λ = 0.6

Synthetic: [2.0, 3.0] + 0.6 × ([4.0, 5.0] - [2.0, 3.0])
         = [2.0, 3.0] + 0.6 × [2.0, 2.0]
         = [2.0, 3.0] + [1.2, 1.2]
         = [3.2, 4.2]
```

```python
from imblearn.over_sampling import SMOTE

# SMOTE
smote = SMOTE(random_state=42, k_neighbors=5)  # k_neighbors is critical parameter
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("After SMOTE:")
analyze_imbalance(y_train_smote)

# Train model
model_smote = LogisticRegression(random_state=42)
model_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = model_smote.predict(X_test)

print("\nResults on Test Set:")
print(classification_report(y_test, y_pred_smote))

# Visualize SMOTE
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_train_smote_pca = pca.transform(X_train_smote)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Before SMOTE
for label in [0, 1]:
    mask = y_train == label
    axes[0].scatter(X_train_pca[mask, 0], X_train_pca[mask, 1],
                   alpha=0.5, label=f'Class {label}')
axes[0].set_title('Before SMOTE')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# After SMOTE
for label in [0, 1]:
    mask = y_train_smote == label
    axes[1].scatter(X_train_smote_pca[mask, 0], X_train_smote_pca[mask, 1],
                   alpha=0.5, label=f'Class {label}')
axes[1].set_title('After SMOTE')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Advantages:**
- ✅ Creates diverse synthetic samples
- ✅ No information loss from majority class
- ✅ Reduces overfitting vs random oversampling

**Disadvantages:**
- ❌ Can create noisy samples (especially if minority class is noisy)
- ❌ May generate unrealistic samples in sparse regions
- ❌ Computationally expensive for large datasets
- ❌ Can amplify noise or outliers in minority class

**Critical warnings:**
- ⚠️ NEVER apply SMOTE before cross-validation (causes data leakage)
- ⚠️ NEVER apply SMOTE to test set (only on training data)
- ⚠️ Works poorly when minority class has multiple distinct clusters
- ⚠️ May not help with tree-based models (they handle imbalance well already)

### 4. SMOTE Variants

```python
from imblearn.over_sampling import SMOTE, SMOTENC, SVMSMOTE, BorderlineSMOTE, ADASYN

# Borderline-SMOTE: Only oversample samples near decision boundary
borderline_smote = BorderlineSMOTE(random_state=42)
X_train_bs, y_train_bs = borderline_smote.fit_resample(X_train, y_train)

# ADASYN: Adaptive Synthetic Sampling
adasyn = ADASYN(random_state=42)
X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)

# Compare
variants = {
    'SMOTE': (X_train_smote, y_train_smote),
    'Borderline-SMOTE': (X_train_bs, y_train_bs),
    'ADASYN': (X_train_adasyn, y_train_adasyn)
}

results_variants = {}

for name, (X_tr, y_tr) in variants.items():
    model = LogisticRegression(random_state=42)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_test)

    results_variants[name] = {
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }

df_variants = pd.DataFrame(results_variants).T
print(df_variants)
```

### 5. Combined Sampling: SMOTE + Tomek Links

**Oversample minority class, then clean overlapping samples.**

```python
from imblearn.combine import SMOTETomek

# SMOTE + Tomek Links
smote_tomek = SMOTETomek(random_state=42)
X_train_st, y_train_st = smote_tomek.fit_resample(X_train, y_train)

print("After SMOTE + Tomek Links:")
analyze_imbalance(y_train_st)

model_st = LogisticRegression(random_state=42)
model_st.fit(X_train_st, y_train_st)
y_pred_st = model_st.predict(X_test)

print("\nResults:")
print(classification_report(y_test, y_pred_st))
```

### Resampling Comparison

```python
# Compare all resampling methods
resampling_methods = {
    'No Resampling': (X_train, y_train),
    'Random Oversample': (X_train_ros, y_train_ros),
    'Random Undersample': (X_train_rus, y_train_rus),
    'SMOTE': (X_train_smote, y_train_smote),
    'SMOTE + Tomek': (X_train_st, y_train_st)
}

results_resampling = {}

for name, (X_tr, y_tr) in resampling_methods.items():
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]

    results_resampling[name] = {
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_scores),
        'PR-AUC': average_precision_score(y_test, y_scores)
    }

df_resampling = pd.DataFrame(results_resampling).T
print(df_resampling.round(4))

# Visualize
df_resampling[['Precision', 'Recall', 'F1']].plot(kind='bar', figsize=(12, 6))
plt.ylabel('Score')
plt.title('Resampling Methods Comparison')
plt.xticks(rotation=45, ha='right')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()
```

---

## Algorithm-Level Approaches

### 1. Class Weights

**Penalize mistakes on minority class more heavily.**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

print("Class Weights:")
for label, weight in zip(np.unique(y_train), class_weights):
    print(f"  Class {label}: {weight:.2f}")

# Train with class weights
model_weighted = LogisticRegression(
    class_weight='balanced',
    random_state=42
)
model_weighted.fit(X_train, y_train)
y_pred_weighted = model_weighted.predict(X_test)

print("\nResults with Class Weights:")
print(classification_report(y_test, y_pred_weighted))

# Compare with unweighted
model_unweighted = LogisticRegression(random_state=42)
model_unweighted.fit(X_train, y_train)
y_pred_unweighted = model_unweighted.predict(X_test)

print("\nResults without Class Weights:")
print(classification_report(y_test, y_pred_unweighted))
```

**How 'balanced' class weights work:**
Formula: weight_i = n_samples / (n_classes × n_samples_i)

Example with 990 negatives, 10 positives (1000 total, 2 classes):
- weight_0 = 1000 / (2 × 990) = 0.505 (majority class gets small weight)
- weight_1 = 1000 / (2 × 10) = 50.0 (minority class gets large weight)

Effect: Misclassifying one minority sample costs 50x more than one majority sample, forcing the model to pay attention to rare class.

**When class weights work well:**
- ✅ Linear models (LogisticRegression, LinearSVC)
- ✅ Tree-based models (RandomForest, XGBoost, LightGBM)
- ✅ SVMs
- ✅ Mild to moderate imbalance (up to 100:1)

**When class weights may not help:**
- ❌ Models that don't support sample weights
- ❌ Extreme imbalance (>1000:1) - may need anomaly detection instead
- ❌ When minority class is just noise (class weights amplify noise too)

### 2. Threshold Adjustment

**Adjust decision threshold to favor minority class.**

```python
def find_optimal_threshold(y_true, y_scores, metric='f1'):
    """
    Find optimal threshold for given metric.
    """
    thresholds = np.linspace(0, 1, 101)
    scores = []

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)

        if metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred)

        scores.append(score)

    optimal_idx = np.argmax(scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_score = scores[optimal_idx]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, scores, linewidth=2)
    plt.axvline(x=optimal_threshold, color='r', linestyle='--',
               label=f'Optimal = {optimal_threshold:.3f}')
    plt.axvline(x=0.5, color='k', linestyle='--', alpha=0.3,
               label='Default = 0.5')
    plt.xlabel('Threshold')
    plt.ylabel(f'{metric.upper()} Score')
    plt.title(f'Threshold vs {metric.upper()}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    print(f"Optimal threshold for {metric}: {optimal_threshold:.3f}")
    print(f"Score at optimal threshold: {optimal_score:.4f}")

    return optimal_threshold

# Find optimal threshold for F1
y_scores = model.predict_proba(X_test)[:, 1]
optimal_threshold = find_optimal_threshold(y_test, y_scores, metric='f1')

# Predict with optimal threshold
y_pred_optimal = (y_scores >= optimal_threshold).astype(int)

print("\nResults with Optimal Threshold:")
print(classification_report(y_test, y_pred_optimal))
```

**Important considerations for threshold adjustment:**

1. **When threshold adjustment is appropriate:**
   - ✅ Model outputs well-calibrated probabilities
   - ✅ You have clear metric to optimize (e.g., maximize F1, achieve 90% recall)
   - ✅ Different error costs are known (can optimize threshold based on cost)
   - ✅ Business requirements change over time (easy to adjust without retraining)

2. **When threshold adjustment is NOT enough:**
   - ❌ Model probabilities are poorly calibrated (e.g., all predictions near 0.5 or extremes)
   - ❌ Model hasn't learned minority class patterns (threshold can't fix bad model)
   - ❌ Severe imbalance where model never predicts minority class

3. **Threshold selection on test set is data leakage:**
   - Using test set to find optimal threshold, then evaluating on same test set is WRONG
   - Correct approach: Use validation set to find threshold, evaluate on separate test set
   - Or use nested cross-validation: inner loop finds threshold, outer loop evaluates

4. **Default threshold (0.5) assumes equal costs and balanced classes:**
   - Default 0.5 is optimal only when P(y=1) = 0.5 and errors have equal cost
   - For imbalanced data with 1% positives, optimal threshold is often much lower (0.1-0.3)
   - Example: With 1% positive class and equal error costs, optimal threshold ≈ 0.01

**Example of threshold selection without leakage:**
```python
# Split into train, validation, test
X_tr, X_temp, y_tr, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Train on train set
model.fit(X_tr, y_tr)

# Find optimal threshold on validation set
y_val_scores = model.predict_proba(X_val)[:, 1]
optimal_threshold = find_optimal_threshold(y_val, y_val_scores, metric='f1')

# Evaluate on test set with chosen threshold
y_test_scores = model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_scores >= optimal_threshold).astype(int)
print(classification_report(y_test, y_test_pred))
```

---

## Cost-Sensitive Learning

### Custom Cost Matrix

```python
def cost_sensitive_prediction(model, X, cost_matrix):
    """
    Make predictions using custom cost matrix.

    cost_matrix[i, j] = cost of predicting j when true class is i
    """
    y_proba = model.predict_proba(X)

    # Expected cost for each sample and each class
    expected_costs = y_proba.dot(cost_matrix.T)

    # Predict class with minimum expected cost
    y_pred = np.argmin(expected_costs, axis=1)

    return y_pred

# Define cost matrix
cost_matrix = np.array([
    [0, 1],   # True Negative (correct): cost 0, False Positive: cost 1
    [10, 0]   # False Negative: cost 10, True Positive (correct): cost 0
])

print("Cost Matrix:")
print("  Predicting Negative when True Negative: 0 (correct)")
print("  Predicting Positive when True Negative: 1 (false alarm)")
print("  Predicting Negative when True Positive: 10 (missed fraud!)")
print("  Predicting Positive when True Positive: 0 (correct)")

# Train model
model_cost = LogisticRegression(random_state=42)
model_cost.fit(X_train, y_train)

# Predict with cost matrix
y_pred_cost = cost_sensitive_prediction(model_cost, X_test, cost_matrix)

print("\nResults with Cost-Sensitive Prediction:")
print(classification_report(y_test, y_pred_cost))

# Calculate total cost
def calculate_total_cost(y_true, y_pred, cost_matrix):
    cm = confusion_matrix(y_true, y_pred)
    total_cost = np.sum(cm * cost_matrix)
    return total_cost

cost_standard = calculate_total_cost(y_test, model.predict(X_test), cost_matrix)
cost_optimized = calculate_total_cost(y_test, y_pred_cost, cost_matrix)

print(f"\nTotal cost (standard): {cost_standard}")
print(f"Total cost (cost-sensitive): {cost_optimized}")
print(f"Cost reduction: {((cost_standard - cost_optimized)/cost_standard)*100:.1f}%")
```

---

## Ensemble Methods

### 1. Balanced Random Forest

```python
from imblearn.ensemble import BalancedRandomForestClassifier

# Balanced Random Forest
brf = BalancedRandomForestClassifier(
    n_estimators=100,
    random_state=42
)
brf.fit(X_train, y_train)
y_pred_brf = brf.predict(X_test)

print("Balanced Random Forest:")
print(classification_report(y_test, y_pred_brf))
```

### 2. Easy Ensemble

```python
from imblearn.ensemble import EasyEnsembleClassifier

# Easy Ensemble
eec = EasyEnsembleClassifier(
    n_estimators=10,
    random_state=42
)
eec.fit(X_train, y_train)
y_pred_eec = eec.predict(X_test)

print("Easy Ensemble:")
print(classification_report(y_test, y_pred_eec))
```

### 3. Balanced Bagging

```python
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Balanced Bagging
bbc = BalancedBaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=10,
    random_state=42
)
bbc.fit(X_train, y_train)
y_pred_bbc = bbc.predict(X_test)

print("Balanced Bagging:")
print(classification_report(y_test, y_pred_bbc))
```

### Ensemble Comparison

```python
from sklearn.ensemble import RandomForestClassifier

ensemble_methods = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Balanced RF': BalancedRandomForestClassifier(n_estimators=100, random_state=42),
    'Easy Ensemble': EasyEnsembleClassifier(n_estimators=10, random_state=42),
    'Balanced Bagging': BalancedBaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=10,
        random_state=42
    )
}

results_ensemble = {}

for name, model in ensemble_methods.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]

    results_ensemble[name] = {
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_scores)
    }

df_ensemble = pd.DataFrame(results_ensemble).T
print(df_ensemble.round(4))
```

---

## Anomaly Detection Approaches

**For severe imbalance (e.g., 1000:1), treat as anomaly detection.**

**When to use anomaly detection instead of classification:**

1. **Extreme imbalance (>1000:1):**
   - Minority class < 0.1% of data
   - Example: Network intrusion (1 attack per 10,000 normal requests)
   - Too few minority samples for classifier to learn meaningful patterns

2. **Minority class is fundamentally different:**
   - Minority class = "anything abnormal"
   - Normal class is well-defined, abnormal class is diverse
   - Example: Fraud detection (fraudsters constantly change tactics)

3. **Labeled minority samples are scarce but normal samples are abundant:**
   - Can train on unlabeled normal data only
   - Don't need labeled anomalies for training

**When to stick with classification approaches:**
- Imbalance < 100:1
- Minority class has clear, learnable patterns
- Sufficient labeled examples of minority class (>100 samples)
- Need probabilistic predictions, not just binary anomaly/normal

**Key difference in approach:**
- **Classification:** Learn decision boundary between classes using both classes
- **Anomaly detection:** Learn "normal" behavior, flag deviations as anomalies

### 1. Isolation Forest

```python
from sklearn.ensemble import IsolationForest

# Train on majority class only
X_train_majority = X_train[y_train == 0]

iso_forest = IsolationForest(
    contamination=0.01,  # Expected proportion of outliers
    random_state=42
)
iso_forest.fit(X_train_majority)

# Predict (-1 for anomaly, 1 for normal)
y_pred_iso = iso_forest.predict(X_test)
y_pred_iso = (y_pred_iso == -1).astype(int)  # Convert to 0/1

print("Isolation Forest:")
print(classification_report(y_test, y_pred_iso))
```

### 2. One-Class SVM

```python
from sklearn.svm import OneClassSVM

# Train on majority class only
ocsvm = OneClassSVM(gamma='auto', nu=0.01)
ocsvm.fit(X_train_majority)

# Predict
y_pred_ocsvm = ocsvm.predict(X_test)
y_pred_ocsvm = (y_pred_ocsvm == -1).astype(int)

print("One-Class SVM:")
print(classification_report(y_test, y_pred_ocsvm))
```

### 3. Autoencoder (Neural Network)

```python
import tensorflow as tf
from tensorflow import keras

# Train autoencoder on majority class
autoencoder = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(X_train.shape[1], activation='linear')
])

autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(
    X_train_majority, X_train_majority,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

# Reconstruction error as anomaly score
X_test_pred = autoencoder.predict(X_test)
reconstruction_errors = np.mean((X_test - X_test_pred)**2, axis=1)

# Use threshold
threshold = np.percentile(reconstruction_errors, 99)
y_pred_ae = (reconstruction_errors > threshold).astype(int)

print("Autoencoder:")
print(classification_report(y_test, y_pred_ae))
```

---

## Best Practices

### 1. Always Use Stratified Splitting

```python
# Stratified split maintains class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y  # Important!
)

print("Train distribution:")
analyze_imbalance(y_train)

print("\nTest distribution:")
analyze_imbalance(y_test)
```

### 2. Use Appropriate Metrics

```python
# Don't use accuracy alone!
def comprehensive_evaluation(y_true, y_pred, y_scores):
    """
    Comprehensive evaluation for imbalanced data.
    """
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    print(f"\nROC-AUC: {roc_auc_score(y_true, y_scores):.4f}")
    print(f"PR-AUC: {average_precision_score(y_true, y_scores):.4f}")
    print(f"MCC: {matthews_corrcoef(y_true, y_pred):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

comprehensive_evaluation(y_test, y_pred_smote, model_smote.predict_proba(X_test)[:, 1])
```

### 3. Cross-Validation with Stratification

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Stratified K-Fold
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validation
cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=skfold,
    scoring='f1'
)

print(f"CV F1 scores: {cv_scores}")
print(f"Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

### 4. Pipeline with Resampling

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler

# Create pipeline
pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', LogisticRegression(random_state=42))
])

# Train (SMOTE applied inside each CV fold!)
cv_scores = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=skfold,
    scoring='f1'
)

print(f"Pipeline CV F1 scores: {cv_scores}")
print(f"Mean: {cv_scores.mean():.4f}")
```

### Decision Flowchart

```
Is imbalance ratio < 10:1?
├─ YES → Try class weights first
│   └─ If not enough → Try SMOTE
│
└─ NO → Is ratio < 100:1?
    ├─ YES → Try SMOTE or ensemble methods
    │   └─ Consider threshold tuning
    │
    └─ NO → Severe imbalance (> 100:1)
        └─ Consider anomaly detection
        └─ Or ensemble + heavy class weighting
```

---

## Complete Examples

### Example 1: Credit Card Fraud Detection

```python
# Simulate credit card fraud dataset
np.random.seed(42)
X_fraud, y_fraud = make_classification(
    n_samples=50000,
    n_features=30,
    n_informative=25,
    n_redundant=5,
    n_classes=2,
    weights=[0.998, 0.002],  # 99.8% legitimate, 0.2% fraud
    flip_y=0.01,
    random_state=42
)

print("Credit Card Fraud Dataset:")
analyze_imbalance(y_fraud)

# Split
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_fraud, y_fraud, test_size=0.3, random_state=42, stratify=y_fraud
)

# Scale
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_f = scaler.fit_transform(X_train_f)
X_test_f = scaler.transform(X_test_f)

# Try different approaches
approaches = {
    'Baseline': (X_train_f, y_train_f),
    'Class Weights': (X_train_f, y_train_f),
    'SMOTE': SMOTE(random_state=42).fit_resample(X_train_f, y_train_f),
    'SMOTE + Tomek': SMOTETomek(random_state=42).fit_resample(X_train_f, y_train_f)
}

results_fraud = {}

for name, (X_tr, y_tr) in approaches.items():
    if name == 'Baseline':
        model = LogisticRegression(random_state=42, max_iter=1000)
    elif name == 'Class Weights':
        model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    else:
        model = LogisticRegression(random_state=42, max_iter=1000)

    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_test_f)
    y_scores = model.predict_proba(X_test_f)[:, 1]

    # Calculate costs (False Negative = missed fraud = $100, False Positive = investigation = $1)
    cm = confusion_matrix(y_test_f, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total_cost = fn * 100 + fp * 1

    results_fraud[name] = {
        'Recall': recall_score(y_test_f, y_pred),
        'Precision': precision_score(y_test_f, y_pred),
        'F1': f1_score(y_test_f, y_pred),
        'PR-AUC': average_precision_score(y_test_f, y_scores),
        'False Negatives': fn,
        'Total Cost ($)': total_cost
    }

df_fraud = pd.DataFrame(results_fraud).T
print("\nFraud Detection Results:")
print(df_fraud.round(4))

print(f"\nBest approach by cost: {df_fraud['Total Cost ($)'].idxmin()}")
print(f"Cost savings: ${df_fraud['Total Cost ($)'].max() - df_fraud['Total Cost ($)'].min()}")
```

---

## Summary

### Quick Reference

| Imbalance Ratio | Recommended Approach |
|----------------|---------------------|
| < 5:1 | Class weights |
| 5:1 - 20:1 | SMOTE or class weights |
| 20:1 - 100:1 | SMOTE + ensemble methods |
| > 100:1 | Anomaly detection or ensemble with heavy weighting |

### Key Takeaways

1. **Don't use accuracy** - Use F1, PR-AUC, or MCC
2. **Always stratify** - Train/test split and CV
3. **Try multiple approaches** - No single solution works for all
4. **Consider business costs** - False negatives vs false positives
5. **Use pipelines** - Prevent data leakage
6. **Monitor both metrics** - Precision AND recall
7. **Test on original distribution** - Don't resample test set!

### Checklist

- [ ] Analyze class distribution and imbalance ratio
- [ ] Use stratified train/test split
- [ ] Choose appropriate metrics (not accuracy!)
- [ ] Try class weights first (simplest)
- [ ] If needed, try SMOTE or other resampling
- [ ] Consider ensemble methods
- [ ] Tune decision threshold
- [ ] Use stratified cross-validation
- [ ] Keep test set in original distribution
- [ ] Evaluate with multiple metrics

Handling imbalanced data correctly is crucial for real-world ML applications!
