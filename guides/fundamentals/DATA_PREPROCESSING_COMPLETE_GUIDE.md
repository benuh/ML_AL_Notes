# 🧹 Data Preprocessing: Complete Guide

**"Garbage in, garbage out."** Data preprocessing is the foundation of successful ML projects. This guide covers everything from data loading to final preparation for modeling.

---

## 📋 Table of Contents

- [Why Preprocessing Matters](#why-preprocessing-matters)
- [Data Loading and Inspection](#data-loading-and-inspection)
- [Handling Missing Values](#handling-missing-values)
  - [Statistical Theory of Missing Data](#statistical-theory-of-missing-data)
    - [Formal Framework (Rubin 1976)](#formal-framework-rubin-1976)
    - [Little's MCAR Test](#littles-mcar-test)
    - [Bias-Variance Trade-off in Imputation](#bias-variance-trade-off-in-imputation)
    - [Multiple Imputation Theory](#multiple-imputation-theory)
- [Dealing with Duplicates](#dealing-with-duplicates)
- [Outlier Detection and Treatment](#outlier-detection-and-treatment)
- [Data Type Conversions](#data-type-conversions)
- [Encoding Categorical Variables](#encoding-categorical-variables)
- [Feature Scaling and Normalization](#feature-scaling-and-normalization)
- [Handling Imbalanced Data](#handling-imbalanced-data)
- [Train-Test Split](#train-test-split)
- [Building Preprocessing Pipelines](#building-preprocessing-pipelines)
- [Common Pitfalls](#common-pitfalls)

---

## Why Preprocessing Matters

### Impact on Model Performance

**Example: Before vs After Preprocessing**

```python
# Before preprocessing
model = RandomForestClassifier()
model.fit(raw_data, labels)
# Accuracy: 65%

# After proper preprocessing
preprocessed_data = preprocess(raw_data)
model.fit(preprocessed_data, labels)
# Accuracy: 87%  # 34% improvement!
```

### What Can Go Wrong

**Common issues in raw data:**
- Missing values (model can't handle NaN)
- Different scales (age: 0-100, income: 0-1M)
- Categorical variables as strings (model needs numbers)
- Outliers (skew model predictions)
- Class imbalance (99% one class, 1% other)
- Data leakage (test data info in training)

---

## Data Loading and Inspection

### Loading Data

```python
import pandas as pd
import numpy as np

# CSV files
df = pd.read_csv('data.csv')

# Excel files
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# JSON files
df = pd.read_json('data.json')

# SQL databases
import sqlite3
conn = sqlite3.connect('database.db')
df = pd.read_sql_query("SELECT * FROM table_name", conn)

# Parquet (efficient for large data)
df = pd.read_parquet('data.parquet')

# Large files (chunking)
chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    process(chunk)
```

### Initial Inspection

```python
# Basic info
print(f"Shape: {df.shape}")  # (rows, columns)
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# First/last rows
print(df.head(10))
print(df.tail(10))

# Column info
print(df.info())
# Shows: column names, non-null counts, data types

# Statistical summary
print(df.describe())
# Shows: count, mean, std, min, quartiles, max

# Include categorical columns
print(df.describe(include='all'))

# Check data types
print(df.dtypes)

# Unique values per column
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")
```

### Visual Inspection

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Distribution of numerical features
df.hist(figsize=(16, 12), bins=50)
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()

# Box plots for outlier detection
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
numerical_cols = df.select_dtypes(include=[np.number]).columns[:6]

for idx, col in enumerate(numerical_cols):
    ax = axes[idx // 3, idx % 3]
    df.boxplot(column=col, ax=ax)
    ax.set_title(col)

plt.tight_layout()
plt.show()

# Missing data visualization
import missingno as msno
msno.matrix(df)
plt.show()
```

---

## Handling Missing Values

### Detect Missing Values

```python
# Count missing values
missing_counts = df.isnull().sum()
print(missing_counts[missing_counts > 0])

# Percentage missing
missing_pct = (df.isnull().sum() / len(df)) * 100
print(missing_pct[missing_pct > 0])

# Missing value patterns
print(df.isnull().sum(axis=1).value_counts())  # Rows with X missing values

# Visualize missing patterns
import seaborn as sns
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False)
plt.title('Missing Data Pattern')
plt.show()
```

### Strategy 1: Remove Missing Data

```python
# Remove rows with any missing values
df_clean = df.dropna()

# Remove rows with missing values in specific columns
df_clean = df.dropna(subset=['important_column1', 'important_column2'])

# Remove columns with >50% missing values
threshold = 0.5
missing_pct = df.isnull().sum() / len(df)
cols_to_drop = missing_pct[missing_pct > threshold].index
df_clean = df.drop(columns=cols_to_drop)

# Remove rows with more than 50% values missing
df_clean = df.dropna(thresh=len(df.columns) * 0.5)
```

### Strategy 2: Imputation - Simple Methods

```python
# Mean imputation (numerical)
df['age'].fillna(df['age'].mean(), inplace=True)

# Median imputation (better for skewed data)
df['income'].fillna(df['income'].median(), inplace=True)

# Mode imputation (categorical)
df['category'].fillna(df['category'].mode()[0], inplace=True)

# Forward fill (time series)
df['sensor_reading'].fillna(method='ffill', inplace=True)

# Backward fill
df['sensor_reading'].fillna(method='bfill', inplace=True)

# Constant value
df['optional_field'].fillna('Unknown', inplace=True)
df['count'].fillna(0, inplace=True)
```

### Strategy 3: Advanced Imputation

```python
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Simple imputer (scikit-learn)
imputer = SimpleImputer(strategy='mean')  # or 'median', 'most_frequent', 'constant'
df_numerical = df.select_dtypes(include=[np.number])
df[df_numerical.columns] = imputer.fit_transform(df_numerical)

# KNN imputation (use similar samples)
imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(
    imputer.fit_transform(df_numerical),
    columns=df_numerical.columns
)

# Iterative imputation (MICE - Multiple Imputation by Chained Equations)
imputer = IterativeImputer(max_iter=10, random_state=42)
df_imputed = pd.DataFrame(
    imputer.fit_transform(df_numerical),
    columns=df_numerical.columns
)

# Domain-specific imputation
# Example: Missing zip codes based on city
zip_by_city = df.groupby('city')['zip_code'].apply(lambda x: x.mode()[0])
df['zip_code'].fillna(df['city'].map(zip_by_city), inplace=True)
```

### Strategy 4: Create Missing Indicator

```python
# Add binary column indicating missingness
df['age_missing'] = df['age'].isnull().astype(int)

# Then impute
df['age'].fillna(df['age'].median(), inplace=True)

# This preserves information about which values were missing
```

### Choosing the Right Strategy

**Decision tree:**
```
Is the column important?
├─ No → Drop column
└─ Yes
   ├─ Missing % > 70% → Drop column or collect more data
   └─ Missing % < 70%
      ├─ Missing Completely at Random (MCAR)
      │  └─ Use mean/median/mode imputation
      ├─ Missing at Random (MAR)
      │  └─ Use KNN or iterative imputation
      └─ Missing Not at Random (MNAR)
         └─ Create missing indicator + imputation
```

**Understanding missing data mechanisms:**

1. **MCAR (Missing Completely at Random):**
   - Missingness is unrelated to any data (observed or unobserved)
   - Example: Survey responses lost due to random server crash
   - Test: Missing data pattern shows no correlation with any variable
   - Safe to use: Simple imputation (mean/median/mode)

2. **MAR (Missing at Random):**
   - Missingness related to observed data, but not the missing value itself
   - Example: Younger people more likely to skip income question (missingness relates to age, not income value)
   - Test: Missingness correlates with other observed features
   - Solution: Model-based imputation (KNN, MICE) using correlated features

3. **MNAR (Missing Not at Random):**
   - Missingness related to the unobserved value itself
   - Example: High earners more likely to skip income question (missingness relates to income value)
   - Test: Missingness remains unexplained after accounting for all observed variables
   - Solution: Create missing indicator + imputation, or model missingness explicitly

**Important:** MCAR is rare in practice. Most real-world missing data is MAR or MNAR. Using simple imputation (mean) on MAR/MNAR data can introduce bias.

---

### Statistical Theory of Missing Data

#### Formal Framework (Rubin 1976)

**Complete Data:** X = (X_obs, X_mis)
- X_obs: Observed values
- X_mis: Missing values

**Missingness Indicator:** R = binary matrix where Rᵢⱼ = 1 if xᵢⱼ is missing

**Missingness Mechanism:** Distribution P(R | X, ψ) where ψ are parameters

**Three Mechanisms (Formal Definitions):**

1. **MCAR (Missing Completely at Random):**
   ```
   P(R | X_obs, X_mis, ψ) = P(R | ψ)
   ```
   Missingness independent of both observed and missing values.

2. **MAR (Missing at Random):**
   ```
   P(R | X_obs, X_mis, ψ) = P(R | X_obs, ψ)
   ```
   Missingness may depend on observed data, but not on missing values themselves.

3. **MNAR (Missing Not at Random):**
   ```
   P(R | X_obs, X_mis, ψ) depends on X_mis
   ```
   Missingness depends on the missing values themselves.

#### Little's MCAR Test

**Null Hypothesis:** Data is MCAR

**Test Statistic:**
```
d² = Σⱼ nⱼ(μ̂ⱼ - μ̂_total)ᵀ Σ̂⁻¹ (μ̂ⱼ - μ̂_total)
```

where:
- j indexes different missing patterns
- nⱼ = number of observations with pattern j
- μ̂ⱼ = mean vector for pattern j
- μ̂_total = overall mean vector
- Σ̂ = pooled covariance matrix

**Distribution under H₀:**
```
d² ~ χ²(df)
```

where df = (# patterns - 1) × (# variables)

**Decision:** Reject MCAR if p-value < α (typically 0.05)

**Implementation:**
```python
# Little's MCAR test (requires R package or custom implementation)
from scipy.stats import chi2
import numpy as np

def littles_mcar_test(data):
    """
    Simplified Little's MCAR test

    Returns: chi-square statistic, p-value
    """
    # Identify missing patterns
    missing_patterns = data.isnull().astype(int)
    unique_patterns = missing_patterns.drop_duplicates()

    # For each pattern, compute mean of observed variables
    chi_square = 0
    df = 0

    for _, pattern in unique_patterns.iterrows():
        # Select rows with this pattern
        mask = (missing_patterns == pattern).all(axis=1)
        subset = data[mask]

        # Observed variables in this pattern
        observed_vars = pattern[pattern == 0].index

        if len(observed_vars) > 0:
            # Compute pattern-specific mean
            pattern_mean = subset[observed_vars].mean()
            overall_mean = data[observed_vars].mean()

            # Contribution to chi-square
            diff = pattern_mean - overall_mean
            n = len(subset)
            chi_square += n * np.sum(diff ** 2)  # Simplified
            df += len(observed_vars)

    # P-value
    p_value = 1 - chi2.cdf(chi_square, df)

    return chi_square, p_value, df

# Usage
chi2_stat, p_val, degrees_of_freedom = littles_mcar_test(df)
print(f"Little's MCAR Test: χ² = {chi2_stat:.2f}, df = {degrees_of_freedom}, p = {p_val:.4f}")

if p_val < 0.05:
    print("Reject MCAR: Data is likely MAR or MNAR")
else:
    print("Cannot reject MCAR: Simple imputation may be appropriate")
```

#### Bias-Variance Trade-off in Imputation

**Theorem 1 (Mean Imputation Bias):**

Under MAR, mean imputation for predictor X given outcome Y:
```
E[X_imputed | X_obs, R, Y] ≠ E[X | X_obs, Y]
```

**Bias:**
```
Bias = E[X | Y] - E[X | X_obs]
```

**Consequences:**
1. **Attenuates correlations:** cor(X_imputed, Y) < cor(X_true, Y)
2. **Reduces variance:** Var(X_imputed) < Var(X_true)
3. **Shrinks regression coefficients** toward zero

**Quantitative Example:**
```python
# True model: Y = 2X + ε
# With 50% MCAR data and mean imputation:

# Simulated data
n = 1000
X_true = np.random.normal(0, 1, n)
Y = 2 * X_true + np.random.normal(0, 0.5, n)

# Introduce MCAR missingness (50%)
missing_mask = np.random.rand(n) < 0.5
X_observed = X_true.copy()
X_observed[missing_mask] = np.nan

# Mean imputation
X_imputed = X_observed.copy()
X_imputed[np.isnan(X_imputed)] = np.nanmean(X_observed)

# Compare correlations
cor_true = np.corrcoef(X_true, Y)[0, 1]
cor_imputed = np.corrcoef(X_imputed, Y)[0, 1]

print(f"True correlation: {cor_true:.3f}")
print(f"After mean imputation: {cor_imputed:.3f}")
print(f"Attenuation: {(cor_true - cor_imputed) / cor_true * 100:.1f}%")

# Typical output:
# True correlation: 0.971
# After mean imputation: 0.686
# Attenuation: 29.3%  (severe bias!)
```

**Theorem 2 (KNN Imputation Variance):**

For KNN imputation with k neighbors, the variance of imputed value:
```
Var(X̂_imputed) ≈ Var(X_true) / k + σ²_noise
```

**Trade-off:**
- Small k: Low bias (uses similar neighbors), high variance
- Large k: High bias (averages dissimilar neighbors), low variance
- Optimal k balances bias-variance

**Cross-Validation for k Selection:**
```python
from sklearn.impute import KNNImputer
from sklearn.model_selection import cross_val_score

def select_optimal_k(X_with_missing, y, k_values=range(1, 21)):
    """
    Select optimal k for KNN imputation via CV
    """
    scores = []

    for k in k_values:
        imputer = KNNImputer(n_neighbors=k)
        X_imputed = imputer.fit_transform(X_with_missing)

        # Evaluate downstream task performance
        model = LinearRegression()
        score = cross_val_score(model, X_imputed, y, cv=5,
                                scoring='r2').mean()
        scores.append(score)

    optimal_k = k_values[np.argmax(scores)]

    plt.plot(k_values, scores)
    plt.xlabel('Number of neighbors (k)')
    plt.ylabel('Cross-validated R²')
    plt.title('KNN Imputation: Optimal k Selection')
    plt.axvline(optimal_k, color='r', linestyle='--',
                label=f'Optimal k={optimal_k}')
    plt.legend()
    plt.show()

    return optimal_k
```

#### Multiple Imputation Theory

**Key Idea:** Impute m times to capture uncertainty, analyze each dataset separately, then pool results.

**Rubin's Rules for Combining Estimates:**

Given m imputed datasets, for parameter θ:

**Within-imputation variance:**
```
W̄ = (1/m) Σⱼ Var(θ̂ⱼ)
```

**Between-imputation variance:**
```
B = (1/(m-1)) Σⱼ (θ̂ⱼ - θ̄)²
```

**Total variance:**
```
T = W̄ + (1 + 1/m) · B
```

**Pooled estimate:**
```
θ̂_pooled = (1/m) Σⱼ θ̂ⱼ
```

**Degrees of freedom:**
```
df = (m - 1) · [1 + (W̄ / ((1 + 1/m)·B))]²
```

**Confidence Interval:**
```
θ̂_pooled ± t_{df, α/2} · √T
```

**Fraction of Missing Information:**
```
λ = (B + B/m) / T
```

Interpretation: Proportion of total variance due to missing data

**Efficiency:**
```
Relative Efficiency = (1 + λ/m)⁻¹
```

For m = 5 imputations and λ = 0.5 (50% missing information):
```
RE = (1 + 0.5/5)⁻¹ = 0.909 (91% efficient)
```

**Practical Guideline:** m = 5-20 imputations typically sufficient

**Implementation:**
```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np

def multiple_imputation(X_missing, y, n_imputations=5):
    """
    Multiple imputation with Rubin's rules
    """
    from sklearn.linear_model import LinearRegression

    # Storage for imputed dataset results
    theta_estimates = []
    theta_variances = []

    for m in range(n_imputations):
        # Impute with different random seed
        imputer = IterativeImputer(random_state=m)
        X_imputed = imputer.fit_transform(X_missing)

        # Fit model on imputed data
        model = LinearRegression()
        model.fit(X_imputed, y)

        # Get coefficient estimates and their variances
        theta_estimates.append(model.coef_)

        # Estimate variance (via bootstrap or analytical formula)
        # Simplified: use residual variance
        y_pred = model.predict(X_imputed)
        residuals = y - y_pred
        var_estimate = np.var(residuals) * np.linalg.inv(X_imputed.T @ X_imputed).diagonal()
        theta_variances.append(var_estimate)

    # Apply Rubin's rules
    theta_pooled = np.mean(theta_estimates, axis=0)
    W_bar = np.mean(theta_variances, axis=0)
    B = np.var(theta_estimates, axis=0, ddof=1)
    T = W_bar + (1 + 1/n_imputations) * B

    # Fraction of missing information
    lambda_mi = (B + B/n_imputations) / T

    # Confidence intervals
    from scipy.stats import t as t_dist
    df = (n_imputations - 1) * (1 + W_bar / ((1 + 1/n_imputations) * B))**2
    t_crit = t_dist.ppf(0.975, df)
    ci_lower = theta_pooled - t_crit * np.sqrt(T)
    ci_upper = theta_pooled + t_crit * np.sqrt(T)

    results = {
        'theta_pooled': theta_pooled,
        'total_variance': T,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'fraction_missing_info': lambda_mi
    }

    return results

# Usage
results = multiple_imputation(X_with_missing, y, n_imputations=10)
print(f"Pooled estimates: {results['theta_pooled']}")
print(f"95% CI: [{results['ci_lower']}, {results['ci_upper']}]")
print(f"Fraction of missing info: {results['fraction_missing_info']}")
```

#### Practical Guidelines

**Missing Data Decision Tree (Enhanced):**

```
1. Check missingness percentage:
   ├─ < 5%: Deletion likely OK (if MCAR)
   ├─ 5-40%: Imputation recommended
   └─ > 40%: Collect more data or model missingness explicitly

2. Test mechanism:
   ├─ Run Little's MCAR test
   ├─ If MCAR (p > 0.05): Simple imputation acceptable
   │  └─ Mean/median for numerical, mode for categorical
   └─ If MAR/MNAR (p < 0.05): Use model-based imputation
      ├─ KNN: Good for local structure
      ├─ MICE: Good for complex dependencies
      └─ Multiple imputation: Best for inference (p-values, CIs)

3. Downstream task:
   ├─ Prediction only: Single imputation OK
   └─ Inference (hypothesis tests): Multiple imputation required

4. Validation:
   ├─ Artificially introduce missingness in complete data
   ├─ Compare imputation methods
   └─ Select method with best downstream task performance
```

**Imputation Method Comparison:**

| Method | Bias (MAR) | Variance | Preserves Correlations | Computational Cost |
|--------|------------|----------|------------------------|-------------------|
| Mean | High | Low (too low!) | No | O(n) |
| Median | High | Low | No | O(n log n) |
| KNN | Medium | Medium | Partially | O(n²) |
| MICE | Low | Medium | Yes | O(n · iter) |
| Multiple Imputation | Lowest | Correct | Yes | O(m · n · iter) |

---

## Dealing with Duplicates

### Detect Duplicates

```python
# Check for duplicate rows
duplicates = df.duplicated()
print(f"Number of duplicate rows: {duplicates.sum()}")

# Show duplicate rows
print(df[duplicates])

# Check duplicates based on specific columns
duplicates = df.duplicated(subset=['user_id', 'date'], keep=False)
print(df[duplicates].sort_values(['user_id', 'date']))

# Count duplicates per key
duplicate_counts = df.groupby('user_id').size()
print(duplicate_counts[duplicate_counts > 1])
```

### Remove Duplicates

```python
# Remove all duplicates (keep first occurrence)
df_clean = df.drop_duplicates()

# Keep last occurrence
df_clean = df.drop_duplicates(keep='last')

# Remove duplicates based on specific columns
df_clean = df.drop_duplicates(subset=['user_id', 'date'], keep='first')

# Keep only truly unique rows (remove all duplicates)
df_clean = df.drop_duplicates(keep=False)
```

### Handle Duplicates with Aggregation

```python
# Instead of dropping, aggregate
# Example: Multiple transactions per user per day
df_agg = df.groupby(['user_id', 'date']).agg({
    'amount': 'sum',
    'transaction_count': 'count',
    'category': lambda x: x.mode()[0]  # Most common category
}).reset_index()
```

---

## Outlier Detection and Treatment

### Statistical Theory of Outliers

**Definition and Types:**
```
Outlier: Observation that deviates significantly from other observations

Types:
1. Point (Global) Outliers: Deviate from entire dataset
   Example: Income = $10M when median = $50K

2. Contextual (Conditional) Outliers: Normal in one context, outlier in another
   Example: Temperature = 30°C in summer (normal), winter (outlier)

3. Collective Outliers: Individual points normal, but group is unusual
   Example: Unusual spike pattern in time series

Statistical Framework:
Under normality assumption X ~ N(μ, σ²):
- P(|X - μ| > 2σ) ≈ 0.05 (5% outside ±2σ)
- P(|X - μ| > 3σ) ≈ 0.003 (0.3% outside ±3σ)

If we observe |x - μ| > 3σ:
→ Either rare event (p=0.003) or x is outlier
```

**Theoretical Justification for IQR Method:**
```
Tukey's Fences (1977):
Lower fence: Q₁ - k × IQR
Upper fence: Q₃ + k × IQR

where k = 1.5 (standard), k = 3.0 (extreme outliers only)

Mathematical Basis (for normal distribution):
1. Q₁ ≈ μ - 0.675σ (25th percentile)
2. Q₃ ≈ μ + 0.675σ (75th percentile)
3. IQR = Q₃ - Q₁ ≈ 1.35σ

Lower fence: Q₁ - 1.5×IQR ≈ μ - 0.675σ - 1.5×1.35σ ≈ μ - 2.7σ
Upper fence: Q₃ + 1.5×IQR ≈ μ + 0.675σ + 1.5×1.35σ ≈ μ + 2.7σ

Expected proportion beyond fences (for normal data):
P(X < μ - 2.7σ or X > μ + 2.7σ) ≈ 0.7%

Robustness:
- Quartiles unaffected by values beyond Q₁ and Q₃
- Breakdown point: 25% (can tolerate 25% outliers before failing)
- Compare: Mean/std breakdown point: 0% (single outlier affects them)

Why k=1.5?
- Balance between sensitivity and specificity
- k=1.0: Too sensitive (flags ~7% as outliers)
- k=2.0: Too permissive (flags ~0.1% as outliers)
- k=1.5: Reasonable compromise (~0.7% flagged)
```

**Z-Score Method (Parametric):**
```
Z-score: z = (x - μ) / σ

Measures: How many standard deviations x is from mean

Threshold Selection:
- |z| > 2: Unusual (outside 95% of normal distribution)
- |z| > 2.5: Outlier (outside 98.8%)
- |z| > 3: Strong outlier (outside 99.7%)

Statistical Test Interpretation:
Under H₀: x ~ N(μ, σ²)
P(|Z| > 3) = 0.0027

If observed |z| > 3:
→ Either rare event (p=0.003) OR reject H₀ (x is outlier)

Multiple Testing Correction:
For n observations, probability at least one exceeds threshold by chance:
P(at least one |z| > 3) = 1 - (1 - 0.0027)ⁿ

Examples:
n=100: P ≈ 0.24 (24% chance of false positive!)
n=1000: P ≈ 0.94 (94% chance of false positive!)

Bonferroni correction:
Use threshold z* where P(|Z| > z*) = α/n
For α=0.05, n=100: z* ≈ 3.89 (more stringent)

Limitations:
1. Assumes normality (invalid for skewed/heavy-tailed distributions)
2. Sensitive to outliers (outliers inflate σ, masking themselves!)
3. Not robust (single outlier can corrupt μ and σ)
```

**Modified Z-Score (Robust):**
```
Formula: M = 0.6745 × (x - median) / MAD

where MAD = median(|xᵢ - median(x)|)
and 0.6745 ≈ Φ⁻¹(0.75) makes MAD consistent estimator of σ

Why 0.6745?
For X ~ N(μ, σ²):
MAD → σ × Φ⁻¹(0.75) ≈ 0.6745σ

Therefore: MAD / 0.6745 → σ (consistent estimator)

Advantages over standard z-score:
1. Median: 50% breakdown point (vs 0% for mean)
2. MAD: 50% breakdown point (vs 0% for std dev)
3. Robust to outliers (outliers don't corrupt median or MAD)

Example:
Data: [1, 2, 3, 4, 5, 100]

Standard z-score:
μ = 19.17, σ = 38.9
z(100) = (100 - 19.17)/38.9 = 2.08 (barely flags as outlier!)

Modified z-score:
median = 3.5, MAD = median(|[1-3.5, 2-3.5, ..., 100-3.5]|) = 1.5
M(100) = 0.6745 × (100-3.5)/1.5 = 43.4 (strongly flags as outlier!)

Threshold:
Typically use |M| > 3.5 (slightly higher than z-score threshold)
```

**Grubbs' Test (Single Outlier):**
```
Formal hypothesis test for single outlier

H₀: No outliers in data
H₁: Exactly one outlier

Test statistic:
G = max|xᵢ - x̄| / s

where x̄ = sample mean, s = sample std dev

Critical value (significance level α, sample size n):
G_crit = [(n-1) / √n] × √[t²_{α/(2n), n-2} / (n - 2 + t²_{α/(2n), n-2})]

Decision: Reject H₀ (outlier present) if G > G_crit

Properties:
- Assumes normality
- Tests only most extreme point
- Iterative: After removing outlier, can test again
- Conservative (Type I error controlled at level α)

Limitation: Masking effect
- If multiple outliers present, they mask each other
- Outliers inflate s, reducing G
- May fail to detect any outliers even when present!
```

### Statistical Methods

**IQR Method:**
```python
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)

    print(f"Outliers in {column}: {outliers.sum()} ({outliers.sum()/len(df)*100:.2f}%)")
    print(f"Range: [{lower_bound:.2f}, {upper_bound:.2f}]")

    return outliers

# Detect outliers
outliers = detect_outliers_iqr(df, 'price')
```

**Z-Score Method:**
```python
from scipy import stats

def detect_outliers_zscore(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    outliers = z_scores > threshold

    print(f"Outliers in {column}: {outliers.sum()}")
    return outliers

outliers = detect_outliers_zscore(df, 'price', threshold=3)
```

**Modified Z-Score (robust to outliers):**
```python
def detect_outliers_modified_zscore(df, column, threshold=3.5):
    median = df[column].median()
    mad = np.median(np.abs(df[column] - median))

    modified_z_scores = 0.6745 * (df[column] - median) / mad
    outliers = np.abs(modified_z_scores) > threshold

    print(f"Outliers in {column}: {outliers.sum()}")
    return outliers
```

### Machine Learning Methods

**Isolation Forest:**
```python
from sklearn.ensemble import IsolationForest

# Train isolation forest
clf = IsolationForest(contamination=0.1, random_state=42)
outlier_labels = clf.fit_predict(df[['feature1', 'feature2']])

# -1 indicates outlier, 1 indicates inlier
outliers = outlier_labels == -1
print(f"Outliers detected: {outliers.sum()}")

df['is_outlier'] = outliers
```

**Local Outlier Factor (LOF):**
```python
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
outlier_labels = lof.fit_predict(df[['feature1', 'feature2']])

outliers = outlier_labels == -1
print(f"Outliers detected: {outliers.sum()}")
```

### Treatment Options

**Option 1: Remove outliers**
```python
# Remove using IQR
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1

df_clean = df[
    (df['price'] >= Q1 - 1.5 * IQR) &
    (df['price'] <= Q3 + 1.5 * IQR)
]
```

**Option 2: Cap outliers (Winsorization)**
```python
# Cap at 1st and 99th percentile
lower = df['price'].quantile(0.01)
upper = df['price'].quantile(0.99)

df['price_capped'] = df['price'].clip(lower=lower, upper=upper)

# Or use IQR bounds
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df['price_capped'] = df['price'].clip(lower=lower, upper=upper)
```

**Option 3: Transform data**
```python
# Log transformation (reduces impact of outliers)
df['price_log'] = np.log1p(df['price'])  # log(1 + x)

# Square root transformation
df['price_sqrt'] = np.sqrt(df['price'])

# Box-Cox transformation
from scipy.stats import boxcox
df['price_boxcox'], lambda_param = boxcox(df['price'] + 1)  # +1 if data has zeros
```

**Option 4: Keep but flag**
```python
# Create indicator variable
outliers = detect_outliers_iqr(df, 'price')
df['is_price_outlier'] = outliers.astype(int)

# Model can learn from this information
```

---

## Data Type Conversions

### Convert Data Types

```python
# String to numeric
df['age'] = pd.to_numeric(df['age'], errors='coerce')  # Invalid values become NaN

# String to datetime
df['date'] = pd.to_datetime(df['date'])
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')  # Specify format for speed

# String to categorical (saves memory)
df['category'] = df['category'].astype('category')

# Numeric to categorical
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 100])

# Downcast to save memory
df['age'] = pd.to_numeric(df['age'], downcast='integer')
df['price'] = pd.to_numeric(df['price'], downcast='float')
```

### Handle String Issues

```python
# Strip whitespace
df['name'] = df['name'].str.strip()

# Lowercase
df['email'] = df['email'].str.lower()

# Remove special characters
df['phone'] = df['phone'].str.replace(r'[^0-9]', '', regex=True)

# Extract numbers from string
df['price'] = df['price_text'].str.extract(r'(\d+\.?\d*)', expand=False).astype(float)

# Fix encoding issues
df['text'] = df['text'].str.encode('ascii', errors='ignore').str.decode('ascii')
```

### Boolean Conversions

```python
# String to boolean
df['is_active'] = df['is_active'].map({'Yes': True, 'No': False})

# Numeric to boolean
df['has_children'] = (df['num_children'] > 0)

# Multiple conditions
df['is_senior_citizen'] = (df['age'] >= 65) & (df['country'] == 'USA')
```

---

## Encoding Categorical Variables

### One-Hot Encoding

```python
# Pandas get_dummies
df_encoded = pd.get_dummies(
    df,
    columns=['color', 'size'],
    prefix=['color', 'size'],
    drop_first=True  # Avoid multicollinearity
)

# scikit-learn OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded = encoder.fit_transform(df[['color', 'size']])

# Get feature names
feature_names = encoder.get_feature_names_out(['color', 'size'])
df_encoded = pd.DataFrame(encoded, columns=feature_names)
```

### Label Encoding

```python
from sklearn.preprocessing import LabelEncoder

# For ordinal variables
encoder = LabelEncoder()
df['size_encoded'] = encoder.fit_transform(df['size'])
# S:0, M:1, L:2, XL:3

# Manual mapping (if order is important)
size_mapping = {'S': 0, 'M': 1, 'L': 2, 'XL': 3}
df['size_encoded'] = df['size'].map(size_mapping)
```

### Ordinal Encoding

```python
from sklearn.preprocessing import OrdinalEncoder

# Specify order
encoder = OrdinalEncoder(
    categories=[['Low', 'Medium', 'High']],
    handle_unknown='use_encoded_value',
    unknown_value=-1
)

df['priority_encoded'] = encoder.fit_transform(df[['priority']])
```

### Target Encoding

```python
# Mean target encoding
def target_encode(train_df, val_df, cat_col, target_col, smoothing=10):
    global_mean = train_df[target_col].mean()

    # Calculate encoding on training data
    agg = train_df.groupby(cat_col)[target_col].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']

    # Smoothing (shrinkage towards global mean)
    smooth = (counts * means + smoothing * global_mean) / (counts + smoothing)

    # Apply to both train and validation
    train_encoded = train_df[cat_col].map(smooth)
    val_encoded = val_df[cat_col].map(smooth).fillna(global_mean)

    return train_encoded, val_encoded

# Use with cross-validation to avoid leakage
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
df['city_encoded'] = 0

for train_idx, val_idx in kf.split(df):
    train_enc, val_enc = target_encode(
        df.iloc[train_idx],
        df.iloc[val_idx],
        'city',
        'target'
    )
    df.loc[val_idx, 'city_encoded'] = val_enc
```

**Critical warnings about target encoding:**

1. **Data leakage risk is VERY HIGH:**
   - NEVER use entire dataset to calculate target encoding
   - MUST use cross-validation or train-only encoding
   - Test set should be encoded using train set statistics only

2. **Why smoothing (shrinkage) is essential:**
   - Formula: smooth = (n × mean + m × global_mean) / (n + m)
   - Where n = category count, m = smoothing parameter
   - Without smoothing: Rare categories with 1 sample get encoded as 0 or 1 (overfitting)
   - With smoothing: Rare categories shrink towards global mean (regularization)
   - Example: Category with 1 positive sample → encoded as 1.0 without smoothing vs 0.6 with smoothing

3. **When target encoding works well:**
   - High cardinality categorical variables (100+ categories)
   - Tree-based models (RandomForest, XGBoost) - they capture non-linear relationships
   - Categories with sufficient samples (>20 per category)

4. **When to avoid target encoding:**
   - Linear models (high overfitting risk, prefer one-hot encoding)
   - Categories with very few samples (<10)
   - When interpretability is critical (creates hard-to-explain features)

---

## Feature Scaling and Normalization

### When to Scale

**Need scaling:**
- Linear Regression
- Logistic Regression
- Neural Networks
- SVM
- K-Nearest Neighbors
- PCA

**Don't need scaling:**
- Tree-based models (Decision Trees, Random Forest, XGBoost)

**Why certain algorithms need scaling:**

1. **Gradient descent-based algorithms (Linear/Logistic Regression, Neural Networks):**
   - Features with larger scales dominate gradient updates
   - Example: Feature A (range 0-1), Feature B (range 0-10000)
   - Gradient ∂L/∂weight_B will be ~10000× larger than ∂L/∂weight_A
   - Result: Optimizer takes tiny steps for B, giant steps for A → slow convergence or divergence
   - Scaling makes optimization landscape more spherical, enabling faster convergence

2. **Distance-based algorithms (KNN, SVM with RBF kernel, K-Means):**
   - Distance calculation: √[(x1-y1)² + (x2-y2)² + ...]
   - Feature with larger range dominates distance metric
   - Example: age (0-100) and income (0-100000) → income will dominate Euclidean distance
   - Result: Model ignores features with smaller ranges
   - Scaling ensures all features contribute equally to distance

3. **Regularization (Lasso, Ridge):**
   - Penalty term: λ × Σ(weights²)
   - Features with larger scales get smaller weights to maintain same prediction
   - Example: Feature A (0-1) gets weight=100, Feature B (0-100) gets weight=1
   - Regularization penalizes weight=100 much more than weight=1
   - Result: Model biased towards features with larger scales
   - Scaling ensures fair regularization penalty across all features

**Why tree-based models don't need scaling:**
- Split decisions based on feature thresholds, not feature magnitudes
- Example: "age < 30" works regardless of whether age is 0-100 or 0-1
- Trees are invariant to monotonic transformations of features
- Each feature evaluated independently in split decisions

### Mathematical Theory of Feature Scaling

**Convergence Analysis for Gradient Descent:**
```
Consider quadratic loss: L(w) = (1/2) w^T H w - b^T w

where H is Hessian (second derivative matrix)

Without scaling:
- Condition number κ = λ_max / λ_min can be very large
- Convergence rate: O((1 - 1/κ)^k) → slow when κ >> 1

With scaling (standardization):
- Features have similar scales → H closer to identity matrix
- Condition number κ ≈ 1 → faster convergence
- Convergence rate: O(exp(-k)) → exponentially fast

Quantitative Example:
Feature 1: range [0, 1], variance = 0.1
Feature 2: range [0, 1000], variance = 10000

Hessian eigenvalues: λ₁ ≈ 0.1, λ₂ ≈ 10000
Condition number: κ ≈ 10000 / 0.1 = 100,000

Gradient descent iterations to converge:
- Without scaling: O(100,000) iterations
- With scaling: O(10-100) iterations (1000× speedup!)

Formal Theorem:
For L-smooth, μ-strongly convex loss with κ = L/μ:
- GD converges as: ||w_k - w*|| ≤ (1 - μ/L)^k ||w_0 - w*||
- After scaling: μ ≈ L → κ ≈ 1 → (1 - 1/1)^k → very fast!
```

**Distance Metric Distortion:**
```
Euclidean distance without scaling:
d(x, y) = √[Σ(xᵢ - yᵢ)²]

Problem: Features with large variance dominate distance

Example:
Feature 1 (age): x₁ = 25, y₁ = 30, diff = 5
Feature 2 (income): x₂ = 50000, y₂ = 55000, diff = 5000

d(x,y) = √(5² + 5000²) = √25,000,025 ≈ 5000

Distance contribution:
- Age: 5²/25,000,025 ≈ 0.0001% (effectively ignored!)
- Income: 5000²/25,000,025 ≈ 99.9999% (dominates)

After standardization (assuming σ_age=10, σ_income=20000):
- Age_scaled: (30-25)/10 = 0.5
- Income_scaled: (55000-50000)/20000 = 0.25

d_scaled = √(0.5² + 0.25²) = √0.3125 ≈ 0.56

Distance contribution:
- Age: 0.5²/0.3125 ≈ 80%
- Income: 0.25²/0.3125 ≈ 20%
Both features now contribute meaningfully!
```

**Regularization Bias:**
```
L2 regularization: L(w) = MSE(w) + λ||w||²

Without scaling:
- Large-scale feature f₁ ∈ [0, 1000] gets small weight w₁ ≈ 0.001
- Small-scale feature f₂ ∈ [0, 1] gets large weight w₂ ≈ 1
- Regularization penalty: λ(0.001² + 1²) ≈ λ(1) (dominated by w₂)
- Model unfairly penalizes small-scale features!

Mathematical bias:
Optimal weight scales as w_i* ∝ 1/scale(fᵢ)
→ Features with larger scales get artificially reduced importance

After scaling:
- All features have comparable scale → all weights comparable
- Regularization penalty distributed fairly across features
- Model can learn true feature importance without scale bias
```

### StandardScaler (Z-score normalization)

**Formula:** x_scaled = (x - μ) / σ

**Mathematical Properties:**
```
Transformation: z = (x - μ) / σ

Properties:
1. E[z] = E[(x - μ)/σ] = (E[x] - μ)/σ = 0  (zero mean)
2. Var(z) = Var((x - μ)/σ) = Var(x)/σ² = σ²/σ² = 1  (unit variance)
3. Shape preservation: Does not change distribution shape
4. Linearity: Affine transformation preserves linear relationships
5. Outlier sensitivity: μ and σ affected by outliers

Statistical Interpretation:
z represents number of standard deviations from mean:
- z = 0: at mean
- z = ±1: one std dev from mean (68% of data within [-1,1] for normal)
- z = ±2: two std devs from mean (95% within [-2,2] for normal)
- z = ±3: three std devs from mean (99.7% within [-3,3] for normal)

For normal distribution: z ~ N(0, 1) (standard normal)
```

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit on training data only!
scaler.fit(X_train)

# Transform both train and test
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Result: mean=0, std=1
```

**When to use StandardScaler:**
- Default choice for most ML algorithms
- Features follow approximately normal distribution
- Algorithm assumes features are centered around 0 (e.g., PCA)
- Want to preserve shape of original distribution while standardizing

**Important:** StandardScaler is sensitive to outliers because it uses mean and std.

### MinMaxScaler

**Formula:** x_scaled = (x - x_min) / (x_max - x_min)

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Result: values between 0 and 1
```

**When to use MinMaxScaler:**
- Neural networks (especially with sigmoid/tanh activation)
- Image pixel values (already in 0-255 range)
- Features with hard bounds (e.g., age, percentages)
- When you need specific range (e.g., 0-1 for algorithms sensitive to scale)

**Important:**
- Very sensitive to outliers (single outlier can compress entire distribution)
- Test set values can fall outside [0,1] if test has values beyond train's min/max

### RobustScaler (for outliers)

**Formula:** x_scaled = (x - median) / IQR

Where IQR (Interquartile Range) = Q3 - Q1 (75th percentile - 25th percentile)

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Uses median and IQR instead of mean and std
```

**When to use RobustScaler:**
- Data contains outliers that you want to keep
- Distribution is not Gaussian
- Heavy-tailed distributions
- StandardScaler gives poor results due to extreme values

**Why RobustScaler is robust:**
- Median is unaffected by extreme values (unlike mean)
- IQR uses middle 50% of data, ignoring extreme 25% on each end
- Example: [1, 2, 3, 4, 1000] → median=3, IQR=2 (outlier 1000 doesn't affect scaling)
- Compare StandardScaler: mean=202, std=397 (completely distorted by outlier!)

**Scaler selection guide:**
```
Do you have outliers?
├─ Yes → RobustScaler
└─ No
   ├─ Need specific range (0-1)? → MinMaxScaler
   └─ Otherwise → StandardScaler (default)
```

### MaxAbsScaler

```python
from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()

scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Scales by maximum absolute value
# Good for sparse data
```

### Normalization (L2)

```python
from sklearn.preprocessing import Normalizer

normalizer = Normalizer(norm='l2')  # or 'l1', 'max'

X_normalized = normalizer.fit_transform(X)

# Scales individual samples to unit norm
```

---

## Handling Imbalanced Data

### Detection

```python
# Check class distribution
print(df['target'].value_counts())
print(df['target'].value_counts(normalize=True))

# Visualize
df['target'].value_counts().plot(kind='bar')
plt.title('Class Distribution')
plt.show()

# Calculate imbalance ratio
majority_class = df['target'].value_counts().max()
minority_class = df['target'].value_counts().min()
imbalance_ratio = majority_class / minority_class
print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
```

### Resampling Methods

**Random Oversampling:**
```python
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

print(f"Original: {len(y)}")
print(f"Resampled: {len(y_resampled)}")
print(pd.Series(y_resampled).value_counts())
```

**SMOTE (Synthetic Minority Over-sampling):**
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Creates synthetic examples by interpolating between existing minority samples
```

**Random Undersampling:**
```python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

# Randomly removes majority class samples
```

**Combination (SMOTE + Tomek Links):**
```python
from imblearn.combine import SMOTETomek

smt = SMOTETomek(random_state=42)
X_resampled, y_resampled = smt.fit_resample(X, y)

# Oversample minority, then clean up borderline samples
```

### Class Weights

```python
from sklearn.utils.class_weight import compute_class_weight

# Calculate class weights
classes = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))

# Use in model
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(class_weight='balanced')
# or
model = RandomForestClassifier(class_weight=class_weight_dict)
```

---

## Train-Test Split

### Basic Split

```python
from sklearn.model_selection import train_test_split

# Standard split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# With stratification (maintains class distribution)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,  # Important for imbalanced data!
    random_state=42
)
```

### Train-Validation-Test Split

```python
# First split: train+val vs test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Second split: train vs val
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.25,  # 0.25 of 0.8 = 0.2 of total
    random_state=42
)

# Result: 60% train, 20% val, 20% test
```

### Time Series Split

```python
# Don't shuffle time series data!
# Use temporal split instead

df = df.sort_values('date')

train_size = int(len(df) * 0.8)
train_df = df[:train_size]
test_df = df[train_size:]

# Or use TimeSeriesSplit for cross-validation
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
```

### Group Split (prevent leakage)

```python
from sklearn.model_selection import GroupShuffleSplit

# Example: Medical data with multiple measurements per patient
# Want to ensure same patient isn't in both train and test

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_idx, test_idx in gss.split(X, y, groups=patient_ids):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
```

---

## Building Preprocessing Pipelines

### Simple Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Create pipeline
preprocessing_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Fit on training data
preprocessing_pipeline.fit(X_train)

# Transform both train and test
X_train_processed = preprocessing_pipeline.transform(X_train)
X_test_processed = preprocessing_pipeline.transform(X_test)
```

### Column Transformer (different preprocessing for different columns)

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Define column types
numerical_cols = ['age', 'income', 'credit_score']
categorical_cols = ['city', 'occupation', 'education']

# Create transformers for each column type
numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

# Combine transformers
preprocessor = ColumnTransformer([
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Use in pipeline with model
from sklearn.ensemble import RandomForestClassifier

full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Train
full_pipeline.fit(X_train, y_train)

# Predict
predictions = full_pipeline.predict(X_test)
```

### Custom Transformer

```python
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, quantile_range=(0.01, 0.99)):
        self.quantile_range = quantile_range
        self.lower_bounds = None
        self.upper_bounds = None

    def fit(self, X, y=None):
        self.lower_bounds = np.percentile(X, self.quantile_range[0] * 100, axis=0)
        self.upper_bounds = np.percentile(X, self.quantile_range[1] * 100, axis=0)
        return self

    def transform(self, X):
        X_clipped = np.clip(X, self.lower_bounds, self.upper_bounds)
        return X_clipped

# Use in pipeline
pipeline = Pipeline([
    ('clipper', OutlierClipper()),
    ('scaler', StandardScaler())
])
```

### Save and Load Pipelines

```python
import joblib

# Save pipeline
joblib.dump(preprocessing_pipeline, 'preprocessing_pipeline.pkl')

# Load pipeline
loaded_pipeline = joblib.load('preprocessing_pipeline.pkl')

# Use loaded pipeline
X_new_processed = loaded_pipeline.transform(X_new)
```

---

## Common Pitfalls

### Pitfall 1: Fitting on Test Data

```python
# ❌ WRONG: Leakage!
scaler = StandardScaler()
scaler.fit(pd.concat([X_train, X_test]))  # DON'T DO THIS!
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ CORRECT: Fit only on train
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Pitfall 2: Splitting After Preprocessing

```python
# ❌ WRONG: Scaling before split
X_scaled = StandardScaler().fit_transform(X)
X_train, X_test = train_test_split(X_scaled)

# ✅ CORRECT: Split first, then scale
X_train, X_test = train_test_split(X)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Pitfall 3: Not Handling Categorical Variables

```python
# ❌ WRONG: Model sees 'Red', 'Blue' as strings
model.fit(df[['color', 'size']], y)  # Error!

# ✅ CORRECT: Encode first
df_encoded = pd.get_dummies(df, columns=['color', 'size'])
model.fit(df_encoded, y)
```

### Pitfall 4: Ignoring Data Types

```python
# Check data types
print(df.dtypes)

# ❌ String stored as object
# zip_code    object  (should be category or string, NOT int!)
# age         object  (should be int!)

# ✅ CORRECT: Fix data types
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['zip_code'] = df['zip_code'].astype('category')
```

### Pitfall 5: Not Validating Preprocessing Steps

```python
# ✅ Always validate after preprocessing
print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")
print(f"No missing values: {not X_train.isnull().any().any()}")
print(f"No inf values: {not np.isinf(X_train).any().any()}")
print(f"Feature names match: {list(X_train.columns) == list(X_test.columns)}")
```

---

## Preprocessing Checklist

**Before modeling, ensure:**

- [ ] Data loaded correctly (check shape, dtypes)
- [ ] Missing values handled (imputed or removed)
- [ ] Duplicates removed or aggregated
- [ ] Outliers detected and treated
- [ ] Data types correct (numeric, categorical, datetime)
- [ ] Categorical variables encoded
- [ ] Numerical features scaled (if needed)
- [ ] Train/test split done correctly (split BEFORE preprocessing)
- [ ] No data leakage (fit only on train)
- [ ] Class imbalance addressed (if applicable)
- [ ] No NaN or inf values remain
- [ ] Pipeline saved for production use

---

## Complete Example

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv('data.csv')

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Train-test split (FIRST!)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Define column types
numerical_cols = ['age', 'income', 'credit_score']
categorical_cols = ['city', 'occupation']

# Create preprocessing pipelines
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

# Combine
preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

# Create full pipeline with model
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train
full_pipeline.fit(X_train, y_train)

# Evaluate
train_score = full_pipeline.score(X_train, y_train)
test_score = full_pipeline.score(X_test, y_test)

print(f"Train accuracy: {train_score:.3f}")
print(f"Test accuracy: {test_score:.3f}")

# Save pipeline
import joblib
joblib.dump(full_pipeline, 'model_pipeline.pkl')
```

---

## Key Takeaways

**Golden rules:**
1. **Always split before preprocessing** (fit on train only)
2. **Use pipelines** (ensures consistent preprocessing)
3. **Validate each step** (check for NaN, inf, shape mismatches)
4. **Document decisions** (why you chose imputation method X)
5. **Save pipelines** (for production deployment)

**Common mistakes to avoid:**
- Fitting scalers on test data
- Dropping too many rows/columns
- Not handling categorical variables
- Ignoring class imbalance
- Not using pipelines

**Remember:** Good preprocessing is as important as model selection. Spend time getting it right!

---

**Related guides:**
- [Feature Engineering Guide](./FEATURE_ENGINEERING_GUIDE.md) - Create better features
- [Common ML Mistakes](../../resources/COMMON_ML_MISTAKES.md) - Avoid data mistakes
- [Data Processing Notebook](../../interactive_demos/04_data_processing.ipynb) - Hands-on practice

---

*Last Updated: October 30, 2025*
