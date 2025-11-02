# 🧹 Data Preprocessing: Complete Guide

**"Garbage in, garbage out."** Data preprocessing is the foundation of successful ML projects. This guide covers everything from data loading to final preparation for modeling.

---

## 📋 Table of Contents

- [Why Preprocessing Matters](#why-preprocessing-matters)
- [Data Loading and Inspection](#data-loading-and-inspection)
- [Handling Missing Values](#handling-missing-values)
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

    # Smoothing
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

### StandardScaler (Z-score normalization)

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

### MinMaxScaler

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Result: values between 0 and 1
```

### RobustScaler (for outliers)

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Uses median and IQR instead of mean and std
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
