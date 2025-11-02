# 🔨 Feature Engineering Guide: The Art of Creating Better Features

**"Coming up with features is difficult, time-consuming, requires expert knowledge. Applied machine learning is basically feature engineering."** - Andrew Ng

This guide shows you how to engineer features that significantly improve model performance.

---

## 📋 Table of Contents

- [Why Feature Engineering Matters](#why-feature-engineering-matters)
- [Numerical Features](#numerical-features)
- [Categorical Features](#categorical-features)
- [DateTime Features](#datetime-features)
- [Text Features](#text-features)
- [Geospatial Features](#geospatial-features)
- [Feature Interactions](#feature-interactions)
- [Domain-Specific Patterns](#domain-specific-patterns)
- [Feature Selection](#feature-selection)
- [Validation and Testing](#validation-and-testing)

---

## Why Feature Engineering Matters

### The 80/20 Rule of ML

**80% of model improvement comes from better features, not fancier algorithms.**

**Example: Predicting house prices**

```python
# ❌ Poor features
features = ['bedrooms', 'bathrooms']
model = LinearRegression()
model.fit(features, prices)
# R² = 0.45

# ✅ Engineered features
features = [
    'bedrooms', 'bathrooms', 'sqft',
    'price_per_sqft',  # Derived
    'bedroom_ratio',   # bedrooms / total_rooms
    'age',             # current_year - year_built
    'has_garage',      # Binary indicator
    'neighborhood_avg_price'  # Aggregate feature
]
model = LinearRegression()
model.fit(features, prices)
# R² = 0.82  # 82% better!
```

### When Feature Engineering Helps Most

**High impact:**
- Structured/tabular data
- Small-medium datasets (<1M rows)
- Domain knowledge available
- Interpretability matters

**Lower impact (but still useful):**
- Deep learning on images/text/audio
- Very large datasets (>10M rows)
- Black-box model acceptable

---

## Numerical Features

### 1. Binning / Discretization

Convert continuous values into categorical bins.

**When to use:** Non-linear relationships, interpretability

```python
import pandas as pd
import numpy as np

# Example: Age to age groups
df['age'] = [22, 35, 48, 61, 73, 19, 42]

# Equal-width binning
df['age_bin'] = pd.cut(
    df['age'],
    bins=4,
    labels=['Young', 'Adult', 'Middle', 'Senior']
)

# Custom bins (domain knowledge)
df['age_group'] = pd.cut(
    df['age'],
    bins=[0, 18, 35, 50, 100],
    labels=['Minor', 'Young Adult', 'Adult', 'Senior']
)

# Quantile binning (equal samples per bin)
df['age_quartile'] = pd.qcut(df['age'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
```

**Example: Income binning**
```python
# Income brackets for tax purposes
df['income_bracket'] = pd.cut(
    df['income'],
    bins=[0, 50000, 100000, 200000, np.inf],
    labels=['Low', 'Middle', 'Upper-Middle', 'High']
)
```

### 2. Mathematical Transformations

**Log transformation** (for skewed distributions):
```python
# Income, prices often follow log-normal distribution
df['log_income'] = np.log1p(df['income'])  # log(1 + x) handles zeros

# Compare distributions
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df['income'], bins=50)
axes[0].set_title('Original (Skewed)')
axes[1].hist(df['log_income'], bins=50)
axes[1].set_title('Log-Transformed (Normal)')
```

**Power transformations** (Box-Cox, Yeo-Johnson):
```python
from sklearn.preprocessing import PowerTransformer

# Box-Cox (only for positive values)
pt = PowerTransformer(method='box-cox')
df['income_boxcox'] = pt.fit_transform(df[['income']])

# Yeo-Johnson (works with negative values too)
pt = PowerTransformer(method='yeo-johnson')
df['feature_transformed'] = pt.fit_transform(df[['feature']])
```

**Square root, square, reciprocal**:
```python
# Square root for count data
df['sqrt_num_purchases'] = np.sqrt(df['num_purchases'])

# Square for emphasis
df['sqft_squared'] = df['sqft'] ** 2

# Reciprocal for rates
df['days_per_purchase'] = 1 / (df['num_purchases'] + 1)  # Avoid division by zero
```

### 3. Scaling and Normalization

**StandardScaler** (zero mean, unit variance):
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df['income_scaled'] = scaler.fit_transform(df[['income']])

# Result: mean=0, std=1
# Good for: algorithms sensitive to scale (SVM, neural networks, linear regression)
```

**MinMaxScaler** (scale to range [0,1]):
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df['age_normalized'] = scaler.fit_transform(df[['age']])

# Result: values between 0 and 1
# Good for: neural networks, distance-based algorithms
```

**RobustScaler** (robust to outliers):
```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
df['price_robust'] = scaler.fit_transform(df[['price']])

# Uses median and IQR instead of mean and std
# Good for: data with outliers
```

### 4. Handling Outliers

**Detection**:
```python
# IQR method
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = (df['price'] < lower_bound) | (df['price'] > upper_bound)
print(f"Found {outliers.sum()} outliers")
```

**Treatment**:
```python
# Option 1: Remove outliers
df_clean = df[~outliers]

# Option 2: Cap (winsorize)
df['price_capped'] = df['price'].clip(lower=lower_bound, upper=upper_bound)

# Option 3: Transform (log reduces impact)
df['price_log'] = np.log1p(df['price'])

# Option 4: Create indicator
df['is_outlier'] = outliers.astype(int)
```

### 5. Ratios and Differences

**Ratios** (often more informative than absolutes):
```python
# House price prediction
df['price_per_sqft'] = df['price'] / df['sqft']
df['bedroom_ratio'] = df['bedrooms'] / (df['bedrooms'] + df['bathrooms'])
df['living_to_lot_ratio'] = df['sqft_living'] / df['sqft_lot']

# E-commerce
df['conversion_rate'] = df['purchases'] / df['visits']
df['avg_order_value'] = df['revenue'] / df['num_orders']

# Finance
df['debt_to_income'] = df['total_debt'] / df['annual_income']
df['current_ratio'] = df['current_assets'] / df['current_liabilities']
```

**Differences**:
```python
# Time between events
df['days_since_last_purchase'] = (df['current_date'] - df['last_purchase_date']).dt.days

# Comparing to baseline
df['income_vs_median'] = df['income'] - df['median_income']
df['price_vs_avg'] = df['price'] / df['neighborhood_avg_price'] - 1  # % difference
```

---

## Categorical Features

### 1. Ordinal Encoding

For categories with natural order.

```python
from sklearn.preprocessing import OrdinalEncoder

# Education level (clear order)
education_order = ['High School', 'Bachelor', 'Master', 'PhD']

df['education_encoded'] = df['education'].map({
    'High School': 0,
    'Bachelor': 1,
    'Master': 2,
    'PhD': 3
})

# Or use OrdinalEncoder
encoder = OrdinalEncoder(categories=[education_order])
df['education_encoded'] = encoder.fit_transform(df[['education']])
```

### 2. One-Hot Encoding

For categories without order.

```python
from sklearn.preprocessing import OneHotEncoder

# Color (no natural order)
colors = ['red', 'blue', 'green', 'red', 'blue']

# Pandas method (simplest)
df_encoded = pd.get_dummies(df, columns=['color'], prefix='color')
# Creates: color_red, color_blue, color_green

# scikit-learn method (better for production)
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
color_encoded = encoder.fit_transform(df[['color']])

# Reduce dimensionality for high-cardinality
encoder = OneHotEncoder(
    max_categories=10,  # Keep only top 10 categories
    handle_unknown='infrequent_if_exist'  # Group rare categories
)
```

**Caution with high cardinality:**
```python
# If 'city' has 1000 unique values → 1000 features!
# Solutions:
# 1. Group rare categories
df['city_grouped'] = df['city'].apply(
    lambda x: x if df['city'].value_counts()[x] > 100 else 'Other'
)

# 2. Use target encoding (see below)
# 3. Use feature hashing
```

### 3. Target Encoding

Encode categories by their relationship to target.

```python
# Mean target encoding
category_means = df.groupby('city')['price'].mean()
df['city_avg_price'] = df['city'].map(category_means)

# With regularization (avoid overfitting)
def target_encode(df, cat_col, target_col, smoothing=10):
    """
    Target encoding with smoothing.

    smoothing: Higher values = more regularization toward global mean
    """
    global_mean = df[target_col].mean()
    counts = df.groupby(cat_col)[target_col].count()
    means = df.groupby(cat_col)[target_col].mean()

    # Smoothed encoding
    smooth_encoding = (counts * means + smoothing * global_mean) / (counts + smoothing)

    return df[cat_col].map(smooth_encoding)

df['city_encoded'] = target_encode(df, 'city', 'price', smoothing=10)

# IMPORTANT: Use cross-validation to avoid leakage!
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
df['city_target_encoded'] = 0

for train_idx, val_idx in kf.split(df):
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    # Compute encoding on train fold only
    encoding = target_encode(train_df, 'city', 'price')

    # Apply to validation fold
    df.loc[val_idx, 'city_target_encoded'] = val_df['city'].map(encoding)
```

### 4. Frequency Encoding

Encode by how often category appears.

```python
# Count encoding
category_counts = df['brand'].value_counts()
df['brand_count'] = df['brand'].map(category_counts)

# Frequency encoding (normalized)
category_freq = df['brand'].value_counts(normalize=True)
df['brand_frequency'] = df['brand'].map(category_freq)
```

### 5. Hash Encoding

For very high cardinality.

```python
from sklearn.feature_extraction import FeatureHasher

# Hash 1000 cities into 20 features
hasher = FeatureHasher(n_features=20, input_type='string')
hashed = hasher.transform(df[['city']].values.flatten())

# Add to dataframe
hashed_df = pd.DataFrame(
    hashed.toarray(),
    columns=[f'city_hash_{i}' for i in range(20)]
)
df = pd.concat([df, hashed_df], axis=1)
```

---

## DateTime Features

DateTime features are goldmines for prediction tasks.

### 1. Extract Components

```python
df['date'] = pd.to_datetime(df['date'])

# Basic components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
df['day_of_year'] = df['date'].dt.dayofyear
df['week_of_year'] = df['date'].dt.isocalendar().week
df['quarter'] = df['date'].dt.quarter

# Time components
df['hour'] = df['date'].dt.hour
df['minute'] = df['date'].dt.minute

# Boolean indicators
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)
```

### 2. Cyclical Encoding

Months, hours, days are cyclical (December is close to January).

```python
# Month as cyclical features
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Hour of day
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Day of week
df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
```

### 3. Time Since / Until Event

```python
# Days since last purchase
df['days_since_last_purchase'] = (df['current_date'] - df['last_purchase_date']).dt.days

# Days until next holiday
next_holiday = pd.Timestamp('2025-12-25')
df['days_until_christmas'] = (next_holiday - df['date']).dt.days

# Account age
df['account_age_days'] = (df['current_date'] - df['signup_date']).dt.days
```

### 4. Aggregations Over Time

```python
# Last 7 days
df['purchases_last_7d'] = df.groupby('user_id')['purchase_date'].transform(
    lambda x: ((df['current_date'] - x).dt.days <= 7).sum()
)

# Rolling windows
df = df.sort_values('date')
df['sales_7d_avg'] = df['sales'].rolling(window=7).mean()
df['sales_7d_max'] = df['sales'].rolling(window=7).max()
df['sales_7d_std'] = df['sales'].rolling(window=7).std()

# Lag features
df['sales_lag_1d'] = df['sales'].shift(1)
df['sales_lag_7d'] = df['sales'].shift(7)

# Percent change
df['sales_pct_change'] = df['sales'].pct_change()
```

### 5. Special Dates

```python
import holidays

# US holidays
us_holidays = holidays.US()

df['is_holiday'] = df['date'].apply(lambda x: x in us_holidays).astype(int)
df['is_black_friday'] = ((df['month'] == 11) &
                         (df['day_of_week'] == 4) &
                         (df['day'] >= 22) &
                         (df['day'] <= 28)).astype(int)
df['days_to_holiday'] = df['date'].apply(
    lambda x: min((holiday - x).days for holiday in us_holidays
                  if (holiday - x).days >= 0)
    if any((holiday - x).days >= 0 for holiday in us_holidays)
    else 365
)
```

---

## Text Features

### 1. Basic Text Features

```python
# Length-based features
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()
df['avg_word_length'] = df['text'].apply(lambda x: np.mean([len(w) for w in x.split()]))

# Character counts
df['num_uppercase'] = df['text'].str.count(r'[A-Z]')
df['num_digits'] = df['text'].str.count(r'\d')
df['num_special'] = df['text'].str.count(r'[^A-Za-z0-9\s]')

# Boolean indicators
df['has_url'] = df['text'].str.contains(r'http', case=False).astype(int)
df['has_email'] = df['text'].str.contains(r'@', case=False).astype(int)
df['has_phone'] = df['text'].str.contains(r'\d{3}-\d{3}-\d{4}').astype(int)
```

### 2. TF-IDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Basic TF-IDF
tfidf = TfidfVectorizer(
    max_features=100,  # Keep top 100 words
    min_df=5,          # Word must appear in >=5 documents
    max_df=0.8,        # Word must appear in <=80% of documents
    ngram_range=(1, 2) # Unigrams and bigrams
)

tfidf_matrix = tfidf.fit_transform(df['text'])

# Convert to dataframe
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=[f'tfidf_{word}' for word in tfidf.get_feature_names_out()]
)
```

### 3. Sentiment Analysis

```python
from textblob import TextBlob

# Polarity and subjectivity
df['sentiment_polarity'] = df['text'].apply(
    lambda x: TextBlob(x).sentiment.polarity
)
df['sentiment_subjectivity'] = df['text'].apply(
    lambda x: TextBlob(x).sentiment.subjectivity
)

# Or use VADER (better for social media)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
df['sentiment_vader'] = df['text'].apply(
    lambda x: analyzer.polarity_scores(x)['compound']
)
```

### 4. Custom Keywords

```python
# Domain-specific keywords
positive_words = ['excellent', 'great', 'love', 'best']
negative_words = ['terrible', 'worst', 'hate', 'awful']

df['num_positive'] = df['text'].apply(
    lambda x: sum(1 for word in positive_words if word in x.lower())
)
df['num_negative'] = df['text'].apply(
    lambda x: sum(1 for word in negative_words if word in x.lower())
)
df['sentiment_ratio'] = df['num_positive'] / (df['num_negative'] + 1)
```

---

## Geospatial Features

### 1. Distance Calculations

```python
from geopy.distance import geodesic

# Distance between two points
def calculate_distance(row):
    point1 = (row['lat1'], row['lon1'])
    point2 = (row['lat2'], row['lon2'])
    return geodesic(point1, point2).kilometers

df['distance_km'] = df.apply(calculate_distance, axis=1)

# Distance to important location (e.g., city center)
city_center = (37.7749, -122.4194)  # San Francisco

df['distance_to_center'] = df.apply(
    lambda row: geodesic((row['lat'], row['lon']), city_center).km,
    axis=1
)
```

### 2. Geographic Aggregations

```python
# Average price within radius
from sklearn.neighbors import BallTree
import numpy as np

coords = df[['lat', 'lon']].values
tree = BallTree(np.radians(coords), metric='haversine')

radius_km = 5
radius_rad = radius_km / 6371  # Earth radius in km

# Find neighbors within 5km
indices = tree.query_radius(np.radians(coords), r=radius_rad)

# Average price of neighbors
df['avg_price_5km'] = [
    df.iloc[idx]['price'].mean() for idx in indices
]
```

### 3. Spatial Binning

```python
# Grid-based features (geohash alternative)
def create_spatial_grid(lat, lon, grid_size=0.01):
    """Create spatial grid ID"""
    grid_lat = int(lat / grid_size)
    grid_lon = int(lon / grid_size)
    return f"{grid_lat}_{grid_lon}"

df['grid_id'] = df.apply(
    lambda row: create_spatial_grid(row['lat'], row['lon']),
    axis=1
)

# Then use as categorical feature
grid_stats = df.groupby('grid_id')['price'].agg(['mean', 'count', 'std'])
df = df.merge(grid_stats, on='grid_id', suffixes=('', '_grid'))
```

---

## Feature Interactions

Often 1 + 1 = 3 in machine learning!

### 1. Polynomial Features

```python
from sklearn.preprocessing import PolynomialFeatures

# Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[['sqft', 'bedrooms', 'bathrooms']])

# Features created:
# sqft, bedrooms, bathrooms (original)
# sqft^2, bedrooms^2, bathrooms^2 (squared)
# sqft*bedrooms, sqft*bathrooms, bedrooms*bathrooms (interactions)
```

### 2. Manual Interactions

```python
# Meaningful combinations (domain knowledge)

# House pricing
df['sqft_x_bedrooms'] = df['sqft'] * df['bedrooms']
df['total_rooms'] = df['bedrooms'] + df['bathrooms']
df['bedrooms_per_sqft'] = df['bedrooms'] / df['sqft']

# E-commerce
df['recency_x_frequency'] = df['days_since_purchase'] * df['num_purchases']
df['avg_purchase_value'] = df['total_spent'] / df['num_purchases']

# Finance
df['income_x_credit'] = df['annual_income'] * df['credit_score']
df['debt_service_ratio'] = df['monthly_debt'] / (df['monthly_income'] + 1)
```

### 3. Decision Tree-Based Interactions

```python
# Use tree to find interactions, then create features
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import _tree

def get_tree_paths(tree, feature_names):
    """Extract decision paths from tree"""
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    def recurse(node, path):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            # Left child
            recurse(tree_.children_left[node], path + [(name, '<=', threshold)])
            # Right child
            recurse(tree_.children_right[node], path + [(name, '>', threshold)])
        else:
            paths.append(path)

    recurse(0, [])
    return paths

# Train decision tree
tree = DecisionTreeRegressor(max_depth=3)
tree.fit(df[['sqft', 'bedrooms', 'age']], df['price'])

# Get important interactions
paths = get_tree_paths(tree, ['sqft', 'bedrooms', 'age'])

# Create new features based on paths
# Example: (sqft > 2000) & (bedrooms > 3)
df['large_house'] = ((df['sqft'] > 2000) & (df['bedrooms'] > 3)).astype(int)
```

---

## Domain-Specific Patterns

### E-Commerce / Marketing

```python
# RFM (Recency, Frequency, Monetary)
df['recency'] = (df['current_date'] - df['last_purchase_date']).dt.days
df['frequency'] = df['num_purchases']
df['monetary'] = df['total_spent']

# CLV (Customer Lifetime Value) proxy
df['avg_order_value'] = df['total_spent'] / df['num_purchases']
df['purchase_frequency'] = df['num_purchases'] / df['account_age_days']
df['clv_estimate'] = df['avg_order_value'] * df['purchase_frequency'] * 365

# Engagement
df['session_duration_avg'] = df['total_session_time'] / df['num_sessions']
df['pages_per_session'] = df['total_pages_viewed'] / df['num_sessions']
df['bounce_rate'] = (df['num_sessions'] == df['num_bounces']).astype(int)
```

### Finance / Credit Scoring

```python
# Credit utilization
df['credit_utilization'] = df['credit_balance'] / df['credit_limit']

# Debt-to-income ratio
df['debt_to_income'] = df['total_debt'] / df['annual_income']

# Payment history
df['late_payment_ratio'] = df['num_late_payments'] / df['num_payments']
df['months_since_delinquency'] = (df['current_date'] - df['last_delinquency']).dt.days / 30

# Account diversity
df['num_account_types'] = df[['has_mortgage', 'has_auto_loan', 'has_credit_card']].sum(axis=1)
```

### Healthcare

```python
# BMI
df['bmi'] = df['weight_kg'] / (df['height_m'] ** 2)

# Age groups
df['age_group'] = pd.cut(
    df['age'],
    bins=[0, 18, 30, 50, 65, 100],
    labels=['Pediatric', 'Young', 'Middle', 'Senior', 'Geriatric']
)

# Risk scores
df['diabetes_risk'] = (
    (df['bmi'] > 30).astype(int) * 2 +
    (df['age'] > 45).astype(int) +
    (df['family_history'] == 'Yes').astype(int)
)
```

---

## Feature Selection

After creating features, select the best ones.

### 1. Remove Low-Variance Features

```python
from sklearn.feature_selection import VarianceThreshold

# Remove features with <1% variance
selector = VarianceThreshold(threshold=0.01)
X_high_var = selector.fit_transform(X)

# See which features were removed
removed = X.columns[~selector.get_support()]
print(f"Removed {len(removed)} low-variance features")
```

### 2. Correlation-Based Selection

```python
# Remove highly correlated features
correlation_matrix = df.corr().abs()

# Select upper triangle
upper = correlation_matrix.where(
    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
)

# Find features with correlation > 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
df_reduced = df.drop(columns=to_drop)
print(f"Removed {len(to_drop)} highly correlated features")
```

### 3. Univariate Feature Selection

```python
from sklearn.feature_selection import SelectKBest, f_classif

# Select top K features
selector = SelectKBest(f_classif, k=20)
X_best = selector.fit_transform(X, y)

# Get selected feature names
selected_features = X.columns[selector.get_support()]
print(f"Selected features: {list(selected_features)}")
```

### 4. Recursive Feature Elimination (RFE)

```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# Use Random Forest to rank features
estimator = RandomForestClassifier(n_estimators=100, random_state=42)
selector = RFE(estimator, n_features_to_select=20, step=1)
selector.fit(X, y)

# Get selected features
selected = X.columns[selector.support_]
print(f"Selected: {list(selected)}")
```

### 5. Feature Importance from Trees

```python
from sklearn.ensemble import RandomForestRegressor

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Get feature importance
importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Keep top features
top_features = importances.head(20)['feature'].tolist()
X_selected = X[top_features]
```

---

## Validation and Testing

### Test Feature Engineering Pipeline

```python
# Create reusable feature engineering pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Define transformations
def create_features(df):
    """Feature engineering function"""
    df = df.copy()

    # Numerical features
    df['price_per_sqft'] = df['price'] / df['sqft']
    df['bedroom_ratio'] = df['bedrooms'] / (df['bedrooms'] + df['bathrooms'])

    # DateTime features
    df['age'] = 2025 - df['year_built']
    df['is_new'] = (df['age'] < 5).astype(int)

    # Categorical encoding
    df = pd.get_dummies(df, columns=['neighborhood'], drop_first=True)

    return df

# Test on train/test split
X_train_fe = create_features(X_train)
X_test_fe = create_features(X_test)

# Make sure same features in both
assert list(X_train_fe.columns) == list(X_test_fe.columns), "Feature mismatch!"

# Train model
model = RandomForestRegressor()
model.fit(X_train_fe, y_train)
score = model.score(X_test_fe, y_test)
print(f"R² with feature engineering: {score:.3f}")
```

### A/B Test New Features

```python
# Baseline: without new feature
X_baseline = df[['sqft', 'bedrooms', 'bathrooms']]
model_baseline = RandomForestRegressor(random_state=42)
scores_baseline = cross_val_score(model_baseline, X_baseline, y, cv=5)

# With new feature
df['price_per_sqft'] = df['price'] / df['sqft']
X_new = df[['sqft', 'bedrooms', 'bathrooms', 'price_per_sqft']]
model_new = RandomForestRegressor(random_state=42)
scores_new = cross_val_score(model_new, X_new, y, cv=5)

# Compare
print(f"Baseline: {scores_baseline.mean():.3f} ± {scores_baseline.std():.3f}")
print(f"With new feature: {scores_new.mean():.3f} ± {scores_new.std():.3f}")

# Statistical test
from scipy.stats import ttest_rel
t_stat, p_value = ttest_rel(scores_new, scores_baseline)
print(f"p-value: {p_value:.4f}")

if p_value < 0.05 and scores_new.mean() > scores_baseline.mean():
    print("✅ New feature significantly improves performance!")
else:
    print("❌ New feature doesn't help")
```

---

## Key Takeaways

**Feature engineering is iterative:**
1. Start simple (basic features)
2. Analyze model errors
3. Create features to address errors
4. Validate improvement
5. Repeat

**Best practices:**
- Document all features (what, why, how)
- Version your feature sets
- Test one feature at a time
- Monitor feature importance
- Remove redundant features

**Common mistakes:**
- Using test data to create features (leakage!)
- Creating too many features (overfitting)
- Not validating improvements
- Ignoring domain knowledge
- Making features too complex

**Remember:** Simple features that capture the right pattern beat complex features that don't.

---

**Related guides:**
- [Common ML Mistakes](../../resources/COMMON_ML_MISTAKES.md) - Avoid feature engineering pitfalls
- [Data Processing](../../interactive_demos/04_data_processing.ipynb) - Data preparation basics
- [ML Engineering Best Practices](../production/ML_ENGINEERING_BEST_PRACTICES.md) - Production feature pipelines

---

*Last Updated: October 30, 2025*
