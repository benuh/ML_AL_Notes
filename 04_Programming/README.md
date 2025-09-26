# Module 04: Programming for Machine Learning

## 🎯 Learning Objectives
By the end of this module, you will:
- Master Python fundamentals for ML development
- Use essential libraries: NumPy, Pandas, Matplotlib, Scikit-learn
- Write clean, efficient, and reproducible ML code
- Set up proper development environments and workflows

## 🐍 Why Python for Machine Learning?

*Source: "Hands-On Machine Learning" - Aurélien Géron, Chapter 1*
> "Python has become the go-to language for Machine Learning, thanks to its simplicity, extensive libraries, and vibrant community."

### Python's ML Ecosystem
- **NumPy**: Numerical computing foundation
- **Pandas**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Data visualization
- **Scikit-learn**: Machine learning algorithms
- **TensorFlow/PyTorch**: Deep learning frameworks
- **Jupyter**: Interactive development environment

## 📚 Python Fundamentals for ML

### 4.1 Essential Python Concepts

```python
"""
Python ML Fundamentals
=====================

This section covers Python concepts specifically important for ML development.
Sources: "Python for Data Analysis" - Wes McKinney, Chapters 1-3
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

print("🐍 Python ML Fundamentals")
print("=" * 50)

# 1. List Comprehensions for Data Processing
print("\\n1. 📝 List Comprehensions")
print("-" * 30)

# Traditional approach
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
squares_traditional = []
for num in numbers:
    if num % 2 == 0:
        squares_traditional.append(num ** 2)

# Pythonic approach
squares_pythonic = [num**2 for num in numbers if num % 2 == 0]

print(f"Traditional approach: {squares_traditional}")
print(f"List comprehension:   {squares_pythonic}")

# ML example: Feature engineering
raw_features = ['age', 'income', 'education_years']
normalized_features = [f"{feature}_normalized" for feature in raw_features]
print(f"Feature names: {normalized_features}")

# 2. Lambda Functions and Functional Programming
print("\\n2. 🔧 Lambda Functions")
print("-" * 30)

# Data transformation pipeline
data = [1, 2, 3, 4, 5]

# Apply multiple transformations
pipeline = [
    lambda x: x * 2,        # Scale
    lambda x: x + 1,        # Shift
    lambda x: x ** 0.5      # Square root
]

# Apply pipeline
result = data.copy()
for transform in pipeline:
    result = list(map(transform, result))

print(f"Original: {data}")
print(f"After pipeline: {result}")

# ML example: Feature scaling functions
scalers = {
    'min_max': lambda x: (x - min(x)) / (max(x) - min(x)),
    'z_score': lambda x: (x - np.mean(x)) / np.std(x),
    'robust': lambda x: (x - np.median(x)) / np.percentile(x, 75) - np.percentile(x, 25)
}

sample_data = [1, 2, 3, 4, 5, 100]  # Note the outlier
for name, scaler in scalers.items():
    scaled = scaler(sample_data)
    print(f"{name:10}: {[round(x, 2) for x in scaled]}")

# 3. Exception Handling for Robust ML Code
print("\\n3. 🛡️ Exception Handling")
print("-" * 30)

def safe_model_training(X, y, model_class, **kwargs):
    """
    Safely train a model with proper error handling

    Source: "Effective Python" - Brett Slatkin, Item 65
    """
    try:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Initialize model
        model = model_class(**kwargs)

        # Train model
        model.fit(X_train, y_train)

        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        return {
            'model': model,
            'train_score': train_score,
            'test_score': test_score,
            'success': True,
            'error': None
        }

    except ValueError as e:
        return {'success': False, 'error': f"Data error: {e}"}
    except Exception as e:
        return {'success': False, 'error': f"Unexpected error: {e}"}

# Test with valid data
X, y = make_classification(n_samples=100, n_features=4, random_state=42)
result = safe_model_training(X, y, LogisticRegression, random_state=42)

if result['success']:
    print(f"✅ Training successful!")
    print(f"   Train score: {result['train_score']:.3f}")
    print(f"   Test score: {result['test_score']:.3f}")
else:
    print(f"❌ Training failed: {result['error']}")

# 4. Context Managers for Resource Management
print("\\n4. 📁 Context Managers")
print("-" * 30)

class MLExperimentLogger:
    """
    Context manager for ML experiment logging

    Ensures proper cleanup of resources and logging
    """
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.start_time = None

    def __enter__(self):
        import time
        self.start_time = time.time()
        print(f"🧪 Starting experiment: {self.experiment_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        duration = time.time() - self.start_time
        if exc_type is None:
            print(f"✅ Experiment completed successfully in {duration:.2f}s")
        else:
            print(f"❌ Experiment failed after {duration:.2f}s: {exc_val}")
        return False  # Don't suppress exceptions

# Usage example
with MLExperimentLogger("Logistic Regression Test") as logger:
    # Simulate experiment
    import time
    time.sleep(0.1)  # Simulate work
    accuracy = 0.85
    print(f"   Model accuracy: {accuracy:.3f}")

print("\\n💡 Why These Patterns Matter in ML:")
print("• List comprehensions: Efficient data preprocessing")
print("• Lambda functions: Pipeline transformations")
print("• Exception handling: Robust model training")
print("• Context managers: Resource management and logging")
```

## 🔢 NumPy: The Foundation of ML

### 4.2 NumPy for Machine Learning

```python
"""
NumPy for Machine Learning
=========================

NumPy provides the numerical foundation for all ML libraries.
Source: "Python for Data Analysis" - Wes McKinney, Chapter 4
"""

import numpy as np
import matplotlib.pyplot as plt

print("🔢 NumPy for Machine Learning")
print("=" * 50)

# 1. Array Creation and Manipulation
print("\\n1. 📊 Array Creation")
print("-" * 30)

# Creating arrays for ML
np.random.seed(42)

# Feature matrix (samples × features)
n_samples, n_features = 1000, 5
X = np.random.randn(n_samples, n_features)

# Target vector
y = np.random.randint(0, 2, n_samples)  # Binary classification

print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")
print(f"Memory usage: {X.nbytes / 1024:.1f} KB")

# 2. Broadcasting - Crucial for ML Operations
print("\\n2. 📡 Broadcasting")
print("-" * 30)

# Center the data (subtract mean)
X_centered = X - np.mean(X, axis=0)  # Broadcasting across samples

# Standardize (divide by std)
X_standardized = X_centered / np.std(X, axis=0)  # Broadcasting

print("Original data stats:")
print(f"  Mean: {np.mean(X, axis=0)}")
print(f"  Std:  {np.std(X, axis=0)}")

print("\\nStandardized data stats:")
print(f"  Mean: {np.mean(X_standardized, axis=0)}")
print(f"  Std:  {np.std(X_standardized, axis=0)}")

# 3. Linear Algebra Operations
print("\\n3. 🧮 Linear Algebra")
print("-" * 30)

# Simulate linear regression manually
# y = Xw + b + noise
true_weights = np.array([1.5, -2.0, 0.5, 3.0, -1.0])
true_bias = 2.0

# Generate regression data
X_reg = np.random.randn(100, 5)
y_reg = X_reg @ true_weights + true_bias + 0.1 * np.random.randn(100)

# Solve using normal equation: w = (X^T X)^(-1) X^T y
# Add bias term
X_with_bias = np.column_stack([np.ones(100), X_reg])
true_params = np.concatenate([[true_bias], true_weights])

# Normal equation
XTX = X_with_bias.T @ X_with_bias
XTy = X_with_bias.T @ y_reg
estimated_params = np.linalg.solve(XTX, XTy)

print("Linear Regression Results:")
print(f"True parameters:      {true_params}")
print(f"Estimated parameters: {estimated_params}")
print(f"Estimation error:     {np.abs(true_params - estimated_params)}")

# 4. Advanced Indexing for Data Manipulation
print("\\n4. 🎯 Advanced Indexing")
print("-" * 30)

# Boolean indexing (very common in ML)
data = np.random.randn(1000)
outliers = np.abs(data) > 2  # Boolean mask

print(f"Total data points: {len(data)}")
print(f"Outliers (|x| > 2): {np.sum(outliers)}")
print(f"Outlier percentage: {np.mean(outliers):.1%}")

# Remove outliers
clean_data = data[~outliers]
print(f"Clean data points: {len(clean_data)}")

# Fancy indexing for feature selection
features = np.random.randn(100, 10)
selected_features = [0, 2, 5, 7]  # Feature indices
X_selected = features[:, selected_features]

print(f"Original features: {features.shape}")
print(f"Selected features: {X_selected.shape}")

# 5. Vectorization for Performance
print("\\n5. ⚡ Vectorization")
print("-" * 30)

# Compare vectorized vs loop operations
n = 100000
a = np.random.randn(n)
b = np.random.randn(n)

# Timing vectorized operation
import time

start = time.time()
c_vectorized = a * b + np.sin(a) - np.cos(b)
time_vectorized = time.time() - start

# Timing loop operation
start = time.time()
c_loop = np.zeros(n)
for i in range(n):
    c_loop[i] = a[i] * b[i] + np.sin(a[i]) - np.cos(b[i])
time_loop = time.time() - start

print(f"Vectorized operation: {time_vectorized:.4f} seconds")
print(f"Loop operation:       {time_loop:.4f} seconds")
print(f"Speedup:              {time_loop / time_vectorized:.1f}x")

# Verify results are the same
print(f"Results identical:    {np.allclose(c_vectorized, c_loop)}")

# 6. Memory-Efficient Operations
print("\\n6. 💾 Memory Efficiency")
print("-" * 30)

# In-place operations to save memory
large_array = np.random.randn(10000, 100)
print(f"Original memory: {large_array.nbytes / 1024**2:.1f} MB")

# Bad: Creates new array
# large_array = large_array * 2

# Good: In-place operation
large_array *= 2  # Modifies existing array
print(f"After in-place multiplication - same memory usage")

# Views vs copies
view = large_array[::2, ::2]  # Creates a view (no copy)
copy = large_array[::2, ::2].copy()  # Creates a copy

print(f"View shares memory:   {view.base is large_array}")
print(f"Copy shares memory:   {copy.base is large_array}")

def demonstrate_array_operations():
    """Demonstrate essential NumPy operations for ML"""

    # Create sample dataset
    np.random.seed(42)
    data = np.random.randn(1000, 4)

    # Statistical operations (common in ML)
    print("\\n📊 Statistical Operations:")
    print(f"Mean (features):     {np.mean(data, axis=0)}")
    print(f"Std (features):      {np.std(data, axis=0)}")
    print(f"Min (samples):       {np.min(data, axis=1)[:5]}")  # First 5
    print(f"Max (samples):       {np.max(data, axis=1)[:5]}")  # First 5

    # Correlation matrix (feature relationships)
    correlation_matrix = np.corrcoef(data.T)
    print(f"\\nCorrelation matrix shape: {correlation_matrix.shape}")

    # Eigenvalues (PCA preprocessing)
    eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix)
    print(f"Eigenvalues: {eigenvalues}")

    # Data transformations
    # 1. Standardization
    standardized = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    # 2. Min-max scaling
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    min_max_scaled = (data - min_vals) / (max_vals - min_vals)

    # 3. Robust scaling (using median and IQR)
    median_vals = np.median(data, axis=0)
    q75 = np.percentile(data, 75, axis=0)
    q25 = np.percentile(data, 25, axis=0)
    robust_scaled = (data - median_vals) / (q75 - q25)

    print(f"\\n🔄 Data Transformations:")
    print(f"Original range:     [{np.min(data):.2f}, {np.max(data):.2f}]")
    print(f"Standardized range: [{np.min(standardized):.2f}, {np.max(standardized):.2f}]")
    print(f"Min-max range:      [{np.min(min_max_scaled):.2f}, {np.max(min_max_scaled):.2f}]")
    print(f"Robust range:       [{np.min(robust_scaled):.2f}, {np.max(robust_scaled):.2f}]")

    return data, standardized, min_max_scaled, robust_scaled

# Run demonstration
array_results = demonstrate_array_operations()

print("\\n💡 NumPy Best Practices for ML:")
print("• Use vectorized operations instead of loops")
print("• Leverage broadcasting for efficient computations")
print("• Use views instead of copies when possible")
print("• Take advantage of in-place operations for memory efficiency")
print("• Master array indexing for data manipulation")
```

## 🐼 Pandas: Data Manipulation Powerhouse

### 4.3 Pandas for ML Data Processing

```python
"""
Pandas for Machine Learning
===========================

Pandas excels at data cleaning, transformation, and exploration.
Source: "Python for Data Analysis" - Wes McKinney, Chapters 5-10
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("🐼 Pandas for Machine Learning")
print("=" * 50)

# 1. Creating and Loading Data
print("\\n1. 📥 Data Loading and Creation")
print("-" * 30)

# Create sample dataset
np.random.seed(42)
n_samples = 1000

# Simulate customer data
data = {
    'customer_id': range(1, n_samples + 1),
    'age': np.random.normal(35, 12, n_samples).clip(18, 80).astype(int),
    'income': np.random.lognormal(10, 0.5, n_samples).round(2),
    'spending_score': np.random.normal(50, 15, n_samples).clip(0, 100).round(1),
    'membership_years': np.random.exponential(3, n_samples).round(1),
    'purchase_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], n_samples),
    'satisfaction': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.2, 0.5, 0.3])
}

# Add some missing values (realistic scenario)
missing_indices = np.random.choice(n_samples, size=50, replace=False)
data['income'][missing_indices[:25]] = np.nan
data['satisfaction'][missing_indices[25:]] = np.nan

df = pd.DataFrame(data)

print(f"Dataset shape: {df.shape}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
print("\\nFirst 5 rows:")
print(df.head())

# 2. Data Exploration and Profiling
print("\\n2. 🔍 Data Exploration")
print("-" * 30)

print("Data Info:")
print(df.info())

print("\\nNumerical Summary:")
print(df.describe())

print("\\nCategorical Summary:")
print(df.describe(include=['object']))

print("\\nMissing Values:")
missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing_data,
    'Missing Percentage': missing_percent
})
print(missing_df[missing_df['Missing Count'] > 0])

# 3. Data Cleaning and Preprocessing
print("\\n3. 🧹 Data Cleaning")
print("-" * 30)

# Handle missing values
df_cleaned = df.copy()

# Fill missing income with median by category
median_income_by_category = df_cleaned.groupby('purchase_category')['income'].median()
df_cleaned['income'] = df_cleaned.groupby('purchase_category')['income'].transform(
    lambda x: x.fillna(x.median())
)

# Fill missing satisfaction with mode
mode_satisfaction = df_cleaned['satisfaction'].mode()[0]
df_cleaned['satisfaction'].fillna(mode_satisfaction, inplace=True)

print(f"Missing values after cleaning: {df_cleaned.isnull().sum().sum()}")

# Detect and handle outliers
def detect_outliers_iqr(df, column):
    """Detect outliers using IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    return outliers

# Check for outliers in income
income_outliers = detect_outliers_iqr(df_cleaned, 'income')
print(f"Income outliers: {income_outliers.sum()} ({income_outliers.mean():.1%})")

# Cap outliers instead of removing them
def cap_outliers(df, column, method='iqr'):
    """Cap outliers at specified percentiles"""
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
    else:  # percentile method
        lower_bound = df[column].quantile(0.05)
        upper_bound = df[column].quantile(0.95)

    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df

df_cleaned = cap_outliers(df_cleaned, 'income')
print(f"Income range after capping: [{df_cleaned['income'].min():.0f}, {df_cleaned['income'].max():.0f}]")

# 4. Feature Engineering
print("\\n4. 🔧 Feature Engineering")
print("-" * 30)

# Create new features
df_features = df_cleaned.copy()

# Age groups
df_features['age_group'] = pd.cut(df_features['age'],
                                 bins=[0, 25, 35, 50, 65, 100],
                                 labels=['Young', 'Adult', 'Middle-aged', 'Senior', 'Elderly'])

# Income quartiles
df_features['income_quartile'] = pd.qcut(df_features['income'],
                                        q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])

# Spending efficiency (spending score per income)
df_features['spending_efficiency'] = df_features['spending_score'] / (df_features['income'] / 1000)

# Customer lifetime value estimate
df_features['estimated_clv'] = (df_features['income'] * 0.1 *
                               df_features['membership_years'] *
                               df_features['spending_score'] / 100)

# Binary features
df_features['high_spender'] = (df_features['spending_score'] > df_features['spending_score'].quantile(0.75)).astype(int)
df_features['long_term_customer'] = (df_features['membership_years'] > 5).astype(int)

print("New features created:")
print(df_features[['age_group', 'income_quartile', 'spending_efficiency',
                  'estimated_clv', 'high_spender', 'long_term_customer']].head())

# 5. Encoding Categorical Variables
print("\\n5. 🏷️ Categorical Encoding")
print("-" * 30)

# One-hot encoding
categorical_columns = ['purchase_category', 'satisfaction', 'age_group', 'income_quartile']
df_encoded = pd.get_dummies(df_features, columns=categorical_columns, prefix=categorical_columns)

print(f"Columns before encoding: {len(df_features.columns)}")
print(f"Columns after encoding: {len(df_encoded.columns)}")

# Label encoding (for ordinal variables)
from sklearn.preprocessing import LabelEncoder

# Create ordinal satisfaction encoding
satisfaction_order = {'Low': 0, 'Medium': 1, 'High': 2}
df_features['satisfaction_encoded'] = df_features['satisfaction'].map(satisfaction_order)

print("\\nSatisfaction encoding:")
print(df_features[['satisfaction', 'satisfaction_encoded']].drop_duplicates().sort_values('satisfaction_encoded'))

# 6. Data Aggregation and Grouping
print("\\n6. 📊 Data Aggregation")
print("-" * 30)

# Group by analysis
category_stats = df_features.groupby('purchase_category').agg({
    'age': ['mean', 'std'],
    'income': ['mean', 'median'],
    'spending_score': ['mean', 'std'],
    'membership_years': 'mean',
    'estimated_clv': ['mean', 'sum']
}).round(2)

print("Statistics by Purchase Category:")
print(category_stats)

# Pivot tables for cross-tabulation
pivot_table = pd.pivot_table(df_features,
                            values='estimated_clv',
                            index='age_group',
                            columns='satisfaction',
                            aggfunc='mean').round(2)

print("\\nAverage CLV by Age Group and Satisfaction:")
print(pivot_table)

# 7. Time Series Simulation (for demonstration)
print("\\n7. 📅 Time Series Data")
print("-" * 30)

# Create time series data
dates = pd.date_range('2020-01-01', periods=365, freq='D')
ts_data = pd.DataFrame({
    'date': dates,
    'daily_sales': np.random.poisson(100, 365) + 20 * np.sin(np.arange(365) * 2 * np.pi / 365),
    'temperature': 20 + 15 * np.sin(np.arange(365) * 2 * np.pi / 365) + np.random.normal(0, 3, 365)
})

ts_data.set_index('date', inplace=True)

# Time-based features
ts_data['day_of_week'] = ts_data.index.dayofweek
ts_data['month'] = ts_data.index.month
ts_data['quarter'] = ts_data.index.quarter
ts_data['is_weekend'] = (ts_data['day_of_week'] >= 5).astype(int)

# Rolling statistics
ts_data['sales_7day_avg'] = ts_data['daily_sales'].rolling(window=7).mean()
ts_data['sales_30day_avg'] = ts_data['daily_sales'].rolling(window=30).mean()

print("Time series features:")
print(ts_data.head(10))

# 8. Performance Optimization
print("\\n8. ⚡ Performance Tips")
print("-" * 30)

# Memory optimization
def optimize_memory(df):
    """Optimize DataFrame memory usage"""
    start_mem = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
        else:
            # Convert object to category if beneficial
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if num_unique_values / num_total_values < 0.5:
                df[col] = df[col].astype('category')

    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    reduction = (start_mem - end_mem) / start_mem

    print(f"Memory usage decreased from {start_mem:.1f} MB to {end_mem:.1f} MB")
    print(f"Reduction: {reduction:.1%}")

    return df

# Apply memory optimization
df_optimized = optimize_memory(df_features.copy())

print("\\n💡 Pandas Best Practices for ML:")
print("• Use vectorized operations instead of apply() when possible")
print("• Optimize data types to reduce memory usage")
print("• Use categorical data type for string columns with few unique values")
print("• Leverage groupby for efficient aggregations")
print("• Use method chaining for readable data pipelines")
print("• Always validate data quality after transformations")

return df_cleaned, df_features, df_encoded, ts_data

# Run demonstration
pandas_results = demonstrate_pandas_ml()
```

## 📊 Data Visualization with Matplotlib & Seaborn

### 4.4 Visualization for ML

```python
"""
Data Visualization for Machine Learning
======================================

Effective visualization is crucial for understanding data and model performance.
Source: "Fundamentals of Data Visualization" - Claus Wilke, Chapters 1-5
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, make_classification
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier

print("📊 Data Visualization for ML")
print("=" * 50)

# Set style for consistent, publication-ready plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def create_ml_visualization_guide():
    """Comprehensive guide to ML visualization"""

    # Load sample data
    iris = load_iris()
    df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
    df_iris['species'] = iris.target_names[iris.target]

    # 1. Exploratory Data Analysis Plots
    print("\\n1. 📈 Exploratory Data Analysis")
    print("-" * 30)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('📊 EDA Visualization Guide', fontsize=16, fontweight='bold')

    # Distribution plot
    sns.histplot(data=df_iris, x='sepal length (cm)', hue='species',
                kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Distribution by Class')

    # Box plot for outlier detection
    sns.boxplot(data=df_iris, x='species', y='petal length (cm)', ax=axes[0, 1])
    axes[0, 1].set_title('Box Plot (Outlier Detection)')

    # Correlation heatmap
    correlation_matrix = df_iris.select_dtypes(include=[np.number]).corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[0, 2])
    axes[0, 2].set_title('Feature Correlation Matrix')

    # Scatter plot with color coding
    sns.scatterplot(data=df_iris, x='sepal length (cm)', y='sepal width (cm)',
                   hue='species', style='species', s=100, ax=axes[1, 0])
    axes[1, 0].set_title('Feature Relationships')

    # Pair plot subset (just show in one subplot)
    # For demo, we'll create a simplified pair plot
    features = ['sepal length (cm)', 'petal length (cm)']
    for i, species in enumerate(df_iris['species'].unique()):
        species_data = df_iris[df_iris['species'] == species]
        axes[1, 1].scatter(species_data[features[0]], species_data[features[1]],
                          label=species, alpha=0.7, s=50)
    axes[1, 1].set_xlabel(features[0])
    axes[1, 1].set_ylabel(features[1])
    axes[1, 1].set_title('Pairwise Feature Plot')
    axes[1, 1].legend()

    # Feature importance plot (using a simple model)
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(iris.data, iris.target)

    feature_importance = rf.feature_importances_
    feature_names = iris.feature_names

    bars = axes[1, 2].bar(range(len(feature_importance)), feature_importance)
    axes[1, 2].set_xticks(range(len(feature_names)))
    axes[1, 2].set_xticklabels([name.replace(' (cm)', '') for name in feature_names],
                              rotation=45)
    axes[1, 2].set_title('Feature Importance')
    axes[1, 2].set_ylabel('Importance Score')

    # Add value labels on bars
    for bar, importance in zip(bars, feature_importance):
        axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{importance:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    # 2. Model Performance Visualization
    print("\\n2. 🎯 Model Performance Visualization")
    print("-" * 30)

    # Generate classification data for model evaluation
    X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                              n_informative=2, n_clusters_per_class=1, random_state=42)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('🎯 Model Performance Visualization', fontsize=16, fontweight='bold')

    # Learning curves
    train_sizes, train_scores, val_scores = learning_curve(
        RandomForestClassifier(n_estimators=50, random_state=42), X, y,
        cv=5, train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    axes[0, 0].plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    axes[0, 0].fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                           alpha=0.2, color='blue')

    axes[0, 0].plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
    axes[0, 0].fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                           alpha=0.2, color='red')

    axes[0, 0].set_title('Learning Curves')
    axes[0, 0].set_xlabel('Training Set Size')
    axes[0, 0].set_ylabel('Accuracy Score')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Confusion Matrix
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
    axes[0, 1].set_title('Confusion Matrix')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Actual')

    # ROC Curve
    from sklearn.metrics import roc_curve, auc

    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    axes[1, 0].plot(fpr, tpr, color='darkorange', linewidth=2,
                   label=f'ROC Curve (AUC = {roc_auc:.2f})')
    axes[1, 0].plot([0, 1], [0, 1], color='navy', linewidth=2, linestyle='--')
    axes[1, 0].set_xlim([0.0, 1.0])
    axes[1, 0].set_ylim([0.0, 1.05])
    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].set_title('ROC Curve')
    axes[1, 0].legend(loc="lower right")
    axes[1, 0].grid(True, alpha=0.3)

    # Feature Space Visualization
    # Use PCA for dimensionality reduction if needed
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_vis = pca.fit_transform(X_test)
    else:
        X_vis = X_test

    # Create decision boundary
    h = 0.02
    x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
    y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    # For visualization, we'll use the original 2D data
    model_vis = RandomForestClassifier(n_estimators=50, random_state=42)
    model_vis.fit(X_test, y_test)  # Quick fit for visualization

    Z = model_vis.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axes[1, 1].contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    scatter = axes[1, 1].scatter(X_vis[:, 0], X_vis[:, 1], c=y_test, cmap=plt.cm.RdYlBu)
    axes[1, 1].set_title('Decision Boundary')
    axes[1, 1].set_xlabel('Feature 1')
    axes[1, 1].set_ylabel('Feature 2')

    plt.tight_layout()
    plt.show()

    # 3. Advanced Visualization Techniques
    print("\\n3. 🎨 Advanced Visualization Techniques")
    print("-" * 30)

    create_advanced_visualizations()

def create_advanced_visualizations():
    """Create advanced ML visualizations"""

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 500

    # Multi-class classification data
    from sklearn.datasets import make_blobs
    X_blobs, y_blobs = make_blobs(n_samples=n_samples, centers=4, n_features=2,
                                 random_state=42, cluster_std=1.5)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('🎨 Advanced ML Visualizations', fontsize=16, fontweight='bold')

    # 1. Multi-class Decision Boundaries
    from sklearn.svm import SVC

    svm_model = SVC(kernel='rbf', probability=True, random_state=42)
    svm_model.fit(X_blobs, y_blobs)

    # Create decision boundary
    h = 0.02
    x_min, x_max = X_blobs[:, 0].min() - 1, X_blobs[:, 0].max() + 1
    y_min, y_max = X_blobs[:, 1].min() - 1, X_blobs[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axes[0, 0].contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    scatter = axes[0, 0].scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_blobs, cmap='viridis')
    axes[0, 0].set_title('Multi-class Decision Boundaries')
    axes[0, 0].set_xlabel('Feature 1')
    axes[0, 0].set_ylabel('Feature 2')

    # 2. Model Comparison
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
        'SVM': SVC(random_state=42)
    }

    accuracies = []
    model_names = []

    X_train, X_test, y_train, y_test = train_test_split(X_blobs, y_blobs,
                                                       test_size=0.3, random_state=42)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        model_names.append(name)

    bars = axes[0, 1].bar(model_names, accuracies, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    axes[0, 1].set_title('Model Comparison')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_ylim(0, 1)

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{acc:.3f}', ha='center', va='bottom')

    plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)

    # 3. Hyperparameter Tuning Visualization
    from sklearn.model_selection import validation_curve

    param_range = [1, 2, 5, 10, 20, 50, 100]
    train_scores, val_scores = validation_curve(
        RandomForestClassifier(random_state=42), X_train, y_train,
        param_name='n_estimators', param_range=param_range,
        cv=5, scoring='accuracy', n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    axes[1, 0].plot(param_range, train_mean, 'o-', color='blue', label='Training Score')
    axes[1, 0].fill_between(param_range, train_mean - train_std, train_mean + train_std,
                           alpha=0.2, color='blue')

    axes[1, 0].plot(param_range, val_mean, 'o-', color='red', label='Validation Score')
    axes[1, 0].fill_between(param_range, val_mean - val_std, val_mean + val_std,
                           alpha=0.2, color='red')

    axes[1, 0].set_title('Hyperparameter Tuning: n_estimators')
    axes[1, 0].set_xlabel('Number of Estimators')
    axes[1, 0].set_ylabel('Accuracy Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Feature Importance with Uncertainty
    # Train multiple models to get uncertainty estimates
    rf_models = []
    feature_importances = []

    for i in range(10):
        rf = RandomForestClassifier(n_estimators=50, random_state=i)
        rf.fit(X_train, y_train)
        rf_models.append(rf)
        feature_importances.append(rf.feature_importances_)

    feature_importances = np.array(feature_importances)
    mean_importance = np.mean(feature_importances, axis=0)
    std_importance = np.std(feature_importances, axis=0)

    feature_names = [f'Feature {i+1}' for i in range(X_train.shape[1])]

    bars = axes[1, 1].bar(feature_names, mean_importance, yerr=std_importance,
                         capsize=5, color='lightblue', edgecolor='navy')
    axes[1, 1].set_title('Feature Importance with Uncertainty')
    axes[1, 1].set_ylabel('Importance Score')

    # Add value labels
    for i, (bar, mean_imp, std_imp) in enumerate(zip(bars, mean_importance, std_importance)):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_imp + 0.01,
                       f'{mean_imp:.3f}±{std_imp:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()

def create_publication_ready_plots():
    """Create publication-ready visualization examples"""

    print("\\n4. 📄 Publication-Ready Plots")
    print("-" * 30)

    # Set publication style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })

    # Example: Model performance comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Simulated results from multiple models
    models = ['Linear\\nRegression', 'Decision\\nTree', 'Random\\nForest', 'SVM', 'Neural\\nNetwork']
    accuracy_means = [0.82, 0.85, 0.89, 0.87, 0.91]
    accuracy_stds = [0.03, 0.05, 0.02, 0.04, 0.03]

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

    bars = ax.bar(models, accuracy_means, yerr=accuracy_stds,
                 capsize=5, color=colors, alpha=0.8, edgecolor='black', linewidth=1)

    ax.set_ylabel('Classification Accuracy', fontweight='bold')
    ax.set_title('Model Performance Comparison\\n(5-Fold Cross-Validation)', fontweight='bold', pad=20)
    ax.set_ylim(0.75, 0.95)

    # Add value labels
    for bar, mean_acc, std_acc in zip(bars, accuracy_means, accuracy_stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_acc + 0.005,
               f'{mean_acc:.3f}', ha='center', va='bottom', fontweight='bold')

    # Add statistical significance indicators
    ax.annotate('***', xy=(2, 0.91), xytext=(2, 0.93),
               ha='center', va='bottom', fontsize=14, fontweight='bold')
    ax.annotate('p < 0.001', xy=(2, 0.93), xytext=(2, 0.935),
               ha='center', va='bottom', fontsize=10)

    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("💡 Publication-ready plot features:")
    print("• Clear, descriptive titles and labels")
    print("• Error bars with appropriate statistics")
    print("• Consistent color scheme")
    print("• High DPI for crisp printing")
    print("• Statistical significance indicators")
    print("• Professional styling")

# Run all visualization demonstrations
create_ml_visualization_guide()
create_publication_ready_plots()

print("\\n📊 Visualization Best Practices for ML:")
print("=" * 50)
print("🎯 **Exploratory Data Analysis**:")
print("  • Use histograms for distribution analysis")
print("  • Box plots for outlier detection")
print("  • Correlation heatmaps for feature relationships")
print("  • Scatter plots for bivariate analysis")

print("\\n🎯 **Model Performance**:")
print("  • Learning curves for overfitting analysis")
print("  • Confusion matrices for classification errors")
print("  • ROC curves for threshold selection")
print("  • Feature importance plots for interpretability")

print("\\n🎯 **Communication**:")
print("  • Clear titles and axis labels")
print("  • Consistent color schemes")
print("  • Error bars for uncertainty")
print("  • Publication-ready formatting")

print("\\n🔧 **Technical Tips**:")
print("  • Use appropriate plot types for data types")
print("  • Avoid chart junk and unnecessary decoration")
print("  • Choose colorblind-friendly palettes")
print("  • Save plots in vector formats for papers")
```

## 🚀 Next Steps

This comprehensive setup guide and improved learning materials provide:

### **📋 How to Run the Project:**
1. **Setup**: Follow `SETUP_GUIDE.md` for complete installation
2. **Quick Start**: `jupyter lab` → open `interactive_demos/01_getting_started.ipynb`
3. **Practice**: Run `python exercises/beginner_exercises.py`
4. **Explore**: Browse through all 10 learning modules

### **🆕 New Improvements Added:**
- **Complete Statistics Module** with probability, inference, and Bayesian methods
- **Comprehensive Programming Module** with Python, NumPy, Pandas, and visualization
- **Detailed Setup Guide** with troubleshooting and multiple installation options
- **Enhanced Source Citations** with specific chapter references and free PDF links
- **Interactive Examples** with executable code and visualizations

### **📚 Learning Path:**
1. **Foundations** → **Mathematics** → **Statistics** → **Programming**
2. **Data Processing** → **Classical ML** → **Deep Learning**
3. **Advanced Topics** → **Projects** → **Research**

The project now provides a complete, self-contained learning environment that takes you from complete beginner to ML expert with proper academic sources, interactive visualizations, and hands-on coding experience!