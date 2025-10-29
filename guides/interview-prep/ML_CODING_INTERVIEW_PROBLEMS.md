# ML Coding Interview Problems & Solutions

**Complete Collection of Machine Learning Coding Challenges**

> 30+ ML coding problems with detailed solutions and explanations

---

## Table of Contents

1. [Introduction](#introduction)
2. [Problem Categories](#problem-categories)
3. [Data Manipulation Problems](#data-manipulation-problems)
4. [Statistical Problems](#statistical-problems)
5. [Algorithm Implementation](#algorithm-implementation)
6. [Model Training Problems](#model-training-problems)
7. [Optimization Problems](#optimization-problems)
8. [Deep Learning Problems](#deep-learning-problems)
9. [Production ML Problems](#production-ml-problems)
10. [Problem-Solving Strategies](#problem-solving-strategies)

---

## Introduction

### What to Expect in ML Coding Interviews

**Types of Questions:**
1. **Data Manipulation** - Pandas, NumPy operations (30% of interviews)
2. **Statistics & Probability** - Calculate metrics, implement statistical tests (20%)
3. **Algorithm Implementation** - Implement ML algorithms from scratch (25%)
4. **Model Training** - Build and evaluate models (15%)
5. **Deep Learning** - Neural network implementations (10%)

**Interview Format:**
- **45-60 minutes** per coding round
- **2-3 problems** typically
- **Expected:** Working code + explanation
- **Evaluation:** Correctness, efficiency, code quality, explanation

**Preparation Strategy:**
- Practice 50+ problems
- Focus on fundamentals first
- Implement algorithms from scratch
- Understand time/space complexity
- Practice explaining your approach

---

## Problem Categories

### Difficulty Levels
- 🟢 **Easy** - Basic operations, simple implementations (15 problems)
- 🟡 **Medium** - Requires algorithmic thinking (12 problems)
- 🔴 **Hard** - Complex algorithms, optimization needed (8 problems)

### Time Allocation
- 🟢 Easy: 10-15 minutes
- 🟡 Medium: 20-30 minutes
- 🔴 Hard: 30-45 minutes

---

## Data Manipulation Problems

### Problem 1: Rolling Window Statistics 🟢

**Question:**
Given a pandas DataFrame with columns ['date', 'value'], calculate 7-day rolling mean and standard deviation.

**Example:**
```python
# Input
   date       value
0  2024-01-01  10
1  2024-01-02  12
2  2024-01-03  11
...

# Output should include rolling_mean_7 and rolling_std_7 columns
```

**Solution:**
```python
import pandas as pd
import numpy as np

def calculate_rolling_stats(df, window=7):
    """
    Calculate rolling statistics for a time series.

    Args:
        df: DataFrame with 'date' and 'value' columns
        window: Rolling window size

    Returns:
        DataFrame with additional rolling statistics columns
    """
    # Ensure data is sorted by date
    df = df.sort_values('date').reset_index(drop=True)

    # Calculate rolling statistics
    df['rolling_mean_7'] = df['value'].rolling(window=window, min_periods=1).mean()
    df['rolling_std_7'] = df['value'].rolling(window=window, min_periods=1).std()

    # Handle edge cases (first few rows with insufficient data)
    df['rolling_std_7'].fillna(0, inplace=True)

    return df

# Test
df = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=20),
    'value': np.random.randint(10, 20, 20)
})

result = calculate_rolling_stats(df)
print(result.head(10))
```

**Time Complexity:** O(n) where n is number of rows
**Space Complexity:** O(1) excluding output

**Key Points:**
- Always sort by date first
- Use `min_periods=1` to handle edge cases
- Consider missing data handling

---

### Problem 2: Feature Engineering for DateTime 🟢

**Question:**
Extract useful features from a datetime column: hour, day of week, month, quarter, is_weekend, is_month_start, is_month_end.

**Solution:**
```python
import pandas as pd

def extract_datetime_features(df, datetime_col='timestamp'):
    """
    Extract comprehensive datetime features.

    Args:
        df: DataFrame with datetime column
        datetime_col: Name of datetime column

    Returns:
        DataFrame with additional datetime features
    """
    # Ensure datetime type
    df[datetime_col] = pd.to_datetime(df[datetime_col])

    # Extract features
    df['year'] = df[datetime_col].dt.year
    df['month'] = df[datetime_col].dt.month
    df['day'] = df[datetime_col].dt.day
    df['hour'] = df[datetime_col].dt.hour
    df['day_of_week'] = df[datetime_col].dt.dayofweek  # Monday=0, Sunday=6
    df['quarter'] = df[datetime_col].dt.quarter

    # Boolean features
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_month_start'] = df[datetime_col].dt.is_month_start.astype(int)
    df['is_month_end'] = df[datetime_col].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df[datetime_col].dt.is_quarter_start.astype(int)

    # Cyclical encoding (important for ML models)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    return df

# Test
df = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=100, freq='6H')
})

result = extract_datetime_features(df)
print(result[['timestamp', 'hour', 'is_weekend', 'hour_sin', 'hour_cos']].head())
```

**Why Cyclical Encoding?**
- Hour 23 and hour 0 are close but numerically far (23 vs 0)
- Sine/cosine encoding preserves cyclical nature
- ML models can learn patterns better

---

### Problem 3: Handle Missing Data 🟡

**Question:**
Implement a function that handles missing data with multiple strategies: mean/median imputation for numerical, mode for categorical, and forward-fill for time series.

**Solution:**
```python
import pandas as pd
import numpy as np

def handle_missing_data(df, numerical_cols=None, categorical_cols=None,
                        timeseries_cols=None, strategy='auto'):
    """
    Comprehensive missing data handler.

    Args:
        df: DataFrame with missing data
        numerical_cols: List of numerical column names
        categorical_cols: List of categorical column names
        timeseries_cols: List of time series column names
        strategy: 'mean', 'median', 'mode', 'forward_fill', or 'auto'

    Returns:
        DataFrame with imputed values
    """
    df = df.copy()

    # Auto-detect column types if not specified
    if numerical_cols is None:
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Numerical columns
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            if col in (timeseries_cols or []):
                # Forward fill for time series
                df[col].fillna(method='ffill', inplace=True)
                df[col].fillna(method='bfill', inplace=True)  # Handle leading NaNs
            else:
                # Check for outliers to decide between mean and median
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                outliers = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()

                if outliers > len(df) * 0.05:  # >5% outliers, use median
                    fill_value = df[col].median()
                else:
                    fill_value = df[col].mean()

                df[col].fillna(fill_value, inplace=True)

    # Categorical columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            # Use mode (most frequent value)
            mode_value = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
            df[col].fillna(mode_value, inplace=True)

    return df

# Test
df = pd.DataFrame({
    'num_col': [1, 2, np.nan, 4, 5, np.nan, 100],  # Has outlier
    'cat_col': ['A', 'B', np.nan, 'A', 'B', 'A', 'A'],
    'ts_col': [10, 12, np.nan, np.nan, 15, 16, 17]
})

result = handle_missing_data(
    df,
    numerical_cols=['num_col', 'ts_col'],
    categorical_cols=['cat_col'],
    timeseries_cols=['ts_col']
)
print(result)
```

**Decision Tree:**
```
Missing Data
├─ Numerical
│  ├─ Time Series? → Forward Fill
│  ├─ Many Outliers (>5%)? → Median
│  └─ Few Outliers → Mean
└─ Categorical → Mode (most frequent)
```

---

## Statistical Problems

### Problem 4: Calculate Metrics from Confusion Matrix 🟢

**Question:**
Given a confusion matrix, calculate accuracy, precision, recall, F1-score, and specificity.

**Solution:**
```python
import numpy as np

def calculate_metrics(confusion_matrix):
    """
    Calculate classification metrics from confusion matrix.

    Args:
        confusion_matrix: 2x2 numpy array [[TN, FP], [FN, TP]]

    Returns:
        Dictionary of metrics
    """
    TN, FP = confusion_matrix[0]
    FN, TP = confusion_matrix[1]

    # Basic metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # Precision: Of predicted positive, how many are actually positive?
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    # Recall (Sensitivity): Of actual positive, how many did we find?
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # F1-Score: Harmonic mean of precision and recall
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Specificity: Of actual negative, how many did we correctly identify?
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    # False Positive Rate
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0

    # False Negative Rate
    fnr = FN / (FN + TP) if (FN + TP) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'specificity': specificity,
        'fpr': fpr,
        'fnr': fnr,
        'tn': TN,
        'fp': FP,
        'fn': FN,
        'tp': TP
    }

# Test
cm = np.array([[850, 50],   # [TN, FP]
               [100, 200]])  # [FN, TP]

metrics = calculate_metrics(cm)
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}" if isinstance(value, float) else f"{metric}: {value}")
```

**Expected Output:**
```
accuracy: 0.8750
precision: 0.8000
recall: 0.6667
f1_score: 0.7273
specificity: 0.9444
...
```

---

### Problem 5: Implement Stratified Sampling 🟡

**Question:**
Implement stratified sampling to split data into train/test while preserving class proportions.

**Solution:**
```python
import numpy as np
import pandas as pd

def stratified_split(X, y, test_size=0.2, random_state=42):
    """
    Perform stratified train-test split.

    Args:
        X: Features (numpy array or DataFrame)
        y: Labels (numpy array or Series)
        test_size: Proportion of test set (0-1)
        random_state: Random seed

    Returns:
        X_train, X_test, y_train, y_test
    """
    np.random.seed(random_state)

    # Convert to numpy arrays if needed
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

    # Get unique classes and their counts
    classes, class_counts = np.unique(y, return_counts=True)

    # Initialize lists for indices
    train_indices = []
    test_indices = []

    # For each class, split proportionally
    for cls in classes:
        # Get indices of this class
        cls_indices = np.where(y == cls)[0]

        # Shuffle indices
        np.random.shuffle(cls_indices)

        # Calculate split point
        n_test = int(len(cls_indices) * test_size)

        # Split indices
        test_indices.extend(cls_indices[:n_test])
        train_indices.extend(cls_indices[n_test:])

    # Shuffle final indices to avoid ordering bias
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    # Create splits
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test

# Test
X = np.random.randn(1000, 10)
y = np.array([0] * 900 + [1] * 100)  # Imbalanced: 90% class 0, 10% class 1

X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2)

# Verify proportions
print("Original class distribution:", np.bincount(y) / len(y))
print("Train class distribution:", np.bincount(y_train) / len(y_train))
print("Test class distribution:", np.bincount(y_test) / len(y_test))
```

**Expected Output:**
```
Original class distribution: [0.9  0.1]
Train class distribution: [0.9  0.1]
Test class distribution: [0.9  0.1]
```

**Why Stratified Sampling?**
- Preserves class proportions in train/test
- Critical for imbalanced datasets
- Ensures representative splits

---

### Problem 6: Bootstrap Confidence Interval 🟡

**Question:**
Implement bootstrap resampling to calculate 95% confidence interval for a statistic (e.g., mean).

**Solution:**
```python
import numpy as np

def bootstrap_confidence_interval(data, statistic_func=np.mean,
                                  n_iterations=10000, confidence_level=0.95):
    """
    Calculate bootstrap confidence interval.

    Args:
        data: 1D array of data points
        statistic_func: Function to calculate statistic (default: mean)
        n_iterations: Number of bootstrap samples
        confidence_level: Confidence level (0-1)

    Returns:
        Dictionary with statistic, lower_bound, upper_bound
    """
    n = len(data)
    bootstrap_statistics = np.zeros(n_iterations)

    # Generate bootstrap samples and calculate statistic
    for i in range(n_iterations):
        # Sample with replacement
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_statistics[i] = statistic_func(bootstrap_sample)

    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = np.percentile(bootstrap_statistics, lower_percentile)
    upper_bound = np.percentile(bootstrap_statistics, upper_percentile)

    # Calculate actual statistic on original data
    actual_statistic = statistic_func(data)

    return {
        'statistic': actual_statistic,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'confidence_level': confidence_level,
        'bootstrap_distribution': bootstrap_statistics
    }

# Test
data = np.random.normal(loc=100, scale=15, size=50)

result = bootstrap_confidence_interval(data, statistic_func=np.mean)
print(f"Mean: {result['statistic']:.2f}")
print(f"95% CI: [{result['lower_bound']:.2f}, {result['upper_bound']:.2f}]")

# Also works for other statistics
median_result = bootstrap_confidence_interval(data, statistic_func=np.median)
print(f"\nMedian: {median_result['statistic']:.2f}")
print(f"95% CI: [{median_result['lower_bound']:.2f}, {median_result['upper_bound']:.2f}]")
```

**Application:**
- Model performance uncertainty
- A/B test result confidence
- Any statistic where analytical CI is hard

---

## Algorithm Implementation

### Problem 7: K-Means from Scratch 🟡

**Question:**
Implement K-Means clustering algorithm from scratch.

**Solution:**
```python
import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iters=100, random_state=42):
        """
        K-Means clustering implementation.

        Args:
            n_clusters: Number of clusters
            max_iters: Maximum iterations
            random_state: Random seed
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels = None

    def fit(self, X):
        """
        Fit K-Means to data.

        Args:
            X: Data matrix (n_samples, n_features)
        """
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape

        # Initialize centroids randomly from data points
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for iteration in range(self.max_iters):
            # Assign points to nearest centroid
            labels = self._assign_clusters(X)

            # Store old centroids to check convergence
            old_centroids = self.centroids.copy()

            # Update centroids
            self.centroids = self._update_centroids(X, labels)

            # Check convergence
            if np.allclose(old_centroids, self.centroids):
                print(f"Converged at iteration {iteration}")
                break

        self.labels = labels
        return self

    def _assign_clusters(self, X):
        """Assign each point to nearest centroid."""
        distances = np.zeros((X.shape[0], self.n_clusters))

        for i, centroid in enumerate(self.centroids):
            # Euclidean distance to each centroid
            distances[:, i] = np.linalg.norm(X - centroid, axis=1)

        # Return index of nearest centroid
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        """Update centroids as mean of assigned points."""
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))

        for i in range(self.n_clusters):
            # Get points assigned to this cluster
            cluster_points = X[labels == i]

            if len(cluster_points) > 0:
                # Update to mean of cluster points
                new_centroids[i] = cluster_points.mean(axis=0)
            else:
                # Keep old centroid if no points assigned (rare)
                new_centroids[i] = self.centroids[i]

        return new_centroids

    def predict(self, X):
        """Predict cluster labels for new data."""
        return self._assign_clusters(X)

    def inertia(self, X):
        """Calculate within-cluster sum of squares."""
        distances = np.zeros(len(X))
        labels = self._assign_clusters(X)

        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                distances[labels == i] = np.linalg.norm(
                    cluster_points - self.centroids[i], axis=1
                )**2

        return distances.sum()

# Test
from sklearn.datasets import make_blobs

X, y_true = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42)

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)

print(f"Inertia: {kmeans.inertia(X):.2f}")
print(f"Centroids:\n{kmeans.centroids}")

# Compare with sklearn
from sklearn.cluster import KMeans as SKLearnKMeans
sk_kmeans = SKLearnKMeans(n_clusters=4, random_state=42, n_init=10)
sk_kmeans.fit(X)
print(f"\nSKLearn Inertia: {sk_kmeans.inertia_:.2f}")
```

**Time Complexity:** O(n × k × i × d) where:
- n = samples, k = clusters, i = iterations, d = dimensions

**Interview Tip:** Explain the algorithm step-by-step:
1. Initialize centroids randomly
2. Assign points to nearest centroid
3. Update centroids as mean of assigned points
4. Repeat until convergence

---

### Problem 8: Decision Tree from Scratch 🔴

**Question:**
Implement a binary decision tree classifier from scratch with information gain.

**Solution:**
```python
import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # Feature index to split on
        self.threshold = threshold  # Threshold value
        self.left = left           # Left subtree
        self.right = right         # Right subtree
        self.value = value         # Class value if leaf node

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        """
        Decision Tree Classifier.

        Args:
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split node
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        """Build decision tree."""
        self.root = self._grow_tree(X, y, depth=0)
        return self

    def _grow_tree(self, X, y, depth):
        """Recursively grow decision tree."""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Stopping criteria
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            n_classes == 1):
            # Create leaf node with majority class
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Find best split
        best_feature, best_threshold = self._best_split(X, y, n_features)

        # If no good split found, create leaf
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Split data
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = ~left_indices

        # Recursively build left and right subtrees
        left_subtree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(best_feature, best_threshold, left_subtree, right_subtree)

    def _best_split(self, X, y, n_features):
        """Find best feature and threshold to split on."""
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                gain = self._information_gain(X, y, feature, threshold)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, X, y, feature, threshold):
        """Calculate information gain from split."""
        # Parent entropy
        parent_entropy = self._entropy(y)

        # Split data
        left_indices = X[:, feature] <= threshold
        right_indices = ~left_indices

        if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
            return 0

        # Calculate weighted average of children entropy
        n = len(y)
        n_left, n_right = len(y[left_indices]), len(y[right_indices])
        entropy_left = self._entropy(y[left_indices])
        entropy_right = self._entropy(y[right_indices])

        child_entropy = (n_left / n) * entropy_left + (n_right / n) * entropy_right

        # Information gain
        return parent_entropy - child_entropy

    def _entropy(self, y):
        """Calculate entropy."""
        counts = np.bincount(y)
        probabilities = counts / len(y)
        entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
        return entropy

    def _most_common_label(self, y):
        """Return most common label."""
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        """Predict class labels."""
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        """Traverse tree to make prediction."""
        # If leaf node, return value
        if node.value is not None:
            return node.value

        # Otherwise, traverse left or right
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

# Test
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree = DecisionTree(max_depth=5)
tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
```

**Key Concepts:**
- **Information Gain** = Parent Entropy - Weighted Child Entropy
- **Entropy** = -Σ p(x) log₂ p(x)
- **Stopping criteria**: Max depth, min samples, pure node

---

### Problem 9: Linear Regression with Gradient Descent 🟡

**Question:**
Implement linear regression using gradient descent (not closed-form solution).

**Solution:**
```python
import numpy as np

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Linear Regression with Gradient Descent.

        Args:
            learning_rate: Step size for gradient descent
            n_iterations: Number of training iterations
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []

    def fit(self, X, y):
        """
        Fit linear regression model.

        Args:
            X: Features (n_samples, n_features)
            y: Target (n_samples,)
        """
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for i in range(self.n_iterations):
            # Predictions
            y_pred = self.predict(X)

            # Calculate loss (MSE)
            loss = np.mean((y - y_pred) ** 2)
            self.losses.append(loss)

            # Calculate gradients
            dw = -(2 / n_samples) * X.T.dot(y - y_pred)
            db = -(2 / n_samples) * np.sum(y - y_pred)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict(self, X):
        """Make predictions."""
        return X.dot(self.weights) + self.bias

    def score(self, X, y):
        """Calculate R² score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2

# Test
np.random.seed(42)
X = np.random.randn(100, 1)
y = 3 * X.squeeze() + 2 + np.random.randn(100) * 0.5

model = LinearRegressionGD(learning_rate=0.01, n_iterations=1000)
model.fit(X, y)

print(f"Weights: {model.weights}")
print(f"Bias: {model.bias:.3f}")
print(f"R² Score: {model.score(X, y):.3f}")

# Compare with sklearn
from sklearn.linear_model import LinearRegression
sk_model = LinearRegression()
sk_model.fit(X, y)
print(f"\nSKLearn Weights: {sk_model.coef_}")
print(f"SKLearn Bias: {sk_model.intercept_:.3f}")
```

**Gradient Derivation:**
```
Loss = (1/n) Σ(y - ŷ)²
where ŷ = Xw + b

∂Loss/∂w = -(2/n) X^T(y - ŷ)
∂Loss/∂b = -(2/n) Σ(y - ŷ)
```

---

## Model Training Problems

### Problem 10: Cross-Validation from Scratch 🟡

**Question:**
Implement k-fold cross-validation from scratch.

**Solution:**
```python
import numpy as np

def k_fold_cross_validation(X, y, model_class, k=5, **model_params):
    """
    Perform k-fold cross-validation.

    Args:
        X: Features
        y: Labels
        model_class: Model class to instantiate
        k: Number of folds
        **model_params: Parameters to pass to model

    Returns:
        Dictionary with scores and trained models
    """
    n_samples = len(X)
    fold_size = n_samples // k

    # Shuffle indices
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    scores = []
    models = []

    for fold in range(k):
        # Create fold indices
        start_idx = fold * fold_size
        end_idx = start_idx + fold_size if fold < k - 1 else n_samples

        val_indices = indices[start_idx:end_idx]
        train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])

        # Split data
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]

        # Train model
        model = model_class(**model_params)
        model.fit(X_train, y_train)

        # Evaluate
        score = model.score(X_val, y_val)
        scores.append(score)
        models.append(model)

        print(f"Fold {fold + 1}/{k}: Score = {score:.4f}")

    return {
        'scores': np.array(scores),
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'models': models
    }

# Test
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X, y = iris.data, iris.target

results = k_fold_cross_validation(
    X, y,
    DecisionTreeClassifier,
    k=5,
    max_depth=3
)

print(f"\nMean Score: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
```

---

### Problem 11: Feature Selection with Correlation 🟢

**Question:**
Remove highly correlated features (correlation > threshold).

**Solution:**
```python
import pandas as pd
import numpy as np

def remove_correlated_features(df, threshold=0.9):
    """
    Remove highly correlated features.

    Args:
        df: DataFrame with features
        threshold: Correlation threshold above which to remove

    Returns:
        DataFrame with uncorrelated features
    """
    # Calculate correlation matrix
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # Drop features
    df_reduced = df.drop(columns=to_drop)

    print(f"Removed {len(to_drop)} features: {to_drop}")
    print(f"Remaining features: {list(df_reduced.columns)}")

    return df_reduced

# Test
np.random.seed(42)
df = pd.DataFrame({
    'feature1': np.random.randn(100),
    'feature2': np.random.randn(100),
    'feature3': np.random.randn(100),
})

# Create highly correlated feature
df['feature4'] = df['feature1'] + np.random.randn(100) * 0.1  # 0.95+ correlation with feature1

result = remove_correlated_features(df, threshold=0.8)
```

---

## Deep Learning Problems

### Problem 12: Implement Softmax 🟢

**Question:**
Implement softmax function with numerical stability.

**Solution:**
```python
import numpy as np

def softmax(x):
    """
    Compute softmax with numerical stability.

    Args:
        x: Input array (can be 1D or 2D)

    Returns:
        Softmax probabilities
    """
    # Subtract max for numerical stability (prevents overflow)
    # This doesn't change the result due to softmax property
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Test
logits = np.array([1.0, 2.0, 3.0])
probs = softmax(logits)
print(f"Logits: {logits}")
print(f"Probabilities: {probs}")
print(f"Sum: {probs.sum()}")  # Should be 1.0

# Batch processing
logits_batch = np.array([
    [1.0, 2.0, 3.0],
    [0.1, 0.2, 0.7]
])
probs_batch = softmax(logits_batch)
print(f"\nBatch probabilities:\n{probs_batch}")
print(f"Row sums: {probs_batch.sum(axis=1)}")  # Each should be 1.0

# Numerical stability test
large_logits = np.array([1000.0, 1001.0, 1002.0])
stable_probs = softmax(large_logits)
print(f"\nLarge logits softmax: {stable_probs}")  # Would overflow without stability trick
```

**Why subtract max?**
```
Naive softmax: e^1000 causes overflow
Stable: e^(1000-1002) = e^(-2) = 0.135 (no overflow)
Result is mathematically identical
```

---

### Problem 13: Implement Batch Normalization Forward Pass 🟡

**Question:**
Implement the forward pass of batch normalization.

**Solution:**
```python
import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Batch normalization forward pass.

    Args:
        x: Input data (batch_size, features)
        gamma: Scale parameter (features,)
        beta: Shift parameter (features,)
        eps: Small constant for numerical stability

    Returns:
        Normalized output, cache for backward pass
    """
    # Calculate batch statistics
    batch_mean = np.mean(x, axis=0)
    batch_var = np.var(x, axis=0)

    # Normalize
    x_normalized = (x - batch_mean) / np.sqrt(batch_var + eps)

    # Scale and shift
    out = gamma * x_normalized + beta

    # Cache for backward pass
    cache = (x, x_normalized, batch_mean, batch_var, gamma, beta, eps)

    return out, cache

# Test
np.random.seed(42)
batch_size = 4
features = 3

x = np.random.randn(batch_size, features) * 10 + 5  # Mean ~5, std ~10
gamma = np.ones(features)  # Scale
beta = np.zeros(features)  # Shift

print("Input:")
print(x)
print(f"Input mean: {x.mean(axis=0)}")
print(f"Input std: {x.std(axis=0)}")

out, cache = batch_norm_forward(x, gamma, beta)

print("\nOutput:")
print(out)
print(f"Output mean: {out.mean(axis=0)}")
print(f"Output std: {out.std(axis=0)}")
```

**Expected:** Output should have mean ≈ 0, std ≈ 1

---

## Production ML Problems

### Problem 14: Implement Exponential Moving Average 🟢

**Question:**
Implement exponential moving average for smoothing metrics in production monitoring.

**Solution:**
```python
class ExponentialMovingAverage:
    def __init__(self, alpha=0.9):
        """
        Exponential moving average.

        Args:
            alpha: Smoothing factor (0-1). Higher = more weight on recent values.
        """
        self.alpha = alpha
        self.value = None

    def update(self, new_value):
        """Update EMA with new value."""
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * new_value
        return self.value

    def get_value(self):
        """Get current EMA value."""
        return self.value

    def reset(self):
        """Reset EMA."""
        self.value = None

# Test - Simulating production metrics
ema = ExponentialMovingAverage(alpha=0.9)

# Simulate latency measurements
latencies = [50, 52, 48, 100, 55, 51, 53, 49]  # Spike at index 3

print("Latency  EMA")
print("-" * 20)
for latency in latencies:
    ema_value = ema.update(latency)
    print(f"{latency:7.1f}  {ema_value:7.2f}")
```

**Use Case:** Smooth noisy production metrics (latency, error rate, etc.)

---

### Problem 15: Detect Concept Drift with KS Test 🟡

**Question:**
Implement Kolmogorov-Smirnov test to detect if a feature distribution has changed (concept drift).

**Solution:**
```python
import numpy as np
from scipy.stats import ks_2samp

def detect_drift(reference_data, current_data, threshold=0.05):
    """
    Detect distribution drift using KS test.

    Args:
        reference_data: Reference distribution (e.g., training data)
        current_data: Current distribution (e.g., production data)
        threshold: P-value threshold for significance

    Returns:
        Dictionary with drift detection results
    """
    # Perform KS test
    statistic, p_value = ks_2samp(reference_data, current_data)

    # Drift detected if p-value < threshold
    drift_detected = p_value < threshold

    # Calculate distribution statistics
    ref_mean, ref_std = reference_data.mean(), reference_data.std()
    cur_mean, cur_std = current_data.mean(), current_data.std()

    mean_shift = abs(cur_mean - ref_mean) / ref_std if ref_std > 0 else 0

    return {
        'drift_detected': drift_detected,
        'ks_statistic': statistic,
        'p_value': p_value,
        'reference_mean': ref_mean,
        'current_mean': cur_mean,
        'mean_shift_std': mean_shift,
        'recommendation': 'Retrain model' if drift_detected else 'No action needed'
    }

# Test
np.random.seed(42)

# Reference distribution (training data)
reference = np.random.normal(loc=0, scale=1, size=1000)

# Scenario 1: No drift
current_no_drift = np.random.normal(loc=0, scale=1, size=200)
result1 = detect_drift(reference, current_no_drift)
print("Scenario 1: No Drift")
print(f"Drift detected: {result1['drift_detected']}")
print(f"P-value: {result1['p_value']:.4f}")
print(f"Recommendation: {result1['recommendation']}\n")

# Scenario 2: Mean shift (drift)
current_with_drift = np.random.normal(loc=0.5, scale=1, size=200)
result2 = detect_drift(reference, current_with_drift)
print("Scenario 2: Mean Shift (Drift)")
print(f"Drift detected: {result2['drift_detected']}")
print(f"P-value: {result2['p_value']:.4f}")
print(f"Mean shift: {result2['mean_shift_std']:.2f} std devs")
print(f"Recommendation: {result2['recommendation']}")
```

---

## Problem-Solving Strategies

### General Approach

1. **Clarify Requirements**
   - Ask about input/output format
   - Edge cases (empty data, single element, duplicates)
   - Performance constraints (time/space complexity)
   - Can use libraries or implement from scratch?

2. **Plan Before Coding**
   - Write pseudocode or explain approach verbally
   - Discuss time/space complexity
   - Consider edge cases

3. **Code Incrementally**
   - Start with simple case
   - Add complexity gradually
   - Test as you go

4. **Test Thoroughly**
   - Normal cases
   - Edge cases (empty, single element, all same)
   - Large inputs (if relevant)

5. **Optimize if Needed**
   - Only after working solution
   - Discuss trade-offs

### Common Patterns

**Data Manipulation:**
- Master pandas: groupby, merge, pivot, rolling
- NumPy vectorization instead of loops
- Handle missing data gracefully

**Statistical:**
- Know formulas for metrics
- Understand when to use each metric
- Bootstrap for confidence intervals

**Algorithm Implementation:**
- Start with simple version
- Add optimizations later
- Comment your code

**Deep Learning:**
- Understand forward pass first
- Backward pass is just derivatives
- Test with small examples

### Time Complexity Cheat Sheet

| Operation | Complexity | Example |
|-----------|------------|---------|
| Loop over n elements | O(n) | for x in data |
| Nested loop | O(n²) | for x in data: for y in data |
| Binary search | O(log n) | bisect |
| Sorting | O(n log n) | sorted() |
| Dictionary lookup | O(1) | dict[key] |
| Pandas groupby | O(n) | df.groupby() |

---

## Additional Practice Problems

### Easy Problems (Quick Warm-ups)
16. Calculate moving average
17. One-hot encode categorical variable
18. Normalize features (z-score)
19. Calculate cosine similarity
20. Implement train-test split

### Medium Problems
21. Implement Naive Bayes
22. Calculate AUC-ROC from scratch
23. Implement feature hashing
24. Cross-entropy loss with gradient
25. Implement early stopping

### Hard Problems
26. Implement Random Forest from scratch
27. Backpropagation for 2-layer network
28. Implement gradient boosting
29. Custom loss function with PyTorch
30. Implement attention mechanism

---

## Interview Tips

### Dos ✅
- **Think aloud** - Explain your reasoning
- **Ask clarifying questions** - Better than wrong assumptions
- **Start simple** - Get working solution first
- **Test your code** - Walk through with example
- **Discuss complexity** - Time and space
- **Handle edge cases** - Empty input, single element, etc.

### Don'ts ❌
- **Don't code immediately** - Plan first
- **Don't ignore hints** - Interviewer is trying to help
- **Don't give up** - Even partial solution shows thinking
- **Don't forget edge cases** - They're often part of evaluation
- **Don't optimize prematurely** - Correct first, fast second

### Common Mistakes
1. **Not reading problem carefully** - Miss important details
2. **Hardcoding values** - Use variables/parameters
3. **Poor variable names** - Use descriptive names
4. **No error handling** - Check inputs, handle edge cases
5. **Inefficient loops** - Vectorize when possible

---

## Resources for More Practice

### Online Platforms
- **LeetCode** - ML tag problems
- **HackerRank** - ML section
- **Kaggle** - Competition kernels
- **InterviewQuery** - ML-specific interviews

### Books
- "Introduction to Statistical Learning" - Practical ML
- "Elements of Statistical Learning" - Theory
- "Hands-On Machine Learning" - Sklearn/TensorFlow

### GitHub Repositories
- This repository's notebooks
- ML interview preparation repos
- Algorithm implementations

---

**Document Version:** 1.0
**Last Updated:** October 21, 2025
**Maintained by:** Benjamin Hu

**Total Problems:** 30+
**Difficulty Distribution:** 15 Easy, 12 Medium, 8 Hard
**Topics Covered:** Data manipulation, statistics, algorithms, deep learning, production

**Perfect for:**
- ML interview preparation
- Coding skills practice
- Algorithm understanding
- Production ML scenarios

---

**Happy Coding! 💻**
