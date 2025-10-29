# 💻 ML/AI Code Templates Library

## Production-Ready Code Templates for Common ML Tasks

This library provides copy-paste ready code templates for the most common machine learning tasks. Each template is production-tested and includes best practices.

---

## 📋 Table of Contents

1. [Data Processing](#data-processing)
2. [Model Training](#model-training)
3. [Model Evaluation](#model-evaluation)
4. [Model Deployment](#model-deployment)
5. [Computer Vision](#computer-vision)
6. [Natural Language Processing](#nlp)
7. [Time Series](#time-series)
8. [Utilities](#utilities)

---

## 📊 Data Processing

### Load and Explore Dataset
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
plt.style.use('seaborn-v0_8-darkgrid')

# Load data
df = pd.read_csv('data.csv')

# Quick exploration
print("=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)
print(f"Shape: {df.shape}")
print(f"\nData Types:\n{df.dtypes}\n")
print(f"Missing Values:\n{df.isnull().sum()}\n")
print(f"Numeric Summary:\n{df.describe()}\n")
print(f"First 5 rows:\n{df.head()}\n")

# Memory usage
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Check for duplicates
print(f"Duplicate rows: {df.duplicated().sum()}")

# Class distribution (for classification)
if 'target' in df.columns:
    print(f"\nClass distribution:\n{df['target'].value_counts()}")
    print(f"Class proportions:\n{df['target'].value_counts(normalize=True)}")
```

### Data Cleaning Pipeline
```python
class DataCleaner:
    """Comprehensive data cleaning pipeline"""

    def __init__(self, df):
        self.df = df.copy()
        self.cleaning_report = {}

    def remove_duplicates(self):
        """Remove duplicate rows"""
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        after = len(self.df)
        self.cleaning_report['duplicates_removed'] = before - after
        return self

    def handle_missing_values(self, strategy='auto'):
        """
        Handle missing values

        Strategies:
        - 'auto': Impute numeric with median, categorical with mode
        - 'drop': Drop rows with missing values
        - 'forward_fill': Forward fill
        - dict: Custom strategy per column
        """
        missing_before = self.df.isnull().sum().sum()

        if strategy == 'auto':
            # Numeric columns: impute with median
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = self.df[numeric_cols].fillna(
                self.df[numeric_cols].median()
            )

            # Categorical columns: impute with mode
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

        elif strategy == 'drop':
            self.df = self.df.dropna()

        elif strategy == 'forward_fill':
            self.df = self.df.fillna(method='ffill')

        elif isinstance(strategy, dict):
            for col, method in strategy.items():
                if method == 'median':
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                elif method == 'mean':
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
                elif method == 'mode':
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
                elif isinstance(method, (int, float, str)):
                    self.df[col] = self.df[col].fillna(method)

        missing_after = self.df.isnull().sum().sum()
        self.cleaning_report['missing_values_handled'] = missing_before - missing_after
        return self

    def remove_outliers(self, columns=None, method='iqr', threshold=1.5):
        """
        Remove outliers using IQR or Z-score method

        Parameters:
        -----------
        columns : list
            Columns to check for outliers (None = all numeric)
        method : str
            'iqr' or 'zscore'
        threshold : float
            IQR multiplier (1.5 default) or Z-score threshold (3.0 default)
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns

        before = len(self.df)

        if method == 'iqr':
            for col in columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                self.df = self.df[
                    (self.df[col] >= lower_bound) &
                    (self.df[col] <= upper_bound)
                ]

        elif method == 'zscore':
            from scipy.stats import zscore
            for col in columns:
                z_scores = np.abs(zscore(self.df[col].dropna()))
                self.df = self.df[z_scores < threshold]

        after = len(self.df)
        self.cleaning_report['outliers_removed'] = before - after
        return self

    def encode_categorical(self, columns=None, method='label'):
        """
        Encode categorical variables

        Parameters:
        -----------
        method : str
            'label': Label encoding
            'onehot': One-hot encoding
            'target': Target encoding (requires target column)
        """
        if columns is None:
            columns = self.df.select_dtypes(include=['object']).columns

        if method == 'label':
            from sklearn.preprocessing import LabelEncoder
            for col in columns:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))

        elif method == 'onehot':
            self.df = pd.get_dummies(self.df, columns=columns, drop_first=True)

        self.cleaning_report['categorical_encoded'] = len(columns)
        return self

    def get_clean_data(self):
        """Return cleaned dataframe and report"""
        return self.df, self.cleaning_report


# Usage
cleaner = DataCleaner(df)
df_clean, report = (cleaner
                    .remove_duplicates()
                    .handle_missing_values(strategy='auto')
                    .remove_outliers(method='iqr', threshold=1.5)
                    .encode_categorical(method='label')
                    .get_clean_data())

print("Cleaning Report:", report)
```

### Feature Engineering
```python
class FeatureEngineer:
    """Create new features from existing data"""

    def __init__(self, df):
        self.df = df.copy()
        self.new_features = []

    def create_datetime_features(self, date_column):
        """Extract features from datetime column"""
        self.df[date_column] = pd.to_datetime(self.df[date_column])

        # Extract components
        self.df[f'{date_column}_year'] = self.df[date_column].dt.year
        self.df[f'{date_column}_month'] = self.df[date_column].dt.month
        self.df[f'{date_column}_day'] = self.df[date_column].dt.day
        self.df[f'{date_column}_dayofweek'] = self.df[date_column].dt.dayofweek
        self.df[f'{date_column}_hour'] = self.df[date_column].dt.hour
        self.df[f'{date_column}_quarter'] = self.df[date_column].dt.quarter

        # Is weekend
        self.df[f'{date_column}_is_weekend'] = (
            self.df[f'{date_column}_dayofweek'].isin([5, 6]).astype(int)
        )

        self.new_features.extend([
            f'{date_column}_year', f'{date_column}_month',
            f'{date_column}_day', f'{date_column}_dayofweek',
            f'{date_column}_hour', f'{date_column}_quarter',
            f'{date_column}_is_weekend'
        ])

        return self

    def create_polynomial_features(self, columns, degree=2):
        """Create polynomial and interaction features"""
        from sklearn.preprocessing import PolynomialFeatures

        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_features = poly.fit_transform(self.df[columns])

        # Get feature names
        feature_names = poly.get_feature_names_out(columns)

        # Add to dataframe
        for i, name in enumerate(feature_names):
            if name not in columns:  # Skip original features
                self.df[name] = poly_features[:, i]
                self.new_features.append(name)

        return self

    def create_aggregation_features(self, group_by_col, agg_col, agg_funcs=['mean', 'std', 'min', 'max']):
        """Create aggregation features"""
        for func in agg_funcs:
            feature_name = f'{agg_col}_{func}_by_{group_by_col}'
            self.df[feature_name] = self.df.groupby(group_by_col)[agg_col].transform(func)
            self.new_features.append(feature_name)

        return self

    def create_binned_features(self, column, bins=5, labels=None):
        """Create binned categorical features"""
        feature_name = f'{column}_binned'
        self.df[feature_name] = pd.cut(self.df[column], bins=bins, labels=labels)
        self.new_features.append(feature_name)
        return self

    def get_features(self):
        """Return dataframe with new features"""
        return self.df, self.new_features


# Usage
engineer = FeatureEngineer(df)
df_engineered, new_features = (engineer
                               .create_datetime_features('timestamp')
                               .create_polynomial_features(['feature1', 'feature2'], degree=2)
                               .create_aggregation_features('category', 'value', ['mean', 'std'])
                               .get_features())

print(f"Created {len(new_features)} new features")
```

### Train-Test Split with Stratification
```python
from sklearn.model_selection import train_test_split

def split_data(X, y, test_size=0.2, val_size=0.1, stratify=True, random_state=42):
    """
    Split data into train, validation, and test sets

    Returns:
    --------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First split: separate test set
    stratify_col = y if stratify else None

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=stratify_col,
        random_state=random_state
    )

    # Second split: separate validation set
    if val_size > 0:
        val_size_adjusted = val_size / (1 - test_size)
        stratify_col_temp = y_temp if stratify else None

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            stratify=stratify_col_temp,
            random_state=random_state
        )

        print(f"Train set: {X_train.shape}")
        print(f"Val set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")

        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        print(f"Train set: {X_temp.shape}")
        print(f"Test set: {X_test.shape}")

        return X_temp, X_test, y_temp, y_test


# Usage
X = df.drop('target', axis=1)
y = df['target']

X_train, X_val, X_test, y_train, y_val, y_test = split_data(
    X, y,
    test_size=0.2,
    val_size=0.1,
    stratify=True
)
```

---

## 🤖 Model Training

### scikit-learn Template
```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV

def train_sklearn_model(X_train, y_train, X_val, y_val, model_type='random_forest'):
    """
    Train scikit-learn model with best practices

    Parameters:
    -----------
    model_type : str
        'logistic', 'random_forest', 'gradient_boosting', 'svm'
    """
    # Define model and hyperparameters
    if model_type == 'logistic':
        model = LogisticRegression(random_state=42, max_iter=1000)
        param_grid = {
            'model__C': [0.01, 0.1, 1, 10],
            'model__penalty': ['l1', 'l2'],
            'model__solver': ['liblinear', 'saga']
        }

    elif model_type == 'random_forest':
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        param_grid = {
            'model__n_estimators': [100, 200, 300],
            'model__max_depth': [10, 20, 30, None],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]
        }

    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            'model__n_estimators': [100, 200],
            'model__learning_rate': [0.01, 0.1, 0.3],
            'model__max_depth': [3, 5, 7],
            'model__subsample': [0.8, 1.0]
        }

    elif model_type == 'svm':
        model = SVC(random_state=42)
        param_grid = {
            'model__C': [0.1, 1, 10],
            'model__kernel': ['rbf', 'linear'],
            'model__gamma': ['scale', 'auto']
        }

    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    # Hyperparameter tuning
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    # Train
    print(f"Training {model_type}...")
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_

    # Evaluate
    train_score = best_model.score(X_train, y_train)
    val_score = best_model.score(X_val, y_val)

    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Train accuracy: {train_score:.4f}")
    print(f"Val accuracy: {val_score:.4f}")

    return best_model


# Usage
model = train_sklearn_model(X_train, y_train, X_val, y_val, model_type='random_forest')
```

### PyTorch Neural Network Template
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Custom Dataset
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Define Model
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], num_classes=2, dropout=0.3):
        super().__init__()

        layers = []

        # Input layer
        prev_dim = input_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Training function
def train_pytorch_model(
    X_train, y_train,
    X_val, y_val,
    input_dim,
    num_classes,
    hidden_dims=[128, 64],
    epochs=50,
    batch_size=32,
    learning_rate=0.001,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """Train PyTorch neural network"""

    # Create datasets
    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = NeuralNetwork(input_dim, hidden_dims, num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 10

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        val_loss /= len(val_loader)
        val_acc = correct / total

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Acc: {val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))

    return model, history


# Usage
model, history = train_pytorch_model(
    X_train, y_train,
    X_val, y_val,
    input_dim=X_train.shape[1],
    num_classes=len(np.unique(y_train)),
    hidden_dims=[128, 64, 32],
    epochs=100,
    batch_size=64,
    learning_rate=0.001
)

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training History - Loss')

plt.subplot(1, 2, 2)
plt.plot(history['val_acc'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training History - Accuracy')

plt.tight_layout()
plt.show()
```

### XGBoost Template
```python
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score

def train_xgboost(X_train, y_train, X_val, y_val, task='classification'):
    """
    Train XGBoost model with best practices

    Parameters:
    -----------
    task : str
        'classification' or 'regression'
    """
    # Create DMatrix (XGBoost's internal data structure)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Parameters
    if task == 'classification':
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'seed': 42
        }
    else:  # regression
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42
        }

    # Training with early stopping
    evals = [(dtrain, 'train'), (dval, 'eval')]
    evals_result = {}

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=evals,
        early_stopping_rounds=50,
        evals_result=evals_result,
        verbose_eval=100
    )

    # Predictions
    y_pred = model.predict(dval)

    if task == 'classification':
        y_pred_class = (y_pred > 0.5).astype(int)
        accuracy = accuracy_score(y_val, y_pred_class)
        auc = roc_auc_score(y_val, y_pred)
        print(f"\nValidation Accuracy: {accuracy:.4f}")
        print(f"Validation AUC: {auc:.4f}")
    else:
        from sklearn.metrics import mean_squared_error, r2_score
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        r2 = r2_score(y_val, y_pred)
        print(f"\nValidation RMSE: {rmse:.4f}")
        print(f"Validation R²: {r2:.4f}")

    # Feature importance
    importance = model.get_score(importance_type='gain')
    importance_df = pd.DataFrame({
        'feature': list(importance.keys()),
        'importance': list(importance.values())
    }).sort_values('importance', ascending=False)

    print(f"\nTop 10 Features:")
    print(importance_df.head(10))

    # Plot training history
    epochs = len(evals_result['train'][params['eval_metric']])
    x_axis = range(0, epochs)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_axis, evals_result['train'][params['eval_metric']], label='Train')
    ax.plot(x_axis, evals_result['eval'][params['eval_metric']], label='Validation')
    ax.legend()
    ax.set_ylabel(params['eval_metric'].upper())
    ax.set_xlabel('Boosting Round')
    ax.set_title('XGBoost Training History')
    plt.show()

    return model, importance_df


# Usage
model, feature_importance = train_xgboost(
    X_train, y_train,
    X_val, y_val,
    task='classification'
)
```

---

## 📈 Model Evaluation

### Classification Metrics
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_classification_model(model, X_test, y_test, class_names=None):
    """
    Comprehensive classification model evaluation

    Prints metrics and plots confusion matrix, ROC curve, PR curve
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    print("=" * 80)
    print("CLASSIFICATION METRICS")
    print("=" * 80)

    # Basic metrics
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='binary'):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, average='binary'):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred, average='binary'):.4f}")

    if y_pred_proba is not None:
        print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(15, 5))

    # Plot 1: Confusion Matrix
    plt.subplot(1, 3, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')

    if y_pred_proba is not None:
        # Plot 2: ROC Curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        plt.subplot(1, 3, 2)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")

        # Plot 3: Precision-Recall Curve
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

        plt.subplot(1, 3, 3)
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

    plt.tight_layout()
    plt.show()

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary'),
        'recall': recall_score(y_test, y_pred, average='binary'),
        'f1': f1_score(y_test, y_pred, average='binary'),
        'roc_auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None,
        'confusion_matrix': cm
    }


# Usage
metrics = evaluate_classification_model(
    model, X_test, y_test,
    class_names=['Negative', 'Positive']
)
```

### Regression Metrics
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def evaluate_regression_model(model, X_test, y_test):
    """
    Comprehensive regression model evaluation
    """
    # Predictions
    y_pred = model.predict(X_test)

    print("=" * 80)
    print("REGRESSION METRICS")
    print("=" * 80)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Residuals
    residuals = y_test - y_pred

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Predicted vs Actual
    axes[0, 0].scatter(y_test, y_pred, alpha=0.5)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title('Predicted vs Actual')
    axes[0, 0].grid(True)

    # Plot 2: Residuals vs Predicted
    axes[0, 1].scatter(y_pred, residuals, alpha=0.5)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].grid(True)

    # Plot 3: Residuals Distribution
    axes[1, 0].hist(residuals, bins=50, edgecolor='black')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Residuals Distribution')
    axes[1, 0].grid(True)

    # Plot 4: Q-Q Plot
    from scipy.stats import probplot
    probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


# Usage
metrics = evaluate_regression_model(model, X_test, y_test)
```

### Cross-Validation
```python
from sklearn.model_selection import cross_val_score, cross_validate
import numpy as np

def perform_cross_validation(model, X, y, cv=5, scoring=None):
    """
    Perform k-fold cross-validation with multiple metrics

    Parameters:
    -----------
    model : estimator
        scikit-learn model
    X, y : arrays
        Features and target
    cv : int
        Number of folds
    scoring : list of str
        Metrics to compute
    """
    if scoring is None:
        # Default scoring metrics
        from sklearn.metrics import make_scorer, f1_score
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': make_scorer(f1_score)
        }

    # Perform cross-validation
    cv_results = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )

    print("=" * 80)
    print(f"{cv}-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 80)

    # Print results for each metric
    for metric_name in scoring.keys():
        train_scores = cv_results[f'train_{metric_name}']
        test_scores = cv_results[f'test_{metric_name}']

        print(f"\n{metric_name.upper()}:")
        print(f"  Train: {train_scores.mean():.4f} (+/- {train_scores.std():.4f})")
        print(f"  Test:  {test_scores.mean():.4f} (+/- {test_scores.std():.4f})")

    # Check for overfitting
    train_acc = cv_results['train_accuracy'].mean()
    test_acc = cv_results['test_accuracy'].mean()
    gap = train_acc - test_acc

    print(f"\nTrain-Test Gap: {gap:.4f}")
    if gap > 0.1:
        print("⚠️ Warning: Significant overfitting detected!")
    else:
        print("✅ No significant overfitting")

    return cv_results


# Usage
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
cv_results = perform_cross_validation(model, X, y, cv=5)
```

---

## 🚀 Model Deployment

### Save and Load Models
```python
import joblib
import pickle

# scikit-learn models
def save_sklearn_model(model, filepath, include_scaler=None):
    """Save scikit-learn model and optional preprocessor"""
    model_data = {
        'model': model,
        'scaler': include_scaler,
        'timestamp': pd.Timestamp.now(),
        'sklearn_version': sklearn.__version__
    }
    joblib.dump(model_data, filepath)
    print(f"Model saved to {filepath}")


def load_sklearn_model(filepath):
    """Load scikit-learn model"""
    model_data = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    print(f"Trained on: {model_data['timestamp']}")
    return model_data['model'], model_data.get('scaler')


# Usage
save_sklearn_model(model, 'model.joblib', include_scaler=scaler)
loaded_model, loaded_scaler = load_sklearn_model('model.joblib')


# PyTorch models
def save_pytorch_model(model, filepath, metadata=None):
    """Save PyTorch model"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'metadata': metadata or {},
        'timestamp': pd.Timestamp.now()
    }, filepath)
    print(f"Model saved to {filepath}")


def load_pytorch_model(model_class, filepath, **model_kwargs):
    """Load PyTorch model"""
    checkpoint = torch.load(filepath)
    model = model_class(**model_kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded from {filepath}")
    return model


# Usage
save_pytorch_model(model, 'model.pth', metadata={'accuracy': 0.95})
loaded_model = load_pytorch_model(NeuralNetwork, 'model.pth', input_dim=10, num_classes=2)
```

### FastAPI Deployment
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib

# Initialize FastAPI app
app = FastAPI(title="ML Model API", version="1.0")

# Load model at startup
model, scaler = load_sklearn_model('model.joblib')

# Define request/response models
class PredictionRequest(BaseModel):
    features: list[float]

    class Config:
        schema_extra = {
            "example": {
                "features": [5.1, 3.5, 1.4, 0.2, 1.0, 2.5]
            }
        }

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": pd.Timestamp.now().isoformat()
    }

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Validate input
        if len(request.features) != scaler.n_features_in_:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {scaler.n_features_in_} features, got {len(request.features)}"
            )

        # Preprocess
        features = np.array(request.features).reshape(1, -1)
        features_scaled = scaler.transform(features)

        # Predict
        prediction = int(model.predict(features_scaled)[0])
        probability = float(model.predict_proba(features_scaled)[0][prediction])

        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            model_version="1.0"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Batch prediction endpoint
@app.post("/predict_batch")
async def predict_batch(features_list: list[list[float]]):
    try:
        features = np.array(features_list)
        features_scaled = scaler.transform(features)

        predictions = model.predict(features_scaled).tolist()
        probabilities = model.predict_proba(features_scaled).tolist()

        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "count": len(predictions)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run with: uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Docker Deployment
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and code
COPY model.joblib .
COPY app.py .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  ml-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/model.joblib
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
```

```bash
# Build and run
docker-compose up --build

# Test API
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2, 1.0, 2.5]}'
```

---

## 🖼️ Computer Vision

### Image Classification (PyTorch)
```python
import torch
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader

# Data augmentation and preprocessing
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset
train_dataset = torchvision.datasets.ImageFolder('data/train', transform=train_transform)
val_dataset = torchvision.datasets.ImageFolder('data/val', transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Transfer learning with ResNet50
def create_image_classifier(num_classes, pretrained=True):
    """Create image classification model using transfer learning"""
    model = models.resnet50(pretrained=pretrained)

    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )

    return model


# Training function
def train_image_classifier(model, train_loader, val_loader, epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_acc = correct / total

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}")

        scheduler.step()

    return model


# Usage
model = create_image_classifier(num_classes=10, pretrained=True)
trained_model = train_image_classifier(model, train_loader, val_loader, epochs=10)
```

---

## 📝 NLP

### Text Classification with Transformers
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Prepare data
def prepare_text_classification_data(texts, labels, tokenizer, max_length=128):
    """Tokenize texts for transformer models"""
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt'
    )

    dataset = Dataset.from_dict({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': torch.tensor(labels)
    })

    return dataset


# Train transformer model
def train_text_classifier(train_texts, train_labels, val_texts, val_labels, model_name='bert-base-uncased', num_labels=2):
    """Fine-tune transformer for text classification"""

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # Prepare datasets
    train_dataset = prepare_text_classification_data(train_texts, train_labels, tokenizer)
    val_dataset = prepare_text_classification_data(val_texts, val_labels, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train
    trainer.train()

    return model, tokenizer


# Usage
model, tokenizer = train_text_classifier(
    train_texts=['text1', 'text2', ...],
    train_labels=[0, 1, ...],
    val_texts=['val_text1', ...],
    val_labels=[0, ...],
    model_name='distilbert-base-uncased',
    num_labels=2
)

# Predict
def predict_text(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    probability = torch.softmax(outputs.logits, dim=1).max().item()
    return prediction, probability
```

---

## ⏰ Time Series

### Time Series Forecasting
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Prepare time series data
def create_sequences(data, seq_length):
    """Create sequences for time series prediction"""
    X, y = [], []

    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])

    return np.array(X), np.array(y)


# LSTM Model
import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)

        # Take last time step
        last_time_step = lstm_out[:, -1, :]

        # Fully connected
        output = self.fc(last_time_step)

        return output


# Train time series model
def train_time_series_model(data, seq_length=10, epochs=50):
    """Train LSTM for time series forecasting"""

    # Normalize data
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data.reshape(-1, 1)).flatten()

    # Create sequences
    X, y = create_sequences(data_normalized, seq_length)

    # Train/test split (temporal order!)
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train).unsqueeze(-1)  # Add feature dimension
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test).unsqueeze(-1)
    y_test = torch.FloatTensor(y_test)

    # Model
    model = LSTMForecaster(input_dim=1, hidden_dim=50, num_layers=2, output_dim=1)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(X_train).squeeze()
        loss = criterion(outputs, y_train)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test).squeeze()
                test_loss = criterion(test_outputs, y_test)

            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

    return model, scaler


# Forecast future values
def forecast(model, last_sequence, scaler, steps=10):
    """Forecast future values"""
    model.eval()
    predictions = []

    current_seq = torch.FloatTensor(last_sequence).unsqueeze(0).unsqueeze(-1)

    with torch.no_grad():
        for _ in range(steps):
            pred = model(current_seq).item()
            predictions.append(pred)

            # Update sequence
            current_seq = torch.cat([current_seq[:, 1:, :],
                                      torch.FloatTensor([[[pred]]])], dim=1)

    # Inverse transform
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    return predictions
```

---

## 🛠️ Utilities

### Experiment Tracking
```python
import mlflow

def track_experiment(experiment_name, model, params, metrics):
    """Track ML experiment with MLflow"""

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(params)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Log artifacts (plots, etc.)
        # mlflow.log_artifact("confusion_matrix.png")

        print(f"Experiment logged to MLflow")


# Usage
track_experiment(
    experiment_name="credit_risk_model",
    model=trained_model,
    params={'n_estimators': 100, 'max_depth': 10},
    metrics={'accuracy': 0.95, 'auc': 0.98}
)
```

### Model Comparison
```python
from sklearn.model_selection import cross_val_score
import pandas as pd

def compare_models(models, X, y, cv=5):
    """Compare multiple models using cross-validation"""

    results = []

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

        results.append({
            'Model': name,
            'Mean Accuracy': scores.mean(),
            'Std Accuracy': scores.std(),
            'Min Accuracy': scores.min(),
            'Max Accuracy': scores.max()
        })

    results_df = pd.DataFrame(results).sort_values('Mean Accuracy', ascending=False)

    print("\nModel Comparison Results:")
    print(results_df.to_string(index=False))

    return results_df


# Usage
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100)
}

comparison = compare_models(models, X, y, cv=5)
```

---

## 🎓 Best Practices Summary

1. **Always split data properly**: Train/Val/Test with stratification
2. **Scale features**: Normalize or standardize before training
3. **Handle missing values**: Impute or drop appropriately
4. **Use cross-validation**: For robust model evaluation
5. **Track experiments**: Use MLflow or similar tools
6. **Save models properly**: Include metadata and versioning
7. **Monitor performance**: Both training and production metrics
8. **Document everything**: Parameters, metrics, decisions

---

*These templates provide a solid foundation for most ML projects. Adapt and extend them based on your specific needs!*
