# ML Practical Templates & Checklists

**Complete Collection of Production-Ready Templates for ML Projects**

> Copy-paste templates and checklists for every stage of the ML lifecycle

---

## Table of Contents

1. [Project Initiation Templates](#1-project-initiation-templates)
2. [Data Analysis Checklist](#2-data-analysis-checklist)
3. [Feature Engineering Template](#3-feature-engineering-template)
4. [Model Development Checklist](#4-model-development-checklist)
5. [Experiment Tracking Template](#5-experiment-tracking-template)
6. [Model Evaluation Template](#6-model-evaluation-template)
7. [A/B Testing Plan Template](#7-ab-testing-plan-template)
8. [Production Deployment Checklist](#8-production-deployment-checklist)
9. [Monitoring & Alerting Template](#9-monitoring--alerting-template)
10. [Incident Response Template](#10-incident-response-template)
11. [ML System Design Template](#11-ml-system-design-template)
12. [Interview Preparation Checklist](#12-interview-preparation-checklist)

---

## 1. Project Initiation Templates

### 1.1 ML Project Charter Template

```markdown
# ML Project Charter: [Project Name]

## Executive Summary
- **Project Name:** [Name]
- **Project Owner:** [Name]
- **Start Date:** [Date]
- **Expected Completion:** [Date]
- **Budget:** [Amount]
- **Team Size:** [Number]

## Business Problem
### Problem Statement
[Clear description of the business problem in 2-3 sentences]

### Current State
- Current process: [Description]
- Current metrics: [Baseline metrics]
- Pain points: [List of issues]

### Desired State
- Target metrics: [What success looks like]
- Expected impact: [Business impact]
- Timeline: [When results are needed]

## Success Criteria
### Business Metrics
- Primary: [e.g., Revenue increase by 15%]
- Secondary: [e.g., Customer satisfaction +10 NPS points]
- Constraint: [e.g., Must maintain <100ms latency]

### ML Metrics
- Primary: [e.g., AUC-ROC > 0.85]
- Secondary: [e.g., Precision > 0.90 at 50% recall]
- Constraint: [e.g., False positive rate < 1%]

## Scope
### In Scope
- [ ] Data collection from [sources]
- [ ] Model development for [use case]
- [ ] Integration with [systems]
- [ ] A/B testing framework
- [ ] Production deployment

### Out of Scope
- [ ] [What's not included]
- [ ] [Future considerations]

## Data Requirements
### Data Sources
| Source | Type | Volume | Availability | Quality |
|--------|------|--------|--------------|---------|
| [DB1] | [Structured] | [1M rows] | [Real-time] | [Good] |
| [API] | [JSON] | [100K/day] | [Batch] | [Needs cleaning] |

### Data Labeling
- Labeling required: [Yes/No]
- Label source: [Human annotators/Existing labels/Weak supervision]
- Quality requirements: [Inter-annotator agreement > 90%]

## Technical Requirements
### Infrastructure
- Compute: [GPU/CPU requirements]
- Storage: [Data volume]
- Latency: [Max acceptable latency]
- Throughput: [Requests per second]

### Integration Points
- Input systems: [List]
- Output systems: [List]
- APIs: [List]

## Team & Resources
### Team Composition
- ML Engineer: [Name] - [Responsibilities]
- Data Scientist: [Name] - [Responsibilities]
- Data Engineer: [Name] - [Responsibilities]
- Product Manager: [Name] - [Responsibilities]
- Subject Matter Expert: [Name] - [Responsibilities]

### External Resources
- Cloud budget: [$X/month]
- Annotation budget: [$X]
- Third-party tools: [List]

## Timeline & Milestones
| Milestone | Description | Date | Owner |
|-----------|-------------|------|-------|
| Data Collection | [Details] | [Date] | [Name] |
| EDA Complete | [Details] | [Date] | [Name] |
| Baseline Model | [Details] | [Date] | [Name] |
| Production Model | [Details] | [Date] | [Name] |
| A/B Test | [Details] | [Date] | [Name] |
| Production Launch | [Details] | [Date] | [Name] |

## Risks & Mitigation
| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| [Data quality issues] | [High] | [Medium] | [Implement data validation] |
| [Model accuracy below target] | [High] | [Low] | [Have fallback simpler model] |
| [Integration delays] | [Medium] | [High] | [Start integration early] |

## Budget
### Development Phase
- Data acquisition: [$X]
- Compute resources: [$X]
- Tools & licenses: [$X]
- Personnel: [$X]
- **Total Development:** [$X]

### Production Phase (Annual)
- Inference compute: [$X/year]
- Storage: [$X/year]
- Monitoring: [$X/year]
- Maintenance: [$X/year]
- **Total Annual:** [$X/year]

### ROI Calculation
- Expected benefit: [$X/year]
- Total cost (Year 1): [$X]
- ROI: [X%]
- Payback period: [X months]

## Sign-off
- [ ] Business Sponsor: [Name] - [Date]
- [ ] Technical Lead: [Name] - [Date]
- [ ] Product Owner: [Name] - [Date]
```

### 1.2 Quick Project Kickoff Template (Lean Version)

```markdown
# [Project Name] - Quick Kickoff

**Problem:** [1 sentence]
**Goal:** [1 sentence with metric]
**Timeline:** [X weeks]
**Team:** [Names]

## Week 1: Data & EDA
- [ ] Access data sources
- [ ] EDA notebook
- [ ] Data quality report

## Week 2-3: Baseline Model
- [ ] Feature engineering
- [ ] Train baseline (simple model)
- [ ] Evaluation on holdout set

## Week 4-6: Iteration
- [ ] Try advanced models
- [ ] Hyperparameter tuning
- [ ] Feature selection

## Week 7-8: Production
- [ ] Model deployment
- [ ] Monitoring setup
- [ ] A/B test

**Success Metric:** [Metric] from [baseline] to [target]
```

---

## 2. Data Analysis Checklist

### 2.1 Initial Data Exploration

```markdown
# Data Exploration Checklist

## Dataset Overview
- [ ] Load data successfully
- [ ] Check data shape: [X rows, Y columns]
- [ ] Verify data types (numerical/categorical/datetime)
- [ ] Check memory usage
- [ ] Identify target variable(s)

## Data Quality
### Missing Values
- [ ] Calculate missing percentage per column
- [ ] Identify patterns in missingness (MAR/MCAR/MNAR)
- [ ] Document columns with >50% missing
- [ ] Decision: Drop/Impute/Flag missing values

### Duplicates
- [ ] Check for duplicate rows (exact)
- [ ] Check for duplicate IDs
- [ ] Identify near-duplicates (fuzzy matching)
- [ ] Decision: Keep/Remove/Investigate duplicates

### Outliers
- [ ] Visual inspection (box plots, histograms)
- [ ] Statistical detection (Z-score, IQR)
- [ ] Domain validation (physically impossible values)
- [ ] Decision: Keep/Remove/Cap outliers

### Data Types & Consistency
- [ ] Verify numerical columns are numeric type
- [ ] Check categorical cardinality
- [ ] Validate datetime formats
- [ ] Check text encoding (UTF-8, ASCII)
- [ ] Verify boolean columns

## Univariate Analysis
### Numerical Features
- [ ] Distribution plots (histograms)
- [ ] Summary statistics (mean, median, std, min, max)
- [ ] Identify skewness
- [ ] Check for constants or near-constants

### Categorical Features
- [ ] Value counts
- [ ] Identify high-cardinality features (>100 unique)
- [ ] Check for typos/inconsistencies
- [ ] Document encoding needs

### Target Variable
- [ ] Distribution analysis
- [ ] Class balance (for classification)
- [ ] Range and outliers (for regression)
- [ ] Temporal trends

## Bivariate Analysis
### Feature-Target Relationships
- [ ] Correlation with target (numerical features)
- [ ] Group statistics by category (categorical features)
- [ ] Visual analysis (scatter plots, box plots)
- [ ] Statistical tests (t-test, chi-square)

### Feature-Feature Relationships
- [ ] Correlation matrix
- [ ] Identify multicollinearity (VIF > 10)
- [ ] Check feature interactions
- [ ] Domain knowledge validation

## Temporal Analysis (if applicable)
- [ ] Check data collection period
- [ ] Identify temporal trends
- [ ] Check for seasonality
- [ ] Verify train/test temporal split
- [ ] Check for data leakage across time

## Data Leakage Checks
- [ ] Features that include target information
- [ ] Future data in training set
- [ ] Group leakage (same entity in train/test)
- [ ] Test set statistics don't influence training

## Documentation
- [ ] Create data dictionary
- [ ] Document data quality issues
- [ ] List assumptions made
- [ ] Create summary report

## Decision Points
**Features to Drop:**
- [ ] [Feature name]: [Reason]

**Features to Engineer:**
- [ ] [New feature]: [Description]

**Data Preprocessing Steps:**
- [ ] [Step]: [Details]

**Next Steps:**
- [ ] [Action item]
```

### 2.2 Data Quality Report Template

```markdown
# Data Quality Report: [Dataset Name]

**Date:** [Date]
**Analyst:** [Name]
**Data Version:** [Version]

## Executive Summary
- Total Records: [X]
- Total Features: [Y]
- Overall Quality Score: [X/10]
- Critical Issues: [Number]
- Recommendations: [Key action]

## Dataset Overview
| Metric | Value |
|--------|-------|
| Rows | [X] |
| Columns | [Y] |
| Memory Usage | [X MB] |
| Date Range | [Start] to [End] |

## Quality Metrics

### Completeness (X%)
| Column | Type | Missing % | Action |
|--------|------|-----------|--------|
| [col1] | [num] | [5%] | [Impute with median] |
| [col2] | [cat] | [20%] | [Create 'Unknown' category] |

### Accuracy
| Check | Status | Issues Found |
|-------|--------|--------------|
| Valid ranges | ✅ | 0 |
| Referential integrity | ⚠️ | 152 orphaned records |
| Business rules | ✅ | 0 |

### Consistency
| Check | Status | Details |
|-------|--------|---------|
| Data types | ✅ | All correct |
| Encoding | ⚠️ | Mixed UTF-8/Latin-1 in 'comments' |
| Formatting | ✅ | Standardized |

### Uniqueness
| Check | Expected | Actual | Status |
|-------|----------|--------|--------|
| Primary key duplicates | 0 | 0 | ✅ |
| Near-duplicates | Low | 234 | ⚠️ |

## Critical Issues
1. **[Issue Title]**
   - Severity: [High/Medium/Low]
   - Impact: [Description]
   - Affected records: [X]
   - Recommendation: [Action]

## Recommendations
### Immediate Actions (Before Modeling)
- [ ] [Action 1]
- [ ] [Action 2]

### Monitoring (Ongoing)
- [ ] [Metric to track]
- [ ] [Alert threshold]

## Appendix
### Distribution Plots
[Include key visualizations]

### Correlation Analysis
[Include correlation matrix]
```

---

## 3. Feature Engineering Template

### 3.1 Feature Engineering Documentation

```markdown
# Feature Engineering: [Project Name]

## Feature Inventory

### Raw Features (Input)
| Feature | Type | Description | Source | Quality |
|---------|------|-------------|--------|---------|
| [name] | [numerical] | [Description] | [DB.table] | [Good/Fair/Poor] |

### Engineered Features

#### Feature Group: [Group Name, e.g., "User Engagement"]
| Feature | Formula/Logic | Type | Rationale |
|---------|---------------|------|-----------|
| `user_avg_session_time` | `SUM(session_duration) / COUNT(sessions)` | Numerical | Measures user engagement level |
| `user_days_since_last_visit` | `CURRENT_DATE - MAX(visit_date)` | Numerical | Recency indicator |
| `user_total_purchases` | `COUNT(purchase_id)` | Numerical | Purchase frequency |

#### Feature Group: [Temporal Features]
| Feature | Formula/Logic | Type | Rationale |
|---------|---------------|------|-----------|
| `hour_of_day` | `EXTRACT(HOUR FROM timestamp)` | Categorical | Capture daily patterns |
| `day_of_week` | `EXTRACT(DOW FROM timestamp)` | Categorical | Weekly seasonality |
| `is_weekend` | `day_of_week IN (0,6)` | Boolean | Weekend behavior differs |
| `is_holiday` | `date IN (holiday_calendar)` | Boolean | Special day behavior |

#### Feature Group: [Aggregations]
| Feature | Formula/Logic | Window | Type |
|---------|---------------|--------|------|
| `user_purchases_7d` | `COUNT(purchases)` | 7 days | Numerical |
| `user_purchases_30d` | `COUNT(purchases)` | 30 days | Numerical |
| `user_avg_order_value_30d` | `AVG(order_value)` | 30 days | Numerical |

#### Feature Group: [Ratios & Interactions]
| Feature | Formula/Logic | Type | Rationale |
|---------|---------------|------|-----------|
| `click_through_rate` | `clicks / impressions` | Numerical | Engagement metric |
| `price_to_avg_ratio` | `item_price / user_avg_purchase` | Numerical | Relative price perception |
| `category_affinity` | `user_category_purchases / total_purchases` | Numerical | Category preference |

### Feature Transformations

#### Numerical Transformations
| Original Feature | Transformation | New Feature | Reason |
|------------------|----------------|-------------|--------|
| `price` | `log(price + 1)` | `log_price` | Reduce skewness |
| `age` | `age^2` | `age_squared` | Capture non-linear relationship |
| `distance` | `1/distance` | `proximity` | Inverse relationship |

#### Categorical Encodings
| Feature | Encoding Method | Parameters | Output |
|---------|----------------|------------|--------|
| `category` | One-hot | - | `category_A`, `category_B`, ... |
| `user_id` | Target encoding | smoothing=10 | `user_id_encoded` |
| `city` | Frequency encoding | - | `city_frequency` |

#### Binning
| Feature | Bins | Labels | Rationale |
|---------|------|--------|-----------|
| `age` | `[0, 18, 35, 50, 100]` | `['child', 'young', 'middle', 'senior']` | Life stage segmentation |
| `income` | Quantiles (4) | `['Q1', 'Q2', 'Q3', 'Q4']` | Income quartiles |

## Feature Selection

### Filter Methods
```python
# Correlation with target
correlation_threshold = 0.1
selected_features = features[abs(corr_with_target) > correlation_threshold]

# Variance threshold
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)
high_variance_features = selector.fit_transform(X)
```

### Wrapper Methods
```python
# Recursive Feature Elimination
from sklearn.feature_selection import RFE
rfe = RFE(estimator=model, n_features_to_select=50)
selected_features = rfe.fit_transform(X, y)
```

### Embedded Methods
```python
# L1 Regularization (Lasso)
from sklearn.linear_model import LassoCV
lasso = LassoCV(cv=5)
lasso.fit(X, y)
important_features = X.columns[lasso.coef_ != 0]

# Tree-based importance
feature_importance = model.feature_importances_
top_features = X.columns[np.argsort(feature_importance)[-50:]]
```

## Feature Engineering Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# Numerical features pipeline
numerical_features = ['age', 'income', 'tenure']
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical features pipeline
categorical_features = ['category', 'city', 'device']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier())
])
```

## Feature Validation

### Checks Before Training
- [ ] No data leakage (future information)
- [ ] No target leakage
- [ ] Consistent transformations (train/val/test)
- [ ] Handle unseen categories
- [ ] Missing value handling
- [ ] Outlier treatment
- [ ] Feature scaling if needed

### Feature Importance Analysis
```python
# After training
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Plot
shap.summary_plot(shap_values, X_test, plot_type="bar")
```

## Feature Monitoring (Production)

### Data Drift Detection
```python
from scipy.stats import ks_2samp

def detect_drift(reference_data, current_data, threshold=0.05):
    """Detect feature drift using Kolmogorov-Smirnov test"""
    drifted_features = []

    for column in reference_data.columns:
        statistic, p_value = ks_2samp(
            reference_data[column],
            current_data[column]
        )

        if p_value < threshold:
            drifted_features.append({
                'feature': column,
                'p_value': p_value,
                'statistic': statistic
            })

    return drifted_features
```

### Feature Quality Metrics
| Feature | Metric | Threshold | Alert |
|---------|--------|-----------|-------|
| `user_purchases_30d` | Missing % | < 5% | ⚠️ if exceeded |
| `click_through_rate` | Out of range [0,1] | 0 violations | 🚨 if any |
| `category` | Unknown categories | < 1% | ⚠️ if exceeded |

## Version History
| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | [Date] | Initial features | [Name] |
| 1.1 | [Date] | Added temporal features | [Name] |
| 2.0 | [Date] | Major refactor - removed leaky features | [Name] |
```

---

## 4. Model Development Checklist

```markdown
# Model Development Checklist

## Phase 1: Problem Formulation
- [ ] Define problem type (classification/regression/ranking/clustering)
- [ ] Define success metrics (business + ML)
- [ ] Establish baseline (random/simple heuristic)
- [ ] Set performance targets
- [ ] Define constraints (latency/memory/interpretability)

## Phase 2: Data Preparation
- [ ] Train/validation/test split (temporal if applicable)
- [ ] Verify no data leakage
- [ ] Handle class imbalance (if applicable)
- [ ] Feature engineering complete
- [ ] Data preprocessing pipeline
- [ ] Save data splits with version

## Phase 3: Baseline Model
- [ ] Choose simple baseline model
- [ ] Train on representative sample
- [ ] Evaluate on validation set
- [ ] Document baseline performance
- [ ] Identify failure modes

**Baseline Results:**
- Model: [Name]
- Metric: [Value]
- Training time: [Duration]

## Phase 4: Model Selection
### Models to Try
- [ ] Linear/Logistic Regression
- [ ] Random Forest
- [ ] Gradient Boosting (XGBoost/LightGBM/CatBoost)
- [ ] Neural Network
- [ ] [Domain-specific model]

### Model Comparison
| Model | Val Metric | Train Time | Inference Time | Memory |
|-------|------------|------------|----------------|--------|
| [Model1] | [X] | [Y] | [Z] | [M] |
| [Model2] | [X] | [Y] | [Z] | [M] |

### Selection Criteria
- [ ] Performance on validation set
- [ ] Training time acceptable
- [ ] Inference time meets SLA
- [ ] Memory footprint reasonable
- [ ] Interpretability requirements met
- [ ] Maintenance complexity acceptable

**Selected Model:** [Name]
**Reason:** [Justification]

## Phase 5: Hyperparameter Tuning
### Search Strategy
- [ ] Define search space
- [ ] Choose search method (Grid/Random/Bayesian)
- [ ] Set cross-validation strategy
- [ ] Define optimization metric

### Key Hyperparameters
```python
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200, 500],
    'subsample': [0.8, 1.0]
}
```

### Tuning Results
- Best parameters: [Dict]
- Best CV score: [Value]
- Improvement over default: [+X%]

## Phase 6: Model Training (Final)
- [ ] Train on full training set with best hyperparameters
- [ ] Monitor training metrics (loss, accuracy)
- [ ] Check for overfitting (train vs val performance)
- [ ] Early stopping implemented (if applicable)
- [ ] Save model checkpoints
- [ ] Log training configuration

### Training Logs
```
Epoch 1/100: train_loss=0.523, val_loss=0.551
Epoch 2/100: train_loss=0.412, val_loss=0.445
...
Early stopping at epoch 45
Best val_loss: 0.223 (epoch 43)
```

## Phase 7: Model Evaluation
### Quantitative Evaluation
- [ ] Evaluate on held-out test set
- [ ] Calculate primary metric
- [ ] Calculate secondary metrics
- [ ] Statistical significance test (if applicable)
- [ ] Confidence intervals

**Test Set Results:**
| Metric | Value | 95% CI |
|--------|-------|--------|
| [Primary] | [X] | [CI] |
| [Secondary] | [Y] | [CI] |

### Qualitative Evaluation
- [ ] Error analysis on misclassifications
- [ ] Confusion matrix analysis
- [ ] Precision-recall analysis
- [ ] ROC curve analysis
- [ ] Calibration analysis

### Slice-Based Evaluation
- [ ] Performance by demographic group
- [ ] Performance by time period
- [ ] Performance by category/segment
- [ ] Identify underperforming slices

| Slice | Metric | Sample Size | Notes |
|-------|--------|-------------|-------|
| [Group A] | [X] | [N] | [Performance note] |
| [Group B] | [Y] | [M] | [Performance note] |

## Phase 8: Model Interpretability
- [ ] Feature importance analysis
- [ ] SHAP values calculated
- [ ] Partial dependence plots
- [ ] Individual prediction explanations
- [ ] Model decision boundaries visualized

### Top 10 Important Features
| Rank | Feature | Importance | SHAP Value |
|------|---------|------------|------------|
| 1 | [feature1] | [X] | [Y] |
| 2 | [feature2] | [X] | [Y] |

## Phase 9: Model Validation
### Robustness Checks
- [ ] Performance on different time periods
- [ ] Performance with missing features
- [ ] Performance with noisy inputs
- [ ] Adversarial examples (if applicable)

### Fairness Checks
- [ ] Demographic parity
- [ ] Equal opportunity
- [ ] Calibration across groups
- [ ] Disparate impact analysis

### Business Logic Validation
- [ ] Predictions make domain sense
- [ ] Edge cases handled correctly
- [ ] Model aligns with business rules
- [ ] SME review completed

## Phase 10: Documentation
- [ ] Model card created
- [ ] Training data documented
- [ ] Model architecture documented
- [ ] Hyperparameters documented
- [ ] Performance metrics documented
- [ ] Known limitations documented
- [ ] Ethical considerations documented

## Phase 11: Model Packaging
- [ ] Model serialized (pickle/joblib/ONNX)
- [ ] Preprocessing pipeline included
- [ ] Model version tagged
- [ ] Dependencies listed (requirements.txt)
- [ ] Model size optimized
- [ ] Inference code prepared

## Sign-off
- [ ] ML Engineer: [Name] - [Date]
- [ ] Data Scientist: [Name] - [Date]
- [ ] Product Owner: [Name] - [Date]
- [ ] Ready for deployment: Yes/No
```

---

## 5. Experiment Tracking Template

```markdown
# Experiment Log: [Project Name]

## Experiment #[Number]: [Descriptive Title]

**Date:** [YYYY-MM-DD]
**Experimenter:** [Name]
**Status:** [In Progress / Completed / Failed]
**Duration:** [X hours/days]

### Hypothesis
[What you're testing and why you think it will improve performance]

### Changes from Baseline
```diff
- Old approach: [Description]
+ New approach: [Description]
```

**Specific Changes:**
1. [Change 1]: [Details]
2. [Change 2]: [Details]

### Configuration
```python
config = {
    'model': 'XGBoost',
    'features': ['feature_set_v3'],
    'hyperparameters': {
        'learning_rate': 0.05,
        'max_depth': 7,
        'n_estimators': 500,
        'subsample': 0.8
    },
    'data_version': '2024-01-15',
    'random_seed': 42
}
```

### Results

#### Metrics
| Metric | Baseline | This Experiment | Change |
|--------|----------|----------------|--------|
| AUC-ROC | 0.823 | 0.847 | +2.9% ✅ |
| Precision@0.5 | 0.765 | 0.782 | +2.2% ✅ |
| Recall@0.5 | 0.698 | 0.721 | +3.3% ✅ |
| Training Time | 12 min | 18 min | +50% ⚠️ |
| Inference Time | 23ms | 31ms | +35% ⚠️ |

#### Visualizations
![Confusion Matrix](experiments/exp_015/confusion_matrix.png)
![ROC Curve](experiments/exp_015/roc_curve.png)
![Feature Importance](experiments/exp_015/feature_importance.png)

### Analysis
**What Worked:**
- [Observation 1]
- [Observation 2]

**What Didn't Work:**
- [Observation 1]
- [Observation 2]

**Insights:**
- [Learning 1]
- [Learning 2]

### Decision
- [ ] ✅ Adopt changes (performance improvement justifies cost)
- [ ] ❌ Reject changes (insufficient improvement)
- [ ] 🔄 Iterate (promising but needs refinement)

**Next Steps:**
- [ ] [Action 1]
- [ ] [Action 2]

### Reproducibility
```bash
# Code version
git commit: a3f5b2c

# Command to reproduce
python train.py --config experiments/exp_015/config.yaml

# Data location
s3://ml-data/project-name/data_v20240115/

# Model checkpoint
s3://ml-models/project-name/exp_015/model.pkl
```

### Notes
[Any additional observations, bugs encountered, ideas for future experiments]

---

## Experiment Comparison Table

| Exp # | Date | Key Change | AUC-ROC | Precision | Recall | Decision |
|-------|------|------------|---------|-----------|--------|----------|
| 001 | 2024-01-05 | Baseline (Logistic Regression) | 0.753 | 0.698 | 0.654 | Baseline |
| 002 | 2024-01-08 | Random Forest | 0.789 | 0.731 | 0.682 | ✅ |
| 003 | 2024-01-10 | Feature engineering v2 | 0.812 | 0.755 | 0.693 | ✅ |
| 015 | 2024-01-25 | XGBoost + tuning | 0.847 | 0.782 | 0.721 | ✅ |

## Best Performing Experiments

### Current Production Model
- **Experiment:** #012
- **AUC-ROC:** 0.834
- **Deployed:** 2024-01-18

### Best Candidate for Next Deployment
- **Experiment:** #015
- **AUC-ROC:** 0.847
- **Expected Impact:** +1.5% improvement
- **Next Steps:** A/B test preparation
```

---

## 6. Model Evaluation Template

```markdown
# Model Evaluation Report: [Model Name]

**Date:** [YYYY-MM-DD]
**Model Version:** [X.Y]
**Evaluator:** [Name]
**Evaluation Dataset:** [test_set_v1]

## Executive Summary
- **Primary Metric:** [AUC-ROC = 0.847]
- **vs Baseline:** [+15.3% improvement]
- **Recommendation:** [Deploy / Iterate / Reject]
- **Confidence:** [High / Medium / Low]

## Model Information
| Attribute | Value |
|-----------|-------|
| Model Type | [XGBoost Classifier] |
| Training Data Size | [1.2M samples] |
| Feature Count | [143 features] |
| Training Duration | [2.5 hours] |
| Model Size | [45 MB] |

## Performance Metrics

### Classification Metrics (Threshold = 0.5)
| Metric | Value | Baseline | Change |
|--------|-------|----------|--------|
| Accuracy | 0.876 | 0.812 | +7.9% |
| Precision | 0.823 | 0.745 | +10.5% |
| Recall | 0.791 | 0.723 | +9.4% |
| F1-Score | 0.807 | 0.734 | +9.9% |
| AUC-ROC | 0.912 | 0.856 | +6.5% |
| AUC-PR | 0.887 | 0.823 | +7.8% |

### Threshold Analysis
| Threshold | Precision | Recall | F1 | False Positive Rate |
|-----------|-----------|--------|-----|---------------------|
| 0.3 | 0.712 | 0.891 | 0.792 | 0.187 |
| 0.5 | 0.823 | 0.791 | 0.807 | 0.092 |
| 0.7 | 0.891 | 0.654 | 0.754 | 0.041 |
| 0.9 | 0.945 | 0.412 | 0.573 | 0.012 |

**Recommended Threshold:** 0.5
**Rationale:** [Best balance between precision and recall for this use case]

### Confusion Matrix (at threshold 0.5)
```
                Predicted
                Neg    Pos
Actual  Neg   [8923]  [821]
        Pos   [1054]  [4002]
```

**Analysis:**
- True Negatives: 8923 (91.6%)
- False Positives: 821 (8.4%) - [Impact: minor inconvenience to users]
- False Negatives: 1054 (20.8%) - [Impact: missed opportunities]
- True Positives: 4002 (79.2%) - [Impact: successful interventions]

## Slice-Based Performance

### Performance by Demographic
| Segment | Sample Size | AUC-ROC | Precision | Recall |
|---------|-------------|---------|-----------|--------|
| Overall | 14,800 | 0.912 | 0.823 | 0.791 |
| Age 18-25 | 3,200 | 0.898 | 0.801 | 0.782 |
| Age 26-35 | 5,100 | 0.921 | 0.834 | 0.798 |
| Age 36-50 | 4,200 | 0.915 | 0.829 | 0.793 |
| Age 50+ | 2,300 | 0.905 | 0.817 | 0.785 |

**Findings:**
- Performance consistent across age groups
- Slight degradation for 18-25 segment (investigate further)

### Performance by Time Period
| Period | Sample Size | AUC-ROC | Notes |
|--------|-------------|---------|-------|
| Jan 2024 | 4,800 | 0.914 | Recent data |
| Dec 2023 | 5,000 | 0.911 | Similar performance |
| Nov 2023 | 5,000 | 0.909 | Stable |

## Calibration Analysis
- **Brier Score:** 0.089 (lower is better)
- **Expected Calibration Error (ECE):** 0.034
- **Calibration:** Well-calibrated ✅

[Include calibration plot]

## Feature Importance

### Top 10 Features
| Rank | Feature | Importance | SHAP Value | Description |
|------|---------|------------|------------|-------------|
| 1 | `user_purchase_frequency_30d` | 0.156 | 0.182 | Recent purchase behavior |
| 2 | `time_on_site_avg` | 0.132 | 0.145 | Engagement metric |
| 3 | `click_through_rate` | 0.118 | 0.121 | Interaction rate |
| 4 | `cart_abandonment_rate` | 0.095 | 0.098 | Purchase intent |
| 5 | `days_since_last_visit` | 0.087 | 0.089 | Recency |
| 6 | `category_affinity_electronics` | 0.076 | 0.074 | Category preference |
| 7 | `avg_order_value` | 0.068 | 0.071 | Purchase value |
| 8 | `page_views_per_session` | 0.061 | 0.059 | Engagement |
| 9 | `mobile_vs_desktop_ratio` | 0.053 | 0.051 | Device preference |
| 10 | `email_open_rate` | 0.048 | 0.046 | Email engagement |

**Insights:**
- Recent behavior (30-day window) most predictive
- Engagement metrics highly important
- Demographics less important than behavior

## Error Analysis

### False Positive Analysis
**Sample Size:** 821 false positives

**Patterns:**
1. [Pattern 1]: 32% - [Users with recent spike in activity (not sustained)]
2. [Pattern 2]: 28% - [Users browsing for others (gifts)]
3. [Pattern 3]: 19% - [Window shoppers with no intent]

**Root Cause:** Model over-indexes on short-term activity bursts

**Mitigation:** Add features for activity consistency over longer windows

### False Negative Analysis
**Sample Size:** 1054 false negatives

**Patterns:**
1. [Pattern 1]: 41% - [Infrequent buyers who make large purchases]
2. [Pattern 2]: 31% - [New users with limited history]
3. [Pattern 3]: 18% - [Users who convert via channels not tracked]

**Root Cause:** Model relies heavily on historical data

**Mitigation:** Improve cold-start features, integrate additional data sources

## Model Robustness

### Missing Feature Analysis
| Feature Removed | Impact on AUC-ROC |
|-----------------|-------------------|
| Top feature | -0.023 (0.912 → 0.889) |
| Top 5 features | -0.056 (0.912 → 0.856) |
| Random 10% features | -0.008 (0.912 → 0.904) |

**Finding:** Model reasonably robust to missing features

### Adversarial Robustness
- Tested with ±10% noise in numerical features
- AUC-ROC degradation: -0.012 (0.912 → 0.900)
- **Assessment:** Reasonably robust ✅

## Fairness Evaluation

### Demographic Parity
| Group | Positive Prediction Rate |
|-------|--------------------------|
| Age 18-25 | 32.1% |
| Age 26-35 | 33.8% |
| Age 36-50 | 34.2% |
| Age 50+ | 31.9% |

**Max Difference:** 2.3 percentage points
**Assessment:** Acceptable disparity ✅

### Equal Opportunity
| Group | True Positive Rate (Recall) |
|-------|------------------------------|
| Age 18-25 | 78.2% |
| Age 26-35 | 79.8% |
| Age 36-50 | 79.3% |
| Age 50+ | 78.5% |

**Max Difference:** 1.6 percentage points
**Assessment:** Fair across groups ✅

## Business Impact Estimation

### Confusion Matrix Business Impact
| Outcome | Count | Cost/Benefit | Total Impact |
|---------|-------|--------------|--------------|
| True Positive | 4,002 | +$45/user | +$180,090 |
| False Positive | 821 | -$2/user | -$1,642 |
| False Negative | 1,054 | -$38/user | -$40,052 |
| True Negative | 8,923 | $0 | $0 |

**Net Impact (Test Set):** +$138,396
**Extrapolated Annual Impact:** +$3.4M (assuming similar test set distribution)

## Comparison with Production Model

| Metric | Current Prod | This Model | Change |
|--------|--------------|------------|--------|
| AUC-ROC | 0.856 | 0.912 | +6.5% |
| Precision | 0.745 | 0.823 | +10.5% |
| Recall | 0.723 | 0.791 | +9.4% |
| Inference Time | 18ms | 23ms | +27.8% |
| Model Size | 32MB | 45MB | +40.6% |

**Trade-off Analysis:**
- ✅ Significant performance improvement (+6.5% AUC)
- ⚠️ Slightly slower inference (+5ms, still within SLA of 50ms)
- ✅ Larger model size acceptable (well below 100MB limit)

## Limitations & Risks

### Known Limitations
1. **Cold Start:** Performance degraded for new users (limited history)
2. **Data Coverage:** Lower performance on users from new geographic regions
3. **Temporal:** Model trained on Q4 2023 data; may need retraining for seasonality

### Risks
| Risk | Severity | Probability | Mitigation |
|------|----------|-------------|------------|
| Model drift over time | Medium | High | Implement monitoring, quarterly retraining |
| Performance on new user segments | Low | Medium | A/B test on segment, collect feedback |
| Increased latency | Low | Low | Benchmark on production hardware |

## Recommendations

### Deployment Decision
- **Recommendation:** ✅ **Deploy to Production**
- **Confidence Level:** High
- **Rationale:** Significant performance improvement (+6.5% AUC), well-calibrated, fair across demographics, acceptable trade-offs

### Deployment Plan
1. **A/B Test:** Deploy to 10% of traffic for 2 weeks
2. **Monitor:** Track metrics daily, alert on degradation
3. **Rollout:** Gradual rollout to 100% if A/B test successful
4. **Rollback Plan:** Keep previous model for instant rollback if needed

### Future Improvements
- [ ] Improve cold-start performance with content-based features
- [ ] Investigate 18-25 age segment performance gap
- [ ] Add features for activity consistency (reduce false positives)
- [ ] Integrate additional data sources for false negative reduction
- [ ] Experiment with ensemble methods

## Sign-off

- [ ] Data Scientist: [Name] - [Date]
- [ ] ML Engineer: [Name] - [Date]
- [ ] Product Manager: [Name] - [Date]
- [ ] Deployment Approved: Yes / No
```

---

*This is Part 1 of the ML Practical Templates & Checklists guide. The remaining sections will be continued in Part 2.*

**Coming in Part 2:**
- A/B Testing Plan Template
- Production Deployment Checklist
- Monitoring & Alerting Template
- Incident Response Template
- ML System Design Template
- Interview Preparation Checklist

---

**Document Version:** 1.0
**Last Updated:** October 21, 2025
**Maintained by:** Benjamin Hu
