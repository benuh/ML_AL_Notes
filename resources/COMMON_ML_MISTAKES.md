# ⚠️ Common ML Mistakes and How to Avoid Them

**Learning from others' mistakes is faster than making them yourself.** This guide covers the most common mistakes ML learners and practitioners make, and how to avoid them.

---

## 📋 Table of Contents

- [Learning Phase Mistakes](#learning-phase-mistakes)
- [Data Mistakes](#data-mistakes)
- [Modeling Mistakes](#modeling-mistakes)
- [Code & Engineering Mistakes](#code--engineering-mistakes)
- [Career & Job Search Mistakes](#career--job-search-mistakes)
- [Production & Deployment Mistakes](#production--deployment-mistakes)
- [Research & Experimentation Mistakes](#research--experimentation-mistakes)
- [Quick Reference Checklist](#quick-reference-checklist)

---

## Learning Phase Mistakes

### 1. Tutorial Hell

**Mistake:** Watching endless tutorials without building anything

**Why it happens:**
- Feels like progress (dopamine hit)
- Avoids the discomfort of struggling with code
- Fear of starting a "real" project

**How to avoid:**
- **80/20 rule:** 20% tutorials, 80% building
- After each tutorial section, build something different
- Set a timer: 30 minutes learning → 2 hours coding

**Example:**
```python
# ❌ Bad: Just watching tutorial on linear regression
# Watch video → Move to next video

# ✅ Good: Apply immediately
# Watch video → Implement on different dataset →
# Modify algorithm → Break it → Fix it → Document
```

### 2. Skipping the Fundamentals

**Mistake:** Jumping straight to deep learning without understanding basics

**Why it happens:**
- Deep learning is "sexier"
- Want to work on cutting-edge stuff
- Think fundamentals are boring

**Reality check:**
- 80% of real-world ML uses classical algorithms
- Deep learning interviews require strong fundamentals
- Understanding basics makes debugging 10x faster

**How to avoid:**
Follow this order:
1. **Months 1-2:** Linear algebra, statistics, classical ML
2. **Month 3:** Neural networks from scratch
3. **Month 4+:** Modern deep learning

**Red flag:** You can't implement linear regression from scratch but you're trying to fine-tune BERT.

### 3. Not Understanding the Math

**Mistake:** Treating ML as pure coding without mathematical intuition

**Example misconceptions:**
```python
# ❌ Wrong mental model
"I just call model.fit() and it works somehow"

# ✅ Correct understanding
"Gradient descent minimizes loss by computing derivatives
 of the error with respect to weights, then updating
 weights in the direction that reduces error"
```

**How much math you need:**
- **Minimum:** Understand what gradient descent does
- **Good:** Can derive backpropagation for simple network
- **Excellent:** Can read research papers and understand proofs

**Resource:** [Mathematics for ML](../guides/fundamentals/MATHEMATICS_FOR_ML.md)

### 4. Learning in Isolation

**Mistake:** Never sharing work or getting feedback

**Why it's harmful:**
- Don't know what you don't know
- Miss industry best practices
- No accountability

**How to avoid:**
- Post projects on GitHub (even if imperfect)
- Share progress on LinkedIn/Twitter
- Join ML communities (Discord, Reddit)
- Do pair programming or code reviews
- Contribute to open source

**Reality:** Your "bad" code is better than no code. Ship it.

### 5. Perfectionism Paralysis

**Mistake:** Never finishing projects because they're "not good enough"

**Symptoms:**
- 10 half-finished projects
- Constantly restarting with "better" approach
- Never showing work to others

**How to break the cycle:**
1. Set 2-week deadline for projects
2. Define "done" criteria upfront
3. Timebox polish phase (20% of project time)
4. Ship on deadline, imperfect or not
5. Start next project

**Remember:** Done is better than perfect.

### 6. Not Tracking Progress

**Mistake:** Can't see improvement, feels like not learning

**Better approach:**
```markdown
## Learning Log

### Week 1 (Oct 1-7, 2025)
- ✅ Completed linear regression notebook
- ✅ Built house price predictor
- 📊 Model: R² = 0.72
- 💡 Learned: Feature scaling improves convergence
- ❌ Struggled: Understanding gradient descent math
- ⏭️ Next: Logistic regression

### Week 2 (Oct 8-14, 2025)
...
```

**Benefits:**
- See concrete progress
- Identify patterns in struggles
- Great for interviews ("Tell me about a challenge...")

---

## Data Mistakes

### 7. Not Exploring Data First

**Mistake:** Jumping straight to modeling without EDA

**Real example:**
```python
# ❌ Bad approach
data = pd.read_csv('data.csv')
X = data.drop('target', axis=1)
y = data['target']
model.fit(X, y)  # Why is accuracy so low?!

# ✅ Good approach
data = pd.read_csv('data.csv')

# Explore first
print(data.info())
print(data.describe())
print(data.isnull().sum())
print(data['target'].value_counts())  # Highly imbalanced!

# Visualize
import matplotlib.pyplot as plt
data.hist(figsize=(12, 10))
plt.tight_layout()
plt.show()

# Check correlations
corr_matrix = data.corr()
print(corr_matrix['target'].sort_values(ascending=False))

# Now you know: class imbalance, missing values,
# low correlation features
```

**Checklist before modeling:**
- [ ] Data types correct?
- [ ] Missing values?
- [ ] Outliers?
- [ ] Class imbalance?
- [ ] Feature distributions?
- [ ] Correlations?

### 8. Data Leakage

**Mistake:** Information from test set leaking into training

**Common causes:**

**Type 1: Leaking test data**
```python
# ❌ WRONG: Scaling before split
scaler = StandardScaler()
X_scaled = scaler.fit(X)  # Sees test data!
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

# ✅ CORRECT: Split first, then scale
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Only transform
```

**Type 2: Leaking future information**
```python
# ❌ WRONG: Using future data in time series
# Predicting stock price on day 5 using day 6 data
data['moving_avg'] = data['price'].rolling(window=7).mean()
# This includes future prices!

# ✅ CORRECT: Only use past data
data['moving_avg'] = data['price'].rolling(window=7).mean().shift(1)
```

**Type 3: Leaking target variable**
```python
# ❌ WRONG: Feature derived from target
# Predicting customer churn, but including
# "days_since_cancellation" as feature

# ✅ CORRECT: Only use features available at prediction time
```

### 9. Ignoring Class Imbalance

**Mistake:** 95% accuracy on imbalanced dataset (99% negative class)

**Example:**
```python
# Dataset: 99 non-fraud, 1 fraud transaction

# ❌ Bad: Accuracy looks great!
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.2%}")
# Output: Accuracy: 99%
# But model just predicts "not fraud" for everything!

# ✅ Good: Check appropriate metrics
from sklearn.metrics import classification_report, confusion_matrix

predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
# Fraud (class 1): precision=0.10, recall=0.05, f1=0.06
# Model is terrible at detecting fraud!

print(confusion_matrix(y_test, predictions))
# [[990   0]    # Predicts all non-fraud correctly
#  [ 95   5]]   # Misses 95% of fraud cases!
```

**Solutions:**
1. **Resampling:**
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

2. **Class weights:**
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight='balanced')
```

3. **Different metrics:**
- Precision, Recall, F1-score
- PR-AUC instead of ROC-AUC
- Confusion matrix

### 10. Not Handling Missing Data Properly

**Mistake:** Dropping all rows with any missing values

**Impact:**
```python
# ❌ Bad: Lost 80% of data!
print(f"Original size: {len(data)}")  # 10,000
data_clean = data.dropna()
print(f"After dropna: {len(data_clean)}")  # 2,000
# Lost 8,000 samples!

# ✅ Better: Analyze patterns first
missing_percent = (data.isnull().sum() / len(data)) * 100
print(missing_percent.sort_values(ascending=False))

# Strategy:
# - Drop columns with >50% missing
# - Drop rows with >50% missing
# - Impute rest based on data type
```

**Imputation strategies:**
```python
# Numerical: median (robust to outliers)
data['age'].fillna(data['age'].median(), inplace=True)

# Categorical: mode or new category
data['category'].fillna('Unknown', inplace=True)

# Time series: forward fill
data['sensor_reading'].fillna(method='ffill', inplace=True)

# Advanced: KNN imputation
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
data_imputed = imputer.fit_transform(data)
```

---

## Modeling Mistakes

### 11. Not Using a Baseline

**Mistake:** No point of reference for model performance

**Always start with baseline:**
```python
# For regression: mean/median
baseline_predictions = np.full(len(y_test), y_train.mean())
baseline_mae = mean_absolute_error(y_test, baseline_predictions)
print(f"Baseline MAE: ${baseline_mae:,.0f}")

# Your model
model = RandomForestRegressor()
model.fit(X_train, y_train)
model_predictions = model.predict(X_test)
model_mae = mean_absolute_error(y_test, model_predictions)
print(f"Model MAE: ${model_mae:,.0f}")

# Compare
improvement = (baseline_mae - model_mae) / baseline_mae
print(f"Improvement over baseline: {improvement:.1%}")
```

**Good baselines:**
- **Classification:** Most frequent class
- **Regression:** Mean/median
- **Time series:** Last value (naive forecast)
- **Recommendation:** Popularity-based

### 12. Overfitting Without Realizing

**Mistake:** Great training accuracy, terrible test accuracy

**Example:**
```python
# ❌ Overfit model
model = DecisionTreeClassifier(max_depth=None)  # No limit!
model.fit(X_train, y_train)
print(f"Train: {model.score(X_train, y_train):.2%}")  # 100%
print(f"Test: {model.score(X_test, y_test):.2%}")    # 65%
# Memorized training data!

# ✅ Better
model = DecisionTreeClassifier(max_depth=5)  # Regularized
# Use cross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"CV scores: {scores}")
print(f"Mean: {scores.mean():.2%} (+/- {scores.std():.2%})")
```

**Red flags:**
- Training accuracy >> Test accuracy (>10% gap)
- High variance in cross-validation scores
- Model is very complex (many parameters)

**Solutions:**
1. Regularization (L1/L2)
2. Simpler model
3. More data
4. Feature selection
5. Cross-validation

### 13. Hyperparameter Tuning on Test Set

**Mistake:** Optimizing hyperparameters using test data

```python
# ❌ WRONG: Test set leakage
X_train, X_test, y_train, y_test = train_test_split(X, y)

best_score = 0
for depth in [3, 5, 10, 15]:
    model = DecisionTreeClassifier(max_depth=depth)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)  # Using test set!
    if score > best_score:
        best_score = score
        best_depth = depth
# Test set is no longer "unseen"

# ✅ CORRECT: Use validation set or CV
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2)

# Tune on validation set
best_score = 0
for depth in [3, 5, 10, 15]:
    model = DecisionTreeClassifier(max_depth=depth)
    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)  # Validation set
    if score > best_score:
        best_score = score
        best_depth = depth

# Final evaluation on test set (only once!)
final_model = DecisionTreeClassifier(max_depth=best_depth)
final_model.fit(X_train, y_train)
test_score = final_model.score(X_test, y_test)
```

### 14. Wrong Metric for the Problem

**Mistake:** Optimizing the wrong objective

**Examples:**

| Problem | Wrong Metric | Right Metric |
|---------|-------------|--------------|
| Fraud detection | Accuracy | Precision@90% recall |
| Medical diagnosis | Accuracy | Recall (catch all diseases) |
| Spam filter | Accuracy | Precision (don't lose real emails) |
| Recommendation | MSE | NDCG, MAP |
| Ranking | Accuracy | Precision@K, MRR |

**Example: Medical diagnosis**
```python
# ❌ Bad: Optimizing accuracy
# Missing 10% of cancer cases is unacceptable!

# ✅ Good: Optimize recall
from sklearn.metrics import make_scorer, recall_score
scorer = make_scorer(recall_score)

# Tune model to maximize recall
grid_search = GridSearchCV(model, param_grid, scoring=scorer)
```

### 15. Not Setting Random Seeds

**Mistake:** Can't reproduce results

```python
# ❌ Bad: Different results every time
X_train, X_test = train_test_split(X, y)
model = RandomForestClassifier()
model.fit(X_train, y_train)
# Score: 0.87

# Re-run same code
# Score: 0.83  # Different!

# ✅ Good: Reproducible
SEED = 42
X_train, X_test = train_test_split(X, y, random_state=SEED)
model = RandomForestClassifier(random_state=SEED)
model.fit(X_train, y_train)
# Score: 0.87

# Re-run
# Score: 0.87  # Same!
```

**Set seeds for:**
- train_test_split
- Models (RandomForest, XGBoost, etc.)
- NumPy (np.random.seed)
- PyTorch (torch.manual_seed)
- TensorFlow (tf.random.set_seed)

---

## Code & Engineering Mistakes

### 16. Messy Notebooks

**Mistake:** 5000-line notebook, cells executed out of order

**Example:**
```python
# ❌ Bad notebook structure
[Cell 1] imports
[Cell 2] load data
[Cell 5] train model  # Executed 3rd
[Cell 3] preprocess   # Executed 4th
[Cell 4] EDA         # Executed 2nd
[Cell 6] evaluate
# Results aren't reproducible!

# ✅ Good notebook structure
[Cell 1] # 1. Setup
         import pandas as pd
         import numpy as np
         from sklearn...

[Cell 2] # 2. Load Data
         data = pd.read_csv(...)

[Cell 3] # 3. Exploratory Data Analysis
         data.head()
         data.describe()

[Cell 4] # 4. Preprocessing
         X = preprocess(data)

[Cell 5] # 5. Train/Test Split
         X_train, X_test = ...

[Cell 6] # 6. Model Training
         model.fit(X_train, y_train)

[Cell 7] # 7. Evaluation
         evaluate(model, X_test, y_test)
```

**Best practices:**
- Run "Restart & Run All" before committing
- Clear outputs before committing (reduce file size)
- Use section headers with numbered cells
- Extract repeated code into functions
- Move production code to .py files

### 17. Not Using Version Control

**Mistake:** Files like `model_v2_final_FINAL_actually_final.py`

**Why Git is essential:**
```bash
# Without Git
my_project/
├── model.py
├── model_old.py
├── model_v2.py
├── model_v2_backup.py
├── model_working.py
├── model_final.py
└── model_final_v2.py  # Which one is actually final?!

# With Git
my_project/
├── model.py  # Always the latest
└── .git/     # Full history preserved
```

**Basic Git workflow:**
```bash
# Initial setup
git init
git add .
git commit -m "Initial commit"

# Daily workflow
git add model.py
git commit -m "Add feature engineering"
git push origin main

# Made a mistake? Go back!
git log  # Find commit hash
git checkout abc123 model.py  # Restore old version
```

### 18. Hardcoded Values

**Mistake:** Magic numbers and paths scattered everywhere

```python
# ❌ Bad: Hardcoded values
data = pd.read_csv('/Users/you/Desktop/project/data.csv')
X_train, X_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier(n_estimators=100, max_depth=5)
threshold = 0.73  # Why 0.73?

# ✅ Good: Configuration file
# config.yaml
data:
  path: 'data/processed/features.csv'

preprocessing:
  test_size: 0.2
  random_state: 42

model:
  type: 'random_forest'
  params:
    n_estimators: 100
    max_depth: 5

inference:
  threshold: 0.73  # Optimized for 90% precision

# Python code
import yaml

with open('config.yaml') as f:
    config = yaml.safe_load(f)

data = pd.read_csv(config['data']['path'])
X_train, X_test = train_test_split(
    X, y,
    test_size=config['preprocessing']['test_size'],
    random_state=config['preprocessing']['random_state']
)
```

### 19. Not Writing Tests

**Mistake:** Broken code discovered in production

```python
# ✅ Write tests for critical functions
# tests/test_preprocessing.py

import pytest
import pandas as pd
from src.preprocessing import clean_data, engineer_features

def test_clean_data_removes_nulls():
    """Test that clean_data removes rows with nulls"""
    data = pd.DataFrame({
        'age': [25, None, 35],
        'income': [50000, 60000, None]
    })
    result = clean_data(data)
    assert result.isnull().sum().sum() == 0
    assert len(result) == 1  # Only 1 complete row

def test_engineer_features_creates_age_bins():
    """Test that age bins are created correctly"""
    data = pd.DataFrame({'age': [10, 25, 45, 65, 85]})
    result = engineer_features(data)
    assert 'age_bin' in result.columns
    assert result['age_bin'].nunique() == 4  # 4 bins

def test_scaling_range():
    """Test that scaled features are in [0, 1]"""
    data = pd.DataFrame({'income': [20000, 50000, 100000]})
    result = scale_features(data)
    assert result['income'].min() >= 0
    assert result['income'].max() <= 1

# Run tests
# pytest tests/
```

**What to test:**
- Data preprocessing functions
- Feature engineering logic
- Custom metrics
- API endpoints
- Model inference

### 20. Ignoring Code Quality

**Mistake:** Unreadable code that no one else can use

```python
# ❌ Bad code
def f(d):
    x=d.dropna()
    x['a']=x['b']*x['c']
    return x[x['a']>5]

# ✅ Good code
def prepare_customer_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare customer data for modeling.

    Steps:
    1. Remove rows with missing values
    2. Calculate customer lifetime value (CLV)
    3. Filter high-value customers (CLV > 5000)

    Args:
        raw_data: Raw customer DataFrame with columns:
            - 'annual_revenue': Annual revenue per customer
            - 'retention_years': Years as customer

    Returns:
        Filtered DataFrame with high-value customers
    """
    # Remove incomplete records
    clean_data = raw_data.dropna()

    # Calculate customer lifetime value
    clean_data['customer_lifetime_value'] = (
        clean_data['annual_revenue'] * clean_data['retention_years']
    )

    # Filter for high-value customers (>$5,000 CLV)
    HIGH_VALUE_THRESHOLD = 5000
    high_value_customers = clean_data[
        clean_data['customer_lifetime_value'] > HIGH_VALUE_THRESHOLD
    ]

    return high_value_customers
```

**Code quality checklist:**
- [ ] Descriptive variable names
- [ ] Docstrings for functions
- [ ] Type hints
- [ ] Comments for complex logic
- [ ] Consistent formatting (use `black`)
- [ ] No repeated code (DRY principle)

---

## Career & Job Search Mistakes

### 21. Waiting Until "Ready" to Apply

**Mistake:** "I need 2 more years of learning before applying"

**Reality check:**
```
Your timeline:          Reality:
Year 1: Learn           Month 3-6: Start applying
Year 2: Build portfolio Month 6-9: Interview
Year 3: Start applying  Month 9-12: Get offer
Year 4: Get job?        Month 12+: Continue learning on the job
```

**When to start applying:**
- ✅ Can build end-to-end ML pipeline
- ✅ Have 2-3 portfolio projects
- ✅ Understand fundamentals (linear algebra, statistics, gradient descent)
- ✅ Can implement linear regression from scratch
- ❌ Don't need: PhD, 10 years experience, know everything

**Truth:** You'll learn more in 3 months on the job than in 1 year of solo study.

### 22. Generic Resume/Portfolio

**Mistake:** One resume for all jobs

```markdown
❌ Generic resume bullet:
"Built machine learning models using Python and scikit-learn"

✅ Tailored for Computer Vision role:
"Developed object detection system using YOLOv8, achieving
mAP@0.5 of 0.89 on custom manufacturing defect dataset (12K images)"

✅ Tailored for MLOps role:
"Designed and deployed ML pipeline using Kubernetes and MLflow,
reducing model deployment time from 2 weeks to 2 days"

✅ Tailored for NLP role:
"Fine-tuned BERT for sentiment analysis on 500K customer reviews,
improving F1-score from 0.78 (baseline) to 0.91"
```

**Strategy:**
1. Read job description carefully
2. Identify key requirements (e.g., "experience with transformers")
3. Highlight relevant projects
4. Use their exact terminology

### 23. No Online Presence

**Mistake:** No LinkedIn, no GitHub, no blog

**Impact:**
- Recruiters can't find you
- Can't verify your skills
- Miss networking opportunities

**Minimum viable presence:**
1. **LinkedIn:** Professional photo, 3 project highlights, active
2. **GitHub:** 3-5 pinned repos with good READMEs
3. **Optional but powerful:** Blog (Medium/Dev.to) with 2-3 articles

**Content ideas:**
- "How I built [project]"
- "5 things I wish I knew before learning ML"
- "Debugging [common problem] in PyTorch"

### 24. Not Networking

**Mistake:** Only applying through job portals (2% response rate)

**Reality:**
| Method | Response Rate |
|--------|---------------|
| Cold apply online | 2% |
| Employee referral | 40% |
| Direct message hiring manager | 15% |
| Met at conference/meetup | 60% |

**Networking strategies:**
1. **LinkedIn:** Connect with 5 ML engineers per week, comment on posts
2. **Meetups:** Attend local ML meetups, exchange contacts
3. **Conferences:** NeurIPS, ICML, local ML summits
4. **Twitter/X:** Share learnings, engage with ML community
5. **Open source:** Contribute to popular repos, interact with maintainers

**DM template:**
```
Hi [Name],

I came across your work on [specific project] and was really
impressed by [specific detail].

I'm transitioning into ML engineering and have built [your project].
I'd love to learn about your experience at [Company].

Would you be open to a 15-min chat?

Thanks!
[Your Name]
```

### 25. Underselling in Interviews

**Mistake:** "I just followed a tutorial..."

**Better approach:**

| Underselling | Better Framing |
|-------------|----------------|
| "I followed a Kaggle tutorial" | "I adapted a Kaggle approach and achieved top 20% ranking" |
| "It's just a simple project" | "I built an end-to-end system with 92% accuracy" |
| "I'm still learning" | "I'm continuously improving my skills, recently learned X" |
| "I don't know much about Y" | "I haven't worked with Y yet, but I've worked with similar Z" |

**Framework:**
- What you built (project)
- Why it matters (business impact)
- How you built it (technical approach)
- What you learned (growth)

See [Portfolio Guide](./PORTFOLIO_GUIDE.md#presenting-your-work) for more details.

---

## Production & Deployment Mistakes

### 26. "Works on My Machine"

**Mistake:** Can't reproduce model in production

**Common causes:**
```python
# ❌ Dependencies not specified
pip install scikit-learn  # Which version?

# ✅ Pin versions
# requirements.txt
scikit-learn==1.3.2
pandas==2.1.3
numpy==1.26.2
```

**Better: Use Docker**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ .

CMD ["python", "serve.py"]
```

### 27. Not Monitoring Models

**Mistake:** Model degrades silently in production

**What to monitor:**
1. **Input data distribution:** Detecting data drift
2. **Prediction distribution:** Concept drift
3. **Model performance:** Accuracy over time
4. **System metrics:** Latency, throughput, errors

```python
# Example: Log predictions for monitoring
import logging

logger = logging.getLogger(__name__)

def predict(features):
    prediction = model.predict(features)
    confidence = model.predict_proba(features).max()

    # Log for monitoring
    logger.info({
        'timestamp': datetime.now(),
        'prediction': prediction,
        'confidence': confidence,
        'features': features.to_dict()
    })

    # Alert if low confidence
    if confidence < 0.7:
        logger.warning(f"Low confidence prediction: {confidence}")

    return prediction
```

### 28. Ignoring Edge Cases

**Mistake:** Model breaks on unexpected inputs

```python
# ❌ No input validation
def predict_house_price(sqft, bedrooms):
    return model.predict([[sqft, bedrooms]])[0]

# What if sqft = -1000?
# What if bedrooms = 999?
# What if sqft = None?

# ✅ Validate inputs
def predict_house_price(sqft: float, bedrooms: int) -> float:
    """Predict house price with input validation"""

    # Type checking (use Pydantic in production)
    if not isinstance(sqft, (int, float)):
        raise TypeError("sqft must be numeric")
    if not isinstance(bedrooms, int):
        raise TypeError("bedrooms must be integer")

    # Range checking
    if sqft < 100 or sqft > 50000:
        raise ValueError(f"sqft out of range: {sqft}")
    if bedrooms < 0 or bedrooms > 20:
        raise ValueError(f"bedrooms out of range: {bedrooms}")

    # Make prediction
    features = pd.DataFrame([[sqft, bedrooms]],
                           columns=['sqft', 'bedrooms'])
    prediction = model.predict(features)[0]

    # Sanity check output
    if prediction < 0:
        logger.error(f"Negative price predicted: {prediction}")
        return None

    return prediction
```

### 29. Large Model Files in Git

**Mistake:** 500MB model.pkl in Git repo

**Problem:**
- Bloats repository
- Slow to clone
- Git isn't designed for large binary files

**Solution: Use DVC (Data Version Control)**
```bash
# Install DVC
pip install dvc

# Initialize
dvc init

# Track large files
dvc add models/model.pkl
git add models/model.pkl.dvc models/.gitignore
git commit -m "Add model (DVC tracked)"

# Store remotely (S3, GCS, etc.)
dvc remote add -d myremote s3://mybucket/dvc-cache
dvc push
```

**Alternatives:**
- Model registry (MLflow, Weights & Biases)
- Cloud storage (S3, GCS) with versioning
- Git LFS (for smaller files <100MB)

### 30. Not Handling Model Failures Gracefully

**Mistake:** API returns 500 error when model fails

**Better approach:**
```python
# ✅ Graceful degradation
def predict_with_fallback(features):
    try:
        # Try primary model
        prediction = ml_model.predict(features)
        return {
            'prediction': prediction,
            'model': 'ml_model',
            'confidence': 'high'
        }
    except Exception as e:
        logger.error(f"ML model failed: {e}")

        try:
            # Fallback to simpler model
            prediction = rule_based_model(features)
            return {
                'prediction': prediction,
                'model': 'rule_based',
                'confidence': 'medium'
            }
        except Exception as e:
            logger.error(f"Fallback failed: {e}")

            # Last resort: return safe default
            return {
                'prediction': get_historical_average(),
                'model': 'default',
                'confidence': 'low'
            }
```

---

## Research & Experimentation Mistakes

### 31. Not Tracking Experiments

**Mistake:** "That model worked great! What hyperparameters did I use?"

**Without tracking:**
```
experiment_1.ipynb
experiment_2_updated.ipynb
experiment_2_final.ipynb
experiment_3_THIS_ONE.ipynb
# Which one had the best results?!
```

**With MLflow:**
```python
import mlflow

mlflow.set_experiment("house-price-prediction")

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("n_estimators", 100)

    # Train model
    model = RandomForestRegressor(max_depth=5, n_estimators=100)
    model.fit(X_train, y_train)

    # Log metrics
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    mlflow.log_metric("train_r2", train_score)
    mlflow.log_metric("test_r2", test_score)

    # Log model
    mlflow.sklearn.log_model(model, "model")

# View all experiments in UI
# mlflow ui
```

**Alternatives:**
- Weights & Biases
- Neptune.ai
- TensorBoard
- Simple CSV log file (better than nothing!)

### 32. Chasing SOTA Without Understanding

**Mistake:** Using complex models without understanding simpler ones

**Example:**
```python
# ❌ Bad: Jump to complex model
# "Let me fine-tune GPT-4 for this classification task!"
# (Dataset has 1,000 samples)

# ✅ Good: Start simple, increase complexity
# 1. Logistic Regression: 0.82 accuracy (5 minutes)
# 2. Random Forest: 0.85 accuracy (10 minutes)
# 3. XGBoost: 0.87 accuracy (30 minutes)
# 4. Fine-tuned BERT: 0.88 accuracy (3 hours + GPU costs)
#
# Decision: Random Forest is good enough!
# 2% accuracy gain not worth 18x time increase
```

**Start simple:**
1. **Baseline:** Mean/mode prediction
2. **Simple linear:** Logistic regression
3. **Tree-based:** Random Forest, XGBoost
4. **Deep learning:** Only if above fail or specific use case (images, text, sequences)

### 33. P-Hacking

**Mistake:** Trying many things until test set looks good

**Example:**
```python
# ❌ P-hacking example
# Try 100 different feature combinations
# Report only the one that works best on test set
# Test set is now "contaminated"

for features in all_possible_combinations:  # 100 combinations
    model.fit(X_train[features], y_train)
    score = model.score(X_test[features], y_test)
    if score > best_score:
        best_score = score
        best_features = features

# Report best_score as "test accuracy"
# This is NOT legitimate test accuracy!
```

**Correct approach:**
```python
# ✅ Proper evaluation
# Use validation set for selection
# Test set only for final evaluation (once!)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

# Select features on validation set
for features in all_possible_combinations:
    model.fit(X_train[features], y_train)
    score = model.score(X_val[features], y_val)  # Validation
    if score > best_score:
        best_score = score
        best_features = features

# Final evaluation on test set (ONCE)
final_model.fit(X_train[best_features], y_train)
test_score = final_model.score(X_test[best_features], y_test)
# This is legitimate test accuracy
```

---

## Quick Reference Checklist

### Before Training Model
- [ ] Explored data (EDA)
- [ ] Checked for missing values, outliers
- [ ] Checked class balance
- [ ] Split data (before any preprocessing)
- [ ] Created baseline model
- [ ] Set random seeds

### During Modeling
- [ ] Using correct metric for problem
- [ ] Monitoring for overfitting (train vs val)
- [ ] Using cross-validation
- [ ] Tracking experiments
- [ ] Not tuning on test set

### Before Deployment
- [ ] Model works on hold-out test set
- [ ] Input validation implemented
- [ ] Error handling implemented
- [ ] Monitoring setup
- [ ] Dependencies documented (requirements.txt)
- [ ] Docker container tested
- [ ] API documented

### Code Quality
- [ ] Code in version control (Git)
- [ ] Tests written for critical functions
- [ ] README with setup instructions
- [ ] Configuration file (no hardcoded values)
- [ ] Code follows style guide (black, flake8)
- [ ] Functions have docstrings

### Portfolio/Career
- [ ] Projects on GitHub with good READMEs
- [ ] LinkedIn profile updated
- [ ] Resume tailored to target role
- [ ] Networking actively
- [ ] Applying to jobs (don't wait for "perfect")

---

## Key Takeaways

**Top 5 Mistakes:**
1. **Tutorial hell** → Build projects instead
2. **Data leakage** → Always split before preprocessing
3. **Overfitting** → Use cross-validation, simpler models
4. **Not tracking experiments** → Use MLflow or equivalent
5. **Waiting too long to apply** → Start applying at 70% ready

**Remember:**
- Everyone makes mistakes (including ML experts)
- Mistakes are learning opportunities
- Perfect is the enemy of good
- Ship code, get feedback, iterate

---

**Related guides:**
- [Quick Start Guide](./QUICK_START_GUIDE.md) - Avoiding beginner mistakes from day 1
- [Debugging ML Models](../guides/production/DEBUGGING_ML_MODELS.md) - Troubleshooting specific issues
- [ML Engineering Best Practices](../guides/production/ML_ENGINEERING_BEST_PRACTICES.md) - How to do things right

---

*Last Updated: October 30, 2025*
*Part of: [Complete ML/AI Engineering Curriculum](../README.md)*
