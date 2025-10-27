# ML Experimentation and A/B Testing

**Complete Guide to Experiment Design, A/B Testing, and Causal Inference for ML Systems**

*Last Updated: October 27, 2025*

---

## Table of Contents

1. [Introduction](#introduction)
2. [Statistical Foundations](#statistical-foundations)
3. [Experiment Design](#experiment-design)
4. [A/B Testing for ML Models](#ab-testing-for-ml-models)
5. [Multi-Armed Bandits](#multi-armed-bandits)
6. [Causal Inference](#causal-inference)
7. [Advanced Techniques](#advanced-techniques)
8. [Implementation Guide](#implementation-guide)
9. [Common Pitfalls](#common-pitfalls)
10. [Interview Questions](#interview-questions)

---

## Introduction

ML experimentation and A/B testing are critical for validating model improvements in production. They answer the question: "Does my new model actually improve business metrics?"

**Why Experimentation Matters:**
- **Validate improvements**: Offline metrics don't always correlate with business impact
- **Make data-driven decisions**: Avoid opinions, rely on statistics
- **Reduce risk**: Gradually roll out changes
- **Learn continuously**: Understand what works and what doesn't

**Key Concepts:**
- **A/B Test**: Compare two variants (control vs treatment)
- **Statistical Significance**: Is the difference real or random?
- **Statistical Power**: Probability of detecting real effects
- **Effect Size**: Magnitude of the difference
- **Business Metrics**: What you actually care about (revenue, engagement)

**Example Scenarios:**
- **Recommendation System**: New model increases click-through rate by 5%
- **Search Ranking**: Algorithm change improves user satisfaction
- **Fraud Detection**: Updated model reduces false positives
- **Pricing Model**: Dynamic pricing increases revenue

---

## Statistical Foundations

### 1. Hypothesis Testing

**Null Hypothesis (H₀)**: No difference between control and treatment
**Alternative Hypothesis (H₁)**: There is a difference

**Test Procedure:**
1. Choose significance level (α, typically 0.05)
2. Calculate test statistic
3. Compute p-value
4. Decision: Reject H₀ if p-value < α

```python
import numpy as np
from scipy import stats

def hypothesis_test(control, treatment, alpha=0.05):
    """
    Two-sample t-test for comparing means

    Args:
        control: Control group data
        treatment: Treatment group data
        alpha: Significance level

    Returns:
        Dictionary with test results
    """
    # Calculate statistics
    control_mean = np.mean(control)
    treatment_mean = np.mean(treatment)

    # Perform t-test
    t_stat, p_value = stats.ttest_ind(control, treatment)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(control) + np.var(treatment)) / 2)
    cohens_d = (treatment_mean - control_mean) / pooled_std

    # Confidence interval (95%)
    diff = treatment_mean - control_mean
    se = pooled_std * np.sqrt(1/len(control) + 1/len(treatment))
    ci_lower = diff - 1.96 * se
    ci_upper = diff + 1.96 * se

    # Decision
    significant = p_value < alpha

    return {
        'control_mean': control_mean,
        'treatment_mean': treatment_mean,
        'difference': diff,
        'relative_lift': (diff / control_mean) * 100,
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': significant,
        'cohens_d': cohens_d,
        'ci_95': (ci_lower, ci_upper)
    }

# Example usage
control = np.random.normal(100, 15, 1000)  # Control group
treatment = np.random.normal(105, 15, 1000)  # Treatment group (5% lift)

results = hypothesis_test(control, treatment)
print(f"Control mean: {results['control_mean']:.2f}")
print(f"Treatment mean: {results['treatment_mean']:.2f}")
print(f"Relative lift: {results['relative_lift']:.2f}%")
print(f"P-value: {results['p_value']:.4f}")
print(f"Significant: {results['significant']}")
print(f"Cohen's d: {results['cohens_d']:.3f}")
print(f"95% CI: ({results['ci_95'][0]:.2f}, {results['ci_95'][1]:.2f})")
```

### 2. Sample Size Calculation

**Question**: How many samples do I need to detect a meaningful difference?

**Power Analysis:**
- **α (significance level)**: Probability of false positive (typically 0.05)
- **β (Type II error)**: Probability of false negative (typically 0.20)
- **Power (1-β)**: Probability of detecting real effect (typically 0.80)
- **Effect size**: Minimum detectable effect (MDE)

```python
from statsmodels.stats.power import TTestIndPower

def calculate_sample_size(
    baseline_mean,
    baseline_std,
    mde_percent,
    alpha=0.05,
    power=0.80
):
    """
    Calculate required sample size for A/B test

    Args:
        baseline_mean: Control group mean
        baseline_std: Control group standard deviation
        mde_percent: Minimum detectable effect (%)
        alpha: Significance level
        power: Statistical power

    Returns:
        Required sample size per group
    """
    # Calculate effect size (Cohen's d)
    mde = baseline_mean * (mde_percent / 100)
    effect_size = mde / baseline_std

    # Calculate sample size
    power_analysis = TTestIndPower()
    sample_size = power_analysis.solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        alternative='two-sided'
    )

    return {
        'sample_size_per_group': int(np.ceil(sample_size)),
        'total_sample_size': int(np.ceil(sample_size * 2)),
        'effect_size': effect_size,
        'mde': mde,
        'mde_percent': mde_percent
    }

# Example: Detect 5% improvement in conversion rate
result = calculate_sample_size(
    baseline_mean=0.10,  # 10% baseline conversion
    baseline_std=0.30,   # Binary outcome std = sqrt(p*(1-p))
    mde_percent=5,       # Detect 5% relative improvement
    alpha=0.05,
    power=0.80
)

print(f"Required sample size per group: {result['sample_size_per_group']:,}")
print(f"Total sample size: {result['total_sample_size']:,}")
print(f"MDE: {result['mde']:.4f} ({result['mde_percent']}%)")
```

### 3. Multiple Testing Correction

When running multiple tests, probability of false positives increases.

**Problem**: Run 20 tests at α=0.05, expect 1 false positive even if no real effects.

**Solutions:**

**1. Bonferroni Correction** (Conservative):
```python
def bonferroni_correction(p_values, alpha=0.05):
    """Apply Bonferroni correction for multiple testing"""
    n_tests = len(p_values)
    corrected_alpha = alpha / n_tests
    significant = [p < corrected_alpha for p in p_values]

    return {
        'original_alpha': alpha,
        'corrected_alpha': corrected_alpha,
        'significant': significant,
        'n_significant': sum(significant)
    }
```

**2. Benjamini-Hochberg (FDR)** (Less Conservative):
```python
from statsmodels.stats.multitest import multipletests

def fdr_correction(p_values, alpha=0.05):
    """Apply FDR correction (Benjamini-Hochberg)"""
    reject, p_adjusted, _, _ = multipletests(
        p_values,
        alpha=alpha,
        method='fdr_bh'
    )

    return {
        'significant': reject,
        'adjusted_p_values': p_adjusted,
        'n_significant': sum(reject)
    }

# Example
p_values = [0.001, 0.02, 0.03, 0.15, 0.45]
bonf_result = bonferroni_correction(p_values)
fdr_result = fdr_correction(p_values)

print(f"Bonferroni: {sum(bonf_result['significant'])} significant")
print(f"FDR: {sum(fdr_result['significant'])} significant")
```

---

## Experiment Design

### 1. Randomization

**Goal**: Ensure control and treatment groups are comparable.

**Methods:**

**A. Simple Randomization:**
```python
import random

def simple_randomization(user_ids, treatment_ratio=0.5):
    """Randomly assign users to control or treatment"""
    random.shuffle(user_ids)
    split_point = int(len(user_ids) * treatment_ratio)

    return {
        'treatment': user_ids[:split_point],
        'control': user_ids[split_point:]
    }
```

**B. Stratified Randomization** (Better):
```python
def stratified_randomization(users_df, strata_column, treatment_ratio=0.5):
    """
    Randomize within strata to ensure balance

    Args:
        users_df: DataFrame with user_id and strata columns
        strata_column: Column to stratify by (e.g., 'country', 'age_group')
        treatment_ratio: Proportion in treatment

    Returns:
        DataFrame with assigned group
    """
    def assign_group(group):
        n_treatment = int(len(group) * treatment_ratio)
        shuffled = group.sample(frac=1)
        shuffled['group'] = ['treatment'] * n_treatment + \
                           ['control'] * (len(group) - n_treatment)
        return shuffled

    assigned = users_df.groupby(strata_column).apply(assign_group)
    return assigned.reset_index(drop=True)

# Example
import pandas as pd

users = pd.DataFrame({
    'user_id': range(1000),
    'country': np.random.choice(['US', 'UK', 'DE'], 1000)
})

assigned = stratified_randomization(users, 'country', treatment_ratio=0.5)

# Check balance
print(assigned.groupby(['country', 'group']).size().unstack())
```

**C. Hash-Based Randomization** (Deterministic):
```python
import hashlib

def hash_based_assignment(user_id, treatment_ratio=0.5, salt='experiment_v1'):
    """
    Deterministic assignment based on user_id hash
    Same user always gets same assignment
    """
    hash_input = f"{user_id}_{salt}".encode()
    hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)

    # Map hash to [0, 1]
    normalized = (hash_value % 10000) / 10000

    return 'treatment' if normalized < treatment_ratio else 'control'

# Example
user_ids = [f"user_{i}" for i in range(1000)]
assignments = [hash_based_assignment(uid) for uid in user_ids]

treatment_pct = sum(1 for a in assignments if a == 'treatment') / len(assignments)
print(f"Treatment %: {treatment_pct:.2%}")
```

### 2. Sample Ratio Mismatch (SRM)

**Problem**: Unequal sample sizes between groups can indicate bugs.

```python
def check_sample_ratio_mismatch(control_size, treatment_size, expected_ratio=0.5):
    """
    Check for Sample Ratio Mismatch

    Uses chi-square test to detect if observed ratio differs from expected
    """
    total = control_size + treatment_size
    expected_treatment = total * expected_ratio
    expected_control = total * (1 - expected_ratio)

    # Chi-square test
    chi2 = ((treatment_size - expected_treatment)**2 / expected_treatment +
            (control_size - expected_control)**2 / expected_control)

    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    observed_ratio = treatment_size / total

    return {
        'control_size': control_size,
        'treatment_size': treatment_size,
        'expected_ratio': expected_ratio,
        'observed_ratio': observed_ratio,
        'chi2_statistic': chi2,
        'p_value': p_value,
        'srm_detected': p_value < 0.001,  # Very low threshold for SRM
        'warning': 'POTENTIAL BUG!' if p_value < 0.001 else 'OK'
    }

# Example: Detect SRM
result = check_sample_ratio_mismatch(
    control_size=10000,
    treatment_size=9500,  # Should be ~10000
    expected_ratio=0.5
)

print(f"SRM detected: {result['srm_detected']}")
print(f"P-value: {result['p_value']:.6f}")
print(f"Status: {result['warning']}")
```

### 3. Experiment Duration

**How long to run experiment?**

```python
def calculate_experiment_duration(
    daily_users,
    sample_size_needed,
    traffic_allocation=1.0
):
    """
    Calculate how long to run experiment

    Args:
        daily_users: Number of users per day
        sample_size_needed: Required sample size (per group)
        traffic_allocation: Fraction of traffic in experiment

    Returns:
        Duration in days
    """
    daily_users_in_experiment = daily_users * traffic_allocation
    total_needed = sample_size_needed * 2  # Control + treatment

    duration_days = np.ceil(total_needed / daily_users_in_experiment)

    return {
        'duration_days': int(duration_days),
        'duration_weeks': duration_days / 7,
        'total_users_needed': total_needed,
        'daily_users_in_experiment': daily_users_in_experiment
    }

# Example
result = calculate_experiment_duration(
    daily_users=100000,
    sample_size_needed=10000,
    traffic_allocation=0.1  # 10% of traffic
)

print(f"Run experiment for {result['duration_days']} days ({result['duration_weeks']:.1f} weeks)")
```

---

## A/B Testing for ML Models

### 1. Complete A/B Test Implementation

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class MLModelABTest:
    """Complete A/B testing framework for ML models"""

    def __init__(self, experiment_name, control_model, treatment_model):
        self.experiment_name = experiment_name
        self.control_model = control_model
        self.treatment_model = treatment_model

        # Tracking
        self.results = []
        self.start_time = datetime.now()

    def assign_user(self, user_id, method='hash', treatment_ratio=0.5):
        """Assign user to control or treatment"""
        if method == 'hash':
            return hash_based_assignment(user_id, treatment_ratio)
        elif method == 'random':
            return 'treatment' if np.random.random() < treatment_ratio else 'control'
        else:
            raise ValueError(f"Unknown method: {method}")

    def run_prediction(self, user_id, features):
        """Make prediction and track results"""
        # Assign user
        group = self.assign_user(user_id)

        # Select model
        model = self.treatment_model if group == 'treatment' else self.control_model

        # Make prediction
        prediction = model.predict(features)

        # Track
        self.results.append({
            'user_id': user_id,
            'group': group,
            'prediction': prediction,
            'timestamp': datetime.now()
        })

        return prediction

    def log_outcome(self, user_id, outcome):
        """Log actual outcome for user"""
        # Find prediction
        for result in self.results:
            if result['user_id'] == user_id:
                result['outcome'] = outcome
                break

    def analyze_results(self, metric='conversion_rate'):
        """Analyze A/B test results"""
        df = pd.DataFrame(self.results)

        # Filter complete observations (have outcome)
        df_complete = df[df['outcome'].notna()]

        if len(df_complete) == 0:
            return {"error": "No complete observations yet"}

        # Calculate metrics by group
        if metric == 'conversion_rate':
            control_data = df_complete[df_complete['group'] == 'control']['outcome']
            treatment_data = df_complete[df_complete['group'] == 'treatment']['outcome']

        # Run statistical test
        test_results = hypothesis_test(control_data, treatment_data)

        # Add experiment metadata
        test_results['experiment_name'] = self.experiment_name
        test_results['duration'] = datetime.now() - self.start_time
        test_results['control_n'] = len(control_data)
        test_results['treatment_n'] = len(treatment_data)

        return test_results

    def get_recommendation(self):
        """Get recommendation: launch treatment or stick with control?"""
        results = self.analyze_results()

        if 'error' in results:
            return "Not enough data"

        if results['significant'] and results['relative_lift'] > 0:
            return f"✅ LAUNCH: Treatment is {results['relative_lift']:.2f}% better (p={results['p_value']:.4f})"
        elif results['significant'] and results['relative_lift'] < 0:
            return f"❌ DON'T LAUNCH: Treatment is {abs(results['relative_lift']):.2f}% worse"
        else:
            return f"⏸️  NO DECISION: Not statistically significant (p={results['p_value']:.4f})"

# Usage example
from sklearn.ensemble import RandomForestClassifier

# Train models
control_model = RandomForestClassifier(n_estimators=50)
treatment_model = RandomForestClassifier(n_estimators=100)

# Mock training
X_train = np.random.randn(1000, 10)
y_train = np.random.randint(0, 2, 1000)
control_model.fit(X_train, y_train)
treatment_model.fit(X_train, y_train)

# Run A/B test
ab_test = MLModelABTest('rf_v2_test', control_model, treatment_model)

# Simulate predictions
for i in range(1000):
    user_id = f"user_{i}"
    features = np.random.randn(1, 10)
    prediction = ab_test.run_prediction(user_id, features)

    # Simulate outcome (treatment is 5% better)
    group = ab_test.assign_user(user_id)
    base_prob = 0.10
    prob = base_prob * 1.05 if group == 'treatment' else base_prob
    outcome = 1 if np.random.random() < prob else 0

    ab_test.log_outcome(user_id, outcome)

# Analyze
results = ab_test.analyze_results()
print(f"\n=== A/B Test Results ===")
print(f"Control conversion: {results['control_mean']:.2%}")
print(f"Treatment conversion: {results['treatment_mean']:.2%}")
print(f"Relative lift: {results['relative_lift']:.2f}%")
print(f"P-value: {results['p_value']:.4f}")
print(f"\n{ab_test.get_recommendation()}")
```

### 2. Sequential Testing (Early Stopping)

Traditional A/B tests require waiting for full sample size. Sequential testing allows peeking.

```python
class SequentialABTest:
    """Sequential A/B testing with early stopping"""

    def __init__(self, alpha=0.05, power=0.80):
        self.alpha = alpha
        self.power = power
        self.results = {'control': [], 'treatment': []}

    def add_observation(self, group, value):
        """Add new observation"""
        self.results[group].append(value)

    def sequential_test(self):
        """
        Perform sequential test using Alpha Spending Function
        Allows peeking without inflating Type I error
        """
        control = np.array(self.results['control'])
        treatment = np.array(self.results['treatment'])

        if len(control) < 100 or len(treatment) < 100:
            return {'decision': 'continue', 'reason': 'insufficient data'}

        # Calculate test statistic
        _, p_value = stats.ttest_ind(control, treatment)

        # O'Brien-Fleming boundary (conservative)
        n_looks = len(control) // 100  # Assume we look every 100 samples
        alpha_spending = self.alpha / np.sqrt(n_looks)

        if p_value < alpha_spending:
            lift = (np.mean(treatment) - np.mean(control)) / np.mean(control) * 100
            decision = 'stop' if lift > 0 else 'stop_negative'
            return {
                'decision': decision,
                'p_value': p_value,
                'lift': lift,
                'samples': len(control)
            }

        return {'decision': 'continue', 'p_value': p_value}

# Example
seq_test = SequentialABTest()

# Simulate data collection
for i in range(1000):
    seq_test.add_observation('control', np.random.normal(100, 15))
    seq_test.add_observation('treatment', np.random.normal(105, 15))  # 5% better

    # Check every 100 samples
    if (i + 1) % 100 == 0:
        result = seq_test.sequential_test()
        if result['decision'] == 'stop':
            print(f"Early stop at {i+1} samples!")
            print(f"Lift: {result['lift']:.2f}%")
            break
```

### 3. Monitoring A/B Tests

```python
import matplotlib.pyplot as plt

class ABTestMonitor:
    """Monitor A/B test in real-time"""

    def __init__(self):
        self.history = []

    def log_snapshot(self, control_data, treatment_data):
        """Log current state of A/B test"""
        control_mean = np.mean(control_data)
        treatment_mean = np.mean(treatment_data)
        lift = (treatment_mean - control_mean) / control_mean * 100

        _, p_value = stats.ttest_ind(control_data, treatment_data)

        self.history.append({
            'timestamp': datetime.now(),
            'control_n': len(control_data),
            'treatment_n': len(treatment_data),
            'control_mean': control_mean,
            'treatment_mean': treatment_mean,
            'lift_percent': lift,
            'p_value': p_value,
            'significant': p_value < 0.05
        })

    def plot_progress(self):
        """Plot A/B test progress over time"""
        df = pd.DataFrame(self.history)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Sample size over time
        axes[0, 0].plot(df.index, df['control_n'], label='Control', marker='o')
        axes[0, 0].plot(df.index, df['treatment_n'], label='Treatment', marker='o')
        axes[0, 0].set_title('Sample Size Over Time')
        axes[0, 0].set_xlabel('Check #')
        axes[0, 0].set_ylabel('Sample Size')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Plot 2: Conversion rate over time
        axes[0, 1].plot(df.index, df['control_mean'], label='Control', marker='o')
        axes[0, 1].plot(df.index, df['treatment_mean'], label='Treatment', marker='o')
        axes[0, 1].set_title('Conversion Rate Over Time')
        axes[0, 1].set_xlabel('Check #')
        axes[0, 1].set_ylabel('Conversion Rate')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Plot 3: Lift over time
        axes[1, 0].plot(df.index, df['lift_percent'], marker='o', color='green')
        axes[1, 0].axhline(y=0, color='black', linestyle='--')
        axes[1, 0].set_title('Relative Lift Over Time')
        axes[1, 0].set_xlabel('Check #')
        axes[1, 0].set_ylabel('Lift (%)')
        axes[1, 0].grid(True)

        # Plot 4: P-value over time
        axes[1, 1].plot(df.index, df['p_value'], marker='o', color='red')
        axes[1, 1].axhline(y=0.05, color='black', linestyle='--', label='α=0.05')
        axes[1, 1].set_title('P-value Over Time')
        axes[1, 1].set_xlabel('Check #')
        axes[1, 1].set_ylabel('P-value')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        return fig
```

---

## Multi-Armed Bandits

Multi-armed bandits balance exploration (trying new options) and exploitation (using best known option).

**Use Case**: Dynamically allocate traffic to best-performing variant.

### 1. Epsilon-Greedy

```python
class EpsilonGreedy:
    """Epsilon-greedy multi-armed bandit"""

    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon

        # Track statistics
        self.counts = np.zeros(n_arms)  # Number of pulls per arm
        self.values = np.zeros(n_arms)  # Average reward per arm

    def select_arm(self):
        """Select which arm to pull"""
        if np.random.random() < self.epsilon:
            # Explore: random arm
            return np.random.randint(self.n_arms)
        else:
            # Exploit: best arm
            return np.argmax(self.values)

    def update(self, arm, reward):
        """Update statistics after observing reward"""
        self.counts[arm] += 1

        # Incremental average
        n = self.counts[arm]
        value = self.values[arm]
        self.values[arm] = ((n - 1) / n) * value + (1 / n) * reward

# Simulation
bandit = EpsilonGreedy(n_arms=3, epsilon=0.1)

# True reward probabilities (unknown to algorithm)
true_rewards = [0.1, 0.15, 0.12]  # Arm 1 is best

cumulative_reward = 0
for i in range(1000):
    arm = bandit.select_arm()

    # Simulate reward
    reward = 1 if np.random.random() < true_rewards[arm] else 0

    bandit.update(arm, reward)
    cumulative_reward += reward

print("Arm pulls:", bandit.counts)
print("Estimated values:", bandit.values)
print("True values:", true_rewards)
print(f"Total reward: {cumulative_reward}")
```

### 2. Thompson Sampling (Better)

```python
class ThompsonSampling:
    """Thompson Sampling with Beta priors"""

    def __init__(self, n_arms):
        self.n_arms = n_arms

        # Beta distribution parameters
        self.alpha = np.ones(n_arms)  # Successes + 1
        self.beta = np.ones(n_arms)   # Failures + 1

    def select_arm(self):
        """Sample from posterior and select best"""
        samples = [
            np.random.beta(self.alpha[i], self.beta[i])
            for i in range(self.n_arms)
        ]
        return np.argmax(samples)

    def update(self, arm, reward):
        """Update posterior distribution"""
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

    def get_best_arm(self):
        """Get arm with highest expected value"""
        expected_values = self.alpha / (self.alpha + self.beta)
        return np.argmax(expected_values)

# Simulation
ts = ThompsonSampling(n_arms=3)
true_rewards = [0.1, 0.15, 0.12]

cumulative_reward = 0
for i in range(1000):
    arm = ts.select_arm()
    reward = 1 if np.random.random() < true_rewards[arm] else 0

    ts.update(arm, reward)
    cumulative_reward += reward

print("Expected values:", ts.alpha / (ts.alpha + ts.beta))
print("True values:", true_rewards)
print(f"Best arm: {ts.get_best_arm()}")
print(f"Total reward: {cumulative_reward}")
```

### 3. Contextual Bandits (ML Models)

```python
from sklearn.linear_model import SGDClassifier

class ContextualBandit:
    """Contextual bandit using online learning"""

    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.models = [
            SGDClassifier(loss='log', warm_start=True)
            for _ in range(n_arms)
        ]
        self.epsilon = 0.1

    def select_arm(self, context):
        """Select arm based on context"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)

        # Predict reward for each arm
        predictions = []
        for model in self.models:
            try:
                pred = model.predict_proba([context])[0][1]
            except:
                pred = 0.5  # Default if model not trained
            predictions.append(pred)

        return np.argmax(predictions)

    def update(self, arm, context, reward):
        """Update model for selected arm"""
        self.models[arm].partial_fit([context], [reward], classes=[0, 1])

# Example: Personalized recommendations
cb = ContextualBandit(n_arms=3)

# Simulate 1000 users
for _ in range(1000):
    # User context (features)
    context = np.random.randn(10)

    # Select which model to show
    arm = cb.select_arm(context)

    # Simulate reward (arm 1 is best for high feature 0)
    prob = 0.1 + 0.1 * (arm == 1 and context[0] > 0)
    reward = 1 if np.random.random() < prob else 0

    # Update
    cb.update(arm, context, reward)
```

---

## Causal Inference

Understanding causation (not just correlation) is critical for ML.

**Question**: Does changing X cause Y to change?

### 1. Potential Outcomes Framework

```python
def estimate_causal_effect(df, treatment_col, outcome_col):
    """
    Estimate Average Treatment Effect (ATE)

    ATE = E[Y|T=1] - E[Y|T=0]
    """
    treated = df[df[treatment_col] == 1][outcome_col]
    control = df[df[treatment_col] == 0][outcome_col]

    ate = treated.mean() - control.mean()

    # Standard error
    se = np.sqrt(treated.var()/len(treated) + control.var()/len(control))

    # Confidence interval
    ci_lower = ate - 1.96 * se
    ci_upper = ate + 1.96 * se

    return {
        'ate': ate,
        'se': se,
        'ci_95': (ci_lower, ci_upper),
        'treated_mean': treated.mean(),
        'control_mean': control.mean()
    }
```

### 2. Propensity Score Matching

Match treated and control units with similar characteristics.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

def propensity_score_matching(df, treatment_col, outcome_col, covariates):
    """
    Estimate causal effect using propensity score matching

    Args:
        df: DataFrame
        treatment_col: Treatment indicator column
        outcome_col: Outcome column
        covariates: List of covariate columns

    Returns:
        Estimated ATE
    """
    X = df[covariates].values
    T = df[treatment_col].values
    Y = df[outcome_col].values

    # Step 1: Estimate propensity scores
    ps_model = LogisticRegression()
    ps_model.fit(X, T)
    propensity_scores = ps_model.predict_proba(X)[:, 1]

    # Step 2: Match treated to control units
    treated_idx = np.where(T == 1)[0]
    control_idx = np.where(T == 0)[0]

    # Find nearest neighbor in propensity score
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(propensity_scores[control_idx].reshape(-1, 1))

    matches = []
    for idx in treated_idx:
        ps = propensity_scores[idx]
        match_idx = nn.kneighbors([[ps]], return_distance=False)[0][0]
        control_match = control_idx[match_idx]
        matches.append((idx, control_match))

    # Step 3: Calculate ATE on matched sample
    ate = np.mean([Y[t] - Y[c] for t, c in matches])

    return {
        'ate': ate,
        'n_matches': len(matches),
        'propensity_scores': propensity_scores
    }

# Example
df = pd.DataFrame({
    'treatment': np.random.binomial(1, 0.5, 1000),
    'covariate1': np.random.randn(1000),
    'covariate2': np.random.randn(1000),
})

# Outcome depends on covariates and treatment
df['outcome'] = (
    2 * df['covariate1'] +
    1 * df['covariate2'] +
    5 * df['treatment'] +  # True causal effect = 5
    np.random.randn(1000)
)

result = propensity_score_matching(
    df,
    treatment_col='treatment',
    outcome_col='outcome',
    covariates=['covariate1', 'covariate2']
)

print(f"Estimated ATE: {result['ate']:.2f} (true effect = 5)")
```

### 3. Difference-in-Differences

Estimate causal effect by comparing trends.

```python
def difference_in_differences(df, time_col, treatment_col, outcome_col):
    """
    Difference-in-differences estimation

    Compare change in treatment group to change in control group
    """
    # Before treatment
    before_treated = df[(df[time_col] == 'before') & (df[treatment_col] == 1)][outcome_col].mean()
    before_control = df[(df[time_col] == 'before') & (df[treatment_col] == 0)][outcome_col].mean()

    # After treatment
    after_treated = df[(df[time_col] == 'after') & (df[treatment_col] == 1)][outcome_col].mean()
    after_control = df[(df[time_col] == 'after') & (df[treatment_col] == 0)][outcome_col].mean()

    # Difference-in-differences
    did = (after_treated - before_treated) - (after_control - before_control)

    return {
        'did_estimate': did,
        'before_treated': before_treated,
        'before_control': before_control,
        'after_treated': after_treated,
        'after_control': after_control
    }

# Example
df = pd.DataFrame({
    'time': ['before']*500 + ['after']*500,
    'treatment': [0]*250 + [1]*250 + [0]*250 + [1]*250,
})

# Parallel trends + treatment effect
df['outcome'] = np.where(
    (df['time'] == 'after') & (df['treatment'] == 1),
    100 + np.random.randn(1000),  # Treatment effect = 10
    90 + np.random.randn(1000)
)

result = difference_in_differences(df, 'time', 'treatment', 'outcome')
print(f"DiD estimate: {result['did_estimate']:.2f}")
```

---

## Common Pitfalls

### 1. Peeking Problem

**Problem**: Checking p-value multiple times inflates false positive rate.

**Solution**: Use sequential testing or wait for planned sample size.

### 2. Novelty Effect

**Problem**: Treatment performs better initially due to newness.

**Solution**: Run experiment for 2-4 weeks to account for user adaptation.

### 3. Network Effects

**Problem**: Treatment affects control group (e.g., social networks).

**Solution**: Cluster randomization (randomize by geographic region, not user).

### 4. Seasonality

**Problem**: Day-of-week or seasonal effects confound results.

**Solution**: Run experiment for full weeks, match day-of-week between groups.

### 5. Simpson's Paradox

**Problem**: Aggregate results show opposite effect from subgroups.

**Example**:
```python
# Overall: Treatment looks worse
# But within each subgroup, treatment is better!

data = pd.DataFrame({
    'group': ['control']*100 + ['treatment']*100,
    'segment': ['A']*20 + ['B']*80 + ['A']*80 + ['B']*20,
    'conversion': (
        [1]*18 + [0]*2 +      # Control A: 90% (small)
        [1]*40 + [0]*40 +     # Control B: 50% (large)
        [1]*76 + [0]*4 +      # Treatment A: 95% (large)
        [1]*12 + [0]*8        # Treatment B: 60% (small)
    )
})

# Overall conversion rates
print(data.groupby('group')['conversion'].mean())
# control: 0.58, treatment: 0.44 (treatment looks worse!)

# Segment-level conversion rates
print(data.groupby(['segment', 'group'])['conversion'].mean())
# A: control 90%, treatment 95% (treatment better)
# B: control 50%, treatment 60% (treatment better)
```

**Solution**: Analyze by important segments (stratify).

---

## Interview Questions

### Q1: How do you design an A/B test to evaluate a new recommendation model?

**Answer:**

**Step-by-Step Process:**

**1. Define Success Metrics:**
- **Primary metric**: Click-through rate (CTR) or conversion rate
- **Secondary metrics**: Revenue per user, engagement time
- **Guardrail metrics**: User satisfaction (don't harm), latency (<100ms)

**2. Calculate Sample Size:**
```python
# Want to detect 5% relative improvement in CTR
baseline_ctr = 0.10  # 10% baseline
mde = 0.05  # 5% relative = 0.005 absolute
sample_size = calculate_sample_size(
    baseline_mean=baseline_ctr,
    baseline_std=np.sqrt(baseline_ctr * (1-baseline_ctr)),
    mde_percent=5,
    alpha=0.05,
    power=0.80
)
# Result: ~25,000 users per group
```

**3. Randomization:**
- **Method**: Hash-based randomization (deterministic)
- **Split**: 50/50 control vs treatment
- **Stratification**: Ensure balance across key segments (new vs returning users)

**4. Duration:**
- Minimum 2 weeks to account for day-of-week effects
- Longer if sample size requires it

**5. Analysis Plan:**
```python
# Primary analysis
results = hypothesis_test(control_ctr, treatment_ctr)

# Segment analysis (check for heterogeneous effects)
segments = ['new_users', 'returning_users', 'power_users']
for segment in segments:
    segment_results = analyze_segment(data, segment)

# Check assumptions
check_sample_ratio_mismatch(control_size, treatment_size)

# Multiple testing correction if analyzing many metrics
corrected_results = bonferroni_correction(p_values)
```

**6. Decision Criteria:**
- Launch if: p<0.05 AND lift>0 AND no negative guardrails
- Don't launch if: Treatment worse or guardrails harmed
- Run longer if: Not significant but promising trend

**Common Mistakes to Avoid:**
- ❌ Peeking at results and stopping early
- ❌ Not accounting for multiple comparisons
- ❌ Ignoring guardrail metrics
- ❌ Not checking for Sample Ratio Mismatch

---

### Q2: Explain the difference between A/B testing and multi-armed bandits. When would you use each?

**Answer:**

**A/B Testing:**
- **Goal**: Learn which variant is best
- **Approach**: Fixed traffic split (50/50), analyze at end
- **Pros**: Simple, well-understood statistics, clear conclusion
- **Cons**: Opportunity cost during test (waste traffic on inferior variant)

**Multi-Armed Bandits:**
- **Goal**: Maximize reward while learning
- **Approach**: Dynamic traffic allocation (exploit best, explore others)
- **Pros**: Minimize regret, adapt in real-time
- **Cons**: More complex, harder to interpret, need continuous feedback

**Comparison:**

| Aspect | A/B Test | Bandit |
|--------|----------|--------|
| Traffic split | Fixed | Dynamic |
| Duration | Fixed | Continuous |
| Statistical inference | Clear | Complex |
| Regret | High | Low |
| Best for | One-time changes | Ongoing optimization |

**When to Use A/B Testing:**
- ✅ Major product changes (need clear statistical answer)
- ✅ Regulatory requirements (need proof)
- ✅ Long-term effects (bandits optimize short-term)
- ✅ Small number of variants (2-3)

**When to Use Bandits:**
- ✅ Content optimization (headlines, images)
- ✅ Many variants (>5)
- ✅ High traffic (can learn quickly)
- ✅ Continuous optimization
- ✅ Fast feedback (clicks, not purchases)

**Example:**

**A/B Test**: Testing new recommendation algorithm
- Major change affecting all users
- Need statistical proof before full rollout
- Can wait 2 weeks for results
- **Use A/B test**

**Bandit**: Personalizing notification messages
- 10 different message templates
- Immediate feedback (open rate)
- Want to maximize opens while testing
- **Use Thompson Sampling bandit**

**Hybrid Approach** (Best of Both):
1. Use bandit to quickly narrow down to top 2-3 variants
2. Run A/B test between finalists for statistical rigor

---

### Q3: What is statistical power and why does it matter for A/B tests?

**Answer:**

**Statistical Power = Probability of detecting a real effect when it exists**

**Definition:**
- Power = 1 - β (where β = Type II error rate)
- Typically set to 0.80 (80% power)
- Means: 80% chance of detecting real effect if it exists

**Four Related Concepts:**

1. **α (Significance Level)**: False positive rate (typically 0.05)
2. **β (Type II Error)**: False negative rate (typically 0.20)
3. **Power (1-β)**: True positive rate (typically 0.80)
4. **Effect Size**: Magnitude of difference

**Trade-offs:**
```
Higher Power requires:
- Larger sample size
- Longer experiment
- Or only detect larger effects
```

**Example Calculation:**
```python
# Want 80% power to detect 5% improvement
baseline = 0.10  # 10% conversion rate
effect_size = 0.005 / 0.30  # Cohen's d
power = 0.80
alpha = 0.05

sample_size = calculate_sample_size(
    effect_size=effect_size,
    alpha=alpha,
    power=power
)
# Result: 25,000 per group

# Lower power (60%): only 14,000 per group
# Higher power (90%): 33,000 per group
```

**Why Power Matters:**

**Low Power (< 50%):**
- ❌ Likely to miss real improvements
- ❌ Waste time running inconclusive tests
- ❌ False negatives hurt business

**High Power (> 80%):**
- ✅ Reliably detect real improvements
- ✅ Confident in negative results
- ✅ Make better decisions

**Practical Implications:**

**Scenario 1: Underpowered Test**
- True effect: +5% conversion
- Power: 40%
- Result: 60% chance of "no significant difference"
- Conclusion: Might reject good model!

**Scenario 2: Well-Powered Test**
- True effect: +5% conversion
- Power: 80%
- Result: 80% chance of detecting it
- Conclusion: Likely to launch good model

**How to Increase Power:**
1. **Increase sample size** (most common)
2. **Increase effect size** (improve model more)
3. **Reduce variance** (stratification, better metrics)
4. **Increase α** (accept more false positives)

**Rule of Thumb:**
- Always calculate required sample size BEFORE starting test
- Don't stop early just because p>0.05 (may be underpowered)
- Report power when reporting null results

---

### Q4: How do you handle multiple metrics in an A/B test?

**Answer:**

Testing multiple metrics increases false positive risk. Need careful approach.

**Problem:**
```python
# Test 20 metrics at α=0.05
# Expected false positives: 20 × 0.05 = 1
# Even if no real effects!
```

**Solutions:**

**1. Hierarchical Metrics:**
```
Primary Metric (α=0.05):
  └─ Click-through rate

Secondary Metrics (α=0.05, only if primary significant):
  ├─ Conversion rate
  └─ Revenue per user

Guardrail Metrics (α=0.05, must not worsen):
  ├─ User satisfaction
  ├─ Latency
  └─ Error rate
```

**Decision Logic:**
```python
def make_decision(primary_results, secondary_results, guardrail_results):
    # Must win on primary
    if not primary_results['significant'] or primary_results['lift'] <= 0:
        return "DON'T LAUNCH: No improvement on primary metric"

    # Must not harm guardrails
    for metric, result in guardrail_results.items():
        if result['significant'] and result['lift'] < 0:
            return f"DON'T LAUNCH: Harmed guardrail {metric}"

    # Secondary metrics are bonus
    secondary_wins = sum(1 for r in secondary_results.values()
                         if r['significant'] and r['lift'] > 0)

    return f"LAUNCH: Primary win + {secondary_wins} secondary wins"
```

**2. Multiple Testing Correction:**

**Bonferroni (Conservative):**
```python
n_metrics = 5
corrected_alpha = 0.05 / n_metrics  # 0.01
# Only declare significant if p < 0.01
```

**FDR (Less Conservative):**
```python
from statsmodels.stats.multitest import multipletests

p_values = [0.001, 0.02, 0.03, 0.15, 0.45]
reject, p_adjusted, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
```

**3. Composite Metrics:**

Create single metric combining multiple goals:
```python
def overall_evaluation_criterion(metrics):
    """
    OEC = weighted sum of normalized metrics
    """
    weights = {
        'ctr': 0.4,
        'conversion': 0.3,
        'revenue': 0.2,
        'satisfaction': 0.1
    }

    oec = sum(weights[m] * metrics[m] for m in weights)
    return oec
```

**4. Pre-Registration:**
Decide BEFORE seeing data:
- Which metrics matter
- Which is primary
- How to handle multiples

**Best Practices:**

1. **Always have ONE primary metric**
2. **Separate secondary from guardrails**
3. **Apply corrections if testing many metrics**
4. **Report all metrics (don't cherry-pick)**
5. **Pre-register analysis plan**

**Example Decision Framework:**
```
Metrics:
  [P] Click-through rate: +5% (p=0.01) ✅
  [S] Conversion rate: +2% (p=0.08) ➖
  [S] Revenue: +3% (p=0.05) ✅
  [G] Latency: +10ms (p=0.001) ❌

Decision: DON'T LAUNCH
Reason: Harmed guardrail metric (latency)
```

---

### Q5: What are common reasons an A/B test might show misleading results?

**Answer:**

**1. Sample Ratio Mismatch (SRM)**
- **Problem**: Unequal group sizes due to bugs
- **Example**: Expected 50/50, got 55/45
- **Impact**: Biased results, wrong conclusion
- **Detection**: Chi-square test on sample sizes
- **Solution**: Fix bug before analyzing

**2. Novelty Effect**
- **Problem**: Users try new feature just because it's new
- **Example**: New UI gets more clicks initially, then drops
- **Impact**: Overestimate long-term effect
- **Solution**: Run for 2-4 weeks, look at trends

**3. Primacy Effect**
- **Problem**: Users prefer familiar interface
- **Example**: Redesign performs worse initially
- **Impact**: Underestimate long-term benefit
- **Solution**: Run longer, segment by new vs existing users

**4. Seasonality**
- **Problem**: Day-of-week or calendar effects
- **Example**: Different behavior Friday vs Monday
- **Impact**: Groups experience different conditions
- **Solution**: Run full weeks, stratify by day

**5. Network Effects**
- **Problem**: Treatment affects control group
- **Example**: Users see content from treatment users
- **Impact**: Underestimate effect (dilution)
- **Solution**: Cluster randomization

**6. Selection Bias**
- **Problem**: Non-random attrition from groups
- **Example**: Treatment is slower, impatient users leave
- **Impact**: Remaining users unrepresentative
- **Solution**: Analyze all assigned users (ITT)

**7. Peeking / Multiple Testing**
- **Problem**: Checking p-value repeatedly
- **Example**: Stop when p<0.05
- **Impact**: Inflated false positive rate
- **Solution**: Pre-specify analysis time or use sequential testing

**8. Regression to the Mean**
- **Problem**: Extreme values tend toward average
- **Example**: Target worst-performing users, see "improvement"
- **Impact**: Attribute natural variation to treatment
- **Solution**: Compare to control group

**9. Simpson's Paradox**
- **Problem**: Aggregate shows opposite of subgroups
- **Example**: Treatment better in each segment but worse overall
- **Impact**: Wrong conclusion
- **Solution**: Analyze by important segments

**10. Interaction Effects**
- **Problem**: Multiple concurrent experiments interfere
- **Example**: Two tests both slow down page
- **Impact**: Confounded results
- **Solution**: Orthogonal experiment design

**Checklist to Avoid Misleading Results:**

✅ Check for SRM (sample ratio mismatch)
✅ Run for sufficient duration (2+ weeks)
✅ Account for day-of-week effects
✅ Check for novelty/primacy effects
✅ Verify random assignment worked
✅ Analyze all assigned users (not just active)
✅ Correct for multiple testing
✅ Segment analysis (heterogeneous effects)
✅ Check for concurrent experiments
✅ Pre-register analysis plan

---

## Conclusion

ML experimentation and A/B testing are essential skills for production ML engineers. Key takeaways:

1. **Always use randomized experiments** to validate model improvements
2. **Calculate required sample size** before starting
3. **Pre-register analysis plan** to avoid p-hacking
4. **Account for multiple testing** when analyzing many metrics
5. **Run for sufficient duration** (2+ weeks) to avoid seasonality
6. **Check for common pitfalls** (SRM, novelty effect, peeking)
7. **Use bandits** for ongoing optimization, A/B tests for major changes
8. **Understand causal inference** to make better decisions

**Typical ML A/B Test Workflow:**
```
1. Define metrics and success criteria
2. Calculate sample size (power analysis)
3. Implement randomization (hash-based)
4. Run experiment (2+ weeks)
5. Check data quality (SRM, balance)
6. Analyze results (hypothesis test)
7. Make decision (launch or iterate)
8. Monitor post-launch (validate results)
```

**Tools:**
- **Python**: scipy.stats, statsmodels, scikit-learn
- **Platforms**: Optimizely, VWO, Google Optimize
- **Custom**: Build your own framework (code provided above)

This ensures data-driven decisions and continuous improvement of ML systems.

---

*Last Updated: October 27, 2025*
*Version: 1.0*
*Maintained by: ML Engineering Team*
