# Module 03: Statistics for Machine Learning

## 🎯 Learning Objectives
By the end of this module, you will:
- Master probability theory and distributions essential for ML
- Understand statistical inference and hypothesis testing
- Apply Bayesian thinking to machine learning problems
- Use statistical concepts for model evaluation and selection

## 📚 Why Statistics Matters in ML

*Source: "All of Statistics" - Larry Wasserman, Chapter 1*
> "Statistics is the science of learning from data. Machine learning is, fundamentally, applied statistics with computational focus."

### The Statistical Learning Framework
*Source: "The Elements of Statistical Learning" - Hastie, Tibshirani, Friedman, Chapter 2*

Machine Learning can be viewed as:
- **Input**: Training data D = {(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)}
- **Goal**: Learn function f: X → Y that minimizes expected loss
- **Challenge**: We only have finite sample, not entire population

## 🎲 Probability Theory Foundations

### 3.1 Basic Probability Concepts

#### Sample Spaces and Events
*Source: "Introduction to Mathematical Statistics" - Hogg, McKean, Craig, Chapter 1*

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

# Example: Coin flip experiment
def coin_flip_experiment(n_flips=1000):
    """
    Demonstrate law of large numbers with coin flips
    """
    np.random.seed(42)
    flips = np.random.choice(['H', 'T'], size=n_flips)

    # Calculate running proportion of heads
    cumulative_heads = np.cumsum(flips == 'H')
    proportions = cumulative_heads / np.arange(1, n_flips + 1)

    plt.figure(figsize=(12, 5))

    # Plot 1: Running proportion
    plt.subplot(1, 2, 1)
    plt.plot(proportions, 'b-', alpha=0.7)
    plt.axhline(y=0.5, color='red', linestyle='--', label='True probability = 0.5')
    plt.xlabel('Number of Flips')
    plt.ylabel('Proportion of Heads')
    plt.title('Law of Large Numbers')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Histogram of last 100 flips
    plt.subplot(1, 2, 2)
    last_100 = flips[-100:]
    unique, counts = np.unique(last_100, return_counts=True)
    plt.bar(unique, counts, alpha=0.7)
    plt.xlabel('Outcome')
    plt.ylabel('Frequency')
    plt.title('Last 100 Flips')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"Final proportion of heads: {proportions[-1]:.4f}")
    print(f"Theoretical probability: 0.5000")
    print(f"Difference: {abs(proportions[-1] - 0.5):.4f}")

coin_flip_experiment()
```

#### Conditional Probability and Independence
*Source: "Pattern Recognition and Machine Learning" - Bishop, Chapter 1*

**Bayes' Theorem**: The foundation of probabilistic ML
```
P(A|B) = P(B|A) × P(A) / P(B)
```

```python
def bayes_disease_example():
    """
    Classic medical diagnosis example illustrating Bayes' theorem

    Source: "Think Bayes" - Allen Downey, Chapter 1
    """
    print("🏥 Medical Diagnosis with Bayes' Theorem")
    print("=" * 50)

    # Prior probabilities
    P_disease = 0.01  # 1% of population has disease
    P_no_disease = 0.99

    # Likelihoods (test accuracy)
    P_positive_given_disease = 0.90     # Sensitivity (true positive rate)
    P_negative_given_disease = 0.10     # 1 - sensitivity
    P_positive_given_no_disease = 0.09  # 1 - specificity (false positive rate)
    P_negative_given_no_disease = 0.91  # Specificity (true negative rate)

    # Total probability of positive test (law of total probability)
    P_positive = (P_positive_given_disease * P_disease +
                  P_positive_given_no_disease * P_no_disease)

    # Posterior probabilities (what we actually want to know)
    P_disease_given_positive = (P_positive_given_disease * P_disease) / P_positive
    P_disease_given_negative = (P_negative_given_disease * P_disease) / (1 - P_positive)

    print(f"📊 Test Characteristics:")
    print(f"Sensitivity (true positive rate): {P_positive_given_disease:.1%}")
    print(f"Specificity (true negative rate): {P_negative_given_no_disease:.1%}")
    print(f"Disease prevalence: {P_disease:.1%}")

    print(f"\\n🎯 Results:")
    print(f"P(Disease | Positive Test) = {P_disease_given_positive:.3f} = {P_disease_given_positive:.1%}")
    print(f"P(Disease | Negative Test) = {P_disease_given_negative:.4f} = {P_disease_given_negative:.2%}")

    print(f"\\n💡 Interpretation:")
    print(f"Even with a positive test result, there's only a {P_disease_given_positive:.1%} chance of having the disease!")
    print(f"This is because the disease is rare (low prior probability).")

    # Visualization
    create_bayes_visualization(P_disease, P_positive_given_disease, P_positive_given_no_disease)

    return P_disease_given_positive

def create_bayes_visualization(prior, sensitivity, false_positive_rate):
    """Create visual representation of Bayes' theorem"""

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Population breakdown
    pop_size = 10000
    diseased = int(pop_size * prior)
    healthy = pop_size - diseased

    true_positives = int(diseased * sensitivity)
    false_negatives = diseased - true_positives
    false_positives = int(healthy * false_positive_rate)
    true_negatives = healthy - false_positives

    # Confusion matrix
    cm_data = np.array([[true_negatives, false_positives],
                       [false_negatives, true_positives]])

    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', ax=axes[0],
               xticklabels=['Negative Test', 'Positive Test'],
               yticklabels=['No Disease', 'Disease'])
    axes[0].set_title('Population Breakdown (N=10,000)')
    axes[0].set_ylabel('True Condition')
    axes[0].set_xlabel('Test Result')

    # Posterior probabilities
    categories = ['Disease | Positive', 'Disease | Negative', 'No Disease | Positive', 'No Disease | Negative']
    probabilities = [
        true_positives / (true_positives + false_positives),
        false_negatives / (false_negatives + true_negatives),
        false_positives / (true_positives + false_positives),
        true_negatives / (false_negatives + true_negatives)
    ]

    colors = ['red', 'orange', 'orange', 'green']
    bars = axes[1].bar(range(len(categories)), probabilities, color=colors, alpha=0.7)
    axes[1].set_title('Posterior Probabilities')
    axes[1].set_ylabel('Probability')
    axes[1].set_xticklabels(categories, rotation=45, ha='right')

    # Add value labels on bars
    for bar, prob in zip(bars, probabilities):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{prob:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

# Run the example
bayes_disease_example()
```

### 3.2 Probability Distributions

#### The Normal Distribution - The King of ML
*Source: "Pattern Recognition and Machine Learning" - Bishop, Chapter 2*

```python
def explore_normal_distribution():
    """
    Comprehensive exploration of the normal distribution

    Why it's crucial in ML:
    1. Central Limit Theorem
    2. Maximum likelihood estimation
    3. Error assumptions in regression
    4. Prior distributions in Bayesian methods
    """

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('📊 The Normal Distribution in Machine Learning', fontsize=16, fontweight='bold')

    # 1. Different normal distributions
    x = np.linspace(-6, 6, 1000)

    params = [(0, 1), (0, 2), (2, 1), (-1, 0.5)]
    colors = ['blue', 'red', 'green', 'orange']

    for (mu, sigma), color in zip(params, colors):
        y = stats.norm.pdf(x, mu, sigma)
        axes[0, 0].plot(x, y, color=color, linewidth=2,
                       label=f'μ={mu}, σ={sigma}')

    axes[0, 0].set_title('Normal Distributions with Different Parameters')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('Probability Density')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Central Limit Theorem demonstration
    sample_sizes = [1, 5, 20, 50]

    for i, n in enumerate(sample_sizes):
        # Generate sample means from uniform distribution
        n_experiments = 1000
        sample_means = []

        for _ in range(n_experiments):
            # Draw n samples from uniform distribution
            samples = np.random.uniform(-3, 3, n)
            sample_means.append(np.mean(samples))

        axes[0, 1].hist(sample_means, bins=30, alpha=0.6,
                       label=f'n={n}', density=True)

    axes[0, 1].set_title('Central Limit Theorem')
    axes[0, 1].set_xlabel('Sample Mean')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 68-95-99.7 Rule
    mu, sigma = 0, 1
    x_rule = np.linspace(-4, 4, 1000)
    y_rule = stats.norm.pdf(x_rule, mu, sigma)

    axes[1, 0].plot(x_rule, y_rule, 'b-', linewidth=2)

    # Shade different regions
    x_68 = x_rule[np.abs(x_rule) <= 1]
    y_68 = stats.norm.pdf(x_68, mu, sigma)
    axes[1, 0].fill_between(x_68, y_68, alpha=0.3, color='green', label='68% (μ ± σ)')

    x_95 = x_rule[np.abs(x_rule) <= 2]
    y_95 = stats.norm.pdf(x_95, mu, sigma)
    axes[1, 0].fill_between(x_95, y_95, alpha=0.2, color='orange', label='95% (μ ± 2σ)')

    axes[1, 0].set_title('68-95-99.7 Rule')
    axes[1, 0].set_xlabel('Standard Deviations from Mean')
    axes[1, 0].set_ylabel('Probability Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Q-Q plot for normality testing
    # Generate mixed data: some normal, some not
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, 1000)
    exponential_data = np.random.exponential(1, 1000)

    # Q-Q plot for normal data
    stats.probplot(normal_data, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot: Testing for Normality')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("🧠 Why Normal Distribution is Central to ML:")
    print("=" * 50)
    print("1. **Central Limit Theorem**: Sample means approach normal distribution")
    print("2. **Maximum Likelihood**: Many ML algorithms assume normal errors")
    print("3. **Gaussian Processes**: Entire ML field based on normal distributions")
    print("4. **Neural Networks**: Weight initialization often uses normal distribution")
    print("5. **Feature Engineering**: Many transformations aim to make data more normal")

explore_normal_distribution()
```

#### Other Important Distributions
*Source: "All of Statistics" - Wasserman, Chapters 3-4*

```python
def compare_distributions():
    """Compare important probability distributions used in ML"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('🎲 Important Probability Distributions for ML', fontsize=16, fontweight='bold')

    x_continuous = np.linspace(0, 10, 1000)
    x_discrete = np.arange(0, 21)

    # 1. Exponential Distribution (survival analysis, waiting times)
    lambda_param = 1.5
    exp_pdf = stats.expon.pdf(x_continuous, scale=1/lambda_param)
    axes[0, 0].plot(x_continuous, exp_pdf, 'b-', linewidth=2)
    axes[0, 0].set_title(f'Exponential Distribution (λ={lambda_param})')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('Probability Density')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Beta Distribution (modeling probabilities, Bayesian priors)
    alpha, beta = 2, 5
    x_beta = np.linspace(0, 1, 1000)
    beta_pdf = stats.beta.pdf(x_beta, alpha, beta)
    axes[0, 1].plot(x_beta, beta_pdf, 'r-', linewidth=2)
    axes[0, 1].set_title(f'Beta Distribution (α={alpha}, β={beta})')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('Probability Density')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Gamma Distribution (modeling positive continuous data)
    alpha_gamma = 2
    beta_gamma = 2
    gamma_pdf = stats.gamma.pdf(x_continuous, alpha_gamma, scale=1/beta_gamma)
    axes[0, 2].plot(x_continuous, gamma_pdf, 'g-', linewidth=2)
    axes[0, 2].set_title(f'Gamma Distribution (α={alpha_gamma}, β={beta_gamma})')
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('Probability Density')
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Poisson Distribution (count data, rare events)
    lambda_poisson = 3
    poisson_pmf = stats.poisson.pmf(x_discrete, lambda_poisson)
    axes[1, 0].bar(x_discrete, poisson_pmf, alpha=0.7, color='purple')
    axes[1, 0].set_title(f'Poisson Distribution (λ={lambda_poisson})')
    axes[1, 0].set_xlabel('k')
    axes[1, 0].set_ylabel('Probability Mass')
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Binomial Distribution (binary outcomes, classification)
    n_trials = 20
    p_success = 0.3
    binomial_pmf = stats.binom.pmf(x_discrete, n_trials, p_success)
    axes[1, 1].bar(x_discrete, binomial_pmf, alpha=0.7, color='orange')
    axes[1, 1].set_title(f'Binomial Distribution (n={n_trials}, p={p_success})')
    axes[1, 1].set_xlabel('k')
    axes[1, 1].set_ylabel('Probability Mass')
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Student's t-Distribution (small samples, robust statistics)
    degrees_freedom = [1, 5, 30]
    x_t = np.linspace(-4, 4, 1000)

    for df in degrees_freedom:
        t_pdf = stats.t.pdf(x_t, df)
        axes[1, 2].plot(x_t, t_pdf, linewidth=2, label=f'df={df}')

    # Add normal for comparison
    normal_pdf = stats.norm.pdf(x_t, 0, 1)
    axes[1, 2].plot(x_t, normal_pdf, 'k--', linewidth=2, label='Normal')

    axes[1, 2].set_title('Student\'s t-Distribution')
    axes[1, 2].set_xlabel('x')
    axes[1, 2].set_ylabel('Probability Density')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("🎯 ML Applications of Each Distribution:")
    print("=" * 50)
    print("📈 **Exponential**: Survival analysis, time-to-event modeling")
    print("📊 **Beta**: Bayesian priors for probabilities, click-through rates")
    print("📉 **Gamma**: Modeling positive continuous variables, Bayesian priors")
    print("🔢 **Poisson**: Count data, rare event modeling, recommendation systems")
    print("🎯 **Binomial**: Classification problems, A/B testing")
    print("📏 **t-Distribution**: Small sample inference, robust statistics")

compare_distributions()
```

## 📊 Statistical Inference

### 3.3 Hypothesis Testing
*Source: "Introduction to Statistical Learning" - James et al., Chapter 3*

```python
def hypothesis_testing_ml():
    """
    Demonstrate hypothesis testing in ML context

    Example: Testing if a new ML model performs significantly better
    """
    print("🧪 Hypothesis Testing for Model Comparison")
    print("=" * 50)

    # Simulate model performance (accuracy scores)
    np.random.seed(42)

    # Model A (baseline): mean accuracy = 0.85
    model_a_scores = np.random.normal(0.85, 0.05, 30)

    # Model B (new): mean accuracy = 0.87
    model_b_scores = np.random.normal(0.87, 0.05, 30)

    # Hypothesis test
    # H₀: μ_B - μ_A = 0 (no difference)
    # H₁: μ_B - μ_A > 0 (model B is better)

    # Two-sample t-test
    t_statistic, p_value = stats.ttest_ind(model_b_scores, model_a_scores,
                                          alternative='greater')

    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(model_a_scores) - 1) * np.var(model_a_scores, ddof=1) +
                         (len(model_b_scores) - 1) * np.var(model_b_scores, ddof=1)) /
                        (len(model_a_scores) + len(model_b_scores) - 2))
    cohens_d = (np.mean(model_b_scores) - np.mean(model_a_scores)) / pooled_std

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Distribution comparison
    axes[0].hist(model_a_scores, alpha=0.7, label='Model A (Baseline)', bins=15, color='blue')
    axes[0].hist(model_b_scores, alpha=0.7, label='Model B (New)', bins=15, color='red')
    axes[0].axvline(np.mean(model_a_scores), color='blue', linestyle='--', linewidth=2)
    axes[0].axvline(np.mean(model_b_scores), color='red', linestyle='--', linewidth=2)
    axes[0].set_title('Model Performance Distributions')
    axes[0].set_xlabel('Accuracy Score')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Box plot comparison
    axes[1].boxplot([model_a_scores, model_b_scores],
                   labels=['Model A', 'Model B'])
    axes[1].set_title('Performance Comparison')
    axes[1].set_ylabel('Accuracy Score')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Results
    print(f"📊 Results:")
    print(f"Model A mean accuracy: {np.mean(model_a_scores):.4f} ± {np.std(model_a_scores):.4f}")
    print(f"Model B mean accuracy: {np.mean(model_b_scores):.4f} ± {np.std(model_b_scores):.4f}")
    print(f"\\n🧪 Hypothesis Test:")
    print(f"t-statistic: {t_statistic:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Cohen's d (effect size): {cohens_d:.4f}")

    alpha = 0.05
    if p_value < alpha:
        print(f"\\n✅ **Conclusion**: Reject H₀ (p < {alpha})")
        print(f"Model B performs significantly better than Model A")
    else:
        print(f"\\n❌ **Conclusion**: Fail to reject H₀ (p ≥ {alpha})")
        print(f"No significant difference between models")

    # Practical significance
    if cohens_d < 0.2:
        effect_interpretation = "small"
    elif cohens_d < 0.5:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"

    print(f"\\n💡 **Effect Size**: {effect_interpretation} practical difference")

    return t_statistic, p_value, cohens_d

# Run hypothesis testing example
t_stat, p_val, effect_size = hypothesis_testing_ml()
```

### 3.4 Confidence Intervals
*Source: "All of Statistics" - Wasserman, Chapter 9*

```python
def confidence_intervals_ml():
    """
    Demonstrate confidence intervals for ML model performance
    """
    print("📏 Confidence Intervals for Model Performance")
    print("=" * 50)

    # Simulate cross-validation scores
    np.random.seed(42)
    cv_scores = np.random.normal(0.85, 0.03, 10)  # 10-fold CV

    n = len(cv_scores)
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores, ddof=1)  # Sample standard deviation

    # Different confidence levels
    confidence_levels = [0.90, 0.95, 0.99]

    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean accuracy: {mean_score:.4f}")
    print(f"Standard deviation: {std_score:.4f}")
    print(f"Sample size: {n}")

    intervals = []

    for conf_level in confidence_levels:
        # t-distribution (appropriate for small samples)
        alpha = 1 - conf_level
        t_critical = stats.t.ppf(1 - alpha/2, df=n-1)

        # Margin of error
        margin_error = t_critical * (std_score / np.sqrt(n))

        # Confidence interval
        lower_bound = mean_score - margin_error
        upper_bound = mean_score + margin_error

        intervals.append((conf_level, lower_bound, upper_bound, margin_error))

        print(f"\\n{conf_level:.0%} Confidence Interval:")
        print(f"  [{lower_bound:.4f}, {upper_bound:.4f}]")
        print(f"  Margin of error: ±{margin_error:.4f}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Data and confidence intervals
    axes[0].scatter(range(n), cv_scores, color='blue', s=50, zorder=3, label='CV Scores')
    axes[0].axhline(mean_score, color='red', linestyle='-', linewidth=2, label='Mean')

    colors = ['lightblue', 'lightgreen', 'lightyellow']
    for i, (conf_level, lower, upper, margin) in enumerate(intervals):
        axes[0].axhspan(lower, upper, alpha=0.3, color=colors[i],
                       label=f'{conf_level:.0%} CI')

    axes[0].set_title('Cross-Validation Scores with Confidence Intervals')
    axes[0].set_xlabel('Fold')
    axes[0].set_ylabel('Accuracy Score')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Confidence interval widths
    conf_levels_pct = [int(cl * 100) for cl, _, _, _ in intervals]
    margins = [margin for _, _, _, margin in intervals]

    bars = axes[1].bar(conf_levels_pct, margins, color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[1].set_title('Confidence Interval Widths')
    axes[1].set_xlabel('Confidence Level (%)')
    axes[1].set_ylabel('Margin of Error')

    # Add value labels on bars
    for bar, margin in zip(bars, margins):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                    f'±{margin:.4f}', ha='center', va='bottom')

    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\\n💡 Interpretation:")
    print(f"We are 95% confident that the true model accuracy is between")
    print(f"{intervals[1][1]:.4f} and {intervals[1][2]:.4f}")
    print(f"\\nNote: Higher confidence → Wider intervals")

    return intervals

# Run confidence intervals example
ci_results = confidence_intervals_ml()
```

## 🔄 Resampling Methods

### 3.5 Bootstrap and Cross-Validation
*Source: "Introduction to Statistical Learning" - James et al., Chapter 5*

```python
def bootstrap_demonstration():
    """
    Demonstrate bootstrap resampling for estimating uncertainty
    """
    print("🔄 Bootstrap Resampling for ML")
    print("=" * 40)

    # Original dataset
    np.random.seed(42)
    original_data = np.random.exponential(2, 100)

    # Statistic we want to estimate (median)
    original_median = np.median(original_data)

    # Bootstrap resampling
    n_bootstrap = 1000
    bootstrap_medians = []

    for i in range(n_bootstrap):
        # Resample with replacement
        bootstrap_sample = np.random.choice(original_data, size=len(original_data),
                                          replace=True)
        bootstrap_medians.append(np.median(bootstrap_sample))

    bootstrap_medians = np.array(bootstrap_medians)

    # Bootstrap statistics
    bootstrap_mean = np.mean(bootstrap_medians)
    bootstrap_std = np.std(bootstrap_medians, ddof=1)

    # Bootstrap confidence interval (percentile method)
    ci_lower = np.percentile(bootstrap_medians, 2.5)
    ci_upper = np.percentile(bootstrap_medians, 97.5)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Original data
    axes[0].hist(original_data, bins=20, alpha=0.7, color='blue', density=True)
    axes[0].axvline(original_median, color='red', linestyle='--', linewidth=2,
                   label=f'Original Median = {original_median:.3f}')
    axes[0].set_title('Original Data Distribution')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Density')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Bootstrap distribution
    axes[1].hist(bootstrap_medians, bins=50, alpha=0.7, color='green', density=True)
    axes[1].axvline(bootstrap_mean, color='red', linestyle='-', linewidth=2,
                   label=f'Bootstrap Mean = {bootstrap_mean:.3f}')
    axes[1].axvline(ci_lower, color='orange', linestyle='--', linewidth=2)
    axes[1].axvline(ci_upper, color='orange', linestyle='--', linewidth=2,
                   label=f'95% CI = [{ci_lower:.3f}, {ci_upper:.3f}]')
    axes[1].set_title('Bootstrap Distribution of Median')
    axes[1].set_xlabel('Bootstrap Median')
    axes[1].set_ylabel('Density')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"📊 Bootstrap Results:")
    print(f"Original median: {original_median:.4f}")
    print(f"Bootstrap mean: {bootstrap_mean:.4f}")
    print(f"Bootstrap std: {bootstrap_std:.4f}")
    print(f"95% Bootstrap CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    print(f"\\n💡 Applications in ML:")
    print("• Estimating uncertainty in model predictions")
    print("• Creating confidence intervals for performance metrics")
    print("• Feature importance with uncertainty quantification")
    print("• Model selection with bootstrap validation")

    return bootstrap_medians, ci_lower, ci_upper

# Run bootstrap demonstration
bootstrap_results = bootstrap_demonstration()
```

## 📈 Bayesian Statistics for ML

### 3.6 Bayesian Thinking
*Source: "Pattern Recognition and Machine Learning" - Bishop, Chapters 1-3*

```python
def bayesian_linear_regression():
    """
    Demonstrate Bayesian approach to linear regression

    Shows uncertainty quantification vs point estimates
    """
    print("🎯 Bayesian Linear Regression")
    print("=" * 40)

    # Generate data
    np.random.seed(42)
    n_points = 20
    true_slope = 2.0
    true_intercept = 1.0
    noise_std = 0.3

    x = np.linspace(0, 1, n_points)
    y_true = true_intercept + true_slope * x
    y = y_true + noise_std * np.random.randn(n_points)

    # Bayesian linear regression using conjugate priors
    # Prior: Normal-Gamma distribution
    # Likelihood: Normal

    # Design matrix
    X = np.column_stack([np.ones(n_points), x])

    # Prior parameters (weakly informative)
    alpha_prior = 1e-3  # Precision of prior on weights
    beta_prior = 1e-3   # Precision of noise

    # Posterior parameters
    S_N_inv = alpha_prior * np.eye(2) + beta_prior * X.T @ X
    S_N = np.linalg.inv(S_N_inv)
    m_N = beta_prior * S_N @ X.T @ y

    # Posterior predictive distribution
    x_test = np.linspace(-0.2, 1.2, 100)
    X_test = np.column_stack([np.ones(100), x_test])

    # Mean prediction
    y_mean = X_test @ m_N

    # Predictive variance
    y_var = (1/beta_prior) + np.sum((X_test @ S_N) * X_test, axis=1)
    y_std = np.sqrt(y_var)

    # Sample from posterior for visualization
    n_samples = 10
    weight_samples = np.random.multivariate_normal(m_N, S_N, n_samples)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Data and posterior samples
    axes[0].scatter(x, y, color='blue', s=50, zorder=3, label='Data')
    axes[0].plot(x, y_true, 'g-', linewidth=2, label='True function')

    # Plot posterior samples
    for i, weights in enumerate(weight_samples):
        y_sample = X_test @ weights
        label = 'Posterior samples' if i == 0 else None
        axes[0].plot(x_test, y_sample, 'r-', alpha=0.3, linewidth=1, label=label)

    # Plot mean and uncertainty
    axes[0].plot(x_test, y_mean, 'r-', linewidth=2, label='Posterior mean')
    axes[0].fill_between(x_test, y_mean - 2*y_std, y_mean + 2*y_std,
                        alpha=0.2, color='red', label='95% credible interval')

    axes[0].set_title('Bayesian Linear Regression')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Posterior distribution of parameters
    # Sample more for better visualization
    weight_samples_viz = np.random.multivariate_normal(m_N, S_N, 1000)

    axes[1].scatter(weight_samples_viz[:, 1], weight_samples_viz[:, 0],
                   alpha=0.6, s=20)
    axes[1].axvline(true_slope, color='red', linestyle='--', linewidth=2,
                   label=f'True slope = {true_slope}')
    axes[1].axhline(true_intercept, color='green', linestyle='--', linewidth=2,
                   label=f'True intercept = {true_intercept}')
    axes[1].scatter(m_N[1], m_N[0], color='red', s=100, marker='x',
                   label='Posterior mean')

    axes[1].set_title('Posterior Distribution of Parameters')
    axes[1].set_xlabel('Slope')
    axes[1].set_ylabel('Intercept')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"📊 Results:")
    print(f"True parameters: slope = {true_slope:.3f}, intercept = {true_intercept:.3f}")
    print(f"Posterior mean: slope = {m_N[1]:.3f}, intercept = {m_N[0]:.3f}")
    print(f"Parameter uncertainty (std):")
    print(f"  Slope: ±{np.sqrt(S_N[1,1]):.3f}")
    print(f"  Intercept: ±{np.sqrt(S_N[0,0]):.3f}")

    print(f"\\n💡 Advantages of Bayesian Approach:")
    print("• Quantifies uncertainty in predictions")
    print("• Incorporates prior knowledge")
    print("• Provides full posterior distribution")
    print("• Natural way to handle small datasets")
    print("• Avoid overfitting through regularization")

    return m_N, S_N, y_mean, y_std

# Run Bayesian regression demonstration
bayesian_results = bayesian_linear_regression()
```

## 📚 Sources & Further Reading

### 📖 **Primary Textbooks** (with specific chapter references):

1. **"All of Statistics"** - Larry Wasserman
   - **🆓 Free Access**: University libraries, partial preview on Google Books
   - **📑 Key Chapters**:
     - Chapter 1: Probability *(sample spaces, conditional probability)*
     - Chapter 2: Random Variables *(distributions, expectation)*
     - Chapter 9: Confidence Intervals *(construction, interpretation)*
     - Chapter 10: Hypothesis Testing *(p-values, power)*

2. **"Pattern Recognition and Machine Learning"** - Christopher Bishop
   - **🆓 Free PDF**: [Microsoft Research](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
   - **📑 Key Chapters**:
     - Chapter 1: Introduction *(probability theory, decision theory)*
     - Chapter 2: Probability Distributions *(normal, beta, gamma)*
     - Chapter 3: Linear Models *(Bayesian linear regression)*

3. **"Think Stats"** - Allen Downey
   - **🆓 Free Online**: [https://greenteapress.com/wp/think-stats-2e/](https://greenteapress.com/wp/think-stats-2e/)
   - **💡 Best For**: Practical statistics with Python examples

4. **"Introduction to Statistical Learning"** - James, Witten, Hastie, Tibshirani
   - **🆓 Free PDF**: [https://www.statlearning.com/](https://www.statlearning.com/)
   - **📑 Key Chapters**:
     - Chapter 2: Statistical Learning *(bias-variance, resampling)*
     - Chapter 5: Resampling Methods *(bootstrap, cross-validation)*

### 📑 **Essential Papers**:

1. **"Bootstrap Methods: Another Look at the Jackknife"** - Bradley Efron (1979)
   - **Source**: Annals of Statistics
   - **💡 Impact**: Introduced bootstrap resampling

2. **"An Introduction to the Bootstrap"** - Efron & Tibshirani (1993)
   - **💡 Content**: Comprehensive bootstrap methodology

### 🌐 **Online Resources**:
- **Khan Academy Statistics**: [https://www.khanacademy.org/math/statistics-probability](https://www.khanacademy.org/math/statistics-probability)
- **Seeing Theory**: [https://seeing-theory.brown.edu/](https://seeing-theory.brown.edu/) *(Interactive probability visualizations)*

## ✅ Self-Assessment Exercises

### Exercise 1: Probability Distributions
```python
# Identify the appropriate distribution for each scenario:
scenarios = [
    "Number of website visits per day",
    "Time until next customer arrives",
    "Probability of success in A/B test",
    "Height measurements of adults",
    "Number of defects in manufacturing"
]
# Your answers: Poisson, Exponential, Beta, Normal, Poisson
```

### Exercise 2: Hypothesis Testing
Design a hypothesis test to compare two ML models with these requirements:
- Model A accuracy: 0.82 ± 0.05 (n=50)
- Model B accuracy: 0.85 ± 0.04 (n=50)
- Significance level: α = 0.05

### Exercise 3: Bayesian Thinking
Explain how Bayesian inference differs from frequentist inference in the context of:
- Parameter estimation
- Confidence vs credible intervals
- Handling of uncertainty

## 🚀 Next Steps

Ready to apply these statistical concepts? Continue to [Module 04: Programming for ML](../04_Programming/README.md) where you'll implement these ideas in Python!

---
*Estimated completion time: 6-8 hours*
*Prerequisites: Module 02 (Mathematics) recommended*