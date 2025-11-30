# Hyperparameter Tuning Guide

Master the art and science of finding optimal hyperparameters for your ML models.

## Table of Contents
1. [Understanding Hyperparameters](#understanding-hyperparameters)
2. [Search Strategies](#search-strategies)
3. [Advanced Optimization](#advanced-optimization)
4. [Tools and Frameworks](#tools-and-frameworks)
5. [Best Practices](#best-practices)
6. [Domain-Specific Tuning](#domain-specific-tuning)

---

## Understanding Hyperparameters

### What Are Hyperparameters?

**Hyperparameters** are configuration settings that control the learning process, set BEFORE training begins.

**Hyperparameters vs Parameters:**

| Hyperparameters | Parameters |
|----------------|------------|
| Set by user | Learned during training |
| Control learning process | Model weights/biases |
| Examples: learning rate, batch size | Examples: W, b in neural networks |
| Require tuning/search | Updated by optimization |

### Critical Hyperparameters by Model Type

#### Neural Networks
```python
hyperparameters = {
    # Architecture
    'num_layers': [2, 3, 4, 5],
    'hidden_size': [64, 128, 256, 512],
    'dropout': [0.1, 0.2, 0.3, 0.5],

    # Optimization
    'learning_rate': [1e-5, 1e-4, 1e-3, 1e-2],
    'batch_size': [16, 32, 64, 128],
    'optimizer': ['adam', 'sgd', 'adamw'],
    'weight_decay': [0, 1e-5, 1e-4, 1e-3],

    # Training
    'epochs': [50, 100, 200],
    'lr_scheduler': ['cosine', 'step', 'plateau'],

    # Regularization
    'l2_lambda': [0, 0.01, 0.001],
}
```

#### Tree-Based Models (XGBoost, LightGBM)
```python
xgboost_params = {
    # Tree structure
    'max_depth': [3, 5, 7, 9],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.2, 0.3],

    # Boosting
    'learning_rate': [0.01, 0.05, 0.1, 0.3],
    'n_estimators': [100, 200, 500, 1000],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],

    # Regularization
    'reg_alpha': [0, 0.1, 1, 10],  # L1
    'reg_lambda': [0, 0.1, 1, 10],  # L2
}
```

#### Transformers
```python
transformer_params = {
    # Architecture
    'num_layers': [6, 12, 24],
    'num_heads': [8, 12, 16],
    'hidden_size': [512, 768, 1024],
    'ff_dim': [2048, 3072, 4096],

    # Optimization
    'learning_rate': [1e-5, 3e-5, 5e-5],
    'warmup_steps': [500, 1000, 2000],
    'weight_decay': [0.01, 0.1],

    # Training
    'max_grad_norm': [0.5, 1.0, 2.0],
    'label_smoothing': [0, 0.1, 0.2],
}
```

---

## Search Strategies

**Theoretical Framework:**

**Hyperparameter Optimization as Black-Box Function:**
```
Problem: Find θ* = argmin_θ L(A_θ, D_val)

where:
- θ: Hyperparameter configuration
- A_θ: Model/algorithm with hyperparameters θ
- D_val: Validation dataset
- L: Loss/objective function

Challenges:
1. Non-convex, non-differentiable objective
2. Expensive function evaluations (training takes time)
3. High-dimensional search space
4. Noisy observations (stochastic training)
5. Mixed variable types (continuous, discrete, categorical)
```

#### Statistical Learning Theory for Hyperparameter Optimization

**Generalization Error Decomposition:**

**Theorem 1 (Bias-Variance-Optimization Tradeoff for Model Selection):**

For hyperparameter configuration θ, the expected test error decomposes as:

E[L_test(θ)] = E[L_train(θ)] + Complexity(θ) + Estimation_Error(n, θ)

where:
- E[L_train(θ)]: Training performance
- Complexity(θ): Model complexity penalty
- Estimation_Error(n, θ): Error from finite sample size n

**Detailed:**

E[L_test] = Bias²[A_θ] + Var[A_θ] + σ² + O(1/n)

**Hyperparameter Impact:**

- **Increasing model capacity** (e.g., more layers, larger hidden size):
  - Decreases Bias²
  - Increases Var
  - Increases O(1/n) term (more parameters to estimate)

- **Increasing regularization** (e.g., dropout, weight decay):
  - Increases Bias²
  - Decreases Var
  - Optimal λ balances bias-variance

**Theorem 2 (Validation Set Bias - Rao & Tibshirani, 1997):**

Let θ̂ = argmin_θ L(A_θ, D_val) be hyperparameters chosen via validation set of size n_val. Then:

E[L_test(θ̂)] ≥ min_θ E[L_test(θ)] + O(√(log k / n_val))

where k = number of hyperparameter configurations tried.

**Interpretation:**
- More configurations tried → higher overfitting to validation set
- Validation set acts as "indirect training data"
- Need n_val = Ω(k log k) to avoid significant overfitting

**Corollary (Validation Set Size):**

To select among k configurations with ε-accuracy:

n_val = Ω(k log k / ε²)

Example: k = 100 configurations, ε = 0.01 requires n_val ≥ 46,052

**Hold-out vs Cross-Validation Trade-off:**

**Hold-out:**
- Bias: E[L_val] ≈ E[L_test] (unbiased if val independent of train)
- Variance: High (single split)
- Computation: O(k) for k configurations

**K-fold CV:**
- Bias: Slight optimism (trains on (K-1)/K of data)
- Variance: Lower (K estimates averaged)
- Computation: O(K·k)

**Theorem 3 (K-Fold CV Variance Reduction):**

Var[CV_K] ≈ Var[Hold-out] · (1 + (K-1)ρ) / K²

where ρ = correlation between fold errors.

For independent folds (ρ = 0): Var[CV_K] = Var[Hold-out]/K

Practical: ρ ≈ 0.5, K = 5 → Variance reduction ≈ 40%

#### No Free Lunch Theorem for Hyperparameter Search

**Theorem 4 (NFL for Optimization - Wolpert & Macready, 1997):**

Averaged over all possible objective functions f: Θ → ℝ:

E_f[Performance(Algorithm_A)] = E_f[Performance(Algorithm_B)]

for any two search algorithms A and B.

**Implication:**
- No search strategy is universally better
- Bayesian optimization good for smooth functions
- Random search good for sparse important dimensions
- Grid search good when all dimensions equally important

**Practical Consequence:**

Leverage problem structure:
- Smoothness → Gaussian processes
- Low effective dimension → Random search
- Known bounds → Bounded optimization

#### Sample Complexity of Hyperparameter Search

**Theorem 5 (Random Search Sample Complexity - Bergstra & Bengio, 2012):**

For hyperparameter space Θ with distribution p(θ), define:

ε-good region: G_ε = {θ : L(θ) ≤ L* + ε}

If P(θ ∈ G_ε) = α, then random search with n samples finds ε-good solution with probability:

P(success) = 1 - (1-α)^n ≥ 1 - δ

Solving for n:

n ≥ log(δ) / log(1-α) ≈ log(1/δ) / α

**Example:**
- ε-good region α = 0.1 (10% of space is good)
- Confidence δ = 0.05 (95% success)
- Required samples: n ≥ log(20)/log(10/9) ≈ 28

**Grid Search Sample Complexity:**

For d-dimensional grid with m points per dimension:

Total configurations: m^d

To find ε-optimal with resolution m:

m = O(1/ε) per dimension

Total: n = O((1/ε)^d) - **exponential in d!**

**Curse of Dimensionality:**

Grid search requires exponentially many samples.
Random search requires only O(log(1/δ)/α) - **independent of d!**

**Theorem 6 (Effective Dimensionality - Bergstra & Bengio, 2012):**

If objective f(θ) depends on only d_eff ≪ d dimensions, random search finds ε-good solution in:

n = O(log(1/δ) / α_eff)

where α_eff is probability of hitting good region in effective dimensions.

**Empirical finding:** Most ML objectives have d_eff ≈ 2-5 even when d ≈ 10-20

This explains why random search often outperforms grid search!

#### Regret Bounds for Sequential Search

**Cumulative Regret:**

R_T = Σ_{t=1}^T [L(θ_t) - L(θ*)]

where θ_t is configuration tested at round t.

**Simple Regret:**

r_T = L(θ̂_T) - L(θ*)

where θ̂_T = argmin_{t≤T} L(θ_t) is best found so far.

**Theorem 7 (Random Search Regret):**

For random search over compact space Θ:

E[r_T] = O(1/T)

**Proof:** Each sample i.i.d., minimum of T samples concentrates at rate O(1/T). ∎

**Bayesian Optimization Regret:**

For GP-based Bayesian optimization with GP-UCB acquisition:

**Theorem 8 (GP-UCB Regret - Srinivas et al., 2010):**

R_T = O(√(T γ_T log T))

where γ_T is maximum information gain:

- Squared Exponential kernel: γ_T = O((log T)^{d+1})
- Matérn kernel: γ_T = O(T^{d/(2ν+d)} (log T)^{d/(2ν+d)})

**Interpretation:**
- Sublinear regret: O(√T polylog T)
- Much better than random search for smooth functions!
- Grows with dimension d via γ_T

**Comparison:**

| Method | Cumulative Regret | Simple Regret |
|--------|-------------------|---------------|
| Random | O(T) | O(1/T) |
| BO (GP-UCB) | O(√T polylog T) | O(1/√T polylog T) |
| Grid (d-dim) | O(T^{1-1/d}) | O(T^{-1/d}) |

BO optimal for smooth, low-dimensional problems!

#### Multi-Fidelity Optimization Theory

**Setting:** Evaluate f(θ, s) where s = fidelity (e.g., training epochs, data fraction)

**Goal:** Find θ* = argmin_θ f(θ, s_max) using cheaper evaluations at s < s_max

**Successive Halving (SHA - Jamieson & Talwalkar, 2016):**

**Algorithm:**
1. Start with n configurations, budget B
2. Round r: Evaluate surviving configs at fidelity s_r
3. Keep top 1/η fraction, increase fidelity by η

**Theorem 9 (SHA Sample Complexity):**

For Lipschitz objective with gap Δ between best and second-best:

SHA finds ε-optimal solution with probability ≥ 1-δ using:

B = O((log n / Δ²) · log(log n / δ))

evaluations at maximum fidelity.

**Reduction factor:** O(log n) compared to evaluating all n at max fidelity!

**Hyperband (Li et al., 2018):**

Runs SHA with multiple bracket schedules to handle unknown optimal fidelity allocation.

**Theorem 10 (Hyperband Regret):**

Hyperband achieves:

r_B = O((log B)^c / B^{1/2})

for constant c depending on problem smoothness.

**Practical Impact:**
- 10-100× speedup over full evaluation
- Near-optimal resource allocation
- Robust to fidelity choice

**Search Space Complexity:**
```
Grid Search Complexity:
If each hyperparameter has k values and there are d hyperparameters:
Total configurations: k^d (exponential!)

Example:
- 5 hyperparameters
- 10 values each
- Total: 10^5 = 100,000 configurations

At 1 hour per training → 11.4 years of compute!

Solution: Smart search strategies (Random, Bayesian, etc.)
```

---

### 1. Manual Search

**When to use:** Starting point, small search space, domain expertise.

```python
# Manual hyperparameter search
configs = [
    {'lr': 0.001, 'batch_size': 32, 'dropout': 0.2},
    {'lr': 0.0001, 'batch_size': 64, 'dropout': 0.3},
    {'lr': 0.01, 'batch_size': 16, 'dropout': 0.1},
]

results = []
for config in configs:
    model = create_model(**config)
    accuracy = train_and_evaluate(model)
    results.append({'config': config, 'accuracy': accuracy})

best_config = max(results, key=lambda x: x['accuracy'])
print(f"Best config: {best_config['config']}")
print(f"Best accuracy: {best_config['accuracy']:.4f}")
```

**Pros:** Simple, interpretable
**Cons:** Doesn't scale, misses optimal combinations

---

### 2. Grid Search

**Strategy:** Exhaustively try all combinations.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
}

# Grid search with cross-validation
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring='accuracy',
    verbose=2
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# Get all results
results_df = pd.DataFrame(grid_search.cv_results_)
results_df.sort_values('rank_test_score').head(10)
```

**Grid Search for Neural Networks:**
```python
import itertools
import torch
import torch.nn as nn

def grid_search_neural_net(X_train, y_train, X_val, y_val):
    param_grid = {
        'learning_rate': [0.001, 0.01, 0.1],
        'hidden_size': [64, 128, 256],
        'dropout': [0.2, 0.3, 0.5],
        'batch_size': [32, 64],
    }

    # Generate all combinations
    keys = param_grid.keys()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*param_grid.values())]

    print(f"Total configurations: {len(combinations)}")

    best_acc = 0
    best_config = None
    results = []

    for config in combinations:
        # Create model
        model = NeuralNetwork(
            input_size=X_train.shape[1],
            hidden_size=config['hidden_size'],
            dropout=config['dropout']
        )

        # Train
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=config['batch_size'],
            shuffle=True
        )

        for epoch in range(50):
            train_epoch(model, train_loader, optimizer)

        # Evaluate
        val_acc = evaluate(model, X_val, y_val)
        results.append({'config': config, 'val_acc': val_acc})

        if val_acc > best_acc:
            best_acc = val_acc
            best_config = config

        print(f"Config: {config} -> Val Acc: {val_acc:.4f}")

    return best_config, best_acc, results

best_config, best_acc, all_results = grid_search_neural_net(X_train, y_train, X_val, y_val)
```

**Pros:** Guaranteed to find best in grid, reproducible
**Cons:** Exponentially expensive, wastes computation on bad regions

**Complexity:** O(n^d) where n = values per param, d = number of params

---

### 3. Random Search

**Strategy:** Randomly sample from hyperparameter distributions.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Define distributions
param_distributions = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(10, 50),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': uniform(0.1, 0.9),
    'learning_rate': uniform(0.01, 0.3),
}

# Random search
random_search = RandomizedSearchCV(
    estimator=xgb.XGBClassifier(),
    param_distributions=param_distributions,
    n_iter=100,  # Number of random samples
    cv=5,
    n_jobs=-1,
    scoring='roc_auc',
    random_state=42,
    verbose=2
)

random_search.fit(X_train, y_train)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best score: {random_search.best_score_:.4f}")
```

**Custom Random Search for PyTorch:**
```python
import numpy as np

def random_search_pytorch(X_train, y_train, X_val, y_val, n_iter=50):
    param_distributions = {
        'learning_rate': lambda: 10 ** np.random.uniform(-5, -2),  # Log-uniform
        'hidden_size': lambda: np.random.choice([64, 128, 256, 512]),
        'num_layers': lambda: np.random.randint(2, 6),
        'dropout': lambda: np.random.uniform(0.1, 0.5),
        'batch_size': lambda: np.random.choice([16, 32, 64, 128]),
        'weight_decay': lambda: 10 ** np.random.uniform(-6, -3),
    }

    best_acc = 0
    best_config = None
    results = []

    for i in range(n_iter):
        # Sample configuration
        config = {param: sampler() for param, sampler in param_distributions.items()}

        print(f"\n[{i+1}/{n_iter}] Testing config: {config}")

        # Build model
        model = build_model(
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        )

        # Train
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=config['batch_size'],
            shuffle=True
        )

        # Early stopping
        best_val_acc = 0
        patience = 10
        no_improve = 0

        for epoch in range(100):
            train_epoch(model, train_loader, optimizer)
            val_acc = evaluate(model, X_val, y_val)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                break

        results.append({'config': config, 'val_acc': best_val_acc})

        if best_val_acc > best_acc:
            best_acc = best_val_acc
            best_config = config

        print(f"Val Acc: {best_val_acc:.4f} (Best: {best_acc:.4f})")

    return best_config, best_acc, results
```

**Why Random Search Works:**
- Often outperforms grid search with same budget
- Better explores high-dimensional spaces
- More likely to find good values for important hyperparameters

---

### 4. Bayesian Optimization

**Strategy:** Build probabilistic model of objective function, intelligently choose next points.

```python
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

# Define search space
search_space = {
    'learning_rate': Real(1e-6, 1e-1, prior='log-uniform'),
    'num_layers': Integer(2, 10),
    'hidden_size': Integer(32, 512),
    'dropout': Real(0.1, 0.5),
    'batch_size': Categorical([16, 32, 64, 128]),
    'weight_decay': Real(1e-6, 1e-2, prior='log-uniform'),
}

# Bayesian optimization
bayes_search = BayesSearchCV(
    estimator=create_model_sklearn_wrapper(),
    search_spaces=search_space,
    n_iter=50,
    cv=3,
    n_jobs=-1,
    scoring='accuracy',
    random_state=42
)

bayes_search.fit(X_train, y_train)
```

**Bayesian Optimization with Optuna:**
```python
import optuna

def objective(trial):
    # Define hyperparameter search space
    config = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
        'num_layers': trial.suggest_int('num_layers', 2, 6),
        'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256, 512]),
        'dropout': trial.suggest_uniform('dropout', 0.1, 0.5),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
        'weight_decay': trial.suggest_loguniform('weight_decay', 1e-6, 1e-3),
        'optimizer': trial.suggest_categorical('optimizer', ['adam', 'sgd', 'adamw']),
    }

    # Build model
    model = NeuralNetwork(
        input_size=X_train.shape[1],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)

    # Optimizer
    if config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'],
                                     weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'],
                                    weight_decay=config['weight_decay'], momentum=0.9)
    else:  # adamw
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                      weight_decay=config['weight_decay'])

    # Data loader
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=config['batch_size'],
        shuffle=True
    )

    # Training with pruning
    for epoch in range(100):
        train_loss = train_epoch(model, train_loader, optimizer)
        val_acc = evaluate(model, X_val, y_val)

        # Report intermediate value for pruning
        trial.report(val_acc, epoch)

        # Prune unpromising trials
        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_acc

# Create study
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20)
)

# Optimize
study.optimize(objective, n_trials=100, timeout=3600)

# Results
print(f"Best trial: {study.best_trial.number}")
print(f"Best value: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

# Visualization
optuna.visualization.plot_optimization_history(study).show()
optuna.visualization.plot_param_importances(study).show()
optuna.visualization.plot_parallel_coordinate(study).show()
```

**Mathematical Foundation of Bayesian Optimization:**

**Problem Formulation:**
```
Find: θ* = argmin_θ∈Θ f(θ)

where:
- f: ℝ^d → ℝ is expensive, black-box objective function
- Θ: Feasible hyperparameter space (bounded)
- d: Dimensionality of hyperparameter space

Constraints:
- f has no closed form
- ∇f is unavailable (non-differentiable)
- Each evaluation f(θ) is expensive (hours of training)
- Evaluations may be noisy: y = f(θ) + ε, ε ~ N(0, σ²_noise)
```

**Bayesian Optimization Algorithm:**

**1. Surrogate Model - Gaussian Process (GP):**
```
Model f as a random function with prior:
f ~ GP(μ(θ), k(θ, θ'))

where:
- μ(θ): Prior mean function (often μ(θ) = 0)
- k(θ, θ'): Covariance/kernel function

Common kernels:
a) Squared Exponential (RBF):
   k(θ, θ') = σ²·exp(-||θ - θ'||²/(2ℓ²))

b) Matérn 5/2:
   k(θ, θ') = σ²(1 + √5r/ℓ + 5r²/3ℓ²)·exp(-√5r/ℓ)
   where r = ||θ - θ'||

After observing D_n = {(θ_i, y_i)}^n_{i=1}, posterior is also GP:

f | D_n ~ GP(μ_n(θ), k_n(θ, θ'))

Posterior mean:
μ_n(θ) = k^T(K + σ²_noise I)^(-1)y

Posterior variance:
σ²_n(θ) = k(θ,θ) - k^T(K + σ²_noise I)^(-1)k

where:
- K_ij = k(θ_i, θ_j): n×n Gram matrix
- k = [k(θ, θ_1), ..., k(θ, θ_n)]^T: n×1 vector
- y = [y_1, ..., y_n]^T: observed values
```

**2. Acquisition Function α(θ | D_n):**

**a) Expected Improvement (EI):**
```
α_EI(θ) = E[max(f_min - f(θ), 0) | D_n]

where f_min = min{y_1, ..., y_n}

Closed form (assuming GP posterior):
α_EI(θ) = (f_min - μ_n(θ))·Φ(Z) + σ_n(θ)·φ(Z)

where:
- Z = (f_min - μ_n(θ))/σ_n(θ) if σ_n(θ) > 0, else 0
- Φ: Standard normal CDF
- φ: Standard normal PDF

Intuition:
- High μ_n(θ): Exploit (likely good performance)
- High σ_n(θ): Explore (high uncertainty)
```

**b) Upper Confidence Bound (UCB):**
```
α_UCB(θ) = μ_n(θ) - κ·σ_n(θ)  (for minimization)
         = μ_n(θ) + κ·σ_n(θ)  (for maximization)

where κ > 0 controls exploration-exploitation tradeoff

Typical: κ = √(2·log(n·π²/6δ)) for δ-PAC guarantee

Bounds (Srinivas et al., 2010):
With probability ≥ 1-δ, cumulative regret is:
R_n ≤ O(√(n·γ_n·log(n/δ)))
where γ_n is maximum information gain
```

**c) Probability of Improvement (PI):**
```
α_PI(θ) = P(f(θ) < f_min - ξ | D_n)
        = Φ((f_min - ξ - μ_n(θ))/σ_n(θ))

where ξ ≥ 0 is improvement threshold
```

**3. Optimization Loop:**
```
Algorithm: Bayesian Optimization
Input: f, Θ, T (budget), α (acquisition function)
Output: θ*

1. Initialize D_0 = {(θ_i, f(θ_i))}^n_0_{i=1}  (random or LHS)
2. Fit initial GP to D_0
3. For t = 1 to T:
   a) Find next point: θ_t = argmax_θ∈Θ α(θ | D_{t-1})
   b) Evaluate: y_t = f(θ_t)
   c) Augment data: D_t = D_{t-1} ∪ {(θ_t, y_t)}
   d) Update GP posterior with D_t
4. Return: θ* = argmin_{θ∈D_T} f(θ)

Computational Complexity per iteration:
- GP posterior: O(n³) for matrix inversion (n = number of observations)
- Acquisition optimization: Depends on method (gradient-free, multi-start)
- Total per iteration: O(n³ + m·d) where m = acquisition optimization evals
```

**Convergence Properties:**

**Theorem (GP-UCB, Srinivas et al., 2010):**
```
Under regularity conditions on k, with GP-UCB acquisition:

Cumulative Regret: R_T = Σ^T_{t=1} (f(θ_t) - f(θ*))

Bound: R_T ≤ O(√(T·γ_T·log T))

where γ_T is maximum information gain:
- For SE kernel in d dimensions: γ_T = O((log T)^(d+1))
- For Matérn kernel (ν > 1): γ_T = O(T^(d(d+1)/(2ν+d(d+1))) (log T)^(2))

This implies: R_T = O(poly(log T)) → sublinear regret!
```

**Practical Considerations:**

**Kernel Selection:**
- SE/RBF: Smooth functions, infinite differentiability
- Matérn 3/2, 5/2: Less smooth, more robust
- Choice affects both fit and convergence rate

**Hyperparameter Learning:**
- Kernel hyperparameters θ_k = {σ², ℓ, ...} learned via MLE:
  θ_k* = argmax_θ_k p(y | X, θ_k) = argmax_θ_k N(y | 0, K_θ_k + σ²I)

- Often optimized alongside BO (empirical Bayes)

**Scaling to High Dimensions:**
- Standard GP: O(n³) → infeasible for n > 10,000
- Solutions:
  * Sparse GPs: O(nm²) with m inducing points
  * Random features: O(nm) approximation
  * Deep kernel learning: Use DNN feature extractor

**Pros:**
- Sample efficient (few evaluations needed)
- Principled uncertainty quantification
- Handles noisy objectives
- Works with mixed variable types
- No-regret guarantees

**Cons:**
- O(n³) computational cost per iteration
- Struggles in very high dimensions (d > 20)
- Kernel/acquisition choice matters
- Can get stuck if GP misspecifies f

**When to Use:**
- Expensive function evaluations (>1 min per eval)
- Budget < 1000 evaluations
- Dimensions d < 20
- Need sample efficiency over wall-clock time

---

### 5. Hyperband / ASHA

**Strategy:** Adaptive resource allocation - kill bad trials early.

```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler

def train_function(config):
    """Training function for Ray Tune"""
    model = NeuralNetwork(
        input_size=784,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256)

    for epoch in range(100):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer)

        # Validate
        val_acc = evaluate(model, val_loader)

        # Report to Ray Tune
        tune.report(accuracy=val_acc, loss=train_loss)

# Configuration space
config = {
    'learning_rate': tune.loguniform(1e-5, 1e-2),
    'hidden_size': tune.choice([64, 128, 256, 512]),
    'num_layers': tune.randint(2, 7),
    'dropout': tune.uniform(0.1, 0.5),
    'batch_size': tune.choice([16, 32, 64, 128]),
    'weight_decay': tune.loguniform(1e-6, 1e-3),
}

# ASHA scheduler for early stopping
scheduler = ASHAScheduler(
    metric='accuracy',
    mode='max',
    max_t=100,  # max epochs
    grace_period=10,  # min epochs before stopping
    reduction_factor=3  # halve resources every iteration
)

# Run tuning
analysis = tune.run(
    train_function,
    config=config,
    num_samples=50,
    scheduler=scheduler,
    resources_per_trial={'gpu': 0.5},
    verbose=1
)

# Get best config
best_config = analysis.get_best_config(metric='accuracy', mode='max')
print(f"Best config: {best_config}")
```

**How ASHA Works:**
1. Start many trials with small budget
2. After each rung, keep top 1/η trials and give them more resources
3. Repeat until one trial uses full budget

**Pros:** Very efficient, parallelizable
**Cons:** Assumes smooth learning curves

---

## Advanced Optimization

### 1. Population-Based Training (PBT)

**Strategy:** Evolve a population of models, exploit good hyperparameters.

```python
from ray.tune.schedulers import PopulationBasedTraining

# PBT scheduler
pbt = PopulationBasedTraining(
    time_attr='training_iteration',
    metric='accuracy',
    mode='max',
    perturbation_interval=5,  # Perturb every 5 iterations
    hyperparam_mutations={
        'learning_rate': lambda: np.random.uniform(1e-5, 1e-2),
        'dropout': lambda: np.random.uniform(0.1, 0.5),
        'batch_size': [16, 32, 64, 128],
    }
)

# Run PBT
analysis = tune.run(
    train_function,
    config={
        'learning_rate': tune.uniform(1e-5, 1e-2),
        'dropout': tune.uniform(0.1, 0.5),
        'batch_size': tune.choice([16, 32, 64, 128]),
    },
    scheduler=pbt,
    num_samples=20,  # Population size
    stop={'training_iteration': 100},
)
```

**PBT Process:**
1. Train population in parallel
2. Periodically:
   - **Exploit:** Copy weights from better performers
   - **Explore:** Perturb hyperparameters
3. Continue training

---

### 2. Multi-Fidelity Optimization

**Strategy:** Use cheap approximations (fewer epochs, smaller data) to guide search.

```python
import optuna

def objective_with_fidelity(trial):
    # Suggest fidelity parameters
    n_epochs = trial.suggest_int('n_epochs', 10, 100, step=10)
    data_fraction = trial.suggest_uniform('data_fraction', 0.1, 1.0)

    # Regular hyperparameters
    config = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
        'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256]),
    }

    # Subsample data based on fidelity
    n_samples = int(len(X_train) * data_fraction)
    X_train_subset = X_train[:n_samples]
    y_train_subset = y_train[:n_samples]

    # Train for n_epochs
    model = create_model(**config)
    for epoch in range(n_epochs):
        train_epoch(model, X_train_subset, y_train_subset)

    val_acc = evaluate(model, X_val, y_val)

    # Penalize low fidelity to encourage convergence
    penalty = 0.1 * (1 - data_fraction) + 0.1 * (1 - n_epochs / 100)

    return val_acc - penalty

study = optuna.create_study(direction='maximize')
study.optimize(objective_with_fidelity, n_trials=200)
```

---

### 3. Transfer Learning for Hyperparameters

**Strategy:** Use hyperparameters from similar tasks as starting point.

```python
class HyperparameterTransfer:
    def __init__(self):
        self.task_configs = {}

    def save_best_config(self, task_name, config, performance):
        """Save successful configuration"""
        self.task_configs[task_name] = {
            'config': config,
            'performance': performance
        }

    def get_initial_config(self, new_task, task_similarity_fn):
        """Get warm-start configuration based on similar tasks"""
        if not self.task_configs:
            return None

        # Find most similar task
        similarities = {
            task: task_similarity_fn(new_task, task)
            for task in self.task_configs.keys()
        }

        most_similar_task = max(similarities, key=similarities.get)

        return self.task_configs[most_similar_task]['config']

# Usage
transfer = HyperparameterTransfer()

# Save configs from previous tasks
transfer.save_best_config(
    'mnist_cnn',
    {'learning_rate': 0.001, 'dropout': 0.2, 'batch_size': 64},
    performance=0.99
)

transfer.save_best_config(
    'cifar10_resnet',
    {'learning_rate': 0.01, 'dropout': 0.3, 'batch_size': 128},
    performance=0.94
)

# Get initial config for new similar task
def task_similarity(task1, task2):
    # Define similarity based on domain, architecture, etc.
    if 'cnn' in task1 and 'cnn' in task2:
        return 0.8
    elif 'resnet' in task1 and 'resnet' in task2:
        return 0.9
    return 0.1

initial_config = transfer.get_initial_config(
    'fashion_mnist_cnn',
    task_similarity
)

# Use as starting point for optimization
print(f"Starting optimization from: {initial_config}")
```

---

## Tools and Frameworks

### Optuna (Recommended)

**Best for:** PyTorch, flexible, good visualization

```python
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import pytorch_lightning as pl

# Define objective
def objective(trial):
    # Hyperparameters
    config = {
        'lr': trial.suggest_loguniform('lr', 1e-5, 1e-1),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'dropout': trial.suggest_uniform('dropout', 0.1, 0.5),
    }

    # Model
    model = LightningModel(**config)

    # Trainer with pruning
    trainer = pl.Trainer(
        max_epochs=100,
        gpus=1,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor='val_acc')]
    )

    trainer.fit(model)

    return trainer.callback_metrics['val_acc'].item()

# Study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Dashboard
optuna.dashboard.run_server(study)
```

### Ray Tune

**Best for:** Distributed tuning, complex schedules

```python
from ray import tune
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.suggest.hyperopt import HyperOptSearch

# Search algorithm
search_alg = OptunaSearch(metric='accuracy', mode='max')

# Early stopping
scheduler = ASHAScheduler(
    metric='accuracy',
    mode='max',
    max_t=100,
    grace_period=10,
    reduction_factor=3
)

# Run
analysis = tune.run(
    train_fn,
    config=search_space,
    num_samples=100,
    search_alg=search_alg,
    scheduler=scheduler,
    resources_per_trial={'gpu': 1},
    verbose=1
)
```

### Weights & Biases Sweeps

**Best for:** Experiment tracking + tuning

```yaml
# sweep.yaml
program: train.py
method: bayes
metric:
  name: val_accuracy
  goal: maximize
parameters:
  learning_rate:
    distribution: log_uniform
    min: 0.00001
    max: 0.1
  dropout:
    distribution: uniform
    min: 0.1
    max: 0.5
  batch_size:
    values: [16, 32, 64, 128]
```

```python
import wandb

# Initialize sweep
sweep_id = wandb.sweep(sweep_config, project='my-project')

# Training function
def train():
    wandb.init()
    config = wandb.config

    # Train model with config
    model = create_model(config)
    for epoch in range(100):
        train_loss = train_epoch(model, config)
        val_acc = validate(model)

        # Log metrics
        wandb.log({'val_accuracy': val_acc, 'train_loss': train_loss})

# Run sweep
wandb.agent(sweep_id, function=train, count=50)
```

---

## Best Practices

### 1. Start Simple, Then Optimize

```python
# Stage 1: Baseline
baseline_config = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'hidden_size': 128,
}
baseline_acc = train(baseline_config)

# Stage 2: One hyperparameter at a time
for lr in [0.0001, 0.001, 0.01]:
    acc = train({**baseline_config, 'learning_rate': lr})
    print(f"LR {lr}: {acc:.4f}")

# Stage 3: Joint optimization (after finding good ranges)
```

### 2. Use Log Scale for Learning Rates

```python
# BAD: Linear scale
learning_rates = [0.0001, 0.0002, 0.0003, ..., 0.001]

# GOOD: Log scale
learning_rates = [10**x for x in np.linspace(-5, -2, 20)]
# [1e-5, 1.58e-5, 2.51e-5, ..., 1e-2]
```

### 3. Early Stopping for Efficiency

```python
def objective_with_early_stopping(trial):
    model = create_model(trial.suggest_hyperparams())

    best_val_acc = 0
    patience = 10
    no_improve = 0

    for epoch in range(100):
        train_epoch(model)
        val_acc = validate(model)

        # Optuna pruning
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    return best_val_acc
```

### 4. Use Cross-Validation

```python
def objective_with_cv(trial, n_folds=5):
    config = trial.suggest_hyperparams()

    cv_scores = []
    for fold in range(n_folds):
        X_train_fold, X_val_fold = split_data(fold)

        model = create_model(config)
        train(model, X_train_fold)
        score = evaluate(model, X_val_fold)

        cv_scores.append(score)

    # Return mean CV score
    return np.mean(cv_scores)
```

### 5. Track Everything

```python
import mlflow

def train_with_tracking(config):
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params(config)

        # Train
        model = create_model(config)
        for epoch in range(100):
            train_loss = train_epoch(model)
            val_acc = validate(model)

            # Log metrics
            mlflow.log_metric('train_loss', train_loss, step=epoch)
            mlflow.log_metric('val_acc', val_acc, step=epoch)

        # Log final model
        mlflow.pytorch.log_model(model, 'model')

        return val_acc
```

---

## Domain-Specific Tuning

### Computer Vision

**Critical Hyperparameters:**
```python
cv_config = {
    # Data augmentation
    'augmentation_strength': trial.suggest_uniform('aug_strength', 0.0, 1.0),
    'mixup_alpha': trial.suggest_uniform('mixup_alpha', 0.0, 0.4),
    'cutmix_prob': trial.suggest_uniform('cutmix_prob', 0.0, 1.0),

    # Optimizer
    'learning_rate': trial.suggest_loguniform('lr', 1e-5, 1e-2),
    'momentum': trial.suggest_uniform('momentum', 0.85, 0.99),
    'weight_decay': trial.suggest_loguniform('weight_decay', 1e-5, 1e-2),

    # Architecture
    'backbone': trial.suggest_categorical('backbone', ['resnet50', 'efficientnet_b0', 'vit_base']),
    'pretrained': trial.suggest_categorical('pretrained', [True, False]),

    # Training
    'warmup_epochs': trial.suggest_int('warmup_epochs', 0, 10),
    'lr_schedule': trial.suggest_categorical('lr_schedule', ['cosine', 'step', 'plateau']),
}
```

### NLP / Transformers

**Critical Hyperparameters:**
```python
nlp_config = {
    # Architecture
    'num_layers': trial.suggest_int('num_layers', 6, 24),
    'num_heads': trial.suggest_categorical('num_heads', [8, 12, 16]),
    'hidden_size': trial.suggest_categorical('hidden_size', [512, 768, 1024]),

    # Optimization
    'learning_rate': trial.suggest_loguniform('lr', 1e-6, 1e-4),
    'warmup_ratio': trial.suggest_uniform('warmup_ratio', 0.0, 0.2),
    'weight_decay': trial.suggest_loguniform('weight_decay', 1e-4, 1e-1),

    # Regularization
    'dropout': trial.suggest_uniform('dropout', 0.1, 0.3),
    'attention_dropout': trial.suggest_uniform('attention_dropout', 0.0, 0.2),
    'label_smoothing': trial.suggest_uniform('label_smoothing', 0.0, 0.2),

    # Training
    'gradient_accumulation_steps': trial.suggest_categorical('grad_accum', [1, 2, 4, 8]),
    'max_grad_norm': trial.suggest_uniform('max_grad_norm', 0.5, 2.0),
}
```

### Time Series

**Critical Hyperparameters:**
```python
timeseries_config = {
    # Architecture
    'sequence_length': trial.suggest_int('seq_len', 10, 100),
    'num_layers': trial.suggest_int('num_layers', 1, 4),
    'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256]),

    # LSTM/GRU specific
    'bidirectional': trial.suggest_categorical('bidirectional', [True, False]),
    'dropout': trial.suggest_uniform('dropout', 0.0, 0.5),

    # Optimization
    'learning_rate': trial.suggest_loguniform('lr', 1e-5, 1e-2),
    'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),

    # Loss function
    'loss_fn': trial.suggest_categorical('loss', ['mse', 'mae', 'huber']),
}
```

---

## Complete Example: End-to-End Tuning

```python
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow

class HyperparameterTuner:
    """Complete hyperparameter tuning pipeline"""

    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def objective(self, trial):
        # Suggest hyperparameters
        config = {
            'learning_rate': trial.suggest_loguniform('lr', 1e-5, 1e-2),
            'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256, 512]),
            'num_layers': trial.suggest_int('num_layers', 2, 6),
            'dropout': trial.suggest_uniform('dropout', 0.1, 0.5),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'weight_decay': trial.suggest_loguniform('weight_decay', 1e-6, 1e-3),
            'optimizer': trial.suggest_categorical('optimizer', ['adam', 'sgd', 'adamw']),
        }

        # Start MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(config)

            # Create model
            model = self.create_model(config)

            # Optimizer
            if config['optimizer'] == 'adam':
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=config['learning_rate'],
                    weight_decay=config['weight_decay']
                )
            elif config['optimizer'] == 'sgd':
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=config['learning_rate'],
                    weight_decay=config['weight_decay'],
                    momentum=0.9
                )
            else:  # adamw
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=config['learning_rate'],
                    weight_decay=config['weight_decay']
                )

            # Data loader
            train_loader = DataLoader(
                TensorDataset(self.X_train, self.y_train),
                batch_size=config['batch_size'],
                shuffle=True
            )

            # Training loop with pruning
            best_val_acc = 0
            for epoch in range(100):
                # Train
                train_loss = self.train_epoch(model, train_loader, optimizer)

                # Validate
                val_acc = self.evaluate(model, self.X_val, self.y_val)

                # Log to MLflow
                mlflow.log_metric('train_loss', train_loss, step=epoch)
                mlflow.log_metric('val_acc', val_acc, step=epoch)

                # Report to Optuna for pruning
                trial.report(val_acc, epoch)

                # Prune unpromising trials
                if trial.should_prune():
                    raise optuna.TrialPruned()

                # Track best
                if val_acc > best_val_acc:
                    best_val_acc = val_acc

            # Log final metrics
            mlflow.log_metric('best_val_acc', best_val_acc)

            return best_val_acc

    def create_model(self, config):
        layers = []
        input_size = self.X_train.shape[1]

        for i in range(config['num_layers']):
            layers.append(nn.Linear(
                input_size if i == 0 else config['hidden_size'],
                config['hidden_size']
            ))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config['dropout']))

        layers.append(nn.Linear(config['hidden_size'], 10))  # 10 classes

        return nn.Sequential(*layers)

    def train_epoch(self, model, train_loader, optimizer):
        model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = F.cross_entropy(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(train_loader)

    @torch.no_grad()
    def evaluate(self, model, X, y):
        model.eval()
        outputs = model(X)
        preds = outputs.argmax(dim=1)
        accuracy = (preds == y).float().mean().item()
        return accuracy

    def tune(self, n_trials=100):
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=10,
                n_warmup_steps=20
            )
        )

        # Optimize
        study.optimize(self.objective, n_trials=n_trials)

        # Results
        print(f"\nBest trial: {study.best_trial.number}")
        print(f"Best value: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")

        # Save study
        joblib.dump(study, 'study.pkl')

        return study

# Usage
tuner = HyperparameterTuner(X_train, y_train, X_val, y_val)
study = tuner.tune(n_trials=100)

# Train final model with best config
best_config = study.best_params
final_model = tuner.create_model(best_config)
# ... train final_model with best_config
```

---

## Summary: Choosing the Right Strategy

| Budget | Strategy | Tools |
|--------|----------|-------|
| **Very Limited (<10 trials)** | Manual or Random Search | scikit-learn |
| **Small (10-50 trials)** | Random Search | scikit-learn, Optuna |
| **Medium (50-200 trials)** | Bayesian Optimization | Optuna, Ray Tune |
| **Large (200+ trials)** | ASHA + Bayesian | Ray Tune, Optuna |
| **Very Large (1000+ trials)** | PBT or Multi-Fidelity | Ray Tune |

---

## Key Takeaways

1. **Start with random search** before trying sophisticated methods
2. **Use log scale** for learning rates, weight decay
3. **Prune bad trials early** to save compute
4. **Always use cross-validation** for robust estimates
5. **Track everything** with MLflow or W&B
6. **Domain knowledge helps** - constrain search space
7. **Good hyperparameters transfer** across similar tasks
8. **Parallelize** when possible (Ray Tune, Optuna)

**Next Steps:**
- Implement Optuna tuning for your model
- Visualize hyperparameter importance
- Build a hyperparameter database for transfer learning
- Automate tuning in your ML pipeline


---

## 📚 References

**Hyperparameter Optimization:**

1. **Bergstra, J., & Bengio, Y.** (2012). "Random search for hyper-parameter optimization." *Journal of Machine Learning Research*, 13(1), 281-305.
   - Random search beats grid search

2. **Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B.** (2011). "Algorithms for hyper-parameter optimization." *NIPS 2011*.
   - Tree-structured Parzen estimator (TPE)

**Bayesian Optimization:**

3. **Snoek, J., Larochelle, H., & Adams, R. P.** (2012). "Practical Bayesian optimization of machine learning algorithms." *NIPS 2012*.
   - BO for ML hyperparameters

4. **Srinivas, N., Krause, A., Kakade, S. M., & Seeger, M.** (2010). "Gaussian process optimization in the bandit setting: No regret and experimental design." *ICML 2010*.
   - GP-UCB algorithm and regret bounds

5. **Shahriari, B., Swersky, K., Wang, Z., et al.** (2016). "Taking the human out of the loop: A review of Bayesian optimization." *Proceedings of the IEEE*, 104(1), 148-175.
   - Comprehensive BO survey

6. **Frazier, P. I.** (2018). "A tutorial on Bayesian optimization." *arXiv:1807.02811*.
   - Excellent tutorial paper

**Gaussian Processes:**

7. **Rasmussen, C. E., & Williams, C. K. I.** (2006). *Gaussian Processes for Machine Learning*. MIT Press.
   - Definitive GP textbook. Free: http://www.gaussianprocess.org/gpml/

**Advanced Methods:**

8. **Li, L., Jamieson, K., DeSalvo, G., et al.** (2017). "Hyperband: A novel bandit-based approach to hyperparameter optimization." *JMLR*, 18(1), 6765-6816.
   - Hyperband algorithm

9. **Akiba, T., Sano, S., Yanase, T., et al.** (2019). "Optuna: A next-generation hyperparameter optimization framework." *KDD 2019*.
   - Optuna framework

10. **Liaw, R., Liang, E., Nishihara, R., et al.** (2018). "Tune: A research platform for distributed model selection and training." *arXiv:1807.05118*.
    - Ray Tune framework

**AutoML:**

11. **Feurer, M., & Hutter, F.** (2019). "Hyperparameter optimization." In *Automated Machine Learning* (pp. 3-33). Springer.
    - AutoML chapter on HPO

**Online Resources:**
- Optuna: https://optuna.org/
- Ray Tune: https://docs.ray.io/en/latest/tune/
- Hyperopt: http://hyperopt.github.io/hyperopt/
- scikit-optimize: https://scikit-optimize.github.io/

---

*Hyperparameter tuning is crucial for model performance. Use Bayesian optimization for expensive evaluations!*
