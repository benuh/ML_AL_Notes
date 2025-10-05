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

**How Bayesian Optimization Works:**
1. **Surrogate model** (usually Gaussian Process) models objective function
2. **Acquisition function** (e.g., Expected Improvement) suggests next point to try
3. **Update** surrogate with new observation
4. **Repeat** until budget exhausted

**Pros:** Sample efficient, handles noisy objectives
**Cons:** Overhead per iteration, can get stuck in local optima

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
