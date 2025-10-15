# 🚀 Speculative Coding for ML/AI

**Advanced Techniques for Efficient and Reliable ML Development**

> Last Updated: October 2025
> Covers speculative decoding, specification-driven development, and advanced ML coding patterns

---

## 📋 Table of Contents

1. [Speculative Decoding for LLMs](#speculative-decoding)
2. [Specification-Driven ML Development](#specification-driven-development)
3. [Speculative Execution Patterns](#speculative-execution-patterns)
4. [Test-Driven Development for ML](#test-driven-development)
5. [Contract-Based ML Development](#contract-based-development)
6. [Parallel Speculation Strategies](#parallel-speculation)
7. [Production Best Practices](#production-best-practices)

---

## 🎯 Speculative Decoding

### Overview
Speculative decoding is a technique to speed up LLM inference by using a small "draft" model to predict tokens, which are then verified by the large "target" model in parallel.

### The Problem with Autoregressive Generation
```python
# Standard autoregressive decoding (SLOW!)
for i in range(max_length):
    token = large_model.generate_next_token()  # Sequential!
    # Each token depends on previous tokens
    # Can't parallelize across tokens
```

**Bottleneck:** Each token requires a full forward pass through the large model.

### Speculative Decoding Solution

**Key Insight:** Use a small, fast model to generate candidate tokens, then verify multiple tokens at once with the large model.

```
Small Model (draft):    Generates k tokens quickly
                        ↓
Large Model (target):   Verifies all k tokens in parallel
                        ↓
Accept correct tokens, reject and regenerate bad ones
```

### Algorithm

```python
def speculative_decoding(
    draft_model,    # Small, fast model (e.g., 7B params)
    target_model,   # Large, slow model (e.g., 70B params)
    prompt,
    k=5,           # Number of speculative tokens
    max_length=100
):
    """
    Speculative decoding algorithm.

    Speed improvement: 2-3x faster!
    No quality degradation: mathematically equivalent to target model alone.
    """
    tokens = tokenize(prompt)

    while len(tokens) < max_length:
        # Step 1: Draft model generates k candidate tokens
        draft_tokens = []
        draft_probs = []

        for _ in range(k):
            probs = draft_model(tokens)
            next_token = sample(probs)
            draft_tokens.append(next_token)
            draft_probs.append(probs[next_token])
            tokens.append(next_token)

        # Step 2: Target model verifies all k tokens in ONE forward pass
        target_probs_all = target_model(tokens[:-k])  # Parallel!

        # Step 3: Accept or reject each token
        num_accepted = 0
        for i in range(k):
            target_prob = target_probs_all[i][draft_tokens[i]]
            draft_prob = draft_probs[i]

            # Acceptance probability
            accept_prob = min(1, target_prob / draft_prob)

            if random.random() < accept_prob:
                # Accept this token
                num_accepted += 1
            else:
                # Reject this token and all after it
                tokens = tokens[:-(k-i)]  # Rollback

                # Resample from adjusted distribution
                adjusted_probs = normalize(
                    max(0, target_probs_all[i] - draft_probs[i])
                )
                new_token = sample(adjusted_probs)
                tokens.append(new_token)
                break

        # Step 4: If all k tokens accepted, continue
        if num_accepted == k:
            # Bonus: generate one more token with target model
            target_probs = target_model(tokens)
            next_token = sample(target_probs)
            tokens.append(next_token)

    return tokens
```

### Mathematical Guarantee

**Theorem:** Speculative decoding produces the **exact same distribution** as sampling from the target model alone.

**Why it works:**
- Acceptance-rejection sampling maintains target distribution
- No approximation or quality loss
- Pure speed improvement!

### Practical Implementation

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load models
draft_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B")
target_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")

def speculative_generate(prompt, k=5, max_new_tokens=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    for _ in range(max_new_tokens // k):
        # Draft phase: generate k candidates
        draft_outputs = draft_model.generate(
            input_ids,
            max_new_tokens=k,
            do_sample=True,
            temperature=0.7,
            return_dict_in_generate=True,
            output_scores=True
        )

        draft_tokens = draft_outputs.sequences[0, len(input_ids[0]):]

        # Verification phase: check with target model
        combined_ids = torch.cat([input_ids, draft_tokens.unsqueeze(0)], dim=1)

        with torch.no_grad():
            target_outputs = target_model(combined_ids)
            target_logits = target_outputs.logits[0, len(input_ids[0])-1:-1]

        # Accept/reject logic
        accepted = []
        for i, token in enumerate(draft_tokens):
            target_probs = torch.softmax(target_logits[i], dim=-1)

            if torch.rand(1) < target_probs[token]:
                accepted.append(token)
            else:
                # Reject and resample
                resampled = torch.multinomial(target_probs, 1)[0]
                accepted.append(resampled)
                break

        input_ids = torch.cat([
            input_ids,
            torch.tensor(accepted).unsqueeze(0)
        ], dim=1)

    return tokenizer.decode(input_ids[0])

# Usage
output = speculative_generate("Once upon a time", k=5)
```

### Performance Gains

| Setup | Speedup | Notes |
|-------|---------|-------|
| 7B draft + 70B target | 2.5x | Best for long-form generation |
| 1B draft + 13B target | 2.0x | Good for consumer hardware |
| Same architecture required | Yes | Draft and target must use same tokenizer |

### When to Use

**Good for:**
- Long-form generation (stories, articles, code)
- Latency-sensitive applications
- Cost reduction (fewer large model calls)

**Not ideal for:**
- Single token generation
- Very short sequences
- When draft model is too different from target

**Reference:** *Fast Inference from Transformers via Speculative Decoding* - Leviathan et al. (2023)

---

## 📐 Specification-Driven ML Development

### Overview
Define formal specifications for ML models before implementation, ensuring correctness and reliability.

### Components of ML Specifications

#### 1. **Input Specifications**
```python
from typing import TypedDict, Literal
import numpy as np
from numpy.typing import NDArray

class ImageClassifierInput(TypedDict):
    """Specification for image classifier input."""
    image: NDArray[np.uint8]  # Shape: (H, W, 3)
    preprocess: Literal["resize", "crop", "pad"]

    @staticmethod
    def validate(data: dict) -> bool:
        """Validate input specification."""
        assert data["image"].ndim == 3, "Image must be 3D"
        assert data["image"].shape[2] == 3, "Image must have 3 channels"
        assert data["image"].dtype == np.uint8, "Image must be uint8"
        assert 0 <= data["image"].min() <= data["image"].max() <= 255
        return True

# Usage
def preprocess_image(input_data: ImageClassifierInput):
    ImageClassifierInput.validate(input_data)
    # ... implementation
```

#### 2. **Output Specifications**
```python
from pydantic import BaseModel, Field, validator

class ClassificationOutput(BaseModel):
    """Specification for classification output."""
    predictions: list[float] = Field(..., min_items=1)
    class_names: list[str] = Field(..., min_items=1)
    confidence: float = Field(..., ge=0.0, le=1.0)

    @validator("predictions")
    def probabilities_sum_to_one(cls, v):
        """Ensure probabilities sum to 1."""
        assert abs(sum(v) - 1.0) < 1e-6, f"Probabilities must sum to 1, got {sum(v)}"
        return v

    @validator("predictions")
    def all_valid_probabilities(cls, v):
        """Ensure all probabilities in [0, 1]."""
        assert all(0 <= p <= 1 for p in v), "All probabilities must be in [0, 1]"
        return v

    class Config:
        schema_extra = {
            "example": {
                "predictions": [0.7, 0.2, 0.1],
                "class_names": ["cat", "dog", "bird"],
                "confidence": 0.7
            }
        }

# Usage
def classify(image: NDArray) -> ClassificationOutput:
    # ... model inference
    output = ClassificationOutput(
        predictions=[0.7, 0.2, 0.1],
        class_names=["cat", "dog", "bird"],
        confidence=0.7
    )
    return output  # Automatically validated!
```

#### 3. **Model Specifications**
```python
class ModelSpec:
    """Specification for ML model behavior."""

    def __init__(self, model):
        self.model = model

    # Invariant: predictions should be deterministic with same seed
    def test_determinism(self, input_data, seed=42):
        """Model should produce same output with same seed."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        output1 = self.model(input_data)

        np.random.seed(seed)
        torch.manual_seed(seed)
        output2 = self.model(input_data)

        assert np.allclose(output1, output2), "Model is not deterministic!"

    # Invariant: small input changes → small output changes (smoothness)
    def test_smoothness(self, input_data, epsilon=1e-3):
        """Model should be Lipschitz continuous."""
        output1 = self.model(input_data)

        # Perturb input slightly
        noise = np.random.randn(*input_data.shape) * epsilon
        output2 = self.model(input_data + noise)

        # Output change should be bounded
        output_diff = np.linalg.norm(output2 - output1)
        input_diff = np.linalg.norm(noise)

        lipschitz_constant = output_diff / input_diff
        assert lipschitz_constant < 100, f"Model too sensitive: L={lipschitz_constant}"

    # Invariant: valid probability distribution
    def test_probability_axioms(self, input_data):
        """Classification output should be valid probability distribution."""
        output = self.model(input_data)

        # All probabilities in [0, 1]
        assert np.all((output >= 0) & (output <= 1)), "Invalid probabilities"

        # Sum to 1
        assert np.allclose(output.sum(axis=-1), 1.0), "Probabilities don't sum to 1"

    # Invariant: invariance to expected transformations
    def test_invariance(self, input_data, transformation):
        """Model should be invariant to certain transformations."""
        output1 = self.model(input_data)
        output2 = self.model(transformation(input_data))

        # Should produce same prediction
        pred1 = np.argmax(output1)
        pred2 = np.argmax(output2)
        assert pred1 == pred2, "Model not invariant to transformation"

# Usage
spec = ModelSpec(classifier)
spec.test_determinism(test_input)
spec.test_smoothness(test_input)
spec.test_probability_axioms(test_input)
spec.test_invariance(test_input, lambda x: x[:, ::-1])  # Horizontal flip
```

#### 4. **Performance Specifications**
```python
class PerformanceSpec:
    """Specification for model performance requirements."""

    def __init__(self,
                 min_accuracy: float = 0.90,
                 max_latency_ms: float = 100,
                 max_memory_mb: float = 1000):
        self.min_accuracy = min_accuracy
        self.max_latency_ms = max_latency_ms
        self.max_memory_mb = max_memory_mb

    def verify_accuracy(self, model, test_loader):
        """Verify model meets accuracy requirements."""
        correct = 0
        total = 0

        for inputs, labels in test_loader:
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total
        assert accuracy >= self.min_accuracy, \
            f"Accuracy {accuracy:.3f} below requirement {self.min_accuracy}"

        return accuracy

    def verify_latency(self, model, sample_input):
        """Verify model meets latency requirements."""
        import time

        # Warmup
        for _ in range(10):
            _ = model(sample_input)

        # Measure
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            _ = model(sample_input)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms

        p95_latency = np.percentile(latencies, 95)
        assert p95_latency <= self.max_latency_ms, \
            f"P95 latency {p95_latency:.1f}ms exceeds {self.max_latency_ms}ms"

        return p95_latency

    def verify_memory(self, model):
        """Verify model meets memory requirements."""
        import torch

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        memory_mb = (total_params * 4) / (1024 ** 2)  # Assume float32

        assert memory_mb <= self.max_memory_mb, \
            f"Model size {memory_mb:.1f}MB exceeds {self.max_memory_mb}MB"

        return memory_mb

# Usage
spec = PerformanceSpec(
    min_accuracy=0.95,
    max_latency_ms=50,
    max_memory_mb=500
)

# Verify all specs
accuracy = spec.verify_accuracy(model, test_loader)
latency = spec.verify_latency(model, sample_input)
memory = spec.verify_memory(model)

print(f"✓ Accuracy: {accuracy:.3f}")
print(f"✓ P95 Latency: {latency:.1f}ms")
print(f"✓ Memory: {memory:.1f}MB")
```

### Design by Contract for ML

```python
from functools import wraps
import inspect

def precondition(check_func):
    """Decorator for preconditions (input validation)."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Check precondition
            if not check_func(*bound_args.args, **bound_args.kwargs):
                raise ValueError(f"Precondition failed for {func.__name__}")

            return func(*args, **kwargs)
        return wrapper
    return decorator

def postcondition(check_func):
    """Decorator for postconditions (output validation)."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            # Check postcondition
            if not check_func(result, *args, **kwargs):
                raise ValueError(f"Postcondition failed for {func.__name__}")

            return result
        return wrapper
    return decorator

# Example: Image preprocessing with contracts
@precondition(lambda image, size: image.ndim == 3 and image.shape[2] == 3)
@precondition(lambda image, size: size > 0)
@postcondition(lambda result, image, size: result.shape == (size, size, 3))
@postcondition(lambda result, image, size: 0 <= result.min() and result.max() <= 1)
def preprocess_image(image: np.ndarray, size: int) -> np.ndarray:
    """
    Preprocess image for model input.

    Preconditions:
        - image must be 3D (H, W, C)
        - image must have 3 channels
        - size must be positive

    Postconditions:
        - output shape is (size, size, 3)
        - output values in [0, 1]
    """
    from PIL import Image

    # Resize
    pil_image = Image.fromarray(image)
    resized = pil_image.resize((size, size))

    # Normalize
    normalized = np.array(resized).astype(np.float32) / 255.0

    return normalized

# Usage - contracts are checked automatically!
try:
    result = preprocess_image(image, size=224)
    print("✓ All contracts satisfied")
except ValueError as e:
    print(f"✗ Contract violation: {e}")
```

---

## ⚡ Speculative Execution Patterns

### 1. Parallel Model Inference

Execute multiple model variants in parallel and select best result:

```python
import concurrent.futures
import time

class SpeculativeModelEnsemble:
    """Execute multiple models in parallel, return fastest/best result."""

    def __init__(self, models: list):
        self.models = models
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=len(models)
        )

    def predict_fastest(self, input_data, timeout=5.0):
        """Return prediction from fastest model."""
        futures = {
            self.executor.submit(model.predict, input_data): i
            for i, model in enumerate(self.models)
        }

        # Return first completed
        done, pending = concurrent.futures.wait(
            futures,
            timeout=timeout,
            return_when=concurrent.futures.FIRST_COMPLETED
        )

        # Cancel pending
        for future in pending:
            future.cancel()

        # Get result from first completed
        if done:
            result = list(done)[0].result()
            model_idx = futures[list(done)[0]]
            return result, model_idx
        else:
            raise TimeoutError("No model completed in time")

    def predict_best(self, input_data, scoring_func):
        """Execute all models, return best according to scoring_func."""
        futures = {
            self.executor.submit(model.predict, input_data): model
            for model in self.models
        }

        results = []
        for future in concurrent.futures.as_completed(futures):
            model = futures[future]
            try:
                prediction = future.result()
                score = scoring_func(prediction)
                results.append((prediction, score, model))
            except Exception as e:
                print(f"Model {model} failed: {e}")

        # Return best result
        best = max(results, key=lambda x: x[1])
        return best[0], best[2]

    def predict_ensemble(self, input_data):
        """Execute all models and ensemble predictions."""
        futures = [
            self.executor.submit(model.predict, input_data)
            for model in self.models
        ]

        predictions = []
        for future in concurrent.futures.as_completed(futures):
            try:
                predictions.append(future.result())
            except Exception as e:
                print(f"Prediction failed: {e}")

        # Ensemble (average for regression, vote for classification)
        ensembled = np.mean(predictions, axis=0)
        return ensembled

# Usage
models = [FastModel(), AccurateModel(), BalancedModel()]
ensemble = SpeculativeModelEnsemble(models)

# Get fastest prediction
prediction, model_idx = ensemble.predict_fastest(input_data)
print(f"Got prediction from model {model_idx}")

# Get best prediction
prediction, model = ensemble.predict_best(
    input_data,
    scoring_func=lambda p: -p.uncertainty()  # Lower uncertainty is better
)

# Get ensembled prediction
prediction = ensemble.predict_ensemble(input_data)
```

### 2. Speculative Data Loading

Load and preprocess data speculatively while model is training:

```python
import threading
import queue

class SpeculativeDataLoader:
    """Load and preprocess data ahead of time."""

    def __init__(self, dataset, batch_size, prefetch_batches=3):
        self.dataset = dataset
        self.batch_size = batch_size
        self.prefetch_batches = prefetch_batches
        self.queue = queue.Queue(maxsize=prefetch_batches)
        self.stop_event = threading.Event()

    def _loader_worker(self):
        """Background worker to load data."""
        idx = 0
        while not self.stop_event.is_set():
            # Load batch
            batch = self._load_batch(idx)

            # Preprocess speculatively
            batch = self._preprocess(batch)

            # Put in queue (blocks if queue is full)
            try:
                self.queue.put(batch, timeout=1.0)
                idx += 1
            except queue.Full:
                continue

    def _load_batch(self, idx):
        """Load batch from dataset."""
        start = idx * self.batch_size
        end = start + self.batch_size
        return self.dataset[start:end]

    def _preprocess(self, batch):
        """Preprocess batch (expensive operations here)."""
        # Augmentation, normalization, etc.
        return batch

    def start(self):
        """Start background loading."""
        self.worker = threading.Thread(target=self._loader_worker, daemon=True)
        self.worker.start()

    def __iter__(self):
        """Iterate over batches."""
        self.start()
        while True:
            try:
                batch = self.queue.get(timeout=1.0)
                yield batch
            except queue.Empty:
                if self.stop_event.is_set():
                    break

    def stop(self):
        """Stop background loading."""
        self.stop_event.set()

# Usage
loader = SpeculativeDataLoader(dataset, batch_size=32, prefetch_batches=5)

for epoch in range(num_epochs):
    for batch in loader:
        # Batch is already preprocessed and ready!
        loss = train_step(model, batch)

    loader.stop()
```

### 3. Speculative Hyperparameter Optimization

Speculatively try multiple hyperparameter configurations:

```python
import optuna

class SpeculativeHPO:
    """Speculatively prune bad hyperparameter trials."""

    def __init__(self, model_class, train_data, val_data):
        self.model_class = model_class
        self.train_data = train_data
        self.val_data = val_data

    def objective(self, trial: optuna.Trial):
        """Optuna objective with early stopping."""
        # Suggest hyperparameters
        lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        hidden_dim = trial.suggest_int("hidden_dim", 64, 512)

        # Create model
        model = self.model_class(hidden_dim=hidden_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Train with early stopping
        for epoch in range(100):
            # Train
            train_loss = self._train_epoch(model, optimizer, batch_size)

            # Validate
            val_loss = self._validate(model, batch_size)

            # Report intermediate value
            trial.report(val_loss, epoch)

            # Speculative pruning: stop if trial is not promising
            if trial.should_prune():
                raise optuna.TrialPruned()

        return val_loss

    def optimize(self, n_trials=100, n_jobs=4):
        """Run speculative hyperparameter optimization."""
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,    # Don't prune first 5 trials
                n_warmup_steps=10,     # Don't prune first 10 epochs
                interval_steps=1       # Check every epoch
            )
        )

        # Run parallel trials
        study.optimize(
            self.objective,
            n_trials=n_trials,
            n_jobs=n_jobs,  # Parallel trials!
            show_progress_bar=True
        )

        print(f"Best trial: {study.best_trial.params}")
        print(f"Best value: {study.best_value}")

        return study.best_params

# Usage
hpo = SpeculativeHPO(MyModel, train_data, val_data)
best_params = hpo.optimize(n_trials=100, n_jobs=4)
```

---

## 🧪 Test-Driven Development for ML

### ML Testing Pyramid

```
        /\
       /  \        Unit Tests (80%)
      /    \       - Data validators
     /------\      - Transform functions
    /        \     - Model components
   /          \
  /------------\   Integration Tests (15%)
 /              \  - Pipeline tests
/                \ - End-to-end data flow
------------------
     System Tests (5%)
     - Model performance
     - Production scenarios
```

### 1. Data Testing

```python
import pytest
import pandas as pd
import great_expectations as ge

class TestDataQuality:
    """Test data quality and schema."""

    @pytest.fixture
    def sample_data(self):
        """Load sample training data."""
        return pd.read_csv("data/train.csv")

    def test_no_missing_values(self, sample_data):
        """Critical columns should have no missing values."""
        critical_cols = ["user_id", "timestamp", "label"]
        for col in critical_cols:
            assert sample_data[col].notna().all(), \
                f"Column {col} has missing values"

    def test_schema(self, sample_data):
        """Data should match expected schema."""
        expected_columns = ["user_id", "feature1", "feature2", "label"]
        assert list(sample_data.columns) == expected_columns

        # Check data types
        assert sample_data["user_id"].dtype == "int64"
        assert sample_data["feature1"].dtype == "float64"
        assert sample_data["label"].dtype == "int64"

    def test_value_ranges(self, sample_data):
        """Features should be in expected ranges."""
        assert (sample_data["feature1"] >= 0).all()
        assert (sample_data["feature1"] <= 1).all()

        assert sample_data["label"].isin([0, 1]).all()

    def test_statistical_properties(self, sample_data):
        """Data should have expected statistical properties."""
        # Check for class balance
        class_dist = sample_data["label"].value_counts(normalize=True)
        assert class_dist.min() > 0.1, "Severe class imbalance"

        # Check for outliers
        from scipy import stats
        z_scores = np.abs(stats.zscore(sample_data["feature1"]))
        assert (z_scores < 5).all(), "Extreme outliers detected"

    def test_data_consistency(self, sample_data):
        """Data should be internally consistent."""
        # No duplicate user_ids
        assert not sample_data["user_id"].duplicated().any()

        # Timestamps should be increasing
        assert sample_data["timestamp"].is_monotonic_increasing

# Run with: pytest test_data.py -v
```

### 2. Model Testing

```python
class TestModel:
    """Test model behavior and properties."""

    @pytest.fixture
    def model(self):
        """Initialize model for testing."""
        return MyClassifier(input_dim=10, output_dim=2)

    @pytest.fixture
    def sample_input(self):
        """Create sample input."""
        return torch.randn(32, 10)  # Batch of 32

    def test_forward_pass_shape(self, model, sample_input):
        """Model should output correct shape."""
        output = model(sample_input)
        assert output.shape == (32, 2)

    def test_forward_pass_dtypes(self, model, sample_input):
        """Model should output correct dtype."""
        output = model(sample_input)
        assert output.dtype == torch.float32

    def test_probability_output(self, model, sample_input):
        """Classification model should output valid probabilities."""
        output = model(sample_input)
        probs = torch.softmax(output, dim=-1)

        # All probabilities in [0, 1]
        assert (probs >= 0).all() and (probs <= 1).all()

        # Sum to 1
        assert torch.allclose(probs.sum(dim=-1), torch.ones(32))

    def test_determinism(self, model, sample_input):
        """Model should be deterministic with same seed."""
        torch.manual_seed(42)
        output1 = model(sample_input)

        torch.manual_seed(42)
        output2 = model(sample_input)

        assert torch.allclose(output1, output2)

    def test_backpropagation(self, model, sample_input):
        """Gradients should flow through all parameters."""
        output = model(sample_input)
        loss = output.sum()
        loss.backward()

        # All parameters should have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

    def test_overfitting_small_batch(self, model):
        """Model should be able to overfit small batch (sanity check)."""
        # Create small, easy dataset
        X = torch.randn(10, 10)
        y = torch.randint(0, 2, (10,))

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        # Train for 100 steps
        for _ in range(100):
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        # Should achieve near-perfect accuracy
        predictions = model(X).argmax(dim=1)
        accuracy = (predictions == y).float().mean()
        assert accuracy > 0.9, f"Can't overfit small batch (acc={accuracy})"

    def test_invariance_to_batch_size(self, model):
        """Model should produce same results regardless of batch size."""
        # Create data
        X = torch.randn(64, 10)

        # Predict with batch_size=64
        model.eval()
        with torch.no_grad():
            output_batch = model(X)

        # Predict with batch_size=1
        outputs_single = []
        for i in range(64):
            with torch.no_grad():
                output = model(X[i:i+1])
            outputs_single.append(output)
        outputs_single = torch.cat(outputs_single, dim=0)

        # Should be identical
        assert torch.allclose(output_batch, outputs_single, atol=1e-5)
```

### 3. Pipeline Testing

```python
class TestMLPipeline:
    """Test end-to-end ML pipeline."""

    def test_full_pipeline(self):
        """Test complete training pipeline."""
        # 1. Load data
        train_data, val_data = load_data()
        assert len(train_data) > 0
        assert len(val_data) > 0

        # 2. Preprocess
        preprocessor = Preprocessor()
        X_train = preprocessor.fit_transform(train_data)
        X_val = preprocessor.transform(val_data)

        # 3. Train model
        model = MyModel()
        history = model.fit(X_train, epochs=2)  # Quick test

        # 4. Evaluate
        metrics = model.evaluate(X_val)
        assert "accuracy" in metrics
        assert metrics["accuracy"] > 0.5  # Better than random

        # 5. Save/load
        model.save("test_model.pkl")
        loaded_model = MyModel.load("test_model.pkl")

        # 6. Verify loaded model
        assert np.allclose(
            model.predict(X_val),
            loaded_model.predict(X_val)
        )
```

---

## 📋 Contract-Based ML Development

### Property-Based Testing for ML

```python
from hypothesis import given, strategies as st
import hypothesis.extra.numpy as npst

class TestModelProperties:
    """Property-based tests for model."""

    @given(
        X=npst.arrays(
            dtype=np.float32,
            shape=(st.integers(1, 100), 10),  # Variable batch size
            elements=st.floats(-10, 10, allow_nan=False)
        )
    )
    def test_output_shape_property(self, X):
        """Model should output correct shape for any valid input."""
        model = MyClassifier(input_dim=10, output_dim=2)
        output = model(torch.tensor(X))
        assert output.shape == (X.shape[0], 2)

    @given(
        X=npst.arrays(
            dtype=np.float32,
            shape=(10, 10),
            elements=st.floats(-10, 10, allow_nan=False)
        ),
        epsilon=st.floats(0, 0.1)
    )
    def test_continuity_property(self, X, epsilon):
        """Small input changes should lead to small output changes."""
        model = MyClassifier(input_dim=10, output_dim=2)
        model.eval()

        X_torch = torch.tensor(X)
        noise = torch.randn_like(X_torch) * epsilon

        with torch.no_grad():
            output1 = model(X_torch)
            output2 = model(X_torch + noise)

        # Lipschitz continuity
        output_diff = torch.norm(output2 - output1)
        input_diff = torch.norm(noise)

        # Lipschitz constant should be bounded
        if input_diff > 1e-6:  # Avoid division by zero
            lipschitz = output_diff / input_diff
            assert lipschitz < 100, f"Model too sensitive: L={lipschitz}"

    @given(
        X=npst.arrays(
            dtype=np.float32,
            shape=(10, 10),
            elements=st.floats(-10, 10, allow_nan=False)
        )
    )
    def test_valid_probability_property(self, X):
        """Classification output should always be valid probability."""
        model = MyClassifier(input_dim=10, output_dim=2)
        model.eval()

        with torch.no_grad():
            output = model(torch.tensor(X))
            probs = torch.softmax(output, dim=-1)

        # Properties of probability distribution
        assert (probs >= 0).all()  # Non-negative
        assert (probs <= 1).all()  # Bounded by 1
        assert torch.allclose(probs.sum(dim=-1), torch.ones(10))  # Sum to 1

# Run with: pytest test_properties.py
# Hypothesis will generate 100s of random test cases!
```

---

## 🎯 Production Best Practices

### 1. Model Versioning and Contracts

```python
from dataclasses import dataclass
from typing import Dict, Any
import json

@dataclass
class ModelContract:
    """Formal contract for model in production."""
    model_name: str
    version: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    performance_requirements: Dict[str, float]
    dependencies: Dict[str, str]

    def save(self, path: str):
        """Save contract to JSON."""
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, path: str):
        """Load contract from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def validate_input(self, input_data):
        """Validate input matches contract."""
        # Implement validation logic
        pass

    def validate_output(self, output_data):
        """Validate output matches contract."""
        # Implement validation logic
        pass

# Example contract
contract = ModelContract(
    model_name="image_classifier_v2",
    version="2.1.0",
    input_schema={
        "image": {"type": "array", "shape": [224, 224, 3], "dtype": "uint8"},
        "preprocessing": {"type": "string", "enum": ["resize", "crop"]}
    },
    output_schema={
        "predictions": {"type": "array", "dtype": "float32"},
        "confidence": {"type": "float", "range": [0, 1]}
    },
    performance_requirements={
        "accuracy": 0.95,
        "latency_p95_ms": 50,
        "throughput_qps": 1000
    },
    dependencies={
        "torch": "2.0.0",
        "torchvision": "0.15.0",
        "numpy": "1.24.0"
    }
)

contract.save("model_contract.json")
```

### 2. Monitoring and Alerting

```python
class ModelMonitor:
    """Monitor model performance in production."""

    def __init__(self, model, contract: ModelContract):
        self.model = model
        self.contract = contract
        self.predictions = []
        self.latencies = []

    def predict_with_monitoring(self, input_data):
        """Make prediction with monitoring."""
        import time

        # Validate input
        self.contract.validate_input(input_data)

        # Measure latency
        start = time.perf_counter()
        output = self.model(input_data)
        latency = (time.perf_counter() - start) * 1000

        # Validate output
        self.contract.validate_output(output)

        # Log
        self.predictions.append(output)
        self.latencies.append(latency)

        # Check alerts
        self._check_alerts()

        return output

    def _check_alerts(self):
        """Check for alert conditions."""
        # Latency alert
        if len(self.latencies) >= 100:
            p95 = np.percentile(self.latencies[-100:], 95)
            if p95 > self.contract.performance_requirements["latency_p95_ms"]:
                self._send_alert(f"Latency spike: {p95:.1f}ms")

        # Distribution shift alert
        if len(self.predictions) >= 1000:
            recent_dist = self._compute_distribution(self.predictions[-1000:])
            baseline_dist = self._load_baseline_distribution()

            kl_div = self._kl_divergence(recent_dist, baseline_dist)
            if kl_div > 0.1:  # Threshold
                self._send_alert(f"Distribution shift detected: KL={kl_div:.3f}")

    def _send_alert(self, message: str):
        """Send alert to monitoring system."""
        print(f"ALERT: {message}")
        # Send to PagerDuty, Slack, etc.
```

---

## 📚 References

1. **Speculative Decoding:**
   - *Fast Inference from Transformers via Speculative Decoding* - Leviathan et al. (2023)
   - *SpecInfer: Accelerating Generative LLM Serving* - Miao et al. (2023)

2. **Formal Methods for ML:**
   - *Formal Verification of Neural Networks* - Singh et al. (2019)
   - *Property-Based Testing for ML* - Zhang et al. (2020)

3. **Testing ML Systems:**
   - *Testing Machine Learning Systems* - Breck et al. (2017)
   - *What's your ML Test Score?* - Sculley et al. (2016)

---

## 🎯 Interview Questions

1. **What is speculative decoding and how does it achieve speedup without quality loss?**
2. **How would you write specifications for an ML model's behavior?**
3. **What are the key differences between testing traditional software vs ML systems?**
4. **Explain property-based testing and how it applies to ML.**
5. **How do you ensure a model meets performance requirements in production?**

---

*For more advanced techniques, see [MODERN_ML_AI_TECHNIQUES_2024_2025.md](./MODERN_ML_AI_TECHNIQUES_2024_2025.md)*
