# Explainable AI (XAI) Guide

Make AI transparent, interpretable, and trustworthy.

## Table of Contents
1. [Why Explainability Matters](#why-explainability-matters)
2. [Model-Agnostic Methods](#model-agnostic-methods)
3. [Deep Learning Interpretability](#deep-learning-interpretability)
4. [Feature Importance](#feature-importance)
5. [Counterfactual Explanations](#counterfactual-explanations)
6. [Evaluation of Explanations](#evaluation-of-explanations)
7. [Production XAI](#production-xai)

---

## Why Explainability Matters

### The XAI Imperative

**Why we need explainability:**
1. **Trust** - Users need to understand AI decisions
2. **Debugging** - Find and fix model errors
3. **Compliance** - Regulations (GDPR, EU AI Act) require explanations
4. **Fairness** - Detect and mitigate bias
5. **Scientific Discovery** - Learn from model insights

**Types of Explanations:**
- **Global** - How the model works overall
- **Local** - Why this specific prediction
- **Model-specific** - For particular architectures (e.g., decision trees)
- **Model-agnostic** - Works for any model

---

## Model-Agnostic Methods

### LIME (Local Interpretable Model-agnostic Explanations)

**Key Idea:** Approximate complex model locally with interpretable model.

```python
from lime import lime_tabular
from lime.lime_text import LimeTextExplainer
import numpy as np

class LIMEExplainer:
    """LIME for tabular and text data"""

    def __init__(self, model, feature_names, class_names):
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names

    def explain_tabular(self, X_train, instance, num_features=10):
        """Explain prediction for tabular data"""

        explainer = lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode='classification'
        )

        # Generate explanation
        exp = explainer.explain_instance(
            data_row=instance,
            predict_fn=self.model.predict_proba,
            num_features=num_features
        )

        return exp

    def explain_text(self, text, num_features=10):
        """Explain prediction for text"""

        explainer = LimeTextExplainer(class_names=self.class_names)

        exp = explainer.explain_instance(
            text_instance=text,
            classifier_fn=self.model.predict_proba,
            num_features=num_features
        )

        return exp

# Usage
from sklearn.ensemble import RandomForestClassifier

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Create explainer
explainer = LIMEExplainer(
    model=model,
    feature_names=['age', 'income', 'credit_score'],
    class_names=['approved', 'denied']
)

# Explain instance
explanation = explainer.explain_tabular(X_train, X_test[0])
explanation.show_in_notebook()  # Visualize

# Get feature weights
weights = explanation.as_list()
for feature, weight in weights:
    print(f"{feature}: {weight:.4f}")
```

**Custom LIME Implementation:**
```python
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import euclidean_distances

class CustomLIME:
    """Custom LIME implementation"""

    def __init__(self, model, num_samples=5000, kernel_width=0.75):
        self.model = model
        self.num_samples = num_samples
        self.kernel_width = kernel_width

    def explain(self, instance, X_train):
        """Generate local explanation"""

        # Generate perturbed samples around instance
        perturbed_samples = self._generate_samples(instance, X_train)

        # Get predictions for perturbed samples
        predictions = self.model.predict_proba(perturbed_samples)[:, 1]

        # Compute sample weights (closer samples weighted more)
        distances = euclidean_distances([instance], perturbed_samples)[0]
        kernel_weights = np.exp(-(distances ** 2) / (self.kernel_width ** 2))

        # Fit linear model
        linear_model = Ridge(alpha=1.0)
        linear_model.fit(perturbed_samples, predictions, sample_weight=kernel_weights)

        # Feature importances
        coefficients = linear_model.coef_

        return coefficients

    def _generate_samples(self, instance, X_train):
        """Generate samples by perturbing features"""
        samples = []

        for _ in range(self.num_samples):
            sample = instance.copy()

            # Randomly perturb some features
            num_perturb = np.random.randint(1, len(instance))
            features_to_perturb = np.random.choice(len(instance), num_perturb, replace=False)

            for feature_idx in features_to_perturb:
                # Sample from training data distribution
                sample[feature_idx] = np.random.choice(X_train[:, feature_idx])

            samples.append(sample)

        return np.array(samples)

# Usage
lime = CustomLIME(model, num_samples=5000)
explanation = lime.explain(X_test[0], X_train)

# Visualize
import matplotlib.pyplot as plt
plt.barh(range(len(explanation)), explanation)
plt.yticks(range(len(explanation)), feature_names)
plt.xlabel('Feature Importance')
plt.title('LIME Explanation')
plt.show()
```

---

### SHAP (SHapley Additive exPlanations)

**Key Idea:** Use Shapley values from game theory to fairly distribute prediction credit among features.

**Mathematical Foundation - Shapley Values:**

**Cooperative Game Theory:**
```
A cooperative game is defined as (N, v) where:
- N = {1, 2, ..., n}: Set of players (features)
- v: 2^N → ℝ: Value function mapping coalitions to real values

For ML: v(S) = E[f(x) | x_S] - E[f(x)]
where:
- S ⊆ N: Coalition of features
- x_S: Features in S set to their values, others marginalized out
- f: Model prediction function
```

**Shapley Value Definition:**
```
The Shapley value φ_i for feature i is:

φ_i(v) = Σ_{S⊆N\{i}} [|S|!(n-|S|-1)!]/n! · [v(S∪{i}) - v(S)]

Intuition:
- Sum over all possible feature coalitions S not containing i
- For each coalition S:
  * Add feature i: v(S∪{i})
  * Marginal contribution: v(S∪{i}) - v(S)
  * Weight by probability of that coalition order

Alternative form:
φ_i(v) = Σ_{S⊆N\{i}} w(|S|) · [v(S∪{i}) - v(S)]

where w(|S|) = [|S|!(n-|S|-1)!]/n! is the weight

Properties:
- Weights sum to 1: Σ_{S⊆N\{i}} w(|S|) = 1
- 2^(n-1) terms to compute (exponential in features!)
```

**Uniqueness Theorem (Shapley, 1953):**
```
The Shapley value is the UNIQUE attribution method satisfying:

1. Efficiency (Local Accuracy):
   Σ^n_{i=1} φ_i = f(x) - E[f(x)]

   The sum of attributions equals prediction - baseline

2. Symmetry (Equal Treatment):
   If v(S∪{i}) = v(S∪{j}) for all S not containing i,j
   then φ_i = φ_j

   Features with equal marginal contributions get equal credit

3. Dummy (No Effect):
   If v(S∪{i}) = v(S) for all S
   then φ_i = 0

   Features with no effect get zero attribution

4. Linearity (Additivity):
   For games v, w: φ_i(v + w) = φ_i(v) + φ_i(w)

   Attributions are additive across models

These axioms uniquely determine the Shapley value!
```

**SHAP as an Additive Feature Attribution:**
```
SHAP represents prediction as:

f(x) = φ_0 + Σ^n_{i=1} φ_i

where:
- φ_0 = E[f(x)]: Base value (expected prediction)
- φ_i: SHAP value for feature i
- Σ^n_{i=1} φ_i = f(x) - E[f(x)]: Total attribution

This is an additive explanation model:
g(z') = φ_0 + Σ^M_{i=1} φ_i · z'_i

where z' ∈ {0,1}^M indicates feature presence
```

**Computational Challenge:**
```
Exact Shapley values require 2^n evaluations!

Example: For 20 features
- Number of coalitions: 2^20 = 1,048,576
- For 100 instances: 104.8M model evaluations

Computational Complexity:
- Exact: O(2^n · T_f) where T_f = time per prediction
- Approximation via sampling: O(m · T_f) where m = samples << 2^n
```

**SHAP Approximation Methods:**

**1. KernelSHAP (Model-Agnostic):**
```
Weighted linear regression in coalition space:

min_{φ} Σ_{z'∈Z} [f(h_x(z')) - g(z')]² · π_{x}(z')

where:
- z': Binary coalition vector
- h_x(z'): Maps coalition to original feature space
- π_x(z'): SHAP kernel weight

SHAP Kernel:
π_x(z') = (n-1) / [C(n, |z'|) · |z'| · (n - |z'|)]

where C(n, |z'|) = n!/(|z'|!(n-|z'|)!) is binomial coefficient

This approximates Shapley values via weighted least squares.
Complexity: O(2^n) worst case, O(n²) with sampling
```

**2. TreeSHAP (For Tree Models):**
```
Exploits tree structure for exact computation in polynomial time!

Algorithm:
1. For each tree path, compute:
   - All possible feature subsets along path
   - Their marginal contributions
2. Weight by path probability
3. Sum over all paths

Complexity: O(TL D²) where:
- T: Number of trees
- L: Max leaves per tree
- D: Max depth

For typical trees: O(TL D²) << O(2^n)
Enables exact SHAP for XGBoost, Random Forest, etc.
```

**3. DeepSHAP (For Neural Networks):**
```
Approximation using DeepLIFT + Shapley sampling:

SHAP_i ≈ Σ_{refs} DeepLIFT(x, ref) / |refs|

where refs are reference/background samples

Uses backpropagation of Shapley values through network.
Complexity: O(m · p) where m = references, p = parameters
```

**Interpretation:**

**SHAP Value Meaning:**
```
φ_i > 0: Feature i pushes prediction HIGHER
φ_i < 0: Feature i pushes prediction LOWER
|φ_i|: Magnitude of impact

Example:
Base prediction (average): 0.5
Feature "age=65": φ_age = +0.15
Feature "income=low": φ_income = -0.10
Final prediction: 0.5 + 0.15 - 0.10 = 0.55

Interpretation: Age increases risk by 15%, income decreases by 10%
```

**Global Importance:**
```
Average absolute SHAP value:
Importance_i = (1/n) Σ^n_{j=1} |φ_i^(j)|

This measures average impact across all instances.
More principled than permutation importance!
```

**Interaction Values:**
```
Shapley Interaction Index between features i, j:

φ_{ij} = Σ_{S⊆N\{i,j}} w(|S|) · [v(S∪{i,j}) - v(S∪{i}) - v(S∪{j}) + v(S)]

Measures synergistic/antagonistic feature effects.
```

**Advantages of SHAP:**
- ✅ Theoretically grounded (game theory)
- ✅ Consistent and locally accurate
- ✅ Unique solution (satisfies axioms)
- ✅ Model-agnostic (works for any f)
- ✅ Efficient for trees (TreeSHAP)
- ✅ Handles feature interactions

**Limitations:**
- ❌ Computationally expensive (exponential without approximation)
- ❌ Assumes feature independence (problematic for correlated features)
- ❌ Requires many background samples for accuracy
- ❌ Can be unstable with small reference sets
- ❌ Difficult to interpret for high-dimensional data

```python
import shap

class SHAPExplainer:
    """SHAP explanations for any model"""

    def __init__(self, model, X_background):
        self.model = model
        self.X_background = X_background

    def explain_tree_model(self, X):
        """Explain tree-based model (fast)"""
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        return shap_values

    def explain_deep_model(self, X):
        """Explain deep learning model"""
        explainer = shap.DeepExplainer(self.model, self.X_background)
        shap_values = explainer.shap_values(X)
        return shap_values

    def explain_kernel(self, X):
        """Model-agnostic SHAP (slower but universal)"""
        explainer = shap.KernelExplainer(
            self.model.predict,
            self.X_background
        )
        shap_values = explainer.shap_values(X)
        return shap_values

    def visualize_summary(self, shap_values, X, feature_names):
        """Summary plot showing feature importances"""
        shap.summary_plot(shap_values, X, feature_names=feature_names)

    def visualize_force(self, shap_values, X, feature_names, instance_idx=0):
        """Force plot for single prediction"""
        shap.force_plot(
            explainer.expected_value,
            shap_values[instance_idx],
            X[instance_idx],
            feature_names=feature_names
        )

    def visualize_waterfall(self, shap_values, X, feature_names, instance_idx=0):
        """Waterfall plot for single prediction"""
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[instance_idx],
                base_values=explainer.expected_value,
                data=X[instance_idx],
                feature_names=feature_names
            )
        )

    def visualize_dependence(self, shap_values, X, feature_names, feature_idx=0):
        """Dependence plot showing feature interactions"""
        shap.dependence_plot(
            feature_idx,
            shap_values,
            X,
            feature_names=feature_names
        )

# Usage with XGBoost
import xgboost as xgb

# Train model
model = xgb.XGBClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Create explainer
explainer = SHAPExplainer(model, X_train[:100])  # Background samples

# Get SHAP values
shap_values = explainer.explain_tree_model(X_test)

# Visualizations
explainer.visualize_summary(shap_values, X_test, feature_names)
explainer.visualize_force(shap_values, X_test, feature_names, instance_idx=0)
explainer.visualize_waterfall(shap_values, X_test, feature_names, instance_idx=0)
explainer.visualize_dependence(shap_values, X_test, feature_names, feature_idx=0)
```

**Computing SHAP Values from Scratch:**
```python
from itertools import combinations

def shapley_value(model, instance, feature_idx, X_background):
    """Compute Shapley value for a feature"""

    num_features = len(instance)
    other_features = [i for i in range(num_features) if i != feature_idx]

    shapley_val = 0

    # Iterate over all subsets
    for r in range(len(other_features) + 1):
        for subset in combinations(other_features, r):
            subset = list(subset)

            # Marginal contribution
            # Prediction with feature
            instance_with = instance.copy()
            pred_with = model.predict([instance_with])[0]

            # Prediction without feature (marginalize)
            instance_without = instance.copy()
            instance_without[feature_idx] = X_background[:, feature_idx].mean()
            pred_without = model.predict([instance_without])[0]

            # Marginal contribution
            marginal = pred_with - pred_without

            # Weight by subset size
            weight = 1.0 / (num_features * comb(num_features - 1, len(subset)))
            shapley_val += weight * marginal

    return shapley_val

from scipy.special import comb

# Compute for all features
shapley_values = []
for feature_idx in range(X_test.shape[1]):
    shap_val = shapley_value(model, X_test[0], feature_idx, X_train)
    shapley_values.append(shap_val)

print("Shapley values:", shapley_values)
```

---

## Deep Learning Interpretability

### Grad-CAM (Gradient-weighted Class Activation Mapping)

**Key Idea:** Visualize which parts of image contribute to prediction.

```python
import torch
import torch.nn.functional as F

class GradCAM:
    """Grad-CAM for CNN interpretability"""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks"""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, image, class_idx=None):
        """Generate Grad-CAM heatmap"""

        # Forward pass
        self.model.eval()
        output = self.model(image)

        # Get target class
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        class_loss = output[0, class_idx]
        class_loss.backward()

        # Compute weights (global average pooling of gradients)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        # ReLU
        cam = F.relu(cam)

        # Normalize
        cam = cam - cam.min()
        cam = cam / cam.max()

        return cam

    def visualize(self, image, class_idx=None):
        """Visualize Grad-CAM overlay"""
        import cv2
        import numpy as np

        # Generate CAM
        cam = self.generate_cam(image, class_idx)

        # Resize CAM to image size
        cam = cam.squeeze().cpu().numpy()
        image_np = image.squeeze().permute(1, 2, 0).cpu().numpy()

        cam_resized = cv2.resize(cam, (image_np.shape[1], image_np.shape[0]))

        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Overlay
        overlay = heatmap * 0.4 + image_np * 255 * 0.6

        return overlay.astype(np.uint8)

# Usage
import torchvision.models as models

model = models.resnet50(pretrained=True)
gradcam = GradCAM(model, target_layer=model.layer4[-1])

# Load and preprocess image
image = load_image('cat.jpg')  # Returns tensor
overlay = gradcam.visualize(image.unsqueeze(0))

# Display
import matplotlib.pyplot as plt
plt.imshow(overlay)
plt.title('Grad-CAM Visualization')
plt.show()
```

---

### Attention Visualization

```python
class AttentionVisualizer:
    """Visualize attention weights in Transformers"""

    def __init__(self, model):
        self.model = model
        self.attention_weights = []

    def extract_attention(self, input_ids):
        """Extract attention weights from all layers"""

        # Forward pass with attention output
        outputs = self.model(
            input_ids,
            output_attentions=True
        )

        # Get attention weights from all layers
        attention = outputs.attentions  # Tuple of (batch, heads, seq, seq)

        return attention

    def visualize_attention_head(self, attention, layer=0, head=0):
        """Visualize single attention head"""
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Get attention for specific layer and head
        attn = attention[layer][0, head].detach().cpu().numpy()

        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(attn, cmap='viridis', square=True)
        plt.title(f'Attention Head {head} (Layer {layer})')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.show()

    def visualize_all_heads(self, attention, layer=0):
        """Visualize all attention heads in a layer"""
        import matplotlib.pyplot as plt

        num_heads = attention[layer].shape[1]
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))

        for head in range(num_heads):
            ax = axes[head // 4, head % 4]
            attn = attention[layer][0, head].detach().cpu().numpy()

            im = ax.imshow(attn, cmap='viridis')
            ax.set_title(f'Head {head}')
            ax.set_xlabel('Key')
            ax.set_ylabel('Query')

        plt.tight_layout()
        plt.show()

    def visualize_attention_flow(self, attention, tokens):
        """Visualize attention flow across layers"""
        import matplotlib.pyplot as plt
        import numpy as np

        num_layers = len(attention)
        num_tokens = attention[0].shape[-1]

        # Average attention across heads
        avg_attention = []
        for layer_attn in attention:
            avg_attn = layer_attn.mean(dim=1)[0].detach().cpu().numpy()
            avg_attention.append(avg_attn)

        # Create flow visualization
        fig, ax = plt.subplots(figsize=(12, 8))

        for layer_idx, attn in enumerate(avg_attention):
            # Visualize connections
            for i in range(num_tokens):
                for j in range(num_tokens):
                    if attn[i, j] > 0.1:  # Threshold
                        ax.plot([layer_idx, layer_idx + 1],
                               [i, j],
                               alpha=attn[i, j],
                               color='blue')

        ax.set_xlabel('Layer')
        ax.set_ylabel('Token Position')
        ax.set_title('Attention Flow Across Layers')
        plt.show()

# Usage with BERT
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Tokenize input
text = "The cat sat on the mat"
inputs = tokenizer(text, return_tensors='pt')

# Extract and visualize attention
visualizer = AttentionVisualizer(model)
attention = visualizer.extract_attention(inputs['input_ids'])

visualizer.visualize_attention_head(attention, layer=0, head=0)
visualizer.visualize_all_heads(attention, layer=11)
```

---

### Integrated Gradients

**Key Idea:** Attribute prediction to input features via path integral.

```python
class IntegratedGradients:
    """Integrated Gradients for feature attribution"""

    def __init__(self, model):
        self.model = model

    def compute(self, input_tensor, target_class, baseline=None, steps=50):
        """Compute integrated gradients"""

        if baseline is None:
            baseline = torch.zeros_like(input_tensor)

        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, steps).view(-1, 1, 1, 1)
        interpolated = baseline + alphas * (input_tensor - baseline)
        interpolated.requires_grad = True

        # Compute gradients for interpolated inputs
        gradients = []
        for interpolated_input in interpolated:
            self.model.zero_grad()

            output = self.model(interpolated_input.unsqueeze(0))
            class_score = output[0, target_class]

            class_score.backward()
            gradients.append(interpolated_input.grad.clone())

        # Average gradients
        avg_gradients = torch.stack(gradients).mean(dim=0)

        # Integrated gradients
        integrated_grads = (input_tensor - baseline) * avg_gradients

        return integrated_grads

    def visualize(self, input_image, attributions):
        """Visualize attributions"""
        import matplotlib.pyplot as plt
        import numpy as np

        # Convert to numpy
        image_np = input_image.squeeze().permute(1, 2, 0).cpu().numpy()
        attr_np = attributions.squeeze().permute(1, 2, 0).cpu().numpy()

        # Normalize attributions
        attr_np = np.abs(attr_np).sum(axis=2)
        attr_np = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min())

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(image_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(attr_np, cmap='hot')
        axes[1].set_title('Attributions')
        axes[1].axis('off')

        axes[2].imshow(image_np)
        axes[2].imshow(attr_np, alpha=0.5, cmap='hot')
        axes[2].set_title('Overlay')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

# Usage
ig = IntegratedGradients(model)

# Compute attributions
attributions = ig.compute(
    input_tensor=image,
    target_class=predicted_class,
    steps=50
)

# Visualize
ig.visualize(image, attributions)
```

---

## Feature Importance

### Permutation Importance

```python
from sklearn.metrics import accuracy_score
import numpy as np

class PermutationImportance:
    """Permutation feature importance"""

    def __init__(self, model, X, y, metric=accuracy_score):
        self.model = model
        self.X = X
        self.y = y
        self.metric = metric

    def compute(self, n_repeats=10):
        """Compute permutation importance"""

        # Baseline score
        baseline_score = self.metric(self.y, self.model.predict(self.X))

        importances = []

        for feature_idx in range(self.X.shape[1]):
            feature_importances = []

            for _ in range(n_repeats):
                # Permute feature
                X_permuted = self.X.copy()
                np.random.shuffle(X_permuted[:, feature_idx])

                # Score with permuted feature
                permuted_score = self.metric(self.y, self.model.predict(X_permuted))

                # Importance = drop in performance
                importance = baseline_score - permuted_score
                feature_importances.append(importance)

            importances.append({
                'mean': np.mean(feature_importances),
                'std': np.std(feature_importances)
            })

        return importances

    def visualize(self, importances, feature_names):
        """Visualize permutation importance"""
        import matplotlib.pyplot as plt

        means = [imp['mean'] for imp in importances]
        stds = [imp['std'] for imp in importances]

        indices = np.argsort(means)[::-1]

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(means)), [means[i] for i in indices],
               yerr=[stds[i] for i in indices], capsize=5)
        plt.xticks(range(len(means)), [feature_names[i] for i in indices], rotation=45)
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.title('Permutation Feature Importance')
        plt.tight_layout()
        plt.show()

# Usage
perm_imp = PermutationImportance(model, X_test, y_test)
importances = perm_imp.compute(n_repeats=10)
perm_imp.visualize(importances, feature_names)
```

---

### Partial Dependence Plots

```python
from sklearn.inspection import partial_dependence, PartialDependenceDisplay

class PartialDependence:
    """Partial dependence analysis"""

    def __init__(self, model, X, feature_names):
        self.model = model
        self.X = X
        self.feature_names = feature_names

    def plot_1d(self, feature_idx):
        """1D partial dependence plot"""
        import matplotlib.pyplot as plt

        pd_result = partial_dependence(
            self.model,
            self.X,
            features=[feature_idx],
            grid_resolution=50
        )

        plt.figure(figsize=(8, 6))
        plt.plot(pd_result['grid_values'][0], pd_result['average'][0])
        plt.xlabel(self.feature_names[feature_idx])
        plt.ylabel('Partial Dependence')
        plt.title(f'PDP: {self.feature_names[feature_idx]}')
        plt.show()

    def plot_2d(self, feature_idx1, feature_idx2):
        """2D partial dependence plot"""

        display = PartialDependenceDisplay.from_estimator(
            self.model,
            self.X,
            features=[(feature_idx1, feature_idx2)],
            feature_names=self.feature_names
        )

        display.figure_.suptitle('2D Partial Dependence Plot')
        display.figure_.tight_layout()

# Usage
pd = PartialDependence(model, X_train, feature_names)
pd.plot_1d(feature_idx=0)
pd.plot_2d(feature_idx1=0, feature_idx2=1)
```

---

## Counterfactual Explanations

### DiCE (Diverse Counterfactual Explanations)

```python
import dice_ml
from dice_ml.utils import helpers

class CounterfactualExplainer:
    """Generate counterfactual explanations"""

    def __init__(self, model, X_train, continuous_features, outcome_name):
        # Create DiCE data object
        self.data = dice_ml.Data(
            dataframe=X_train,
            continuous_features=continuous_features,
            outcome_name=outcome_name
        )

        # Create model object
        self.model = dice_ml.Model(model=model, backend='sklearn')

        # Create DiCE explainer
        self.explainer = dice_ml.Dice(self.data, self.model)

    def generate(self, query_instance, total_CFs=5, desired_class=1):
        """Generate counterfactual explanations"""

        dice_exp = self.explainer.generate_counterfactuals(
            query_instance,
            total_CFs=total_CFs,
            desired_class=desired_class
        )

        return dice_exp

    def visualize(self, dice_exp):
        """Visualize counterfactuals"""
        dice_exp.visualize_as_dataframe(show_only_changes=True)

# Usage
cf_explainer = CounterfactualExplainer(
    model=model,
    X_train=df_train,
    continuous_features=['age', 'income'],
    outcome_name='approved'
)

# Generate counterfactuals for denied application
cf_exp = cf_explainer.generate(
    query_instance=denied_application,
    total_CFs=5,
    desired_class='approved'
)

cf_explainer.visualize(cf_exp)

# Example output:
# "To get approved, increase income from $40K to $55K and reduce debt ratio from 0.6 to 0.4"
```

---

## Evaluation of Explanations

### Faithfulness Metrics

```python
class ExplanationEvaluator:
    """Evaluate quality of explanations"""

    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y

    def faithfulness(self, importance_scores, top_k=5):
        """Measure faithfulness by removing top features"""

        baseline_preds = self.model.predict(self.X)
        baseline_acc = (baseline_preds == self.y).mean()

        # Remove top-k important features
        top_features = np.argsort(importance_scores)[-top_k:]
        X_masked = self.X.copy()
        X_masked[:, top_features] = 0

        masked_preds = self.model.predict(X_masked)
        masked_acc = (masked_preds == self.y).mean()

        # Faithfulness = drop in accuracy
        faithfulness_score = baseline_acc - masked_acc

        return faithfulness_score

    def stability(self, explainer, instance, n_samples=10):
        """Measure stability of explanations"""

        explanations = []
        for _ in range(n_samples):
            # Add small noise
            noisy_instance = instance + np.random.normal(0, 0.01, instance.shape)
            exp = explainer.explain(noisy_instance)
            explanations.append(exp)

        # Compute variance across explanations
        explanations = np.array(explanations)
        stability_score = 1 / (1 + np.mean(np.var(explanations, axis=0)))

        return stability_score
```

---

## Production XAI

### Model Cards

```python
class ModelCard:
    """Model card for documentation and transparency"""

    def __init__(self, model_name, version):
        self.model_name = model_name
        self.version = version
        self.card = {
            'model_details': {},
            'intended_use': {},
            'factors': {},
            'metrics': {},
            'evaluation_data': {},
            'training_data': {},
            'quantitative_analyses': {},
            'ethical_considerations': {},
            'caveats_and_recommendations': {}
        }

    def add_model_details(self, architecture, training_date, developers):
        self.card['model_details'] = {
            'architecture': architecture,
            'training_date': training_date,
            'developers': developers,
            'version': self.version
        }

    def add_intended_use(self, primary_use, out_of_scope_uses):
        self.card['intended_use'] = {
            'primary_intended_uses': primary_use,
            'primary_intended_users': 'Data scientists, ML engineers',
            'out_of_scope_use_cases': out_of_scope_uses
        }

    def add_metrics(self, performance_metrics):
        self.card['metrics'] = performance_metrics

    def add_fairness_analysis(self, fairness_metrics):
        self.card['quantitative_analyses']['fairness'] = fairness_metrics

    def generate_report(self):
        """Generate model card report"""
        import json
        report = f"""
# Model Card: {self.model_name} v{self.version}

## Model Details
{json.dumps(self.card['model_details'], indent=2)}

## Intended Use
{json.dumps(self.card['intended_use'], indent=2)}

## Performance Metrics
{json.dumps(self.card['metrics'], indent=2)}

## Fairness Analysis
{json.dumps(self.card['quantitative_analyses'].get('fairness', {}), indent=2)}

## Ethical Considerations
{json.dumps(self.card.get('ethical_considerations', {}), indent=2)}
        """

        return report

# Usage
card = ModelCard(model_name='Credit Risk Classifier', version='2.0')

card.add_model_details(
    architecture='XGBoost',
    training_date='2024-01-15',
    developers='ML Team'
)

card.add_intended_use(
    primary_use='Assess credit risk for loan applications',
    out_of_scope_uses=['Criminal justice', 'Medical diagnosis']
)

card.add_metrics({
    'accuracy': 0.89,
    'precision': 0.87,
    'recall': 0.85,
    'auc_roc': 0.92
})

card.add_fairness_analysis({
    'demographic_parity_difference': 0.03,
    'equal_opportunity_difference': 0.02
})

# Generate report
report = card.generate_report()
print(report)
```

---

## Summary

| Method | Type | Model | Scope | Cost |
|--------|------|-------|-------|------|
| **LIME** | Model-agnostic | Any | Local | Medium |
| **SHAP** | Model-agnostic | Any | Local/Global | High |
| **Grad-CAM** | Model-specific | CNN | Local | Low |
| **Integrated Gradients** | Model-specific | Differentiable | Local | Medium |
| **Attention** | Model-specific | Transformer | Local | Low |
| **Permutation Importance** | Model-agnostic | Any | Global | High |
| **Counterfactuals** | Model-agnostic | Any | Local | High |

---

## Key Takeaways

1. **SHAP** is the most theoretically sound method (Shapley values)
2. **LIME** is fast and intuitive for local explanations
3. **Grad-CAM** for CNN visualization
4. **Attention weights** for Transformers
5. **Counterfactuals** provide actionable explanations
6. **Model cards** for transparency and documentation
7. **Evaluate explanations** for faithfulness and stability

**Best Practices:**
- Use multiple explanation methods
- Validate explanations with domain experts
- Document limitations clearly
- Provide both local and global explanations
- Consider fairness in explanations
- Make explanations accessible to non-experts

**Next Steps:**
- Implement SHAP for your production model
- Create model cards for all deployed models
- Add explainability dashboards
- Conduct fairness audits
- Build user-facing explanation interfaces


---

## 📚 References

**XAI Foundations:**

1. **Ribeiro, M. T., Singh, S., & Guestrin, C.** (2016). "'Why should I trust you?': Explaining the predictions of any classifier." *KDD 2016*, 1135-1144.
   - LIME algorithm

2. **Lundberg, S. M., & Lee, S. I.** (2017). "A unified approach to interpreting model predictions." *NIPS 2017*, 4765-4774.
   - SHAP framework

3. **Shapley, L. S.** (1953). "A value for n-person games." In *Contributions to the Theory of Games*, 2(28), 307-317.
   - Original Shapley value (game theory)

**SHAP Theory & Methods:**

4. **Lundberg, S. M., Erion, G. G., & Lee, S. I.** (2018). "Consistent individualized feature attribution for tree ensembles." *arXiv:1802.03888*.
   - TreeSHAP algorithm

5. **Lundberg, S. M., Erion, G., Chen, H., et al.** (2020). "From local explanations to global understanding with explainable AI for trees." *Nature Machine Intelligence*, 2(1), 56-67.
   - TreeSHAP applications and theory

6. **Janzing, D., Minorics, L., & Blöbaum, P.** (2020). "Feature relevance quantification in explainable AI: A causal problem." *AISTATS 2020*.
   - Causal perspective on SHAP

**Deep Learning Interpretability:**

7. **Selvaraju, R. R., Cogswell, M., Das, A., et al.** (2017). "Grad-CAM: Visual explanations from deep networks via gradient-based localization." *ICCV 2017*, 618-626.
   - Grad-CAM for CNNs

8. **Sundararajan, M., Taly, A., & Yan, Q.** (2017). "Axiomatic attribution for deep networks." *ICML 2017*, 3319-3328.
   - Integrated Gradients

9. **Shrikumar, A., Greenside, P., & Kundaje, A.** (2017). "Learning important features through propagating activation differences." *ICML 2017*, 3145-3153.
   - DeepLIFT algorithm

**Attention & Transformers:**

10. **Jain, S., & Wallace, B. C.** (2019). "Attention is not explanation." *NAACL-HLT 2019*, 3543-3556.
    - Limitations of attention as explanation

11. **Wiegreffe, S., & Pinter, Y.** (2019). "Attention is not not explanation." *EMNLP-IJCNLP 2019*, 11-20.
    - Defense of attention weights

**Counterfactual Explanations:**

12. **Wachter, S., Mittelstadt, B., & Russell, C.** (2017). "Counterfactual explanations without opening the black box: Automated decisions and the GDPR." *Harvard Journal of Law & Technology*, 31, 841.
    - Legal perspective on XAI

13. **Mothilal, R. K., Sharma, A., & Tan, C.** (2020). "Explaining machine learning classifiers through diverse counterfactual explanations." *FAT* 2020*, 607-617.
    - DiCE algorithm

**Evaluation of Explanations:**

14. **Doshi-Velez, F., & Kim, B.** (2017). "Towards a rigorous science of interpretable machine learning." *arXiv:1702.08608*.
    - Framework for evaluating interpretability

15. **Adebayo, J., Gilmer, J., Muelly, M., et al.** (2018). "Sanity checks for saliency maps." *NeurIPS 2018*, 9505-9515.
    - Critical evaluation of attribution methods

**Books:**

- **Molnar, C.** (2022). *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable* (2nd ed.).
  - Free online: https://christophm.github.io/interpretable-ml-book/

- **Samek, W., Montavon, G., Vedaldi, A., et al.** (Eds.) (2019). *Explainable AI: Interpreting, Explaining and Visualizing Deep Learning*. Springer.

**Online Resources:**
- SHAP documentation: https://shap.readthedocs.io/
- LIME: https://github.com/marcotcr/lime  
- Captum (PyTorch): https://captum.ai/
- InterpretML: https://interpret.ml/
- AI Explainability 360: https://aix360.mybluemix.net/

**Regulatory & Ethics:**

- **EU AI Act**: https://artificialintelligenceact.eu/
- **GDPR Right to Explanation**: Article 22
- **NIST AI Risk Management Framework**: https://www.nist.gov/itl/ai-risk-management-framework

---

*Explainability is not optional—it's essential for trustworthy, debuggable, and compliant AI systems!*
