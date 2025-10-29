# Few-Shot & Meta-Learning Guide

Learn from limited data: master meta-learning algorithms that enable rapid adaptation.

## Table of Contents
1. [Introduction to Few-Shot Learning](#introduction-to-few-shot-learning)
2. [Metric-Based Methods](#metric-based-methods)
3. [Model-Based Methods](#model-based-methods)
4. [Optimization-Based Methods](#optimization-based-methods)
5. [Advanced Meta-Learning](#advanced-meta-learning)
6. [Applications](#applications)

---

## Introduction to Few-Shot Learning

### The Few-Shot Learning Problem

**Goal:** Learn to classify new categories from only a few examples (1-5 samples per class).

**Notation:**
- **N-way K-shot:** N classes, K examples per class
- **Support Set:** Training examples (K examples × N classes)
- **Query Set:** Test examples

**Example:**
```
5-way 1-shot classification:
- Support: 1 image each of {cat, dog, bird, fish, horse}
- Query: New image → classify into one of 5 classes
```

**Why Few-Shot Learning?**
1. **Limited labeled data** - Expensive to annotate
2. **Rare classes** - Long-tail distribution
3. **Rapid adaptation** - New tasks/domains
4. **Human-like learning** - Learn from few examples

---

## Metric-Based Methods

### Siamese Networks

**Key Idea:** Learn a similarity metric between examples.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    """Siamese network for one-shot learning"""

    def __init__(self, embedding_dim=128):
        super().__init__()

        # Shared embedding network
        self.embedding = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(256 * 28 * 28, embedding_dim)
        )

    def forward(self, x1, x2):
        """Compute embeddings for pair of inputs"""
        emb1 = self.embedding(x1)
        emb2 = self.embedding(x2)
        return emb1, emb2

    def predict(self, x1, x2):
        """Predict similarity"""
        emb1, emb2 = self.forward(x1, x2)
        distance = F.pairwise_distance(emb1, emb2)
        similarity = torch.exp(-distance)
        return similarity

class ContrastiveLoss(nn.Module):
    """Contrastive loss for Siamese networks"""

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1, emb2, label):
        """
        label: 1 if same class, 0 if different
        """
        distance = F.pairwise_distance(emb1, emb2)

        # Similar pairs: minimize distance
        # Dissimilar pairs: maximize distance up to margin
        loss = label * distance.pow(2) + \
               (1 - label) * F.relu(self.margin - distance).pow(2)

        return loss.mean()

# Training
def train_siamese(model, train_loader, epochs=10):
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        for (img1, img2), label in train_loader:
            optimizer.zero_grad()

            emb1, emb2 = model(img1, img2)
            loss = criterion(emb1, emb2, label)

            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# Few-shot prediction
def few_shot_predict(model, support_set, query_image):
    """Predict class of query image using support set"""

    similarities = []

    for support_img, support_label in support_set:
        sim = model.predict(query_image.unsqueeze(0), support_img.unsqueeze(0))
        similarities.append((sim.item(), support_label))

    # Return class of most similar support example
    predicted_class = max(similarities, key=lambda x: x[0])[1]
    return predicted_class

# Usage
model = SiameseNetwork(embedding_dim=128)
train_siamese(model, train_loader, epochs=10)

prediction = few_shot_predict(model, support_set, query_image)
```

---

### Prototypical Networks

**Key Idea:** Represent each class by its prototype (mean embedding).

```python
class PrototypicalNetwork(nn.Module):
    """Prototypical Networks for few-shot learning"""

    def __init__(self):
        super().__init__()

        # Embedding network
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten()
        )

    def forward(self, support_images, support_labels, query_images):
        """
        support_images: (N*K, C, H, W) where N=classes, K=shots
        support_labels: (N*K,)
        query_images: (Q, C, H, W) where Q=query examples
        """

        # Compute embeddings
        support_embeddings = self.encoder(support_images)
        query_embeddings = self.encoder(query_images)

        # Compute class prototypes (mean of support embeddings per class)
        unique_labels = support_labels.unique()
        prototypes = []

        for label in unique_labels:
            class_embeddings = support_embeddings[support_labels == label]
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)

        prototypes = torch.stack(prototypes)  # (N, embedding_dim)

        # Compute distances from queries to prototypes
        distances = self._euclidean_distance(query_embeddings, prototypes)

        # Convert to log probabilities
        log_probs = F.log_softmax(-distances, dim=1)

        return log_probs

    def _euclidean_distance(self, x, y):
        """
        x: (Q, D)
        y: (N, D)
        Returns: (Q, N) distance matrix
        """
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        return torch.pow(x - y, 2).sum(2)

# Episodic training
def train_prototypical(model, dataset, n_way=5, k_shot=1, q_query=15, episodes=1000):
    """Train with episodic sampling"""

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for episode in range(episodes):
        # Sample N classes
        classes = np.random.choice(dataset.num_classes, n_way, replace=False)

        support_images = []
        support_labels = []
        query_images = []
        query_labels = []

        # Sample K support + Q query examples per class
        for i, cls in enumerate(classes):
            class_images = dataset.get_class_images(cls)
            indices = np.random.choice(len(class_images), k_shot + q_query, replace=False)

            # Support set
            support_images.append(class_images[indices[:k_shot]])
            support_labels.extend([i] * k_shot)

            # Query set
            query_images.append(class_images[indices[k_shot:]])
            query_labels.extend([i] * q_query)

        support_images = torch.cat(support_images)
        support_labels = torch.tensor(support_labels)
        query_images = torch.cat(query_images)
        query_labels = torch.tensor(query_labels)

        # Forward pass
        optimizer.zero_grad()
        log_probs = model(support_images, support_labels, query_images)
        loss = F.nll_loss(log_probs, query_labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        if episode % 100 == 0:
            accuracy = (log_probs.argmax(dim=1) == query_labels).float().mean()
            print(f'Episode {episode}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

# Usage
model = PrototypicalNetwork()
train_prototypical(model, dataset, n_way=5, k_shot=1, q_query=15, episodes=10000)
```

---

### Matching Networks

**Key Idea:** Attention mechanism over support set.

```python
class MatchingNetwork(nn.Module):
    """Matching Networks with attention"""

    def __init__(self, embedding_dim=64):
        super().__init__()

        # Embedding function
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64 * 56 * 56, embedding_dim)
        )

        # Full context embeddings (LSTM)
        self.lstm = nn.LSTM(embedding_dim, embedding_dim, batch_first=True, bidirectional=True)

    def forward(self, support_images, support_labels, query_image):
        """
        Attend over support set to classify query
        """

        # Encode support and query
        support_embeds = self.encoder(support_images)  # (K*N, D)
        query_embed = self.encoder(query_image.unsqueeze(0))  # (1, D)

        # Full context embeddings with LSTM
        support_embeds, _ = self.lstm(support_embeds.unsqueeze(0))
        support_embeds = support_embeds.squeeze(0)  # (K*N, 2*D)

        # Attention weights
        attention = F.softmax(
            torch.matmul(query_embed, support_embeds.t()),
            dim=1
        )  # (1, K*N)

        # Weighted combination of support labels
        num_classes = support_labels.max().item() + 1
        class_probs = torch.zeros(1, num_classes)

        for i, label in enumerate(support_labels):
            class_probs[0, label] += attention[0, i]

        return class_probs

# Training
def train_matching(model, dataset, episodes=10000):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for episode in range(episodes):
        # Sample episode
        support_images, support_labels, query_image, query_label = sample_episode(dataset)

        # Forward
        optimizer.zero_grad()
        probs = model(support_images, support_labels, query_image)
        loss = F.cross_entropy(probs, query_label.unsqueeze(0))

        # Backward
        loss.backward()
        optimizer.step()

        if episode % 100 == 0:
            accuracy = (probs.argmax(dim=1) == query_label).float().mean()
            print(f'Episode {episode}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')
```

---

## Model-Based Methods

### Memory-Augmented Neural Networks (MANN)

```python
class NeuralTuringMachine(nn.Module):
    """Neural Turing Machine for few-shot learning"""

    def __init__(self, input_dim, output_dim, memory_size=128, memory_dim=40):
        super().__init__()

        self.memory_size = memory_size
        self.memory_dim = memory_dim

        # Controller (LSTM)
        self.controller = nn.LSTM(input_dim, 200, batch_first=True)

        # Read head
        self.read_key = nn.Linear(200, memory_dim)
        self.read_strength = nn.Linear(200, 1)

        # Write head
        self.write_key = nn.Linear(200, memory_dim)
        self.write_strength = nn.Linear(200, 1)
        self.write_vector = nn.Linear(200, memory_dim)

        # Output
        self.output = nn.Linear(200 + memory_dim, output_dim)

        # Memory
        self.register_buffer('memory', torch.zeros(memory_size, memory_dim))

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Initialize memory
        memory = self.memory.unsqueeze(0).repeat(batch_size, 1, 1)

        outputs = []
        hidden = None

        for t in range(seq_len):
            # Controller
            controller_out, hidden = self.controller(x[:, t:t+1], hidden)
            controller_out = controller_out.squeeze(1)

            # Read from memory
            read_key = self.read_key(controller_out)
            read_strength = torch.sigmoid(self.read_strength(controller_out))

            # Attention over memory
            read_attention = F.softmax(
                read_strength * F.cosine_similarity(
                    read_key.unsqueeze(1),
                    memory,
                    dim=2
                ),
                dim=1
            )

            # Read vector
            read_vector = (read_attention.unsqueeze(2) * memory).sum(dim=1)

            # Write to memory
            write_key = self.write_key(controller_out)
            write_strength = torch.sigmoid(self.write_strength(controller_out))
            write_vector = torch.tanh(self.write_vector(controller_out))

            # Write attention
            write_attention = F.softmax(
                write_strength * F.cosine_similarity(
                    write_key.unsqueeze(1),
                    memory,
                    dim=2
                ),
                dim=1
            )

            # Update memory
            memory = memory + write_attention.unsqueeze(2) * write_vector.unsqueeze(1)

            # Output
            output = self.output(torch.cat([controller_out, read_vector], dim=1))
            outputs.append(output)

        return torch.stack(outputs, dim=1)
```

---

## Optimization-Based Methods

### MAML (Model-Agnostic Meta-Learning)

**Key Idea:** Learn initialization that quickly adapts to new tasks.

```python
class MAML:
    """Model-Agnostic Meta-Learning"""

    def __init__(self, model, inner_lr=0.01, meta_lr=0.001):
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)

    def inner_loop(self, support_x, support_y, num_steps=5):
        """
        Adapt to task using support set
        Returns adapted parameters
        """
        # Clone model parameters
        adapted_params = [p.clone() for p in self.model.parameters()]

        for step in range(num_steps):
            # Forward pass with adapted parameters
            logits = self._forward_with_params(support_x, adapted_params)
            loss = F.cross_entropy(logits, support_y)

            # Compute gradients
            grads = torch.autograd.grad(loss, adapted_params, create_graph=True)

            # Inner loop update (gradient descent)
            adapted_params = [p - self.inner_lr * g for p, g in zip(adapted_params, grads)]

        return adapted_params

    def meta_update(self, tasks):
        """
        Meta-update using multiple tasks

        tasks: List of (support_x, support_y, query_x, query_y)
        """
        self.meta_optimizer.zero_grad()
        meta_loss = 0

        for support_x, support_y, query_x, query_y in tasks:
            # Inner loop: adapt to task
            adapted_params = self.inner_loop(support_x, support_y, num_steps=5)

            # Evaluate on query set
            logits = self._forward_with_params(query_x, adapted_params)
            task_loss = F.cross_entropy(logits, query_y)

            meta_loss += task_loss

        # Meta-update
        meta_loss /= len(tasks)
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()

    def _forward_with_params(self, x, params):
        """Forward pass with custom parameters"""
        # This is a simplified version - actual implementation
        # needs to handle the model architecture properly
        x = F.linear(x, params[0], params[1])
        # ... (rest of the model)
        return x

# Training MAML
def train_maml(model, task_sampler, meta_iterations=10000):
    """Meta-train MAML"""

    maml = MAML(model, inner_lr=0.01, meta_lr=0.001)

    for iteration in range(meta_iterations):
        # Sample batch of tasks
        tasks = task_sampler.sample_batch(batch_size=4, n_way=5, k_shot=1, q_query=15)

        # Meta-update
        meta_loss = maml.meta_update(tasks)

        if iteration % 100 == 0:
            print(f'Iteration {iteration}, Meta-loss: {meta_loss:.4f}')

    return maml

# Fine-tuning on new task
def finetune_maml(maml, support_x, support_y, num_steps=10):
    """Fine-tune MAML on new task"""

    adapted_params = maml.inner_loop(support_x, support_y, num_steps=num_steps)
    return adapted_params
```

**Complete MAML Implementation:**
```python
import higher  # Library for meta-learning

class MAMLTrainer:
    """Complete MAML implementation using higher library"""

    def __init__(self, model, inner_lr=0.01, meta_lr=0.001, inner_steps=5):
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.inner_steps = inner_steps
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)

    def train_step(self, task_batch):
        """Single meta-training step"""

        self.meta_optimizer.zero_grad()
        meta_loss = 0

        for task in task_batch:
            support_x, support_y, query_x, query_y = task

            # Inner loop
            with higher.innerloop_ctx(
                self.model,
                torch.optim.SGD(self.model.parameters(), lr=self.inner_lr)
            ) as (fmodel, diffopt):

                # Adapt to support set
                for _ in range(self.inner_steps):
                    support_loss = F.cross_entropy(fmodel(support_x), support_y)
                    diffopt.step(support_loss)

                # Evaluate on query set
                query_loss = F.cross_entropy(fmodel(query_x), query_y)
                meta_loss += query_loss

        # Meta-update
        meta_loss /= len(task_batch)
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()

# Usage
model = SimpleCNN(num_classes=5)
trainer = MAMLTrainer(model, inner_lr=0.01, meta_lr=0.001, inner_steps=5)

for iteration in range(10000):
    task_batch = sample_task_batch(n_tasks=4, n_way=5, k_shot=1)
    loss = trainer.train_step(task_batch)

    if iteration % 100 == 0:
        print(f'Iteration {iteration}, Loss: {loss:.4f}')
```

---

## Advanced Meta-Learning

### Meta-SGD

**Extension of MAML:** Learn learning rates per parameter.

```python
class MetaSGD(MAML):
    """Meta-SGD: Learn inner loop learning rates"""

    def __init__(self, model, meta_lr=0.001):
        super().__init__(model, inner_lr=0.01, meta_lr=meta_lr)

        # Learnable inner learning rates (one per parameter)
        self.inner_lrs = nn.ParameterList([
            nn.Parameter(torch.ones_like(p) * 0.01)
            for p in model.parameters()
        ])

    def inner_loop(self, support_x, support_y, num_steps=5):
        """Adapt with learned learning rates"""

        adapted_params = [p.clone() for p in self.model.parameters()]

        for step in range(num_steps):
            logits = self._forward_with_params(support_x, adapted_params)
            loss = F.cross_entropy(logits, support_y)

            grads = torch.autograd.grad(loss, adapted_params, create_graph=True)

            # Update with learned learning rates
            adapted_params = [
                p - lr * g
                for p, lr, g in zip(adapted_params, self.inner_lrs, grads)
            ]

        return adapted_params
```

---

### Reptile

**Simpler alternative to MAML** - directly average adapted parameters.

```python
class Reptile:
    """Reptile meta-learning"""

    def __init__(self, model, inner_lr=0.01, meta_lr=0.1, inner_steps=5):
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.inner_steps = inner_steps

    def train_step(self, task):
        """Single Reptile step"""

        support_x, support_y = task

        # Save initial parameters
        initial_params = [p.clone() for p in self.model.parameters()]

        # Inner loop (standard SGD on task)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.inner_lr)

        for _ in range(self.inner_steps):
            optimizer.zero_grad()
            loss = F.cross_entropy(self.model(support_x), support_y)
            loss.backward()
            optimizer.step()

        # Meta-update: move towards adapted parameters
        with torch.no_grad():
            for p_initial, p_adapted in zip(initial_params, self.model.parameters()):
                p_adapted.data = p_initial + self.meta_lr * (p_adapted - p_initial)

# Training
reptile = Reptile(model, inner_lr=0.01, meta_lr=0.1, inner_steps=10)

for iteration in range(10000):
    task = sample_task(n_way=5, k_shot=5)
    reptile.train_step(task)
```

---

## Applications

### Few-Shot Image Classification

```python
class FewShotClassifier:
    """Production few-shot classification system"""

    def __init__(self, model_type='prototypical'):
        if model_type == 'prototypical':
            self.model = PrototypicalNetwork()
        elif model_type == 'maml':
            self.model = MAML(SimpleCNN())

        self.model.eval()

    def predict(self, support_set, query_image):
        """
        support_set: List of (image, label) tuples
        query_image: Image to classify
        """

        with torch.no_grad():
            if isinstance(self.model, PrototypicalNetwork):
                support_images = torch.stack([img for img, _ in support_set])
                support_labels = torch.tensor([label for _, label in support_set])

                log_probs = self.model(
                    support_images,
                    support_labels,
                    query_image.unsqueeze(0)
                )

                predicted_class = log_probs.argmax(dim=1).item()

            return predicted_class

    def add_new_class(self, class_name, examples):
        """Add new class with few examples"""
        # Store examples for this class
        self.support_set[class_name] = examples

# Usage
classifier = FewShotClassifier(model_type='prototypical')

# Add new classes with 5 examples each
classifier.add_new_class('dog', dog_images[:5])
classifier.add_new_class('cat', cat_images[:5])

# Classify new image
prediction = classifier.predict(support_set, query_image)
```

---

### Few-Shot Object Detection

```python
class FewShotDetector:
    """Few-shot object detection"""

    def __init__(self, base_detector):
        self.base_detector = base_detector
        self.prototypes = {}

    def compute_prototypes(self, support_images, support_boxes, support_labels):
        """Compute prototypes for each class"""

        for img, boxes, labels in zip(support_images, support_boxes, support_labels):
            features = self.base_detector.extract_features(img)

            for box, label in zip(boxes, labels):
                # ROI pooling
                roi_features = self.base_detector.roi_pool(features, box)

                # Update prototype
                if label not in self.prototypes:
                    self.prototypes[label] = []
                self.prototypes[label].append(roi_features)

        # Average to get final prototypes
        for label in self.prototypes:
            self.prototypes[label] = torch.stack(self.prototypes[label]).mean(dim=0)

    def detect(self, query_image):
        """Detect objects in query image"""

        # Extract features
        features = self.base_detector.extract_features(query_image)

        # Generate proposals
        proposals = self.base_detector.generate_proposals(features)

        detections = []
        for proposal in proposals:
            roi_features = self.base_detector.roi_pool(features, proposal)

            # Compare with prototypes
            best_score = 0
            best_label = None

            for label, prototype in self.prototypes.items():
                score = F.cosine_similarity(roi_features, prototype, dim=0)
                if score > best_score:
                    best_score = score
                    best_label = label

            if best_score > 0.5:  # Threshold
                detections.append({
                    'box': proposal,
                    'label': best_label,
                    'score': best_score
                })

        return detections
```

---

## Summary

| Method | Type | Pros | Cons | Use Case |
|--------|------|------|------|----------|
| **Siamese** | Metric | Simple, interpretable | Limited capacity | Verification tasks |
| **Prototypical** | Metric | Fast, effective | Assumes Euclidean space | Image classification |
| **Matching** | Metric | Attention mechanism | Complex | Classification with context |
| **MAML** | Optimization | General, powerful | Slow training | Any domain |
| **Reptile** | Optimization | Simple, stable | Less effective than MAML | Quick experiments |
| **Meta-SGD** | Optimization | Learns LRs | More parameters | Fine-tuning heavy |

---

## Key Takeaways

1. **Prototypical Networks** are the go-to for few-shot classification
2. **MAML** is most general but computationally expensive
3. **Metric learning** (Siamese, Prototypical) is faster and simpler
4. **Episodic training** is essential for meta-learning
5. **Support set quality** matters more than quantity
6. **Domain shift** can hurt few-shot performance
7. **Pre-training** on large dataset helps significantly

**Best Practices:**
- Start with Prototypical Networks
- Use episodic sampling during training
- Pre-train on related data when possible
- Augment support set carefully
- Validate on held-out classes
- Consider task distribution

**Next Steps:**
- Implement Prototypical Networks for your data
- Try MAML for rapid adaptation
- Experiment with different architectures
- Apply to real-world few-shot problems
- Combine with transfer learning
- Explore meta-learning for reinforcement learning
