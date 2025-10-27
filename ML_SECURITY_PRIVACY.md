# ML Security and Privacy

**Complete Guide to Secure and Privacy-Preserving Machine Learning Systems**

*Last Updated: October 27, 2025*

---

## Table of Contents

1. [Introduction](#introduction)
2. [Threat Model](#threat-model)
3. [Data Security](#data-security)
4. [Model Security](#model-security)
5. [Privacy-Preserving ML](#privacy-preserving-ml)
6. [Compliance & Regulations](#compliance--regulations)
7. [Secure Deployment](#secure-deployment)
8. [Monitoring & Incident Response](#monitoring--incident-response)
9. [Best Practices](#best-practices)
10. [Interview Questions](#interview-questions)

---

## Introduction

ML security and privacy are critical for production systems. A breach can lead to:
- **Data leaks**: Sensitive training data exposed
- **Model theft**: Competitors steal proprietary models
- **Adversarial attacks**: Malicious inputs fool models
- **Privacy violations**: User data misuse, regulatory fines
- **Reputational damage**: Loss of customer trust

**Key Principles:**
1. **Defense in Depth**: Multiple layers of security
2. **Least Privilege**: Minimal access rights
3. **Privacy by Design**: Build privacy into systems from start
4. **Assume Breach**: Plan for compromises
5. **Continuous Monitoring**: Detect attacks early

---

## Threat Model

### Attack Types

**1. Data Poisoning**
- **Goal**: Corrupt training data to degrade model
- **Example**: Inject fake reviews to bias sentiment model
- **Impact**: Model makes wrong predictions

**2. Model Inversion**
- **Goal**: Extract training data from model
- **Example**: Reconstruct faces from face recognition model
- **Impact**: Privacy violation

**3. Membership Inference**
- **Goal**: Determine if specific data point was in training set
- **Example**: Was patient X in medical training data?
- **Impact**: Privacy leak

**4. Model Stealing**
- **Goal**: Replicate model using only API access
- **Example**: Query model repeatedly to learn decision boundary
- **Impact**: Intellectual property theft

**5. Adversarial Examples**
- **Goal**: Craft inputs that fool model
- **Example**: Add imperceptible noise to image to change prediction
- **Impact**: Security bypass (e.g., facial recognition)

**6. Backdoor Attacks**
- **Goal**: Trigger specific behavior with secret input
- **Example**: Model misclassifies images with specific watermark
- **Impact**: Targeted misclassification

### Threat Matrix

| Threat | Likelihood | Impact | Mitigation Difficulty |
|--------|------------|--------|----------------------|
| Data poisoning | Medium | High | Medium |
| Model inversion | Low | High | Hard |
| Membership inference | Medium | Medium | Hard |
| Model stealing | High | Medium | Medium |
| Adversarial examples | High | High | Hard |
| Backdoor attacks | Low | Very High | Very Hard |

---

## Data Security

### 1. Data Encryption

**At Rest:**
```python
from cryptography.fernet import Fernet
import pandas as pd

class DataEncryption:
    """Encrypt sensitive data at rest"""

    def __init__(self, key=None):
        if key is None:
            key = Fernet.generate_key()
        self.cipher = Fernet(key)
        self.key = key

    def encrypt_dataframe(self, df, sensitive_columns):
        """Encrypt sensitive columns in DataFrame"""
        df_encrypted = df.copy()

        for col in sensitive_columns:
            df_encrypted[col] = df_encrypted[col].apply(
                lambda x: self.cipher.encrypt(str(x).encode()).decode()
            )

        return df_encrypted

    def decrypt_dataframe(self, df_encrypted, sensitive_columns):
        """Decrypt sensitive columns"""
        df_decrypted = df_encrypted.copy()

        for col in sensitive_columns:
            df_decrypted[col] = df_decrypted[col].apply(
                lambda x: self.cipher.decrypt(x.encode()).decode()
            )

        return df_decrypted

    def save_key(self, filepath):
        """Save encryption key securely"""
        with open(filepath, 'wb') as f:
            f.write(self.key)

# Usage
encryptor = DataEncryption()

# Encrypt sensitive data
df = pd.DataFrame({
    'user_id': [1, 2, 3],
    'email': ['user1@email.com', 'user2@email.com', 'user3@email.com'],
    'age': [25, 30, 35]
})

df_encrypted = encryptor.encrypt_dataframe(df, sensitive_columns=['email'])
df_encrypted.to_csv('data_encrypted.csv', index=False)

# Save key securely (separate from data!)
encryptor.save_key('encryption_key.key')

# Decrypt when needed
df_decrypted = encryptor.decrypt_dataframe(df_encrypted, sensitive_columns=['email'])
```

**In Transit (HTTPS):**
```python
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import ssl

# Always use HTTPS in production
app = FastAPI()

security = HTTPBearer()

@app.post("/predict")
async def predict(
    data: dict,
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    # Verify token
    if not verify_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid token")

    # Make prediction
    prediction = model.predict(data)
    return {"prediction": prediction}

# Run with HTTPS
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8443,
        ssl_keyfile="path/to/privkey.pem",
        ssl_certfile="path/to/fullchain.pem"
    )
```

### 2. Access Control

**Role-Based Access Control (RBAC):**
```python
from enum import Enum
from typing import List

class Role(Enum):
    DATA_SCIENTIST = "data_scientist"
    ML_ENGINEER = "ml_engineer"
    ADMIN = "admin"
    VIEWER = "viewer"

class Permission(Enum):
    READ_DATA = "read_data"
    WRITE_DATA = "write_data"
    TRAIN_MODEL = "train_model"
    DEPLOY_MODEL = "deploy_model"
    DELETE_MODEL = "delete_model"

# Define role permissions
ROLE_PERMISSIONS = {
    Role.VIEWER: [Permission.READ_DATA],
    Role.DATA_SCIENTIST: [
        Permission.READ_DATA,
        Permission.TRAIN_MODEL
    ],
    Role.ML_ENGINEER: [
        Permission.READ_DATA,
        Permission.TRAIN_MODEL,
        Permission.DEPLOY_MODEL
    ],
    Role.ADMIN: list(Permission)  # All permissions
}

class AccessControl:
    """RBAC for ML systems"""

    def __init__(self):
        self.user_roles = {}

    def assign_role(self, user_id: str, role: Role):
        """Assign role to user"""
        self.user_roles[user_id] = role

    def check_permission(self, user_id: str, permission: Permission) -> bool:
        """Check if user has permission"""
        if user_id not in self.user_roles:
            return False

        role = self.user_roles[user_id]
        return permission in ROLE_PERMISSIONS[role]

    def require_permission(self, user_id: str, permission: Permission):
        """Raise exception if user lacks permission"""
        if not self.check_permission(user_id, permission):
            raise PermissionError(
                f"User {user_id} lacks permission: {permission.value}"
            )

# Usage
ac = AccessControl()

# Assign roles
ac.assign_role("alice", Role.DATA_SCIENTIST)
ac.assign_role("bob", Role.ML_ENGINEER)
ac.assign_role("charlie", Role.VIEWER)

# Check permissions
def deploy_model(user_id: str, model):
    ac.require_permission(user_id, Permission.DEPLOY_MODEL)
    # Deploy model
    print(f"{user_id} deployed model")

try:
    deploy_model("alice", model)  # Should fail
except PermissionError as e:
    print(e)

deploy_model("bob", model)  # Should succeed
```

### 3. Data Anonymization

**Remove PII:**
```python
import re
import hashlib

class PIIRemover:
    """Remove or mask PII from data"""

    # Regex patterns for PII
    PATTERNS = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        'credit_card': r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b'
    }

    def detect_pii(self, text: str) -> dict:
        """Detect PII in text"""
        detected = {}
        for pii_type, pattern in self.PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                detected[pii_type] = matches
        return detected

    def mask_pii(self, text: str) -> str:
        """Mask PII in text"""
        masked = text

        # Email: show first 2 chars + domain
        masked = re.sub(
            self.PATTERNS['email'],
            lambda m: m.group(0)[:2] + '***@' + m.group(0).split('@')[1],
            masked
        )

        # Phone: show last 4 digits
        masked = re.sub(
            self.PATTERNS['phone'],
            lambda m: '***-***-' + m.group(0)[-4:],
            masked
        )

        # SSN: show last 4
        masked = re.sub(
            self.PATTERNS['ssn'],
            lambda m: '***-**-' + m.group(0)[-4:],
            masked
        )

        # Credit card: show last 4
        masked = re.sub(
            self.PATTERNS['credit_card'],
            lambda m: '**** **** **** ' + m.group(0)[-4:],
            masked
        )

        return masked

    def hash_identifier(self, identifier: str, salt: str = '') -> str:
        """Hash identifier (irreversible)"""
        combined = f"{identifier}{salt}".encode()
        return hashlib.sha256(combined).hexdigest()

# Usage
pii_remover = PIIRemover()

text = "Contact me at john@example.com or 555-123-4567"

# Detect
detected = pii_remover.detect_pii(text)
print(f"Detected PII: {detected}")

# Mask
masked = pii_remover.mask_pii(text)
print(f"Masked: {masked}")

# Hash IDs
user_id = "user_12345"
hashed = pii_remover.hash_identifier(user_id, salt="my_secret_salt")
print(f"Hashed ID: {hashed}")
```

---

## Model Security

### 1. Defending Against Adversarial Examples

**Adversarial Training:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdversarialTrainer:
    """Train model robust to adversarial examples"""

    def __init__(self, model, epsilon=0.1):
        self.model = model
        self.epsilon = epsilon

    def fgsm_attack(self, images, labels, criterion):
        """
        Fast Gradient Sign Method (FGSM) attack

        Args:
            images: Input images
            labels: True labels
            criterion: Loss function

        Returns:
            Adversarial examples
        """
        images.requires_grad = True

        # Forward pass
        outputs = self.model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        self.model.zero_grad()
        loss.backward()

        # Create adversarial example
        grad_sign = images.grad.sign()
        adversarial_images = images + self.epsilon * grad_sign
        adversarial_images = torch.clamp(adversarial_images, 0, 1)

        return adversarial_images.detach()

    def adversarial_train_step(self, images, labels, optimizer, criterion):
        """
        Training step with adversarial examples

        Train on mix of clean and adversarial examples
        """
        self.model.train()

        # Clean examples
        optimizer.zero_grad()
        outputs_clean = self.model(images)
        loss_clean = criterion(outputs_clean, labels)

        # Generate adversarial examples
        adv_images = self.fgsm_attack(images, labels, criterion)

        # Adversarial examples
        outputs_adv = self.model(adv_images)
        loss_adv = criterion(outputs_adv, labels)

        # Combined loss
        loss = 0.5 * loss_clean + 0.5 * loss_adv

        loss.backward()
        optimizer.step()

        return loss.item()

# Usage
model = MyModel()
adversarial_trainer = AdversarialTrainer(model, epsilon=0.1)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(10):
    for images, labels in train_loader:
        loss = adversarial_trainer.adversarial_train_step(
            images, labels, optimizer, criterion
        )

    print(f"Epoch {epoch}, Loss: {loss:.4f}")
```

**Input Validation:**
```python
class InputValidator:
    """Validate inputs to detect adversarial examples"""

    def __init__(self, model, train_data):
        self.model = model
        # Compute statistics on clean data
        self.mean_activation = self._compute_mean_activation(train_data)
        self.std_activation = self._compute_std_activation(train_data)

    def _compute_mean_activation(self, data):
        """Compute mean activation on clean data"""
        activations = []
        self.model.eval()
        with torch.no_grad():
            for x, _ in data:
                act = self.model.get_activations(x)
                activations.append(act)
        return torch.mean(torch.cat(activations), dim=0)

    def _compute_std_activation(self, data):
        """Compute std activation on clean data"""
        activations = []
        self.model.eval()
        with torch.no_grad():
            for x, _ in data:
                act = self.model.get_activations(x)
                activations.append(act)
        return torch.std(torch.cat(activations), dim=0)

    def is_adversarial(self, x, threshold=3.0):
        """
        Detect adversarial example by checking if activations are anomalous

        Args:
            x: Input
            threshold: Number of std deviations

        Returns:
            True if likely adversarial
        """
        self.model.eval()
        with torch.no_grad():
            activation = self.model.get_activations(x)

        # Z-score
        z_score = torch.abs((activation - self.mean_activation) / self.std_activation)

        # Flag if any activation is > threshold std away
        return torch.any(z_score > threshold).item()

# Usage
validator = InputValidator(model, train_loader)

# Check input
if validator.is_adversarial(suspicious_input):
    print("WARNING: Potential adversarial example detected!")
    # Reject or flag for review
```

### 2. Model Watermarking

Embed watermark to prove model ownership:

```python
class ModelWatermark:
    """Embed watermark in model to prove ownership"""

    def __init__(self, model, watermark_data, watermark_labels):
        """
        Args:
            model: Model to watermark
            watermark_data: Special inputs (trigger set)
            watermark_labels: Expected outputs for trigger set
        """
        self.model = model
        self.watermark_data = watermark_data
        self.watermark_labels = watermark_labels

    def embed_watermark(self, train_loader, optimizer, criterion, epochs=10):
        """
        Embed watermark during training

        Mix regular training data with watermark data
        """
        self.model.train()

        for epoch in range(epochs):
            # Regular training
            for data, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Watermark training (ensure model memorizes watermark)
            for _ in range(10):  # Train more on watermark
                optimizer.zero_grad()
                outputs = self.model(self.watermark_data)
                loss = criterion(outputs, self.watermark_labels)
                loss.backward()
                optimizer.step()

    def verify_watermark(self, suspected_model, threshold=0.9):
        """
        Verify if model contains watermark

        Args:
            suspected_model: Model to check
            threshold: Accuracy threshold on watermark data

        Returns:
            True if watermark detected
        """
        suspected_model.eval()
        correct = 0

        with torch.no_grad():
            outputs = suspected_model(self.watermark_data)
            pred = outputs.argmax(dim=1)
            correct = (pred == self.watermark_labels).sum().item()

        accuracy = correct / len(self.watermark_data)

        return accuracy >= threshold

# Usage
# Create watermark (100 special examples)
watermark_data = torch.randn(100, 3, 32, 32)
watermark_labels = torch.randint(0, 10, (100,))

watermarker = ModelWatermark(model, watermark_data, watermark_labels)

# Embed watermark during training
watermarker.embed_watermark(train_loader, optimizer, criterion)

# Later: verify ownership
if watermarker.verify_watermark(suspected_stolen_model):
    print("Watermark detected! This model was stolen from us.")
```

### 3. Rate Limiting (Prevent Model Stealing)

```python
from collections import defaultdict
from datetime import datetime, timedelta
import time

class RateLimiter:
    """Rate limit API requests to prevent model stealing"""

    def __init__(self, max_requests_per_minute=100, max_requests_per_day=10000):
        self.max_rpm = max_requests_per_minute
        self.max_rpd = max_requests_per_day

        self.request_times = defaultdict(list)  # user_id -> [timestamps]

    def is_allowed(self, user_id: str) -> bool:
        """Check if user is allowed to make request"""
        now = datetime.now()
        user_requests = self.request_times[user_id]

        # Remove old requests (> 24 hours)
        user_requests = [
            t for t in user_requests
            if now - t < timedelta(days=1)
        ]
        self.request_times[user_id] = user_requests

        # Check daily limit
        if len(user_requests) >= self.max_rpd:
            return False

        # Check per-minute limit
        recent_requests = [
            t for t in user_requests
            if now - t < timedelta(minutes=1)
        ]

        if len(recent_requests) >= self.max_rpm:
            return False

        # Allow request
        self.request_times[user_id].append(now)
        return True

    def get_wait_time(self, user_id: str) -> int:
        """Get seconds user must wait before next request"""
        now = datetime.now()
        user_requests = self.request_times[user_id]

        if len(user_requests) == 0:
            return 0

        # Check if daily limit exceeded
        daily_requests = [
            t for t in user_requests
            if now - t < timedelta(days=1)
        ]

        if len(daily_requests) >= self.max_rpd:
            oldest = min(daily_requests)
            return int((oldest + timedelta(days=1) - now).total_seconds())

        # Check if per-minute limit exceeded
        recent = [
            t for t in user_requests
            if now - t < timedelta(minutes=1)
        ]

        if len(recent) >= self.max_rpm:
            oldest_recent = min(recent)
            return int((oldest_recent + timedelta(minutes=1) - now).total_seconds())

        return 0

# Usage in API
rate_limiter = RateLimiter(max_requests_per_minute=100, max_requests_per_day=10000)

@app.post("/predict")
async def predict(user_id: str, data: dict):
    # Check rate limit
    if not rate_limiter.is_allowed(user_id):
        wait_time = rate_limiter.get_wait_time(user_id)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Wait {wait_time} seconds."
        )

    # Make prediction
    prediction = model.predict(data)
    return {"prediction": prediction}
```

---

## Privacy-Preserving ML

### 1. Differential Privacy

Add noise to protect individual privacy:

```python
import numpy as np

class DifferentialPrivacy:
    """Implement differential privacy for ML"""

    def __init__(self, epsilon=1.0):
        """
        Args:
            epsilon: Privacy budget (smaller = more private)
                    ε=0.1: Strong privacy
                    ε=1.0: Moderate privacy
                    ε=10.0: Weak privacy
        """
        self.epsilon = epsilon

    def laplace_mechanism(self, true_value, sensitivity):
        """
        Add Laplace noise for differential privacy

        Args:
            true_value: Original value
            sensitivity: How much one person can change result

        Returns:
            Noisy value
        """
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        return true_value + noise

    def private_mean(self, data, data_range):
        """
        Compute differentially private mean

        Args:
            data: Array of values
            data_range: (min, max) of possible values

        Returns:
            Private mean
        """
        true_mean = np.mean(data)

        # Sensitivity of mean
        sensitivity = (data_range[1] - data_range[0]) / len(data)

        # Add noise
        private_mean = self.laplace_mechanism(true_mean, sensitivity)

        return private_mean

    def private_histogram(self, data, bins):
        """
        Compute differentially private histogram

        Args:
            data: Array of values
            bins: Histogram bins

        Returns:
            Private histogram
        """
        # True histogram
        hist, _ = np.histogram(data, bins=bins)

        # Add noise to each bin (sensitivity = 1 per bin)
        private_hist = [
            max(0, self.laplace_mechanism(count, sensitivity=1))
            for count in hist
        ]

        return private_hist

# Usage
dp = DifferentialPrivacy(epsilon=1.0)

# Salaries (sensitive data)
salaries = np.array([50000, 60000, 55000, 70000, 65000])

# Compute private mean
true_mean = np.mean(salaries)
private_mean = dp.private_mean(salaries, data_range=(0, 200000))

print(f"True mean: ${true_mean:,.0f}")
print(f"Private mean: ${private_mean:,.0f}")
print(f"Error: ${abs(true_mean - private_mean):,.0f}")

# Private histogram
private_hist = dp.private_histogram(
    salaries,
    bins=[0, 50000, 60000, 70000, 100000]
)
print(f"Private histogram: {private_hist}")
```

**Differential Privacy in SGD:**
```python
class DPSGDOptimizer:
    """Differentially Private Stochastic Gradient Descent"""

    def __init__(self, model, epsilon, delta, max_grad_norm=1.0, noise_multiplier=1.0):
        """
        Args:
            epsilon: Privacy budget
            delta: Privacy parameter (typically 1/n)
            max_grad_norm: Gradient clipping threshold
            noise_multiplier: Scale of noise
        """
        self.model = model
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier

    def clip_gradients(self, gradients):
        """Clip gradients to bound sensitivity"""
        total_norm = torch.norm(
            torch.stack([torch.norm(g) for g in gradients])
        )

        clip_coef = self.max_grad_norm / (total_norm + 1e-6)

        if clip_coef < 1:
            clipped = [g * clip_coef for g in gradients]
        else:
            clipped = gradients

        return clipped

    def add_noise(self, gradients):
        """Add Gaussian noise for privacy"""
        noisy = []
        for g in gradients:
            noise = torch.normal(
                0,
                self.noise_multiplier * self.max_grad_norm,
                size=g.shape
            )
            noisy.append(g + noise)

        return noisy

    def dp_step(self, loss, optimizer):
        """Perform DP-SGD update"""
        # Compute gradients
        optimizer.zero_grad()
        loss.backward()

        # Get gradients
        gradients = [p.grad for p in self.model.parameters() if p.grad is not None]

        # Clip gradients (per-sample, but approximated here)
        clipped = self.clip_gradients(gradients)

        # Add noise
        noisy = self.add_noise(clipped)

        # Set noisy gradients
        for p, noisy_grad in zip(self.model.parameters(), noisy):
            p.grad = noisy_grad

        # Update
        optimizer.step()

# Usage (requires careful privacy accounting!)
model = MyModel()
dp_optimizer = DPSGDOptimizer(
    model,
    epsilon=1.0,
    delta=1e-5,
    max_grad_norm=1.0,
    noise_multiplier=1.1
)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10):
    for data, labels in train_loader:
        outputs = model(data)
        loss = criterion(outputs, labels)
        dp_optimizer.dp_step(loss, optimizer)
```

### 2. Federated Learning

Train on decentralized data without sharing:

```python
class FederatedLearning:
    """Federated learning coordinator"""

    def __init__(self, global_model):
        self.global_model = global_model

    def aggregate_models(self, client_models, client_weights):
        """
        Federated Averaging (FedAvg)

        Args:
            client_models: List of trained client models
            client_weights: List of client dataset sizes

        Returns:
            Aggregated global model
        """
        # Normalize weights
        total_weight = sum(client_weights)
        weights = [w / total_weight for w in client_weights]

        # Average model parameters
        global_dict = self.global_model.state_dict()

        for key in global_dict.keys():
            # Weighted average of client parameters
            global_dict[key] = sum(
                w * client_models[i].state_dict()[key]
                for i, w in enumerate(weights)
            )

        self.global_model.load_state_dict(global_dict)

        return self.global_model

    def train_round(self, clients, num_epochs=5):
        """
        One round of federated training

        Args:
            clients: List of (data_loader, model) tuples
            num_epochs: Local training epochs

        Returns:
            Updated global model
        """
        client_models = []
        client_weights = []

        for client_data, client_model in clients:
            # Copy global model to client
            client_model.load_state_dict(self.global_model.state_dict())

            # Train locally
            optimizer = torch.optim.SGD(client_model.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()

            client_model.train()
            for epoch in range(num_epochs):
                for data, labels in client_data:
                    optimizer.zero_grad()
                    outputs = client_model(data)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

            # Collect trained model and weight
            client_models.append(client_model)
            client_weights.append(len(client_data.dataset))

        # Aggregate
        self.global_model = self.aggregate_models(client_models, client_weights)

        return self.global_model

# Usage
global_model = MyModel()
fl = FederatedLearning(global_model)

# Simulate 5 clients with local data
clients = [
    (client1_data, MyModel()),
    (client2_data, MyModel()),
    (client3_data, MyModel()),
    (client4_data, MyModel()),
    (client5_data, MyModel())
]

# Train for 10 rounds
for round in range(10):
    global_model = fl.train_round(clients, num_epochs=5)
    print(f"Round {round} complete")

# Global model now trained on all client data without sharing data!
```

### 3. Secure Multi-Party Computation (SMPC)

Compute on encrypted data:

```python
# Simplified example using PySyft (install: pip install syft)
import syft as sy
import torch

# Setup
hook = sy.TorchHook(torch)
alice = sy.VirtualWorker(hook, id="alice")
bob = sy.VirtualWorker(hook, id="bob")

# Alice and Bob have private data
alice_data = torch.tensor([1.0, 2.0, 3.0]).send(alice)
bob_data = torch.tensor([4.0, 5.0, 6.0]).send(bob)

# Compute on encrypted data (shared between alice and bob)
alice_data_encrypted = alice_data.fix_precision().share(alice, bob)
bob_data_encrypted = bob_data.fix_precision().share(alice, bob)

# Perform operations on encrypted data
result_encrypted = alice_data_encrypted + bob_data_encrypted

# Decrypt result
result = result_encrypted.get().float_precision()

print(f"Result: {result}")  # [5, 7, 9]
# Neither Alice nor Bob saw the other's data!
```

---

## Compliance & Regulations

### 1. GDPR Compliance

**Key Requirements:**
- **Right to be forgotten**: Delete user data on request
- **Data portability**: Export user data
- **Consent**: Get explicit consent for data use
- **Purpose limitation**: Use data only for stated purpose
- **Data minimization**: Collect only necessary data

**Implementation:**
```python
class GDPRCompliance:
    """GDPR compliance utilities"""

    def __init__(self, database):
        self.db = database

    def right_to_access(self, user_id):
        """Export all data for user (GDPR Article 15)"""
        user_data = {
            'personal_info': self.db.get_user_info(user_id),
            'transactions': self.db.get_transactions(user_id),
            'predictions': self.db.get_predictions(user_id),
            'consent_records': self.db.get_consent(user_id)
        }

        # Export as JSON
        import json
        return json.dumps(user_data, indent=2)

    def right_to_erasure(self, user_id):
        """Delete all data for user (GDPR Article 17)"""
        # Delete from all tables
        self.db.delete_user_info(user_id)
        self.db.delete_transactions(user_id)
        self.db.delete_predictions(user_id)
        self.db.anonymize_logs(user_id)

        # Retrain model without user's data (if possible)
        # Or mark model for retraining

        return {"status": "deleted", "user_id": user_id}

    def record_consent(self, user_id, purpose, consented):
        """Record user consent"""
        self.db.insert_consent({
            'user_id': user_id,
            'purpose': purpose,
            'consented': consented,
            'timestamp': datetime.now()
        })

    def check_consent(self, user_id, purpose):
        """Check if user consented for purpose"""
        consent_record = self.db.get_consent(user_id, purpose)
        return consent_record and consent_record['consented']

# Usage
gdpr = GDPRCompliance(database)

# User requests data export
user_data = gdpr.right_to_access(user_id="user_123")

# User requests deletion
gdpr.right_to_erasure(user_id="user_123")

# Before using data, check consent
if gdpr.check_consent(user_id, purpose="model_training"):
    # Use data for training
    pass
```

### 2. Model Cards

Document model details for transparency:

```python
MODEL_CARD = {
    "model_name": "Credit Risk Predictor",
    "version": "2.1.0",
    "date": "2025-10-27",

    "intended_use": {
        "primary": "Predict credit default risk for loan applications",
        "users": ["Credit analysts", "Automated lending system"],
        "out_of_scope": ["Employment decisions", "Insurance pricing"]
    },

    "training_data": {
        "dataset": "Internal credit bureau data",
        "size": "500,000 applications",
        "timeframe": "2020-2024",
        "demographics": {
            "age": "18-80",
            "geography": "United States",
            "protected_attributes": ["age", "race", "gender"]
        }
    },

    "model_details": {
        "architecture": "Gradient Boosting (XGBoost)",
        "features": 47,
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1
        }
    },

    "performance": {
        "metrics": {
            "auc": 0.82,
            "precision": 0.75,
            "recall": 0.70
        },
        "fairness": {
            "demographic_parity": {
                "gender": 0.05,  # <0.1 is good
                "race": 0.08
            }
        }
    },

    "ethical_considerations": {
        "fairness": "Model tested for demographic parity across protected groups",
        "explainability": "SHAP values provided for each prediction",
        "privacy": "Trained with differential privacy (ε=1.0)"
    },

    "limitations": {
        "known_issues": [
            "Lower accuracy for applicants <25 years old",
            "Requires minimum credit history of 1 year"
        ],
        "biases": [
            "May underperform for recent immigrants (limited credit history)"
        ]
    },

    "monitoring": {
        "metrics": ["AUC", "false positive rate", "demographic parity"],
        "frequency": "Weekly",
        "alert_threshold": "AUC drops below 0.75"
    }
}
```

---

## Best Practices

### Security Checklist

✅ **Data Security:**
- [ ] Encrypt data at rest (AES-256)
- [ ] Encrypt data in transit (TLS 1.3)
- [ ] Implement access control (RBAC)
- [ ] Remove/mask PII
- [ ] Secure key management

✅ **Model Security:**
- [ ] Adversarial training
- [ ] Input validation
- [ ] Rate limiting (prevent model stealing)
- [ ] Model watermarking
- [ ] Monitor for unusual queries

✅ **Privacy:**
- [ ] Differential privacy (if applicable)
- [ ] Federated learning (if applicable)
- [ ] Data minimization
- [ ] Anonymization
- [ ] Privacy policy

✅ **Compliance:**
- [ ] GDPR compliance (EU)
- [ ] CCPA compliance (California)
- [ ] HIPAA compliance (healthcare)
- [ ] Model card documentation
- [ ] Bias testing
- [ ] Regular audits

✅ **Deployment:**
- [ ] HTTPS only
- [ ] API authentication
- [ ] Request logging
- [ ] Anomaly detection
- [ ] Incident response plan

---

## Interview Questions

### Q1: How would you defend an ML model against adversarial examples?

**Answer:**

Multiple defense layers:

**1. Adversarial Training (Most Effective):**
```python
# Train on mix of clean and adversarial examples
for epoch in range(epochs):
    for images, labels in dataloader:
        # Clean examples
        loss_clean = train_step(images, labels)

        # Generate adversarial examples (FGSM/PGD)
        adv_images = generate_adversarial(images, labels)

        # Adversarial examples
        loss_adv = train_step(adv_images, labels)

        # Combined loss
        loss = 0.5 * loss_clean + 0.5 * loss_adv
        loss.backward()
```

**Benefits**: Robust to wide range of attacks
**Drawbacks**: Slower training, may reduce clean accuracy

**2. Input Preprocessing:**
- **Image compression**: JPEG compression removes adversarial noise
- **Random resizing**: Disrupt pixel-level perturbations
- **Bit depth reduction**: Quantize input values

**3. Defensive Distillation:**
Train student model on soft outputs of teacher:
```python
# Teacher outputs probabilities (soft targets)
soft_targets = F.softmax(teacher(x) / temperature, dim=1)

# Student trained on soft targets
loss = KL(student(x), soft_targets)
```

Makes model smoother, harder to fool.

**4. Anomaly Detection:**
```python
# Detect if input is out-of-distribution
if is_anomalous(input, threshold=3.0):
    reject_input()
```

**5. Ensemble Methods:**
Multiple models harder to fool than single model.

**6. Certified Defenses:**
Mathematical guarantees of robustness (e.g., randomized smoothing).

**Best Practice: Combine multiple defenses (defense in depth)**

---

### Q2: What is differential privacy and when would you use it?

**Answer:**

**Definition:**
Differential privacy ensures that removing any single person's data doesn't significantly change the output.

**Mathematical Guarantee:**
```
Pr[M(D) ∈ S] ≤ e^ε × Pr[M(D') ∈ S]
```
where D and D' differ by one person, ε is privacy budget.

**Mechanism:**
Add calibrated noise to results:
```python
# True mean
true_mean = np.mean(salaries)

# Sensitivity (max one person can change mean)
sensitivity = (max_salary - min_salary) / n

# Add Laplace noise
noise = np.random.laplace(0, sensitivity / epsilon)
private_mean = true_mean + noise
```

**When to Use:**

✅ **Good for:**
- **Public statistics**: Census data, aggregate analytics
- **Research**: Medical research with patient data
- **Model training**: Protect training examples (DP-SGD)
- **APIs**: Release aggregate data without privacy risk

❌ **Not good for:**
- **Individual predictions**: Adds too much noise
- **Small datasets**: Noise overwhelms signal
- **Real-time systems**: Computational overhead
- **When privacy not required**: Adds complexity

**Example:**
```python
# Hospital wants to publish average patient age
# Without exposing any individual

ages = [25, 30, 35, 40, 45, ...]  # 10,000 patients

# With ε=1.0 differential privacy
dp = DifferentialPrivacy(epsilon=1.0)
private_avg = dp.private_mean(ages, data_range=(0, 100))

# Now safe to publish private_avg
# No one can tell if any individual was included
```

**Trade-off:**
- **More privacy (smaller ε)** → More noise → Less accurate
- **Less privacy (larger ε)** → Less noise → More accurate

**Typical values:**
- ε = 0.1: Strong privacy (very noisy)
- ε = 1.0: Moderate privacy (recommended)
- ε = 10: Weak privacy (less noisy)

**Production Use:**
- Google Chrome (telemetry)
- Apple (keyboard suggestions)
- U.S. Census 2020

---

### Q3: How do you ensure GDPR compliance for an ML system?

**Answer:**

**Key GDPR Requirements:**

**1. Right to Access (Article 15):**
Users can request all their data.

**Implementation:**
```python
def export_user_data(user_id):
    """Export all user data as JSON"""
    return {
        'personal_info': get_user_profile(user_id),
        'training_data': get_training_samples(user_id),
        'predictions': get_prediction_history(user_id),
        'model_versions_used': get_model_versions(user_id)
    }
```

**2. Right to Erasure (Article 17):**
Users can request deletion ("right to be forgotten").

**Implementation:**
```python
def delete_user_data(user_id):
    # Delete from all systems
    delete_from_database(user_id)
    delete_from_logs(user_id)  # Or anonymize
    delete_from_backups(user_id)  # Or mark deleted

    # Handle ML models
    if user_data_in_training_set(user_id):
        # Option 1: Retrain without user's data
        retrain_model(exclude=[user_id])

        # Option 2: Mark model as needing retrain
        mark_for_retrain(reason=f"GDPR deletion: {user_id}")

        # Option 3: Use machine unlearning (if available)
        unlearn_user(model, user_id)
```

**3. Data Minimization (Article 5):**
Collect only necessary data.

**Before:**
```python
# Bad: Collecting everything
user_data = {
    'name', 'email', 'address', 'ssn', 'phone',
    'browsing_history', 'location_history', ...
}
```

**After:**
```python
# Good: Only what's needed for recommendation
user_data = {
    'user_id',  # Anonymous ID
    'purchase_history',  # Needed for recommendations
    'preferences'  # Needed for personalization
}
```

**4. Purpose Limitation (Article 5):**
Use data only for stated purpose.

**Implementation:**
```python
# Record consent for each purpose
record_consent(user_id, purpose='recommendations', consented=True)
record_consent(user_id, purpose='marketing', consented=False)

# Before using data, check consent
if check_consent(user_id, purpose='recommendations'):
    use_for_recommendations(user_id)

# Don't use for other purposes
if not check_consent(user_id, purpose='marketing'):
    dont_send_marketing(user_id)
```

**5. Lawful Basis (Article 6):**
Have legal basis for processing:
- Consent
- Contract
- Legal obligation
- Legitimate interest

**6. Data Protection Impact Assessment (DPIA):**
For high-risk processing, conduct assessment:
```markdown
## DPIA for Recommendation System

**Purpose:** Personalize product recommendations
**Data:** User purchases, clicks, ratings
**Risks:**
  - Profiling: Creating detailed user profiles
  - Inference: Inferring sensitive attributes

**Mitigations:**
  - Data minimization: Only use necessary data
  - Anonymization: Remove direct identifiers
  - Encryption: Encrypt data at rest/transit
  - Access control: Limit who can access
  - Regular audits: Check for misuse
```

**7. Model Cards & Documentation:**
Document how models work:
- What data used
- How decisions made
- Performance by demographic
- Known biases

**Best Practices:**
- ✅ Privacy by design (build in from start)
- ✅ Document everything (consent, processing, purposes)
- ✅ Regular audits (check compliance)
- ✅ Appoint DPO (Data Protection Officer)
- ✅ Incident response plan (for breaches)
- ✅ Training for team (on GDPR requirements)

**Penalties:** Up to €20M or 4% of global revenue (whichever is higher)!

---

## Conclusion

ML security and privacy are critical for production systems. Key principles:

1. **Defense in Depth**: Multiple security layers
2. **Least Privilege**: Minimal access rights
3. **Privacy by Design**: Build privacy in from start
4. **Continuous Monitoring**: Detect threats early
5. **Compliance First**: Follow regulations (GDPR, CCPA, HIPAA)

**Security Priorities:**
1. Data encryption (at rest and in transit)
2. Access control (RBAC)
3. Input validation (prevent adversarial examples)
4. Rate limiting (prevent model stealing)
5. Monitoring (detect anomalies)

**Privacy Priorities:**
1. Data minimization (collect only what's needed)
2. Anonymization (remove PII)
3. Differential privacy (for statistics)
4. Federated learning (for distributed data)
5. User rights (access, deletion, portability)

**Remember:** Security and privacy are not optional—they're essential for user trust and legal compliance.

---

*Last Updated: October 27, 2025*
*Version: 1.0*
*Maintained by: ML Engineering Team*
