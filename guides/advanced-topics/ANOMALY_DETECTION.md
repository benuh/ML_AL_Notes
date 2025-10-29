# Anomaly Detection: Complete Guide

## Table of Contents
1. [Introduction to Anomaly Detection](#introduction)
2. [Statistical Methods](#statistical-methods)
3. [Machine Learning Approaches](#machine-learning-approaches)
4. [Deep Learning Methods](#deep-learning-methods)
5. [Time Series Anomaly Detection](#time-series-anomaly-detection)
6. [Multivariate Anomaly Detection](#multivariate-anomaly-detection)
7. [Production Systems](#production-systems)
8. [Evaluation Metrics](#evaluation-metrics)

---

## Introduction to Anomaly Detection

Anomaly detection identifies unusual patterns that deviate from expected behavior.

### Types of Anomalies

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F

class AnomalyTypes:
    """Demonstrate different types of anomalies"""

    def __init__(self):
        self.types = {
            'point': 'Individual data points that are anomalous',
            'contextual': 'Normal in one context, anomalous in another',
            'collective': 'Collection of points that together form anomaly'
        }

    def generate_examples(self, n_samples=1000):
        """Generate synthetic data with different anomaly types"""

        # Normal data
        normal_data = np.random.normal(0, 1, n_samples)

        # Point anomalies: extreme values
        point_anomalies = np.random.choice(n_samples, 10, replace=False)
        normal_data[point_anomalies] = np.random.normal(5, 1, 10)

        # Contextual anomaly: value normal overall but unusual at specific time
        # E.g., high temperature in winter
        time = np.arange(n_samples)
        seasonal = 10 * np.sin(2 * np.pi * time / 365)
        contextual_data = normal_data + seasonal
        contextual_anomaly = n_samples // 2
        contextual_data[contextual_anomaly] = -5  # Cold in summer

        # Collective anomaly: subsequence that's unusual
        collective_start = n_samples // 3
        collective_end = collective_start + 50
        collective_data = normal_data.copy()
        collective_data[collective_start:collective_end] = np.random.normal(3, 0.5, 50)

        return {
            'point': normal_data,
            'contextual': contextual_data,
            'collective': collective_data,
            'point_indices': point_anomalies,
            'contextual_index': contextual_anomaly,
            'collective_range': (collective_start, collective_end)
        }

    def visualize(self):
        """Visualize different anomaly types"""
        examples = self.generate_examples()

        fig, axes = plt.subplots(3, 1, figsize=(12, 8))

        # Point anomalies
        axes[0].plot(examples['point'], alpha=0.7)
        axes[0].scatter(examples['point_indices'],
                       examples['point'][examples['point_indices']],
                       color='red', s=100, label='Anomalies')
        axes[0].set_title('Point Anomalies')
        axes[0].legend()

        # Contextual anomalies
        axes[1].plot(examples['contextual'], alpha=0.7)
        axes[1].scatter(examples['contextual_index'],
                       examples['contextual'][examples['contextual_index']],
                       color='red', s=100, label='Anomaly')
        axes[1].set_title('Contextual Anomaly')
        axes[1].legend()

        # Collective anomalies
        axes[2].plot(examples['collective'], alpha=0.7)
        start, end = examples['collective_range']
        axes[2].axvspan(start, end, alpha=0.3, color='red', label='Anomaly')
        axes[2].set_title('Collective Anomaly')
        axes[2].legend()

        plt.tight_layout()
        # plt.show()

# Demo
demo = AnomalyTypes()
print("Anomaly Types:")
for type_name, description in demo.types.items():
    print(f"- {type_name}: {description}")
```

---

## Statistical Methods

Classical statistical approaches for anomaly detection.

### Z-Score Method

**Statistical Foundation:**

**Z-Score (Standard Score):**
```
z = (x - μ) / σ

where:
- x: Observed value
- μ: Population mean (estimated from data)
- σ: Population standard deviation (estimated from data)
- z: Number of standard deviations from mean

Under normal distribution N(μ, σ²):
- |z| > 2: ~95% confidence (5% false positive rate)
- |z| > 3: ~99.7% confidence (0.3% false positive rate)
- |z| > 4: ~99.994% confidence (0.006% false positive rate)

Probability of observing |z| > k in standard normal:
P(|Z| > k) = 2 · (1 - Φ(k))

where Φ is the cumulative distribution function (CDF) of N(0,1)
```

**Assumptions:**
1. Data follows normal distribution (Gaussian)
2. Independence of observations
3. Mean and variance are well-defined and stable

**Limitations:**
- ⚠️ Sensitive to outliers (mean and std are not robust)
- ⚠️ Assumes normality (fails for skewed distributions)
- ⚠️ Fixed threshold may not work for all contexts
- ⚠️ Not suitable for multimodal distributions

**Solution for Non-Normal Data:**
Use Modified Z-Score with Median Absolute Deviation (MAD):
```
Modified Z-Score: M_i = 0.6745 · (x_i - x̃) / MAD

where:
- x̃: Median (robust to outliers)
- MAD = median(|x_i - x̃|)
- 0.6745: Constant to make MAD consistent with σ for normal distribution

Threshold: |M_i| > 3.5 (recommended by Iglewicz & Hoaglin, 1993)
```

```python
class ZScoreDetector:
    """Z-score based anomaly detection with robust variant"""

    def __init__(self, threshold=3.0, method='standard'):
        """
        Args:
            threshold: Number of standard deviations for anomaly
                      - standard: typically 2-3
                      - modified: typically 3.5
            method: 'standard' or 'modified' (robust to outliers)
        """
        self.threshold = threshold
        self.method = method
        self.mean = None
        self.std = None
        self.median = None
        self.mad = None

    def fit(self, data):
        """Fit the detector on normal data"""
        if self.method == 'standard':
            self.mean = np.mean(data)
            self.std = np.std(data, ddof=1)  # Sample std (unbiased)
        elif self.method == 'modified':
            self.median = np.median(data)
            self.mad = np.median(np.abs(data - self.median))
        else:
            raise ValueError("method must be 'standard' or 'modified'")

    def predict(self, data):
        """
        Predict anomalies

        Returns:
            Binary array (1 = anomaly, 0 = normal)
        """
        scores = self.anomaly_scores(data)
        anomalies = (scores > self.threshold).astype(int)
        return anomalies

    def anomaly_scores(self, data):
        """Return anomaly scores (z-scores or modified z-scores)"""
        if self.method == 'standard':
            # Standard z-score
            if self.std == 0:
                return np.zeros_like(data)  # No variance, no anomalies
            return np.abs((data - self.mean) / self.std)
        else:
            # Modified z-score (robust)
            if self.mad == 0:
                return np.zeros_like(data)
            return 0.6745 * np.abs((data - self.median) / self.mad)

    def p_values(self, data):
        """
        Return p-values for each data point

        P-value: Probability of observing value this extreme under null hypothesis
        Small p-value (< 0.05) suggests anomaly
        """
        from scipy import stats

        if self.method == 'standard':
            z_scores = (data - self.mean) / self.std
            # Two-tailed test
            p_vals = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
            return p_vals
        else:
            # For modified z-score, approximate using standard normal
            m_scores = self.anomaly_scores(data)
            # Convert modified z-score to approximate p-value
            z_equiv = m_scores / 0.6745
            p_vals = 2 * (1 - stats.norm.cdf(z_equiv))
            return p_vals

# Example usage
data = np.random.normal(0, 1, 1000)
data[100] = 5  # Anomaly

# Standard Z-Score
detector_std = ZScoreDetector(threshold=3.0, method='standard')
detector_std.fit(data)
anomalies_std = detector_std.predict(data)
print(f"Standard Z-Score: {anomalies_std.sum()} anomalies detected")

# Modified Z-Score (robust)
detector_mod = ZScoreDetector(threshold=3.5, method='modified')
detector_mod.fit(data)
anomalies_mod = detector_mod.predict(data)
print(f"Modified Z-Score: {anomalies_mod.sum()} anomalies detected")

# P-values
p_vals = detector_std.p_values(data)
print(f"Anomalies with p < 0.001: {(p_vals < 0.001).sum()}")
```

**Computational Complexity:**
```
Time Complexity:
- fit(): O(n) for mean/std or O(n log n) for median/MAD
- predict(): O(n) for computing scores
- p_values(): O(n) for CDF lookups

Space Complexity: O(1) (only store mean/std or median/MAD)

where n = number of samples
```
data[500] = -4.5  # Anomaly

detector = ZScoreDetector(threshold=3.0)
detector.fit(data)
anomalies = detector.predict(data)

print(f"Detected {anomalies.sum()} anomalies")
print(f"Anomaly indices: {np.where(anomalies == 1)[0]}")
```

### Modified Z-Score (Robust)

```python
class RobustZScoreDetector:
    """Modified Z-score using median and MAD"""

    def __init__(self, threshold=3.5):
        self.threshold = threshold
        self.median = None
        self.mad = None  # Median Absolute Deviation

    def fit(self, data):
        """Fit using median and MAD"""
        self.median = np.median(data)
        mad = np.median(np.abs(data - self.median))
        self.mad = mad if mad != 0 else 1.0  # Avoid division by zero

    def predict(self, data):
        """Predict anomalies using modified z-score"""
        # Modified z-score: 0.6745 * (x - median) / MAD
        modified_z_scores = np.abs(0.6745 * (data - self.median) / self.mad)
        anomalies = (modified_z_scores > self.threshold).astype(int)
        return anomalies
```

### Interquartile Range (IQR)

```python
class IQRDetector:
    """IQR-based anomaly detection"""

    def __init__(self, k=1.5):
        """
        Args:
            k: IQR multiplier (1.5 for outliers, 3.0 for extreme outliers)
        """
        self.k = k
        self.q1 = None
        self.q3 = None
        self.iqr = None

    def fit(self, data):
        """Fit IQR bounds"""
        self.q1 = np.percentile(data, 25)
        self.q3 = np.percentile(data, 75)
        self.iqr = self.q3 - self.q1

        self.lower_bound = self.q1 - self.k * self.iqr
        self.upper_bound = self.q3 + self.k * self.iqr

    def predict(self, data):
        """Predict anomalies"""
        anomalies = ((data < self.lower_bound) | (data > self.upper_bound)).astype(int)
        return anomalies
```

### Grubbs' Test

```python
from scipy import stats

class GrubbsTest:
    """Grubbs' test for outlier detection"""

    def __init__(self, alpha=0.05):
        """
        Args:
            alpha: Significance level
        """
        self.alpha = alpha

    def detect_outlier(self, data):
        """
        Detect one outlier using Grubbs' test

        Returns:
            Index of outlier if found, None otherwise
        """
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)

        # Calculate Grubbs' statistic for each point
        G_values = np.abs((data - mean) / std)
        max_idx = np.argmax(G_values)
        G_max = G_values[max_idx]

        # Critical value
        t_dist = stats.t.ppf(1 - self.alpha / (2 * n), n - 2)
        G_critical = ((n - 1) / np.sqrt(n)) * np.sqrt(t_dist**2 / (n - 2 + t_dist**2))

        if G_max > G_critical:
            return max_idx
        return None

    def detect_all_outliers(self, data):
        """Iteratively detect all outliers"""
        data_copy = data.copy()
        outlier_indices = []

        while True:
            outlier_idx = self.detect_outlier(data_copy)
            if outlier_idx is None:
                break

            # Map back to original indices
            original_idx = np.where(data == data_copy[outlier_idx])[0][0]
            outlier_indices.append(original_idx)

            # Remove outlier and continue
            data_copy = np.delete(data_copy, outlier_idx)

        return outlier_indices
```

---

## Machine Learning Approaches

Traditional ML methods for anomaly detection.

### Isolation Forest

```python
class SimpleIsolationTree:
    """Simple implementation of isolation tree"""

    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        self.split_feature = None
        self.split_value = None
        self.left = None
        self.right = None
        self.depth = 0
        self.size = 0

    def fit(self, X, depth=0):
        """Build isolation tree"""
        self.depth = depth
        self.size = len(X)

        if depth >= self.max_depth or len(X) <= 1:
            return

        # Random feature and split value
        n_features = X.shape[1]
        self.split_feature = np.random.randint(0, n_features)

        feature_values = X[:, self.split_feature]
        min_val, max_val = feature_values.min(), feature_values.max()

        if min_val == max_val:
            return

        self.split_value = np.random.uniform(min_val, max_val)

        # Split data
        left_mask = X[:, self.split_feature] < self.split_value
        right_mask = ~left_mask

        # Recursively build subtrees
        if left_mask.any():
            self.left = SimpleIsolationTree(self.max_depth)
            self.left.fit(X[left_mask], depth + 1)

        if right_mask.any():
            self.right = SimpleIsolationTree(self.max_depth)
            self.right.fit(X[right_mask], depth + 1)

    def path_length(self, x):
        """Compute path length for a point"""
        if self.split_feature is None:
            # Leaf node
            return self.depth + self._c(self.size)

        if x[self.split_feature] < self.split_value:
            if self.left is None:
                return self.depth + 1
            return self.left.path_length(x)
        else:
            if self.right is None:
                return self.depth + 1
            return self.right.path_length(x)

    def _c(self, n):
        """Average path length of unsuccessful search in BST"""
        if n <= 1:
            return 0
        return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n


class IsolationForest:
    """Isolation Forest for anomaly detection"""

    def __init__(self, n_estimators=100, max_samples=256, max_depth=10):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X):
        """Build isolation forest"""
        self.trees = []
        n_samples = len(X)

        for _ in range(self.n_estimators):
            # Sample data
            sample_size = min(self.max_samples, n_samples)
            sample_indices = np.random.choice(n_samples, sample_size, replace=False)
            X_sample = X[sample_indices]

            # Build tree
            tree = SimpleIsolationTree(max_depth=self.max_depth)
            tree.fit(X_sample)
            self.trees.append(tree)

    def anomaly_score(self, X):
        """Compute anomaly scores"""
        n_samples = len(X)
        scores = np.zeros(n_samples)

        for i, x in enumerate(X):
            # Average path length across all trees
            avg_path_length = np.mean([tree.path_length(x) for tree in self.trees])

            # Normalize
            c = self._c(self.max_samples)
            scores[i] = 2 ** (-avg_path_length / c)

        return scores

    def predict(self, X, contamination=0.1):
        """Predict anomalies"""
        scores = self.anomaly_score(X)
        threshold = np.percentile(scores, 100 * (1 - contamination))
        predictions = (scores >= threshold).astype(int)
        return predictions

    def _c(self, n):
        """Average path length of unsuccessful search"""
        if n <= 1:
            return 0
        return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n

# Example
X = np.random.randn(1000, 2)
X[0] = [5, 5]  # Anomaly
X[1] = [-5, -5]  # Anomaly

iforest = IsolationForest(n_estimators=100)
iforest.fit(X)
scores = iforest.anomaly_score(X)
anomalies = iforest.predict(X, contamination=0.01)

print(f"Detected {anomalies.sum()} anomalies")
```

### One-Class SVM

```python
from sklearn.svm import OneClassSVM as SklearnOCSVM

class OneClassSVMDetector:
    """One-Class SVM for anomaly detection"""

    def __init__(self, nu=0.1, kernel='rbf', gamma='auto'):
        """
        Args:
            nu: Upper bound on fraction of outliers
            kernel: Kernel type
            gamma: Kernel coefficient
        """
        self.model = SklearnOCSVM(nu=nu, kernel=kernel, gamma=gamma)

    def fit(self, X):
        """Fit One-Class SVM"""
        self.model.fit(X)

    def predict(self, X):
        """Predict anomalies (1 = normal, -1 = anomaly)"""
        predictions = self.model.predict(X)
        # Convert to binary: 0 = normal, 1 = anomaly
        return (predictions == -1).astype(int)

    def decision_function(self, X):
        """Get anomaly scores"""
        return -self.model.decision_function(X)  # Negative = more anomalous
```

### Local Outlier Factor (LOF)

```python
class LocalOutlierFactor:
    """Local Outlier Factor for anomaly detection"""

    def __init__(self, k=20):
        """
        Args:
            k: Number of neighbors
        """
        self.k = k
        self.X_train = None

    def fit(self, X):
        """Fit LOF"""
        self.X_train = X

    def _k_distance(self, point):
        """Find k-distance and k-nearest neighbors"""
        distances = np.linalg.norm(self.X_train - point, axis=1)
        k_nearest_indices = np.argsort(distances)[1:self.k+1]  # Exclude self
        k_distance = distances[k_nearest_indices[-1]]
        return k_distance, k_nearest_indices

    def _reachability_distance(self, point_idx, neighbor_idx):
        """Compute reachability distance"""
        point = self.X_train[point_idx]
        neighbor = self.X_train[neighbor_idx]

        k_dist_neighbor, _ = self._k_distance(neighbor)
        actual_dist = np.linalg.norm(point - neighbor)

        return max(k_dist_neighbor, actual_dist)

    def _local_reachability_density(self, point_idx):
        """Compute local reachability density"""
        _, k_neighbors = self._k_distance(self.X_train[point_idx])

        reachability_sum = sum(
            self._reachability_distance(point_idx, neighbor_idx)
            for neighbor_idx in k_neighbors
        )

        lrd = self.k / reachability_sum if reachability_sum > 0 else 0
        return lrd

    def local_outlier_factor(self, point_idx):
        """Compute LOF score"""
        _, k_neighbors = self._k_distance(self.X_train[point_idx])

        lrd_point = self._local_reachability_density(point_idx)
        if lrd_point == 0:
            return 1.0

        lrd_ratio_sum = sum(
            self._local_reachability_density(neighbor_idx) / lrd_point
            for neighbor_idx in k_neighbors
        )

        lof = lrd_ratio_sum / self.k
        return lof

    def predict(self, threshold=1.5):
        """Predict anomalies based on LOF threshold"""
        n_samples = len(self.X_train)
        lof_scores = np.array([
            self.local_outlier_factor(i) for i in range(n_samples)
        ])

        anomalies = (lof_scores > threshold).astype(int)
        return anomalies, lof_scores
```

---

## Deep Learning Methods

Neural network-based anomaly detection.

### Autoencoder for Anomaly Detection

```python
class Autoencoder(nn.Module):
    """Autoencoder for anomaly detection"""

    def __init__(self, input_dim, encoding_dim=32):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)


class AutoencoderDetector:
    """Anomaly detection using autoencoder"""

    def __init__(self, input_dim, encoding_dim=32):
        self.model = Autoencoder(input_dim, encoding_dim).cuda()
        self.threshold = None

    def fit(self, X_train, epochs=100, batch_size=128):
        """Train autoencoder on normal data"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        X_tensor = torch.FloatTensor(X_train).cuda()
        dataset = torch.utils.data.TensorDataset(X_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                x = batch[0]

                # Forward pass
                reconstructed = self.model(x)
                loss = criterion(reconstructed, x)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

        # Set threshold based on training data
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            reconstruction_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
            self.threshold = reconstruction_errors.mean() + 3 * reconstruction_errors.std()

    def anomaly_score(self, X):
        """Compute reconstruction error as anomaly score"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).cuda()

        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)

        return errors.cpu().numpy()

    def predict(self, X):
        """Predict anomalies"""
        scores = self.anomaly_score(X)
        predictions = (scores > self.threshold.cpu().numpy()).astype(int)
        return predictions

# Usage
# X_train = np.random.randn(1000, 10)  # Normal data
# detector = AutoencoderDetector(input_dim=10)
# detector.fit(X_train, epochs=100)
# anomalies = detector.predict(X_test)
```

### Variational Autoencoder (VAE)

```python
class VAE(nn.Module):
    """Variational Autoencoder for anomaly detection"""

    def __init__(self, input_dim, latent_dim=20):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar

    def vae_loss(self, reconstructed, x, mu, logvar):
        """VAE loss = reconstruction loss + KL divergence"""
        reconstruction_loss = F.mse_loss(reconstructed, x, reduction='sum')
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return reconstruction_loss + kl_divergence


class VAEDetector:
    """VAE-based anomaly detector"""

    def __init__(self, input_dim, latent_dim=20):
        self.model = VAE(input_dim, latent_dim).cuda()
        self.threshold = None

    def fit(self, X_train, epochs=100):
        """Train VAE"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        X_tensor = torch.FloatTensor(X_train).cuda()
        self.model.train()

        for epoch in range(epochs):
            reconstructed, mu, logvar = self.model(X_tensor)
            loss = self.model.vae_loss(reconstructed, X_tensor, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss.item()/len(X_train):.4f}")

        # Set threshold
        self.model.eval()
        with torch.no_grad():
            reconstructed, mu, logvar = self.model(X_tensor)
            losses = F.mse_loss(reconstructed, X_tensor, reduction='none').mean(dim=1)
            self.threshold = losses.mean() + 3 * losses.std()

    def anomaly_score(self, X):
        """Compute anomaly scores"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).cuda()

        with torch.no_grad():
            reconstructed, _, _ = self.model(X_tensor)
            scores = F.mse_loss(reconstructed, X_tensor, reduction='none').mean(dim=1)

        return scores.cpu().numpy()

    def predict(self, X):
        """Predict anomalies"""
        scores = self.anomaly_score(X)
        return (scores > self.threshold.cpu().numpy()).astype(int)
```

### LSTM Autoencoder for Time Series

```python
class LSTMAutoencoder(nn.Module):
    """LSTM Autoencoder for time series anomaly detection"""

    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()

        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=0.2
        )

        # Decoder
        self.decoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, dropout=0.2
        )

        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        """
        # Encode
        _, (hidden, cell) = self.encoder_lstm(x)

        # Decode
        batch_size, seq_len, _ = x.shape
        decoder_input = torch.zeros_like(x)
        decoded, _ = self.decoder_lstm(decoder_input, (hidden, cell))

        # Output
        reconstructed = self.output_layer(decoded)

        return reconstructed


class LSTMAutoencoderDetector:
    """Time series anomaly detection with LSTM autoencoder"""

    def __init__(self, input_dim, hidden_dim=64):
        self.model = LSTMAutoencoder(input_dim, hidden_dim).cuda()
        self.threshold = None

    def fit(self, X_train, epochs=50):
        """
        Train on normal sequences

        Args:
            X_train: (n_samples, seq_len, input_dim)
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        X_tensor = torch.FloatTensor(X_train).cuda()

        self.model.train()
        for epoch in range(epochs):
            reconstructed = self.model(X_tensor)
            loss = criterion(reconstructed, X_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

        # Set threshold
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=(1, 2))
            self.threshold = errors.mean() + 3 * errors.std()

    def anomaly_score(self, X):
        """Compute reconstruction error"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).cuda()

        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=(1, 2))

        return errors.cpu().numpy()

    def predict(self, X):
        """Predict anomalies"""
        scores = self.anomaly_score(X)
        return (scores > self.threshold.cpu().numpy()).astype(int)
```

---

## Time Series Anomaly Detection

Specialized methods for detecting anomalies in time series data.

### STL Decomposition + Residual Analysis

```python
from statsmodels.tsa.seasonal import STL

class STLAnomalyDetector:
    """STL decomposition for time series anomaly detection"""

    def __init__(self, period, threshold=3.0):
        """
        Args:
            period: Seasonal period
            threshold: Z-score threshold for anomalies
        """
        self.period = period
        self.threshold = threshold
        self.residual_mean = None
        self.residual_std = None

    def fit(self, time_series):
        """Fit on normal time series"""
        # STL decomposition
        stl = STL(time_series, period=self.period)
        result = stl.fit()

        # Analyze residuals
        self.residual_mean = result.resid.mean()
        self.residual_std = result.resid.std()

        return result

    def detect(self, time_series):
        """Detect anomalies in time series"""
        # Decompose
        stl = STL(time_series, period=self.period)
        result = stl.fit()

        # Z-scores of residuals
        z_scores = np.abs((result.resid - self.residual_mean) / self.residual_std)

        # Anomalies
        anomalies = (z_scores > self.threshold).astype(int)

        return anomalies, result

# Usage
# time_series = load_time_series()
# detector = STLAnomalyDetector(period=7)  # Weekly seasonality
# detector.fit(time_series_train)
# anomalies, decomposition = detector.detect(time_series_test)
```

### ARIMA Residual Analysis

```python
from statsmodels.tsa.arima.model import ARIMA

class ARIMAAnomalyDetector:
    """ARIMA-based anomaly detection"""

    def __init__(self, order=(1, 1, 1), threshold=3.0):
        """
        Args:
            order: ARIMA(p, d, q) order
            threshold: Z-score threshold
        """
        self.order = order
        self.threshold = threshold
        self.model = None

    def fit(self, time_series):
        """Fit ARIMA model"""
        self.model = ARIMA(time_series, order=self.order)
        self.fitted_model = self.model.fit()

    def detect(self, time_series):
        """Detect anomalies"""
        # Forecast
        forecast = self.fitted_model.forecast(steps=len(time_series))

        # Residuals
        residuals = time_series - forecast

        # Z-scores
        z_scores = np.abs((residuals - residuals.mean()) / residuals.std())

        # Anomalies
        anomalies = (z_scores > self.threshold).astype(int)

        return anomalies, forecast
```

### Sliding Window + Statistics

```python
class SlidingWindowDetector:
    """Sliding window anomaly detection"""

    def __init__(self, window_size=50, threshold=3.0):
        self.window_size = window_size
        self.threshold = threshold

    def detect(self, time_series):
        """Detect anomalies using sliding window"""
        n = len(time_series)
        anomalies = np.zeros(n, dtype=int)

        for i in range(n):
            # Get window
            start = max(0, i - self.window_size)
            end = min(n, i + self.window_size + 1)
            window = time_series[start:end]

            # Exclude current point
            window_without_current = np.concatenate([
                window[:i-start],
                window[i-start+1:]
            ])

            if len(window_without_current) == 0:
                continue

            # Z-score relative to window
            mean = window_without_current.mean()
            std = window_without_current.std()

            if std == 0:
                continue

            z_score = abs((time_series[i] - mean) / std)

            if z_score > self.threshold:
                anomalies[i] = 1

        return anomalies
```

---

## Multivariate Anomaly Detection

Detect anomalies in multivariate data.

### Mahalanobis Distance

```python
class MahalanobisDetector:
    """Mahalanobis distance for multivariate anomaly detection"""

    def __init__(self, threshold=3.0):
        self.threshold = threshold
        self.mean = None
        self.cov_inv = None

    def fit(self, X):
        """Fit on normal data"""
        self.mean = np.mean(X, axis=0)
        cov = np.cov(X.T)

        # Add small value to diagonal for numerical stability
        cov += np.eye(cov.shape[0]) * 1e-6

        self.cov_inv = np.linalg.inv(cov)

    def mahalanobis_distance(self, x):
        """Compute Mahalanobis distance"""
        diff = x - self.mean
        distance = np.sqrt(diff @ self.cov_inv @ diff.T)
        return distance

    def anomaly_score(self, X):
        """Compute anomaly scores"""
        scores = np.array([
            self.mahalanobis_distance(x) for x in X
        ])
        return scores

    def predict(self, X):
        """Predict anomalies"""
        scores = self.anomaly_score(X)
        return (scores > self.threshold).astype(int)
```

### Principal Component Analysis (PCA)

```python
from sklearn.decomposition import PCA

class PCAAnomalyDetector:
    """PCA-based anomaly detection"""

    def __init__(self, n_components=0.95, threshold=3.0):
        """
        Args:
            n_components: Number of components or variance ratio
            threshold: Z-score threshold
        """
        self.pca = PCA(n_components=n_components)
        self.threshold = threshold

    def fit(self, X):
        """Fit PCA"""
        self.pca.fit(X)

    def anomaly_score(self, X):
        """Reconstruction error as anomaly score"""
        # Transform to PCA space
        X_transformed = self.pca.transform(X)

        # Reconstruct
        X_reconstructed = self.pca.inverse_transform(X_transformed)

        # Reconstruction error
        errors = np.sum((X - X_reconstructed) ** 2, axis=1)

        return errors

    def predict(self, X):
        """Predict anomalies"""
        scores = self.anomaly_score(X)

        # Use threshold on normalized scores
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        z_scores = (scores - mean_score) / std_score

        return (z_scores > self.threshold).astype(int)
```

---

## Production Systems

Deploy anomaly detection in production.

### Real-Time Anomaly Detection

```python
class RealTimeAnomalyDetector:
    """Real-time streaming anomaly detector"""

    def __init__(self, window_size=1000, threshold=3.0):
        self.window_size = window_size
        self.threshold = threshold

        # Sliding statistics
        self.window = []
        self.mean = 0
        self.M2 = 0  # For Welford's online variance
        self.count = 0

    def update(self, value):
        """Update statistics with new value"""
        # Add to window
        self.window.append(value)
        if len(self.window) > self.window_size:
            old_value = self.window.pop(0)
            self._remove_from_stats(old_value)

        # Update statistics (Welford's method)
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.M2 += delta * delta2

    def _remove_from_stats(self, value):
        """Remove old value from statistics"""
        if self.count <= 1:
            return

        delta = value - self.mean
        self.mean -= delta / (self.count - 1)
        delta2 = value - self.mean
        self.M2 -= delta * delta2
        self.count -= 1

    def is_anomaly(self, value):
        """Check if value is anomaly"""
        if self.count < 2:
            return False

        # Compute z-score
        variance = self.M2 / (self.count - 1)
        std = np.sqrt(variance)

        if std == 0:
            return False

        z_score = abs((value - self.mean) / std)

        return z_score > self.threshold

    def process(self, value):
        """Process new value"""
        is_anomaly = self.is_anomaly(value)
        self.update(value)

        return is_anomaly

# Usage
detector = RealTimeAnomalyDetector(window_size=1000, threshold=3.0)

# Stream processing
# for value in data_stream:
#     if detector.process(value):
#         print(f"Anomaly detected: {value}")
```

### Ensemble Detector

```python
class EnsembleAnomalyDetector:
    """Ensemble of multiple anomaly detectors"""

    def __init__(self, detectors, voting='soft', weights=None):
        """
        Args:
            detectors: List of anomaly detectors
            voting: 'hard' (majority vote) or 'soft' (weighted average)
            weights: Weights for soft voting
        """
        self.detectors = detectors
        self.voting = voting
        self.weights = weights if weights else [1.0] * len(detectors)

    def fit(self, X):
        """Fit all detectors"""
        for detector in self.detectors:
            detector.fit(X)

    def predict(self, X):
        """Ensemble prediction"""
        if self.voting == 'hard':
            # Majority voting
            predictions = np.array([detector.predict(X) for detector in self.detectors])
            ensemble_pred = (predictions.sum(axis=0) > len(self.detectors) / 2).astype(int)

        else:  # soft voting
            # Weighted average of anomaly scores
            scores = []
            for detector, weight in zip(self.detectors, self.weights):
                if hasattr(detector, 'anomaly_score'):
                    score = detector.anomaly_score(X)
                else:
                    score = detector.predict(X).astype(float)
                scores.append(score * weight)

            ensemble_score = np.mean(scores, axis=0)

            # Normalize and threshold
            threshold = np.percentile(ensemble_score, 90)  # Top 10% as anomalies
            ensemble_pred = (ensemble_score > threshold).astype(int)

        return ensemble_pred

# Usage
# detectors = [
#     IsolationForest(),
#     OneClassSVMDetector(),
#     AutoencoderDetector(input_dim=10)
# ]
# ensemble = EnsembleAnomalyDetector(detectors, voting='soft')
# ensemble.fit(X_train)
# anomalies = ensemble.predict(X_test)
```

---

## Evaluation Metrics

Properly evaluate anomaly detection systems.

### Metrics for Imbalanced Data

```python
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix
)

class AnomalyDetectionMetrics:
    """Comprehensive metrics for anomaly detection"""

    @staticmethod
    def evaluate(y_true, y_pred, y_scores=None):
        """
        Evaluate anomaly detector

        Args:
            y_true: True labels (0 = normal, 1 = anomaly)
            y_pred: Predicted labels
            y_scores: Anomaly scores (optional, for ROC-AUC)
        """
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Basic metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # False positive rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        results = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'false_positive_rate': fpr,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }

        # Score-based metrics
        if y_scores is not None:
            results['roc_auc'] = roc_auc_score(y_true, y_scores)
            results['pr_auc'] = average_precision_score(y_true, y_scores)

        return results

    @staticmethod
    def print_evaluation(results):
        """Print evaluation results"""
        print("Anomaly Detection Evaluation")
        print("=" * 50)
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1-Score: {results['f1_score']:.4f}")
        print(f"Specificity: {results['specificity']:.4f}")
        print(f"False Positive Rate: {results['false_positive_rate']:.4f}")

        if 'roc_auc' in results:
            print(f"ROC-AUC: {results['roc_auc']:.4f}")
            print(f"PR-AUC: {results['pr_auc']:.4f}")

        print("\nConfusion Matrix:")
        print(f"True Positives: {results['true_positives']}")
        print(f"False Positives: {results['false_positives']}")
        print(f"True Negatives: {results['true_negatives']}")
        print(f"False Negatives: {results['false_negatives']}")

# Usage
# y_true = np.array([0, 0, 1, 0, 1, 0, 0, 1])
# y_pred = np.array([0, 0, 1, 0, 0, 1, 0, 1])
# y_scores = np.array([0.1, 0.2, 0.9, 0.3, 0.7, 0.8, 0.1, 0.95])

# metrics = AnomalyDetectionMetrics()
# results = metrics.evaluate(y_true, y_pred, y_scores)
# metrics.print_evaluation(results)
```

---

## Summary

Anomaly detection identifies unusual patterns in data:

1. **Statistical Methods**: Z-score, IQR, Grubbs' test
2. **ML Approaches**: Isolation Forest, One-Class SVM, LOF
3. **Deep Learning**: Autoencoders, VAE, LSTM-AE
4. **Time Series**: STL, ARIMA residuals, sliding windows
5. **Multivariate**: Mahalanobis distance, PCA

**Key Considerations:**
- Choose method based on data type and distribution
- Handle imbalanced data appropriately
- Set appropriate thresholds
- Use ensemble methods for robustness
- Monitor false positive rates in production

**Applications:**
- Fraud detection
- Network intrusion detection
- Equipment failure prediction
- Quality control
- Health monitoring


---

## 📚 References

**Statistical Methods:**

1. **Iglewicz, B., & Hoaglin, D. C.** (1993). "How to detect and handle outliers." *The ASQC Basic References in Quality Control: Statistical Techniques*, Vol. 16.
   - Modified Z-score with MAD

2. **Grubbs, F. E.** (1969). "Procedures for detecting outlying observations in samples." *Technometrics*, 11(1), 1-21.
   - Grubbs' test for outliers

3. **Tukey, J. W.** (1977). *Exploratory Data Analysis*. Addison-Wesley.
   - IQR method for outliers

**Machine Learning Methods:**

4. **Liu, F. T., Ting, K. M., & Zhou, Z. H.** (2008). "Isolation forest." *ICDM 2008*, 413-422.
   - Isolation Forest algorithm

5. **Schölkopf, B., Williamson, R. C., Smola, A. J., et al.** (2000). "Support vector method for novelty detection." *NIPS 2000*, 582-588.
   - One-Class SVM

6. **Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J.** (2000). "LOF: Identifying density-based local outliers." *SIGMOD 2000*, 93-104.
   - Local Outlier Factor

**Deep Learning Methods:**

7. **Sakurada, M., & Yairi, T.** (2014). "Anomaly detection using autoencoders with nonlinear dimensionality reduction." *MLSDA Workshop, ACM SIGKDD 2014*.
   - Autoencoder for anomaly detection

8. **An, J., & Cho, S.** (2015). "Variational autoencoder based anomaly detection using reconstruction probability." *SNU Data Mining Center Technical Report*.
   - VAE for anomaly detection

9. **Malhotra, P., Vig, L., Shroff, G., & Agarwal, P.** (2015). "Long short term memory networks for anomaly detection in time series." *ESANN 2015*.
   - LSTM for time series anomalies

**Time Series:**

10. **Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I.** (1990). "STL: A seasonal-trend decomposition procedure based on LOESS." *Journal of Official Statistics*, 6(1), 3-73.
    - STL decomposition

**Evaluation:**

11. **Davis, J., & Goadrich, M.** (2006). "The relationship between Precision-Recall and ROC curves." *ICML 2006*, 233-240.
    - PR curves for imbalanced data

12. **Emmott, A. F., Das, S., Dietterich, T., et al.** (2013). "Systematic construction of anomaly detection benchmarks from real data." *ACM SIGKDD Workshop on Outlier Detection and Description*, 16-21.
    - Benchmark evaluation

**Books:**

- **Aggarwal, C. C.** (2017). *Outlier Analysis* (2nd ed.). Springer.
  - Comprehensive textbook on anomaly detection

- **Chandola, V., Banerjee, A., & Kumar, V.** (2009). "Anomaly detection: A survey." *ACM Computing Surveys*, 41(3), 1-58.
  - Survey of anomaly detection methods

**Online Resources:**
- scikit-learn Novelty and Outlier Detection: https://scikit-learn.org/stable/modules/outlier_detection.html
- PyOD (Python Outlier Detection): https://pyod.readthedocs.io/
- Anomaly Detection Learning Resources: https://github.com/yzhao062/anomaly-detection-resources

---

*Anomaly detection is context-dependent. Always validate methods on domain-specific data and adjust thresholds based on business requirements.*
