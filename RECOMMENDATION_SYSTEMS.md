# Recommendation Systems: Complete Guide

## Table of Contents
1. [Introduction to Recommendation Systems](#introduction)
2. [Collaborative Filtering](#collaborative-filtering)
3. [Content-Based Filtering](#content-based-filtering)
4. [Matrix Factorization](#matrix-factorization)
5. [Deep Learning for RecSys](#deep-learning-for-recsys)
6. [Hybrid Approaches](#hybrid-approaches)
7. [Context-Aware Recommendations](#context-aware-recommendations)
8. [Production Systems](#production-systems)

---

## Introduction to Recommendation Systems

Recommendation systems predict user preferences and suggest relevant items.

### Types of Recommendation Systems

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

class RecommendationSystemTypes:
    """Overview of different recommendation approaches"""

    def __init__(self):
        self.approaches = {
            'collaborative_filtering': 'Use user-item interactions',
            'content_based': 'Use item features',
            'hybrid': 'Combine multiple approaches',
            'context_aware': 'Consider context (time, location, etc.)'
        }

    def demonstrate_types(self):
        """Compare different recommendation types"""

        # Sample user-item matrix (users x items)
        ratings = np.array([
            [5, 3, 0, 1],
            [4, 0, 0, 1],
            [1, 1, 0, 5],
            [1, 0, 0, 4],
            [0, 1, 5, 4],
        ])

        print("User-Item Ratings Matrix:")
        print(ratings)
        print("\nApproaches:")
        for name, desc in self.approaches.items():
            print(f"- {name}: {desc}")

# Usage
demo = RecommendationSystemTypes()
demo.demonstrate_types()
```

### Evaluation Metrics

```python
class RecSysMetrics:
    """Evaluation metrics for recommendation systems"""

    @staticmethod
    def precision_at_k(recommended, relevant, k=10):
        """Precision@K: Fraction of recommended items that are relevant"""
        recommended_at_k = set(recommended[:k])
        relevant_set = set(relevant)
        return len(recommended_at_k & relevant_set) / k

    @staticmethod
    def recall_at_k(recommended, relevant, k=10):
        """Recall@K: Fraction of relevant items that are recommended"""
        recommended_at_k = set(recommended[:k])
        relevant_set = set(relevant)
        return len(recommended_at_k & relevant_set) / len(relevant_set)

    @staticmethod
    def mean_average_precision(recommended_list, relevant_list):
        """MAP: Mean of average precision across all users"""
        aps = []
        for recommended, relevant in zip(recommended_list, relevant_list):
            relevant_set = set(relevant)
            precision_sum = 0
            num_relevant = 0

            for i, item in enumerate(recommended, 1):
                if item in relevant_set:
                    num_relevant += 1
                    precision_sum += num_relevant / i

            ap = precision_sum / len(relevant_set) if relevant_set else 0
            aps.append(ap)

        return np.mean(aps)

    @staticmethod
    def ndcg_at_k(recommended, relevant_scores, k=10):
        """Normalized Discounted Cumulative Gain@K"""
        dcg = 0
        for i, item in enumerate(recommended[:k], 1):
            rel = relevant_scores.get(item, 0)
            dcg += (2**rel - 1) / np.log2(i + 1)

        # Ideal DCG
        ideal_scores = sorted(relevant_scores.values(), reverse=True)
        idcg = sum((2**rel - 1) / np.log2(i + 1)
                   for i, rel in enumerate(ideal_scores[:k], 1))

        return dcg / idcg if idcg > 0 else 0

    @staticmethod
    def hit_rate_at_k(recommended_list, relevant_list, k=10):
        """Hit Rate@K: Fraction of users with at least one hit in top-k"""
        hits = 0
        for recommended, relevant in zip(recommended_list, relevant_list):
            recommended_at_k = set(recommended[:k])
            if len(recommended_at_k & set(relevant)) > 0:
                hits += 1
        return hits / len(recommended_list)

# Test metrics
metrics = RecSysMetrics()
recommended = [1, 2, 3, 4, 5]
relevant = [2, 4, 7]

print(f"Precision@5: {metrics.precision_at_k(recommended, relevant, 5):.2f}")
print(f"Recall@5: {metrics.recall_at_k(recommended, relevant, 5):.2f}")
```

---

## Collaborative Filtering

Recommend items based on user similarity or item similarity.

### User-Based Collaborative Filtering

```python
class UserBasedCF:
    """User-based collaborative filtering"""

    def __init__(self, ratings_matrix):
        """
        Args:
            ratings_matrix: User-item ratings (users x items)
        """
        self.ratings = ratings_matrix
        self.user_similarity = self._compute_user_similarity()

    def _compute_user_similarity(self):
        """Compute pairwise user similarity (cosine)"""
        # Replace zeros with NaN for proper similarity computation
        ratings_for_sim = self.ratings.copy()
        ratings_for_sim[ratings_for_sim == 0] = np.nan

        # Fill NaN with 0 for cosine similarity
        ratings_filled = np.nan_to_num(ratings_for_sim)
        similarity = cosine_similarity(ratings_filled)

        # Set self-similarity to 0
        np.fill_diagonal(similarity, 0)

        return similarity

    def predict(self, user_id, item_id, k=5):
        """
        Predict rating for user-item pair

        Args:
            user_id: User index
            item_id: Item index
            k: Number of similar users to consider
        """
        # Find k most similar users who rated this item
        similarities = self.user_similarity[user_id]
        rated_users = np.where(self.ratings[:, item_id] > 0)[0]

        # Filter for users who rated the item
        valid_similarities = [(u, similarities[u]) for u in rated_users]
        valid_similarities.sort(key=lambda x: x[1], reverse=True)

        # Get top-k
        top_k = valid_similarities[:k]

        if not top_k:
            # No similar users, return global mean
            return np.mean(self.ratings[self.ratings > 0])

        # Weighted average of ratings
        numerator = sum(sim * self.ratings[u, item_id] for u, sim in top_k)
        denominator = sum(abs(sim) for _, sim in top_k)

        if denominator == 0:
            return np.mean(self.ratings[self.ratings > 0])

        return numerator / denominator

    def recommend(self, user_id, n=10):
        """
        Recommend top-n items for user

        Args:
            user_id: User index
            n: Number of recommendations
        """
        # Find items user hasn't rated
        unrated_items = np.where(self.ratings[user_id] == 0)[0]

        # Predict ratings for unrated items
        predictions = [(item, self.predict(user_id, item))
                       for item in unrated_items]

        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:n]

# Example usage
ratings = np.array([
    [5, 3, 0, 1, 0],
    [4, 0, 0, 1, 1],
    [1, 1, 0, 5, 0],
    [1, 0, 0, 4, 0],
    [0, 1, 5, 4, 0],
])

cf = UserBasedCF(ratings)
recommendations = cf.recommend(user_id=0, n=3)
print("Recommendations for user 0:")
for item_id, score in recommendations:
    print(f"  Item {item_id}: {score:.2f}")
```

### Item-Based Collaborative Filtering

```python
class ItemBasedCF:
    """Item-based collaborative filtering"""

    def __init__(self, ratings_matrix):
        self.ratings = ratings_matrix
        self.item_similarity = self._compute_item_similarity()

    def _compute_item_similarity(self):
        """Compute pairwise item similarity"""
        # Transpose to get items x users
        ratings_t = self.ratings.T
        ratings_filled = np.nan_to_num(ratings_t)

        similarity = cosine_similarity(ratings_filled)
        np.fill_diagonal(similarity, 0)

        return similarity

    def predict(self, user_id, item_id, k=5):
        """Predict rating for user-item pair"""
        # Find k most similar items that user has rated
        similarities = self.item_similarity[item_id]
        rated_items = np.where(self.ratings[user_id] > 0)[0]

        # Filter and sort
        valid_similarities = [(i, similarities[i]) for i in rated_items]
        valid_similarities.sort(key=lambda x: x[1], reverse=True)

        top_k = valid_similarities[:k]

        if not top_k:
            return np.mean(self.ratings[self.ratings > 0])

        # Weighted average
        numerator = sum(sim * self.ratings[user_id, i] for i, sim in top_k)
        denominator = sum(abs(sim) for _, sim in top_k)

        if denominator == 0:
            return np.mean(self.ratings[self.ratings > 0])

        return numerator / denominator

    def recommend(self, user_id, n=10):
        """Recommend top-n items for user"""
        unrated_items = np.where(self.ratings[user_id] == 0)[0]

        predictions = [(item, self.predict(user_id, item))
                       for item in unrated_items]
        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:n]
```

---

## Content-Based Filtering

Recommend items similar to those the user liked based on item features.

### Content-Based Recommender

```python
class ContentBasedRecommender:
    """Content-based filtering using item features"""

    def __init__(self, item_features, user_ratings):
        """
        Args:
            item_features: Item feature matrix (items x features)
            user_ratings: User-item ratings
        """
        self.item_features = item_features
        self.user_ratings = user_ratings

        # Compute item similarity
        self.item_similarity = cosine_similarity(item_features)
        np.fill_diagonal(self.item_similarity, 0)

    def build_user_profile(self, user_id):
        """Build user profile from rated items"""
        # Get items user has rated
        rated_items = np.where(self.user_ratings[user_id] > 0)[0]
        ratings = self.user_ratings[user_id, rated_items]

        # Weighted average of item features
        if len(rated_items) == 0:
            return np.zeros(self.item_features.shape[1])

        weighted_features = (
            self.item_features[rated_items].T * ratings
        ).T.sum(axis=0)

        user_profile = weighted_features / ratings.sum()

        return user_profile

    def predict(self, user_id, item_id):
        """Predict user rating for item"""
        user_profile = self.build_user_profile(user_id)
        item_feature = self.item_features[item_id]

        # Cosine similarity
        similarity = np.dot(user_profile, item_feature) / (
            np.linalg.norm(user_profile) * np.linalg.norm(item_feature) + 1e-8
        )

        # Scale to rating range (e.g., 1-5)
        predicted_rating = 1 + 4 * (similarity + 1) / 2

        return predicted_rating

    def recommend(self, user_id, n=10):
        """Recommend top-n items for user"""
        # Find unrated items
        unrated_items = np.where(self.user_ratings[user_id] == 0)[0]

        # Predict ratings
        predictions = [(item, self.predict(user_id, item))
                       for item in unrated_items]
        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:n]

# Example with TF-IDF features
from sklearn.feature_extraction.text import TfidfVectorizer

class TextContentRecommender:
    """Content-based recommender for text items"""

    def __init__(self, item_descriptions, user_ratings):
        # Create TF-IDF features
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.item_features = self.vectorizer.fit_transform(item_descriptions).toarray()
        self.user_ratings = user_ratings

        # Recommender
        self.recommender = ContentBasedRecommender(
            self.item_features, user_ratings
        )

    def recommend(self, user_id, n=10):
        return self.recommender.recommend(user_id, n)
```

---

## Matrix Factorization

Decompose user-item matrix into latent factor matrices.

### SVD for Recommendations

```python
class SVDRecommender:
    """Singular Value Decomposition for recommendations"""

    def __init__(self, n_factors=20):
        self.n_factors = n_factors
        self.user_factors = None
        self.item_factors = None
        self.global_mean = 0

    def fit(self, ratings_matrix):
        """
        Fit SVD model

        Args:
            ratings_matrix: User-item ratings (users x items)
        """
        # Center ratings
        self.global_mean = ratings_matrix[ratings_matrix > 0].mean()
        centered_ratings = ratings_matrix.copy()
        centered_ratings[centered_ratings > 0] -= self.global_mean

        # SVD
        from scipy.sparse.linalg import svds
        U, sigma, Vt = svds(centered_ratings, k=self.n_factors)

        # Store factors
        self.user_factors = U
        self.item_factors = Vt.T
        self.sigma = np.diag(sigma)

    def predict(self, user_id, item_id):
        """Predict rating"""
        pred = (
            self.global_mean +
            self.user_factors[user_id] @ self.sigma @ self.item_factors[item_id]
        )
        return pred

    def recommend(self, user_id, n=10, rated_items=None):
        """Recommend top-n items"""
        # Predict all items
        predictions = self.user_factors[user_id] @ self.sigma @ self.item_factors.T
        predictions += self.global_mean

        # Exclude rated items
        if rated_items is not None:
            predictions[list(rated_items)] = -np.inf

        # Get top-n
        top_items = np.argsort(predictions)[::-1][:n]
        top_scores = predictions[top_items]

        return list(zip(top_items, top_scores))
```

### Alternating Least Squares (ALS)

```python
class ALSRecommender:
    """Alternating Least Squares for implicit feedback"""

    def __init__(self, n_factors=20, reg_lambda=0.01, alpha=40):
        """
        Args:
            n_factors: Number of latent factors
            reg_lambda: Regularization parameter
            alpha: Confidence weight for implicit feedback
        """
        self.n_factors = n_factors
        self.reg_lambda = reg_lambda
        self.alpha = alpha

        self.user_factors = None
        self.item_factors = None

    def fit(self, interactions, n_iterations=15):
        """
        Fit ALS model on implicit feedback

        Args:
            interactions: User-item interaction matrix (0/1 or counts)
            n_iterations: Number of ALS iterations
        """
        n_users, n_items = interactions.shape

        # Initialize factors randomly
        self.user_factors = np.random.normal(
            scale=0.01, size=(n_users, self.n_factors)
        )
        self.item_factors = np.random.normal(
            scale=0.01, size=(n_items, self.n_factors)
        )

        # Confidence matrix: C = 1 + alpha * R
        confidence = 1 + self.alpha * interactions

        for iteration in range(n_iterations):
            # Fix item factors, update user factors
            self.user_factors = self._solve_factors(
                confidence, self.item_factors, axis=0
            )

            # Fix user factors, update item factors
            self.item_factors = self._solve_factors(
                confidence.T, self.user_factors, axis=0
            )

            if (iteration + 1) % 5 == 0:
                loss = self._compute_loss(interactions, confidence)
                print(f"Iteration {iteration + 1}, Loss: {loss:.4f}")

    def _solve_factors(self, confidence, factors, axis):
        """Solve for one set of factors"""
        n = confidence.shape[axis]
        new_factors = np.zeros((n, self.n_factors))

        for i in range(n):
            # Get confidence values for this user/item
            c = confidence[i].toarray().flatten() if hasattr(confidence, 'toarray') else confidence[i]

            # Preference (binarize)
            p = (c > 1).astype(float)

            # Weighted regularization term
            A = factors.T @ np.diag(c) @ factors + self.reg_lambda * np.eye(self.n_factors)
            b = factors.T @ (c * p)

            # Solve: Ax = b
            new_factors[i] = np.linalg.solve(A, b)

        return new_factors

    def _compute_loss(self, interactions, confidence):
        """Compute reconstruction loss"""
        pred = self.user_factors @ self.item_factors.T
        diff = interactions - pred
        loss = np.sum(confidence * (diff ** 2))
        loss += self.reg_lambda * (
            np.sum(self.user_factors ** 2) + np.sum(self.item_factors ** 2)
        )
        return loss

    def recommend(self, user_id, n=10, rated_items=None):
        """Recommend top-n items"""
        scores = self.user_factors[user_id] @ self.item_factors.T

        if rated_items is not None:
            scores[list(rated_items)] = -np.inf

        top_items = np.argsort(scores)[::-1][:n]
        top_scores = scores[top_items]

        return list(zip(top_items, top_scores))
```

---

## Deep Learning for RecSys

Neural network-based recommendation models.

### Neural Collaborative Filtering (NCF)

```python
class NCF(nn.Module):
    """Neural Collaborative Filtering"""

    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim=32,
        hidden_layers=[64, 32, 16]
    ):
        super().__init__()

        # Embeddings for GMF (Generalized Matrix Factorization)
        self.user_embedding_gmf = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_gmf = nn.Embedding(num_items, embedding_dim)

        # Embeddings for MLP
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(num_items, embedding_dim)

        # MLP layers
        mlp_layers = []
        input_size = embedding_dim * 2
        for hidden_size in hidden_layers:
            mlp_layers.append(nn.Linear(input_size, hidden_size))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(0.2))
            input_size = hidden_size

        self.mlp = nn.Sequential(*mlp_layers)

        # Final prediction layer
        self.output = nn.Linear(embedding_dim + hidden_layers[-1], 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.01)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, user_ids, item_ids):
        """
        Forward pass

        Args:
            user_ids: User indices (B,)
            item_ids: Item indices (B,)
        """
        # GMF part
        user_emb_gmf = self.user_embedding_gmf(user_ids)
        item_emb_gmf = self.item_embedding_gmf(item_ids)
        gmf_output = user_emb_gmf * item_emb_gmf

        # MLP part
        user_emb_mlp = self.user_embedding_mlp(user_ids)
        item_emb_mlp = self.item_embedding_mlp(item_ids)
        mlp_input = torch.cat([user_emb_mlp, item_emb_mlp], dim=-1)
        mlp_output = self.mlp(mlp_input)

        # Combine
        combined = torch.cat([gmf_output, mlp_output], dim=-1)
        prediction = torch.sigmoid(self.output(combined))

        return prediction.squeeze()

# Training NCF
def train_ncf(model, train_loader, epochs=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for user_ids, item_ids, labels in train_loader:
            user_ids = user_ids.cuda()
            item_ids = item_ids.cuda()
            labels = labels.float().cuda()

            # Forward pass
            predictions = model(user_ids, item_ids)
            loss = criterion(predictions, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
```

### Deep & Cross Network (DCN)

```python
class CrossNetwork(nn.Module):
    """Cross Network for explicit feature crossing"""

    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers

        # Weight and bias for each cross layer
        self.cross_weights = nn.ParameterList([
            nn.Parameter(torch.Tensor(input_dim, 1))
            for _ in range(num_layers)
        ])
        self.cross_biases = nn.ParameterList([
            nn.Parameter(torch.Tensor(input_dim, 1))
            for _ in range(num_layers)
        ])

        self._init_weights()

    def _init_weights(self):
        for weight in self.cross_weights:
            nn.init.xavier_uniform_(weight)
        for bias in self.cross_biases:
            nn.init.zeros_(bias)

    def forward(self, x):
        """
        Args:
            x: Input features (B, D)
        """
        x0 = x.unsqueeze(2)  # (B, D, 1)
        xi = x0

        for i in range(self.num_layers):
            # Cross: x_{i+1} = x_0 * x_i^T * w_i + b_i + x_i
            xw = torch.matmul(xi.transpose(1, 2), self.cross_weights[i])  # (B, 1, 1)
            xi = torch.matmul(x0, xw) + self.cross_biases[i] + xi

        return xi.squeeze(2)  # (B, D)


class DCN(nn.Module):
    """Deep & Cross Network"""

    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim=32,
        num_cross_layers=3,
        deep_layers=[256, 128, 64]
    ):
        super().__init__()

        # Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        input_dim = embedding_dim * 2

        # Cross network
        self.cross_net = CrossNetwork(input_dim, num_cross_layers)

        # Deep network
        deep = []
        for hidden_size in deep_layers:
            deep.append(nn.Linear(input_dim, hidden_size))
            deep.append(nn.ReLU())
            deep.append(nn.Dropout(0.2))
            input_dim = hidden_size

        self.deep_net = nn.Sequential(*deep)

        # Combination layer
        self.combination = nn.Linear(
            embedding_dim * 2 + deep_layers[-1], 1
        )

    def forward(self, user_ids, item_ids):
        # Embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        features = torch.cat([user_emb, item_emb], dim=-1)

        # Cross network
        cross_output = self.cross_net(features)

        # Deep network
        deep_output = self.deep_net(features)

        # Combine
        combined = torch.cat([cross_output, deep_output], dim=-1)
        prediction = torch.sigmoid(self.combination(combined))

        return prediction.squeeze()
```

### Wide & Deep

```python
class WideAndDeep(nn.Module):
    """Wide & Deep model for recommendations"""

    def __init__(
        self,
        num_users,
        num_items,
        num_wide_features,
        embedding_dim=32,
        deep_layers=[256, 128, 64]
    ):
        super().__init__()

        # Wide part (linear model on raw features)
        self.wide = nn.Linear(num_wide_features, 1)

        # Deep part
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        deep = []
        input_dim = embedding_dim * 2
        for hidden_size in deep_layers:
            deep.append(nn.Linear(input_dim, hidden_size))
            deep.append(nn.ReLU())
            deep.append(nn.BatchNorm1d(hidden_size))
            deep.append(nn.Dropout(0.3))
            input_dim = hidden_size

        self.deep = nn.Sequential(*deep)
        self.deep_output = nn.Linear(deep_layers[-1], 1)

    def forward(self, user_ids, item_ids, wide_features):
        """
        Args:
            user_ids: User IDs (B,)
            item_ids: Item IDs (B,)
            wide_features: Wide features (B, num_wide_features)
        """
        # Wide part
        wide_output = self.wide(wide_features)

        # Deep part
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        deep_input = torch.cat([user_emb, item_emb], dim=-1)
        deep_features = self.deep(deep_input)
        deep_output = self.deep_output(deep_features)

        # Combine
        output = torch.sigmoid(wide_output + deep_output)

        return output.squeeze()
```

---

## Hybrid Approaches

Combine multiple recommendation techniques.

### Hybrid Recommender

```python
class HybridRecommender:
    """Hybrid recommendation combining multiple approaches"""

    def __init__(
        self,
        collaborative_model,
        content_model,
        weights={'collaborative': 0.6, 'content': 0.4}
    ):
        self.collaborative_model = collaborative_model
        self.content_model = content_model
        self.weights = weights

    def recommend(self, user_id, n=10):
        """Hybrid recommendations"""
        # Get recommendations from both models
        collab_recs = self.collaborative_model.recommend(user_id, n=n*2)
        content_recs = self.content_model.recommend(user_id, n=n*2)

        # Combine scores
        combined_scores = {}

        for item_id, score in collab_recs:
            combined_scores[item_id] = score * self.weights['collaborative']

        for item_id, score in content_recs:
            if item_id in combined_scores:
                combined_scores[item_id] += score * self.weights['content']
            else:
                combined_scores[item_id] = score * self.weights['content']

        # Sort and return top-n
        recommendations = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n]

        return recommendations

# Weighted hybrid with learned weights
class LearnedHybrid(nn.Module):
    """Learn optimal combination of recommendation models"""

    def __init__(self, num_models):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_models) / num_models)

    def forward(self, model_predictions):
        """
        Args:
            model_predictions: List of predictions from different models
                               Each is (B,) tensor
        """
        # Stack predictions
        stacked = torch.stack(model_predictions, dim=1)  # (B, num_models)

        # Softmax weights
        normalized_weights = F.softmax(self.weights, dim=0)

        # Weighted combination
        combined = (stacked * normalized_weights).sum(dim=1)

        return combined
```

---

## Context-Aware Recommendations

Incorporate context (time, location, device) into recommendations.

### Factorization Machines

```python
class FactorizationMachine(nn.Module):
    """Factorization Machine for context-aware recommendations"""

    def __init__(self, num_features, embedding_dim=10):
        super().__init__()

        # Linear terms
        self.linear = nn.Linear(num_features, 1, bias=True)

        # Embedding for pairwise interactions
        self.embeddings = nn.Embedding(num_features, embedding_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embeddings.weight, std=0.01)

    def forward(self, x):
        """
        Args:
            x: Sparse feature vector (B, num_features)
               Can be one-hot encoded categorical features
        """
        # Linear part
        linear_part = self.linear(x)

        # Pairwise interaction part
        # Get embeddings for all features
        feature_embeddings = self.embeddings.weight  # (num_features, embedding_dim)

        # Compute interactions: 0.5 * sum((sum(x*v))^2 - sum((x*v)^2))
        square_of_sum = torch.pow(x @ feature_embeddings, 2)
        sum_of_square = (x ** 2) @ (feature_embeddings ** 2)
        interaction_part = 0.5 * (square_of_sum - sum_of_square).sum(dim=1, keepdim=True)

        # Combine
        output = linear_part + interaction_part

        return output.squeeze()

# Field-aware Factorization Machine
class FFM(nn.Module):
    """Field-aware Factorization Machine"""

    def __init__(self, num_features, num_fields, embedding_dim=10):
        super().__init__()

        self.num_features = num_features
        self.num_fields = num_fields

        # Linear terms
        self.linear = nn.Linear(num_features, 1)

        # Field-aware embeddings: each feature has embedding for each field
        self.field_embeddings = nn.Parameter(
            torch.randn(num_features, num_fields, embedding_dim) * 0.01
        )

    def forward(self, x, field_indices):
        """
        Args:
            x: Feature values (B, num_features)
            field_indices: Field index for each feature (num_features,)
        """
        batch_size = x.shape[0]

        # Linear part
        linear_part = self.linear(x)

        # Pairwise interactions with field awareness
        interaction_sum = 0

        for i in range(self.num_features):
            for j in range(i + 1, self.num_features):
                # Get field-aware embeddings
                fi = field_indices[i]
                fj = field_indices[j]

                # Feature i with respect to field j
                v_i_fj = self.field_embeddings[i, fj]
                # Feature j with respect to field i
                v_j_fi = self.field_embeddings[j, fi]

                # Interaction
                interaction = (
                    x[:, i].unsqueeze(1) *
                    x[:, j].unsqueeze(1) *
                    (v_i_fj * v_j_fi).sum(dim=-1, keepdim=True)
                )
                interaction_sum += interaction

        output = linear_part + interaction_sum

        return output.squeeze()
```

---

## Production Systems

Build scalable recommendation systems for production.

### Two-Tower Model

```python
class TwoTowerModel(nn.Module):
    """Two-tower model for efficient large-scale retrieval"""

    def __init__(self, user_features_dim, item_features_dim, embedding_dim=128):
        super().__init__()

        # User tower
        self.user_tower = nn.Sequential(
            nn.Linear(user_features_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim)
        )

        # Item tower
        self.item_tower = nn.Sequential(
            nn.Linear(item_features_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim)
        )

        # Temperature for scaling
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, user_features, item_features):
        """
        Args:
            user_features: (B, user_features_dim)
            item_features: (B, item_features_dim)
        """
        user_embeds = self.user_tower(user_features)
        item_embeds = self.item_tower(item_features)

        # Normalize
        user_embeds = F.normalize(user_embeds, dim=1)
        item_embeds = F.normalize(item_embeds, dim=1)

        return user_embeds, item_embeds

    def compute_similarity(self, user_embeds, item_embeds):
        """Compute similarity scores"""
        # Dot product similarity
        scores = (user_embeds * item_embeds).sum(dim=1) / self.temperature
        return scores

# Training with in-batch negatives
def train_two_tower(model, dataloader, epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        for user_features, item_features, labels in dataloader:
            user_features = user_features.cuda()
            item_features = item_features.cuda()
            labels = labels.cuda()

            # Forward
            user_embeds, item_embeds = model(user_features, item_features)

            # Compute similarity matrix (in-batch negatives)
            scores = torch.matmul(user_embeds, item_embeds.T) / model.temperature

            # Loss: cross-entropy with in-batch negatives
            loss = F.cross_entropy(scores, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### Serving Pipeline

```python
import faiss

class RecommendationPipeline:
    """Production recommendation pipeline"""

    def __init__(self, model, item_features, embedding_dim=128):
        self.model = model
        self.model.eval()
        self.embedding_dim = embedding_dim

        # Pre-compute item embeddings
        self.item_embeddings = self._compute_item_embeddings(item_features)

        # Build FAISS index for fast retrieval
        self.index = self._build_faiss_index()

    def _compute_item_embeddings(self, item_features):
        """Pre-compute embeddings for all items"""
        with torch.no_grad():
            _, item_embeds = self.model(
                torch.zeros(len(item_features), 1),  # Dummy user features
                torch.tensor(item_features).float()
            )
        return item_embeds.cpu().numpy()

    def _build_faiss_index(self):
        """Build FAISS index for efficient similarity search"""
        # Inner product index (cosine similarity with normalized vectors)
        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(self.item_embeddings)
        return index

    def recommend(self, user_features, k=10):
        """
        Get top-k recommendations for user

        Args:
            user_features: User feature vector
            k: Number of recommendations
        """
        # Compute user embedding
        with torch.no_grad():
            user_embed, _ = self.model(
                torch.tensor([user_features]).float(),
                torch.zeros(1, 1)  # Dummy item features
            )
        user_embed = user_embed.cpu().numpy()

        # Search in FAISS index
        scores, indices = self.index.search(user_embed, k)

        recommendations = [
            {'item_id': int(idx), 'score': float(score)}
            for idx, score in zip(indices[0], scores[0])
        ]

        return recommendations

    def batch_recommend(self, user_features_list, k=10):
        """Batch recommendations"""
        # Compute user embeddings
        with torch.no_grad():
            user_embeds, _ = self.model(
                torch.tensor(user_features_list).float(),
                torch.zeros(len(user_features_list), 1)
            )
        user_embeds = user_embeds.cpu().numpy()

        # Batch search
        scores, indices = self.index.search(user_embeds, k)

        recommendations = []
        for user_indices, user_scores in zip(indices, scores):
            recs = [
                {'item_id': int(idx), 'score': float(score)}
                for idx, score in zip(user_indices, user_scores)
            ]
            recommendations.append(recs)

        return recommendations
```

### A/B Testing Framework

```python
class ABTestingFramework:
    """A/B testing for recommendation systems"""

    def __init__(self):
        self.experiments = {}

    def create_experiment(self, experiment_id, model_a, model_b, split_ratio=0.5):
        """Create new A/B test"""
        self.experiments[experiment_id] = {
            'model_a': model_a,
            'model_b': model_b,
            'split_ratio': split_ratio,
            'results_a': {'impressions': 0, 'clicks': 0},
            'results_b': {'impressions': 0, 'clicks': 0}
        }

    def get_recommendations(self, experiment_id, user_id, **kwargs):
        """Get recommendations based on A/B split"""
        experiment = self.experiments[experiment_id]

        # Assign user to variant (consistent hashing)
        variant = 'a' if hash(str(user_id)) % 100 < experiment['split_ratio'] * 100 else 'b'

        if variant == 'a':
            recommendations = experiment['model_a'].recommend(user_id, **kwargs)
            experiment['results_a']['impressions'] += 1
        else:
            recommendations = experiment['model_b'].recommend(user_id, **kwargs)
            experiment['results_b']['impressions'] += 1

        return recommendations, variant

    def record_click(self, experiment_id, variant):
        """Record click event"""
        experiment = self.experiments[experiment_id]
        if variant == 'a':
            experiment['results_a']['clicks'] += 1
        else:
            experiment['results_b']['clicks'] += 1

    def get_results(self, experiment_id):
        """Get A/B test results"""
        experiment = self.experiments[experiment_id]

        results_a = experiment['results_a']
        results_b = experiment['results_b']

        ctr_a = results_a['clicks'] / results_a['impressions'] if results_a['impressions'] > 0 else 0
        ctr_b = results_b['clicks'] / results_b['impressions'] if results_b['impressions'] > 0 else 0

        return {
            'variant_a': {
                'impressions': results_a['impressions'],
                'clicks': results_a['clicks'],
                'ctr': ctr_a
            },
            'variant_b': {
                'impressions': results_b['impressions'],
                'clicks': results_b['clicks'],
                'ctr': ctr_b
            },
            'improvement': ((ctr_b - ctr_a) / ctr_a * 100) if ctr_a > 0 else 0
        }
```

---

## Best Practices

### 1. Handling Cold Start

```python
class ColdStartHandler:
    """Handle cold start problem"""

    def __init__(self, main_model, fallback_model):
        self.main_model = main_model
        self.fallback_model = fallback_model  # e.g., popularity-based

    def recommend(self, user_id, user_history_size, n=10):
        """Recommend with cold start handling"""

        # Threshold for cold start
        if user_history_size < 5:
            # Use fallback for new users
            recommendations = self.fallback_model.recommend(user_id, n)
        else:
            # Use main model
            recommendations = self.main_model.recommend(user_id, n)

        return recommendations

# Popularity-based fallback
class PopularityRecommender:
    """Recommend popular items"""

    def __init__(self, item_popularity):
        self.item_popularity = item_popularity

    def recommend(self, user_id, n=10):
        """Return most popular items"""
        sorted_items = sorted(
            self.item_popularity.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_items[:n]
```

### 2. Diversity & Exploration

```python
def diversify_recommendations(recommendations, item_features, lambda_param=0.5):
    """Maximal Marginal Relevance for diversity"""
    diversified = []
    candidates = recommendations.copy()

    # Add most relevant item first
    diversified.append(candidates.pop(0))

    while len(diversified) < len(recommendations) and candidates:
        max_mmr = -float('inf')
        best_idx = 0

        for idx, (item_id, score) in enumerate(candidates):
            # Relevance
            relevance = score

            # Similarity to already selected items
            max_similarity = max([
                cosine_similarity(
                    item_features[item_id].reshape(1, -1),
                    item_features[selected_item].reshape(1, -1)
                )[0][0]
                for selected_item, _ in diversified
            ])

            # MMR score
            mmr = lambda_param * relevance - (1 - lambda_param) * max_similarity

            if mmr > max_mmr:
                max_mmr = mmr
                best_idx = idx

        diversified.append(candidates.pop(best_idx))

    return diversified
```

---

## Summary

Recommendation systems predict user preferences using various techniques:

1. **Collaborative Filtering**: User/item similarity
2. **Content-Based**: Item feature similarity
3. **Matrix Factorization**: SVD, ALS for latent factors
4. **Deep Learning**: NCF, Wide & Deep, Two-Tower
5. **Hybrid**: Combine multiple approaches
6. **Context-Aware**: Factorization Machines

**Key Metrics:**
- Precision/Recall@K
- NDCG@K
- Hit Rate
- Mean Average Precision

**Production Considerations:**
- Cold start handling
- Scalability (FAISS for retrieval)
- A/B testing
- Diversity and exploration
- Real-time serving
