# ML System Design Interview Guide

**Complete guide to acing ML system design interviews at top tech companies**

> 🎯 **Framework-Based Approach** | 💡 **Real Interview Questions** | 🏗️ **Complete Solutions** | ⚡ **Common Pitfalls**

---

## Table of Contents

1. [Introduction](#introduction)
2. [Interview Format & Expectations](#interview-format)
3. [System Design Framework](#framework)
4. [Complete Example Solutions](#example-solutions)
5. [Common ML Systems](#common-systems)
6. [Trade-offs & Decision Making](#tradeoffs)
7. [Practice Problems](#practice-problems)
8. [Evaluation Rubric](#evaluation)
9. [Common Mistakes](#mistakes)
10. [Resources & Tips](#resources)

---

## 1. Introduction

### What is ML System Design?

ML System Design interviews test your ability to:
- Design end-to-end ML systems for real business problems
- Make informed trade-offs between competing constraints
- Understand production ML engineering
- Communicate technical decisions clearly
- Consider scalability, reliability, and maintainability

### Who Asks These Questions?

- **FAANG:** Google, Meta, Amazon, Netflix, Apple
- **Top Tech:** Microsoft, Uber, LinkedIn, Twitter/X, Airbnb
- **ML-Focused:** OpenAI, Anthropic, Cohere, Hugging Face
- **Startups:** Series B+ companies with ML products

### Interview Duration

- **Typical:** 45-60 minutes
- **Format:** Collaborative discussion with interviewer
- **Deliverable:** High-level architecture diagram + discussion

---

## 2. Interview Format & Expectations

### Typical Structure (60 minutes)

```
0-10 min:  Problem clarification & requirements gathering
10-25 min: High-level system architecture
25-40 min: Deep dives into specific components
40-55 min: Trade-offs, scaling, edge cases
55-60 min: Q&A and wrap-up
```

### What Interviewers Look For

✅ **Problem Understanding**
- Ask clarifying questions
- Identify constraints and requirements
- Understand success metrics

✅ **Technical Depth**
- Know ML fundamentals deeply
- Understand production ML challenges
- Make informed trade-offs

✅ **System Thinking**
- Consider end-to-end system
- Think about scale and reliability
- Address monitoring and maintenance

✅ **Communication**
- Explain reasoning clearly
- Draw clear diagrams
- Structure your thinking

✅ **Practical Experience**
- Reference real-world systems
- Discuss trade-offs pragmatically
- Show production awareness

### What You're NOT Expected to Know

❌ Exact API specifications
❌ Specific tool/library syntax
❌ Detailed implementation code
❌ Exact hyperparameters
❌ Latest research papers (unless explicitly relevant)

---

## 3. System Design Framework

### The PROBLEM Framework

**P** - Problem Definition & Requirements
**R** - Requirements (Functional & Non-Functional)
**O** - Outline High-Level Architecture
**B** - Break Down Components
**L** - List Trade-offs
**E** - Edge Cases & Scaling
**M** - Monitoring & Maintenance

Let's break down each step:

---

### Step 1: Problem Definition (5 minutes)

**Key Questions to Ask:**

**Business Context:**
- What problem are we solving?
- Who are the users?
- What's the business impact?
- What are the success metrics?

**Scale:**
- How many users? (MAU, DAU)
- How much data? (TPS, storage)
- What's the latency requirement?
- Geographic distribution?

**ML-Specific:**
- What type of ML problem is this? (classification, regression, ranking, generation)
- Do we have labeled data?
- How often does the model need to update?
- What's the acceptable accuracy?

**Example Clarification:**
```
Interviewer: "Design a recommendation system for an e-commerce platform."

You should ask:
- How many users? → 10M DAU
- How many products? → 100K
- Latency requirement? → <100ms p99
- Cold start? → Yes, new users daily
- Real-time or batch? → Hybrid
- Personalization level? → Highly personalized
- Privacy constraints? → GDPR compliant
```

---

### Step 2: Requirements (Functional & Non-Functional)

**Functional Requirements (What the system does):**

```
Example: Recommendation System
FR1: Recommend 10 relevant products for each user
FR2: Support real-time personalization
FR3: Handle new users (cold start)
FR4: Support multiple recommendation strategies
FR5: Allow A/B testing
```

**Non-Functional Requirements (How well it performs):**

```
NFR1: Latency - p99 < 100ms
NFR2: Availability - 99.9% uptime
NFR3: Scalability - 10M DAU, 1000 TPS
NFR4: Data freshness - Update within 24 hours
NFR5: Cost - Optimize for compute efficiency
```

**ML-Specific Requirements:**

```
MLR1: Accuracy - CTR > 5% (baseline 3%)
MLR2: Diversity - No filter bubble
MLR3: Freshness - Include recent products
MLR4: Explainability - Show why recommended
MLR5: Fairness - No demographic bias
```

---

### Step 3: High-Level Architecture

**Draw the System Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                         User Layer                           │
│  (Mobile App, Web App, API)                                  │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│                   API Gateway / Load Balancer                │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│              Recommendation Service (ML Serving)             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │Feature Store │  │ Model Server │  │  A/B Testing │      │
│  │   (Redis)    │  │(TF Serving)  │  │   Service    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│               Offline Training Pipeline                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Data Pipeline│  │Model Training│  │  Evaluation  │      │
│  │  (Airflow)   │  │  (PyTorch)   │  │   Pipeline   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│                       Data Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  User Events │  │  Product DB  │  │ Training Data│      │
│  │  (Kafka)     │  │ (PostgreSQL) │  │    (S3)      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└──────────────────────────────────────────────────────────────┘
```

**Key Components to Identify:**

1. **Data Collection & Storage**
   - How do we collect data?
   - Where do we store it?
   - Real-time vs batch?

2. **Feature Engineering**
   - What features do we need?
   - How do we compute them?
   - Online vs offline features?

3. **Model Training**
   - Training pipeline
   - Model architecture
   - Hyperparameter tuning

4. **Model Serving**
   - Online inference
   - Batch inference
   - Caching strategy

5. **Monitoring & Feedback**
   - Performance metrics
   - Data quality
   - Model drift

---

### Step 4: Deep Dive into Components

**For each component, discuss:**

#### A. Data Collection & Preprocessing

```python
# Example: User Event Stream
{
  "event_type": "product_view",
  "user_id": "user_123",
  "product_id": "prod_456",
  "timestamp": "2025-10-19T10:30:00Z",
  "session_id": "session_abc",
  "device_type": "mobile",
  "context": {
    "page": "search_results",
    "query": "wireless headphones",
    "position": 3
  }
}
```

**Discussion Points:**
- Event schema design
- Real-time streaming (Kafka, Kinesis)
- Batch processing (Spark, Airflow)
- Data quality checks
- Privacy and compliance (GDPR, CCPA)

#### B. Feature Engineering

**Types of Features:**

1. **User Features:**
   - Demographics (age, location, gender)
   - Behavioral (click history, purchase history)
   - Engagement metrics (session time, bounce rate)
   - Preferences (favorite categories, brands)

2. **Item Features:**
   - Product attributes (category, price, brand)
   - Popularity metrics (views, purchases, ratings)
   - Temporal features (trending score, seasonality)
   - Content features (title, description, images)

3. **Context Features:**
   - Time (hour, day of week, season)
   - Device (mobile, desktop, tablet)
   - Location (country, city)
   - Session (search query, browsing history)

4. **Interaction Features:**
   - User-item affinity
   - Co-occurrence patterns
   - Sequential patterns

**Feature Store:**
```python
# Feature Store Interface
class FeatureStore:
    def get_online_features(self, user_id, product_ids):
        """Get features for real-time serving (<10ms)"""
        pass

    def get_offline_features(self, user_ids, product_ids):
        """Get features for batch training"""
        pass

    def register_features(self, feature_definitions):
        """Register new features"""
        pass
```

**Trade-offs:**
- **Online vs Offline:** Latency vs complexity
- **Precomputed vs Real-time:** Storage vs freshness
- **Granularity:** Accuracy vs cost

#### C. Model Selection

**For Recommendation System, Consider:**

1. **Collaborative Filtering:**
   - Matrix Factorization (ALS, SVD)
   - Neural Collaborative Filtering
   - Pros: Works well with sparse data
   - Cons: Cold start problem

2. **Content-Based:**
   - Item similarity (TF-IDF, embeddings)
   - User preference profiles
   - Pros: No cold start, explainable
   - Cons: Limited serendipity

3. **Hybrid Models:**
   - Two-tower neural networks
   - Deep & Cross networks
   - Pros: Best of both worlds
   - Cons: More complex

4. **Sequential Models:**
   - RNN/LSTM for session-based
   - Transformer for long-term
   - Pros: Captures temporal dynamics
   - Cons: Training complexity

**Model Architecture Example:**

```python
class RecommendationModel(nn.Module):
    """
    Two-tower architecture:
    - User tower: encodes user features
    - Item tower: encodes item features
    - Dot product for similarity
    """

    def __init__(self, user_dim, item_dim, embedding_dim):
        super().__init__()

        # User tower
        self.user_encoder = nn.Sequential(
            nn.Linear(user_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

        # Item tower
        self.item_encoder = nn.Sequential(
            nn.Linear(item_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, user_features, item_features):
        user_emb = self.user_encoder(user_features)
        item_emb = self.item_encoder(item_features)

        # Dot product similarity
        scores = torch.sum(user_emb * item_emb, dim=1)
        return scores
```

#### D. Training Pipeline

**Components:**

1. **Data Preparation:**
   - Generate positive/negative samples
   - Create training/validation splits
   - Balance classes

2. **Model Training:**
   - Loss function selection
   - Optimizer choice
   - Hyperparameter tuning

3. **Evaluation:**
   - Offline metrics (AUC, Precision@K, NDCG)
   - Online metrics (CTR, conversion)
   - A/B testing

4. **Model Registry:**
   - Version control
   - Metadata tracking
   - Model comparison

**Training Schedule:**
```
Daily: Retrain with last 7 days of data
Weekly: Full retrain with historical data
On-demand: When performance degrades
```

#### E. Serving Architecture

**Two Modes:**

**1. Real-Time Serving (Online Inference):**
```
Request → Feature Extraction → Model Inference → Post-processing → Response
 (<5ms)        (<10ms)              (<50ms)           (<10ms)      (<100ms)
```

**Components:**
- Model server (TensorFlow Serving, TorchServe)
- Feature store (Redis, DynamoDB)
- Caching layer (Redis, Memcached)
- Load balancer

**2. Batch Serving (Offline Inference):**
```
Schedule → Generate candidates → Batch inference → Store results → Serve
(daily)      (millions)            (hours)           (Redis)      (ms)
```

**Serving Trade-offs:**

| Approach | Latency | Freshness | Cost | Complexity |
|----------|---------|-----------|------|------------|
| Real-time | High (ms) | Immediate | High | High |
| Batch | Low (precomputed) | Hours/Days | Low | Low |
| Hybrid | Medium | Minutes | Medium | Medium |

**Hybrid Approach (Recommended):**
```python
def get_recommendations(user_id, context):
    # 1. Get candidate pool (batch pre-computed)
    candidates = get_candidate_pool(user_id)  # 1000 items, from Redis

    # 2. Real-time re-ranking (personalization)
    user_features = get_user_features(user_id, context)  # Real-time
    scores = model.score(user_features, candidates)  # Online inference

    # 3. Apply business rules
    filtered = apply_business_rules(candidates, scores)

    # 4. Return top-K
    return filtered[:10]
```

#### F. Monitoring & Observability

**What to Monitor:**

1. **Model Performance:**
   - Online metrics (CTR, conversion)
   - Prediction distribution
   - Confidence scores
   - Model drift

2. **System Performance:**
   - Latency (p50, p95, p99)
   - Throughput (QPS)
   - Error rates
   - Resource utilization

3. **Data Quality:**
   - Feature distribution drift
   - Missing values
   - Outliers
   - Schema violations

4. **Business Metrics:**
   - Revenue impact
   - User engagement
   - A/B test results

**Alerting Thresholds:**
```yaml
alerts:
  - name: high_latency
    condition: p99_latency > 100ms
    action: page_on_call

  - name: low_ctr
    condition: ctr < baseline * 0.8
    action: rollback_model

  - name: model_drift
    condition: prediction_distribution_shift > 0.1
    action: trigger_retraining
```

---

### Step 5: Trade-offs Discussion

**Common Trade-offs in ML Systems:**

#### 1. Accuracy vs Latency

**Scenario:** Complex ensemble model vs simple model

| Metric | Ensemble | Simple Model |
|--------|----------|--------------|
| Accuracy | 95% | 92% |
| Latency | 200ms | 50ms |
| Cost | High | Low |

**Decision Framework:**
- If latency requirement is <100ms → Simple model
- If revenue gain from 3% accuracy > latency cost → Ensemble
- Consider hybrid: Simple model for real-time, ensemble for batch

#### 2. Freshness vs Cost

**Scenario:** How often to retrain?

| Frequency | Freshness | Cost | Complexity |
|-----------|-----------|------|------------|
| Real-time | Immediate | Very High | Very High |
| Hourly | Good | High | High |
| Daily | Acceptable | Medium | Medium |
| Weekly | Stale | Low | Low |

**Decision:**
- Start with daily retraining
- Monitor business metrics
- Increase frequency only if ROI justifies

#### 3. Personalization vs Privacy

**Scenario:** How much user data to collect?

```
More Data → Better Personalization → Privacy Concerns

Options:
1. Full tracking (best accuracy, privacy concerns)
2. Aggregated tracking (good accuracy, better privacy)
3. Minimal tracking (lower accuracy, best privacy)
4. Federated learning (good accuracy, privacy-preserving)
```

**Decision depends on:**
- Regulatory requirements (GDPR, CCPA)
- User trust and brand
- Competitive advantage
- Implementation cost

#### 4. Exploration vs Exploitation

**Scenario:** Recommendation diversity

```
Pure Exploitation:
+ Maximize immediate CTR
- Filter bubble, poor long-term engagement

Pure Exploration:
+ Discover new preferences
- Poor immediate metrics

Balanced Approach:
- ε-greedy: 90% exploit, 10% explore
- Thompson sampling
- Upper confidence bound (UCB)
```

#### 5. Online vs Offline Evaluation

**Challenge:** Offline metrics don't always predict online performance

| Approach | Pros | Cons |
|----------|------|------|
| Offline only | Fast, cheap | May not reflect reality |
| Online only | Ground truth | Slow, expensive, risky |
| Hybrid | Balanced | Requires both systems |

**Best Practice:**
1. Offline evaluation for initial filtering
2. Shadow mode for safety check
3. Small A/B test for validation
4. Full rollout with monitoring

---

### Step 6: Scaling Considerations

**Horizontal Scaling:**

```
User Growth: 1M → 10M → 100M users

Solutions:
1. Sharding: Partition users across servers
2. Caching: Aggressive caching for popular items
3. CDN: Geographical distribution
4. Load balancing: Distribute traffic
```

**Vertical Scaling (Model Complexity):**

```
Simple model → Complex model

Challenges:
1. Training time: Distributed training
2. Serving latency: Model compression
3. Storage: Efficient serialization
4. Cost: Resource optimization
```

**Data Scaling:**

```
1GB → 1TB → 1PB of training data

Solutions:
1. Distributed training (PyTorch DDP, Horovod)
2. Data sampling strategies
3. Feature selection
4. Incremental learning
```

**Global Scaling:**

```
Single region → Multiple regions → Global

Considerations:
1. Data residency (GDPR)
2. Latency (edge computing)
3. Model consistency
4. Cost optimization
```

---

### Step 7: Edge Cases & Failure Modes

**Common Edge Cases:**

1. **Cold Start:**
   - New users: Use demographic-based or popular items
   - New items: Use content-based filtering
   - Hybrid: Gradually transition to personalized

2. **Sparse Data:**
   - Users with few interactions
   - Seasonal products
   - Solution: Regularization, transfer learning

3. **Adversarial Behavior:**
   - Click fraud
   - Fake reviews
   - Bot traffic
   - Solution: Fraud detection, rate limiting

4. **System Failures:**
   - Model server down → Fallback to rule-based
   - Feature store down → Use cached features
   - Database down → Serve popular items

**Fallback Strategy:**

```python
def get_recommendations(user_id):
    try:
        # Try personalized recommendations
        recs = personalized_recommender(user_id)
        if validate_recommendations(recs):
            return recs
    except ModelServerError:
        log_error("Model server unavailable")

    try:
        # Fallback 1: Collaborative filtering
        recs = collaborative_filter(user_id)
        if validate_recommendations(recs):
            return recs
    except Exception:
        log_error("Collaborative filtering failed")

    # Fallback 2: Popular items
    return get_popular_items(user_id.region)
```

---

## 4. Complete Example Solutions

### Example 1: Design YouTube Video Recommendation System

**Problem Statement:**
Design a recommendation system for YouTube that suggests videos to users on the home page.

**Step 1: Clarification (5 minutes)**

**Q:** How many users and videos?
**A:** 2B users, 800M videos, 500M videos uploaded daily

**Q:** Latency requirement?
**A:** <200ms for homepage load

**Q:** Personalization level?
**A:** Highly personalized based on watch history

**Q:** Cold start?
**A:** Yes, new users daily

**Q:** Main objective?
**A:** Maximize watch time

---

**Step 2: Requirements**

**Functional:**
- FR1: Recommend 20 videos for homepage
- FR2: Real-time personalization based on current session
- FR3: Support new users (cold start)
- FR4: Diversity in recommendations
- FR5: Filter inappropriate content

**Non-Functional:**
- NFR1: Latency p99 < 200ms
- NFR2: 99.99% availability
- NFR3: Support 2B users, 100K QPS
- NFR4: Real-time updates within 1 hour

**ML-Specific:**
- MLR1: Watch time > baseline
- MLR2: CTR > 10%
- MLR3: Session length > 20 minutes
- MLR4: Low bounce rate

---

**Step 3: High-Level Architecture**

```
┌──────────────────────────────────────────────────────────────┐
│                       YouTube Client                          │
│                  (Web, Mobile, Smart TV)                      │
└──────────────────────┬───────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────┐
│                   API Gateway (Global)                        │
│              (Geographic Load Balancing)                      │
└──────────────────────┬───────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
┌───────▼────────┐          ┌─────────▼────────┐
│ Candidate      │          │  Ranking         │
│ Generation     │          │  Service         │
│ (Batch/Daily)  │          │  (Real-time)     │
│                │          │                  │
│ - CF model     │          │ - Neural ranker  │
│ - Content      │          │ - Context-aware  │
│ - Trending     │          │ - Multi-objective│
│                │          │                  │
│ Output:        │          │ Input: ~1000     │
│ ~1000 videos   │          │ Output: Top 20   │
└────────────────┘          └──────────────────┘
        │                             │
        └──────────────┬──────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────┐
│                    Feature Store                              │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐             │
│  │ User       │  │ Video      │  │ Context    │             │
│  │ Features   │  │ Features   │  │ Features   │             │
│  │ (Redis)    │  │ (Bigtable) │  │ (Memory)   │             │
│  └────────────┘  └────────────┘  └────────────┘             │
└──────────────────────┬───────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────┐
│                 Offline Training                              │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐             │
│  │ Data       │  │ Training   │  │ Evaluation │             │
│  │ Pipeline   │  │ (TPU pods) │  │ & A/B Test │             │
│  │ (MapReduce)│  │            │  │            │             │
│  └────────────┘  └────────────┘  └────────────┘             │
└──────────────────────┬───────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────┐
│                   Data Warehouse                              │
│  - User watch history (Bigtable)                             │
│  - Video metadata (Spanner)                                   │
│  - User profiles (MySQL)                                      │
│  - Training datasets (GCS)                                    │
└───────────────────────────────────────────────────────────────┘
```

---

**Step 4: Component Deep Dive**

**A. Candidate Generation (Retrieval Stage)**

**Purpose:** Narrow down 800M videos to ~1000 candidates

**Approaches:**

1. **Collaborative Filtering:**
```
User u watched videos V1, V2, V3
Find similar users U' who watched V1, V2, V3
Recommend videos that U' watched but u hasn't
```

2. **Content-Based:**
```
Video features: title, description, category, tags
User preferences: preferred categories, channels
Match videos similar to user's past watches
```

3. **Trending/Popular:**
```
Region-specific trending videos
Time-decay factor (recent = higher weight)
Virality score (view velocity)
```

**Implementation:**
```python
def generate_candidates(user_id, n_candidates=1000):
    candidates = []

    # 1. Collaborative filtering (40%)
    cf_candidates = collaborative_filtering(user_id, n=400)
    candidates.extend(cf_candidates)

    # 2. Content-based (30%)
    cb_candidates = content_based(user_id, n=300)
    candidates.extend(cb_candidates)

    # 3. Trending (20%)
    trending = get_trending_videos(user_id.region, n=200)
    candidates.extend(trending)

    # 4. Subscriptions (10%)
    subscriptions = get_subscription_uploads(user_id, n=100)
    candidates.extend(subscriptions)

    # Deduplicate
    candidates = list(set(candidates))

    return candidates[:n_candidates]
```

**B. Ranking Stage**

**Purpose:** Rank 1000 candidates to find best 20

**Model Architecture:**
```
Two-stage ranking:

Stage 1: Light ranker (fast, approximate)
- Input: 1000 candidates
- Model: Shallow neural network
- Latency: <50ms
- Output: Top 200

Stage 2: Heavy ranker (slow, accurate)
- Input: 200 candidates
- Model: Deep neural network
- Latency: <100ms
- Output: Top 20
```

**Feature Engineering:**

```python
# User features
user_features = {
    'demographics': ['age', 'gender', 'location'],
    'preferences': [
        'favorite_categories',  # [music, gaming, education]
        'watch_time_by_category',  # {music: 100hrs, gaming: 50hrs}
        'avg_video_length',  # 10 minutes
    ],
    'behavior': [
        'total_watch_time',  # 1000 hours
        'avg_daily_sessions',  # 3
        'preferred_time_of_day',  # [18:00-22:00]
    ],
    'recent_activity': [
        'last_10_videos_watched',
        'last_10_searches',
        'current_session_context'
    ]
}

# Video features
video_features = {
    'metadata': [
        'category',  # music
        'tags',  # [pop, 2024, official]
        'language',  # en
        'duration',  # 240 seconds
    ],
    'quality': [
        'resolution',  # 1080p
        'fps',  # 60
        'audio_quality',  # 320kbps
    ],
    'engagement': [
        'views',  # 1M
        'likes_ratio',  # 0.95
        'comments',  # 10K
        'shares',  # 5K
        'watch_time_ratio',  # 0.8 (users watch 80% on average)
    ],
    'temporal': [
        'upload_date',  # 2024-10-19
        'freshness_score',  # high if recent
        'trending_score',  # view velocity
    ]
}

# Context features
context_features = {
    'device': 'mobile',  # vs desktop, TV
    'network': '4G',  # vs WiFi
    'time': '20:30',
    'location': 'home',  # inferred
    'session': {
        'duration': '15_minutes',
        'videos_watched': 3,
        'searches_made': 1
    }
}
```

**Ranking Model:**

```python
class YouTubeRanker(nn.Module):
    """
    Multi-tower neural network for video ranking.
    """

    def __init__(self):
        super().__init__()

        # User tower
        self.user_tower = nn.Sequential(
            nn.Linear(user_feature_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256)
        )

        # Video tower
        self.video_tower = nn.Sequential(
            nn.Linear(video_feature_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256)
        )

        # Context tower
        self.context_tower = nn.Sequential(
            nn.Linear(context_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        # Interaction layer
        self.interaction = nn.Sequential(
            nn.Linear(256 + 256 + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Watch time prediction
        )

    def forward(self, user_features, video_features, context_features):
        user_emb = self.user_tower(user_features)
        video_emb = self.video_tower(video_features)
        context_emb = self.context_tower(context_features)

        # Concatenate all embeddings
        combined = torch.cat([user_emb, video_emb, context_emb], dim=1)

        # Predict watch time
        watch_time = self.interaction(combined)

        return watch_time
```

**Multi-Objective Optimization:**

YouTube cares about multiple objectives:
1. **Watch time** (primary)
2. **CTR** (secondary)
3. **User satisfaction** (likes, shares)
4. **Diversity** (avoid filter bubble)

```python
# Weighted multi-objective loss
loss = (
    0.5 * watch_time_loss +
    0.2 * ctr_loss +
    0.2 * satisfaction_loss +
    0.1 * diversity_loss
)
```

**C. Post-Processing**

After ranking, apply business rules:

```python
def post_process(ranked_videos, user_id):
    filtered = []

    for video in ranked_videos:
        # 1. Remove already watched
        if video.id in user_watch_history:
            continue

        # 2. Content policy filter
        if not passes_content_policy(video):
            continue

        # 3. Diversity: No more than 2 from same channel
        if channel_count[video.channel_id] >= 2:
            continue

        # 4. Freshness: Include at least 3 recent uploads
        if video.upload_date > (now - 1day):
            freshness_count += 1

        # 5. Category diversity
        if category_count[video.category] >= 5:
            continue

        filtered.append(video)

        if len(filtered) >= 20:
            break

    return filtered
```

---

**Step 5: Trade-offs**

**1. Candidate Generation: Batch vs Real-time**

| Approach | Pros | Cons |
|----------|------|------|
| Daily batch | Fast serving, low cost | Stale recommendations |
| Hourly batch | More fresh | Higher cost |
| Real-time | Most fresh | Very high cost, complexity |

**Decision:** Hybrid
- Batch candidate generation (daily)
- Real-time re-ranking with fresh features
- Real-time incorporation of current session

**2. Model Complexity vs Latency**

Simple model: 50ms, 82% accuracy
Complex model: 150ms, 87% accuracy

**Analysis:**
- 200ms budget
- 150ms still within budget
- 5% accuracy gain → significant watch time increase
- **Decision:** Use complex model

**3. Personalization vs Privacy**

**Options:**
- Full tracking: Best recommendations, privacy concerns
- Aggregated tracking: Good recommendations, better privacy
- On-device ML: Privacy-preserving, limited capabilities

**Decision:**
- Use full tracking with consent
- Offer privacy controls to users
- Comply with GDPR/CCPA

---

**Step 6: Scaling**

**Current:** 2B users, 100K QPS
**Future:** 5B users, 500K QPS (5x growth)

**Scaling Strategy:**

1. **Horizontal Scaling:**
```
- Shard users by user_id hash
- 100 model servers → 500 model servers
- Geographic distribution (edge computing)
```

2. **Caching:**
```
- Cache candidate pools (99% hit rate)
- Cache feature values (1 hour TTL)
- CDN for video metadata
```

3. **Model Optimization:**
```
- Quantization (FP32 → INT8)
- Pruning (remove 30% of weights)
- Distillation (student-teacher)
- Result: 3x faster, 2% accuracy drop
```

4. **Database Optimization:**
```
- Read replicas for user data
- Partitioning for video metadata
- In-memory caching (Redis)
```

---

**Step 7: Monitoring**

**Metrics Dashboard:**

```yaml
Model Metrics:
  - Watch time per session: 23 minutes (target: >20)
  - CTR: 12% (target: >10%)
  - Average view duration: 65% (target: >60%)
  - Daily active users: 1.8B

System Metrics:
  - p99 latency: 180ms (target: <200ms)
  - QPS: 95K (capacity: 100K)
  - Error rate: 0.01% (target: <0.1%)
  - Cache hit rate: 99.2%

Data Quality:
  - Feature freshness: <1 hour
  - Missing feature rate: <0.1%
  - Anomaly detection: 0 alerts

Business Metrics:
  - Revenue per user: $2.50/month
  - Ad engagement: 15% higher than baseline
  - User satisfaction: 4.2/5 stars
```

**Alerts:**
- Latency > 200ms for 5 minutes → page on-call
- CTR drops >20% → automatic rollback
- Error rate > 1% → incident response

---

This completes the YouTube recommendation system design!

**Key Takeaways:**
1. ✅ Two-stage retrieval and ranking
2. ✅ Multi-objective optimization
3. ✅ Hybrid batch + real-time approach
4. ✅ Comprehensive feature engineering
5. ✅ Robust monitoring and fallbacks

---

[Continue to more example problems...]
