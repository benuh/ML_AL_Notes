# ML System Design Interview Guide - Part 2

**Practice Problems, Evaluation Rubric, Tips & Tricks**

> Continuation of [ML_SYSTEM_DESIGN_INTERVIEW_GUIDE.md](./ML_SYSTEM_DESIGN_INTERVIEW_GUIDE.md)

---

## 5. Common ML Systems (Quick Reference)

### Quick Solution Patterns

#### Pattern 1: Recommendation Systems

**Examples:** YouTube, Netflix, Amazon, Spotify
**Key Components:**
- Two-stage: Candidate generation → Ranking
- Collaborative filtering + content-based
- Real-time personalization
- Cold start handling

**Architecture:**
```
User → Candidate Generation (batch) → Ranking (real-time) → Business rules → Top-K
```

**Metrics:** CTR, watch time, conversion, diversity

---

#### Pattern 2: Search & Ranking

**Examples:** Google Search, Bing, Amazon Product Search
**Key Components:**
- Query understanding (NLP)
- Document retrieval (IR)
- Learning to rank (LTR)
- Personalization

**Architecture:**
```
Query → Query rewriting → Retrieval → Ranking → Personalization → Results
```

**Metrics:** NDCG, MRR, precision@K, latency

---

#### Pattern 3: Computer Vision

**Examples:** Face recognition, object detection, image classification
**Key Components:**
- Image preprocessing
- CNN/Vision Transformer
- Post-processing
- Model deployment

**Architecture:**
```
Image → Preprocessing → Feature extraction → Classification/Detection → Results
```

**Metrics:** Accuracy, precision, recall, F1, inference time

---

#### Pattern 4: NLP/Text Understanding

**Examples:** Sentiment analysis, chatbots, translation
**Key Components:**
- Text preprocessing
- Embeddings (BERT, GPT)
- Task-specific head
- Generation/Classification

**Architecture:**
```
Text → Tokenization → Embedding → Transformer → Task head → Output
```

**Metrics:** Accuracy, BLEU, ROUGE, perplexity

---

#### Pattern 5: Fraud/Anomaly Detection

**Examples:** Credit card fraud, spam detection, intrusion detection
**Key Components:**
- Real-time feature engineering
- Ensemble models
- Rule engine
- Feedback loop

**Architecture:**
```
Event → Feature extraction → Rules check → ML model → Decision → Feedback
```

**Metrics:** Precision, recall, F1, false positive rate, latency

---

#### Pattern 6: Real-Time Bidding/Pricing

**Examples:** Ad auctions, dynamic pricing, surge pricing
**Key Components:**
- Real-time prediction
- Multi-armed bandit
- Auction mechanism
- Budget optimization

**Architecture:**
```
Request → Context → Value prediction → Bid optimization → Auction → Result
```

**Metrics:** Revenue, fill rate, ROI, latency (<50ms)

---

## 6. Trade-offs & Decision Making

### The Trade-off Matrix

Every ML system design involves trade-offs. Here's how to think about them:

#### 1. Accuracy vs Latency

**Scenario:** Choose between complex vs simple model

**Decision Framework:**
```python
def choose_model(latency_budget, accuracy_requirement, cost_budget):
    if latency_budget < 50ms:
        # Must use simple model
        return SimpleModel()

    elif latency_budget < 200ms:
        # Can use medium complexity
        if accuracy_requirement > 90%:
            # Try model optimization
            return OptimizedComplexModel()  # Quantized, pruned
        else:
            return MediumModel()

    else:
        # Latency not critical
        if cost_budget == "low":
            return CachedBatchPredictions()
        else:
            return MostAccurateModel()
```

**Real Example:**
- **Ad CTR Prediction:** 10ms budget → Logistic regression
- **Medical Diagnosis:** 5s acceptable → Deep ensemble
- **Recommendation:** 100ms → Neural network with caching

---

#### 2. Batch vs Real-time Processing

| Dimension | Batch | Real-time | Hybrid |
|-----------|-------|-----------|---------|
| **Latency** | Hours/days | Milliseconds | Seconds |
| **Freshness** | Stale | Fresh | Medium |
| **Cost** | Low | High | Medium |
| **Complexity** | Low | High | Medium |
| **Use Case** | Reports, analytics | Fraud, ads | Recommendations |

**Decision Tree:**
```
Is immediate decision required?
├─ Yes: Real-time (fraud detection, ad serving)
└─ No: Can results be precomputed?
   ├─ Yes: Batch (daily reports, email campaigns)
   └─ No: Hybrid (recommendations with precomputed candidates)
```

---

#### 3. Model Complexity Spectrum

**Simple → Complex:**

```
Logistic Regression
    ↓ (+10% accuracy, +1ms latency)
Random Forest
    ↓ (+5% accuracy, +5ms latency)
Gradient Boosting (XGBoost)
    ↓ (+3% accuracy, +20ms latency)
Neural Network
    ↓ (+2% accuracy, +50ms latency)
Deep Ensemble
    ↓ (+1% accuracy, +100ms latency)
```

**Rule of Thumb:**
- Start simple (establish baseline)
- Add complexity only if:
  - ROI justifies it
  - Latency budget allows
  - Team can maintain it

---

#### 4. Feature Engineering: Online vs Offline

**Offline Features (Precomputed):**
```python
# Computed during batch processing
offline_features = {
    'user_avg_purchase_30d': 45.50,  # Average over 30 days
    'user_favorite_category': 'electronics',
    'user_lifetime_value': 1250.00,
    'product_popularity_score': 0.85
}
```
**Pros:** Fast to retrieve, complex aggregations
**Cons:** Can be stale (hours/days old)

**Online Features (Real-time):**
```python
# Computed at request time
online_features = {
    'current_cart_value': 89.99,
    'session_duration_minutes': 12,
    'items_viewed_in_session': 5,
    'time_of_day': '14:30',
    'device': 'mobile'
}
```
**Pros:** Fresh, context-aware
**Cons:** Latency impact, limited complexity

**Hybrid Approach (Recommended):**
```python
def get_features(user_id, context):
    # Fast: retrieve precomputed features
    offline = feature_store.get_offline(user_id)  # <5ms

    # Real-time: compute context features
    online = compute_online_features(context)  # <10ms

    # Combine
    return {**offline, **online}  # Total: <15ms
```

---

#### 5. Exploration vs Exploitation

**The Dilemma:**
```
Exploitation (use best known option)
    ↓
Immediate Revenue $$$$
Short-term optimization
Risk of local optimum

    vs

Exploration (try new options)
    ↓
Learning & Discovery
Long-term optimization
Better global optimum
```

**Solutions:**

**1. ε-Greedy:**
```python
def select_action(epsilon=0.1):
    if random.random() < epsilon:
        return explore()  # Random action
    else:
        return exploit()  # Best known action
```
**Use:** Simple, easy to implement
**Tune:** ε=0.05 to 0.20 typical

**2. Thompson Sampling:**
```python
def thompson_sampling(arms):
    # Sample from posterior distribution
    samples = [arm.sample_posterior() for arm in arms]
    return argmax(samples)
```
**Use:** Bayesian, naturally balances explore/exploit
**Pros:** Optimal in theory, adapts automatically

**3. Upper Confidence Bound (UCB):**
```python
def ucb(arm, total_trials, confidence=2):
    exploitation = arm.mean_reward
    exploration = sqrt(confidence * log(total_trials) / arm.trials)
    return exploitation + exploration
```
**Use:** Deterministic, provable guarantees
**Pros:** No hyperparameters to tune

**Recommendation:**
- **Start:** ε-greedy (simple)
- **Production:** Thompson sampling (optimal)
- **High-stakes:** UCB (safe, provable)

---

### 7. Practical Tips & Tricks

#### Tip 1: Start with Clarifying Questions (5 minutes)

**Template:**
```
Scale:
- How many users/requests?
- How much data?
- Geographic distribution?

Requirements:
- Latency requirement?
- Accuracy requirement?
- Cost constraints?

ML-Specific:
- Do we have labeled data?
- How often to retrain?
- Cold start problem?

Business:
- What's the success metric?
- What's the business impact?
- Are there regulations (privacy, fairness)?
```

**Pro Tip:** Write these down! Shows structured thinking.

---

#### Tip 2: Draw the Architecture (10 minutes)

**Components to Include:**
```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
┌──────▼──────┐
│  API/LB     │
└──────┬──────┘
       │
┌──────▼──────────────────────────┐
│  Serving Layer                  │
│  ┌──────┐  ┌──────┐  ┌──────┐  │
│  │Model │  │Cache │  │A/B   │  │
│  │Server│  │      │  │Test  │  │
│  └──────┘  └──────┘  └──────┘  │
└──────┬──────────────────────────┘
       │
┌──────▼──────────────────────────┐
│  Offline Training               │
│  ┌──────┐  ┌──────┐  ┌──────┐  │
│  │Data  │  │Train │  │Eval  │  │
│  └──────┘  └──────┘  └──────┘  │
└──────┬──────────────────────────┘
       │
┌──────▼──────┐
│  Data Store │
└─────────────┘
```

**Pro Tip:** Use boxes and arrows. Keep it clean and labeled.

---

#### Tip 3: Discuss Trade-offs Explicitly

**Bad Answer:**
"We'll use a neural network for better accuracy."

**Good Answer:**
"We have three options:
1. **Logistic regression:** 50ms latency, 85% accuracy, cheap
2. **XGBoost:** 100ms latency, 90% accuracy, moderate cost
3. **Neural network:** 200ms latency, 92% accuracy, expensive

Given our 150ms latency budget and accuracy requirement of 88%, I'd choose **XGBoost** because:
- Meets latency requirement (100ms < 150ms)
- Exceeds accuracy requirement (90% > 88%)
- Lower cost than neural network
- Team familiar with XGBoost (operational advantage)"

---

#### Tip 4: Think About Failure Modes

**Always discuss:**

1. **What if the model server is down?**
   → Fallback to rule-based system

2. **What if features are missing?**
   → Default values or cached features

3. **What if prediction is very slow?**
   → Timeout and serve default

4. **What if data quality degrades?**
   → Data quality monitoring + alerts

**Example Fallback Cascade:**
```
1. Try: Personalized ML model (best accuracy)
   ↓ (if fails)
2. Try: Non-personalized ML model (good accuracy)
   ↓ (if fails)
3. Try: Rule-based system (acceptable accuracy)
   ↓ (if fails)
4. Return: Default/popular items (safe)
```

---

#### Tip 5: Mention Monitoring

**Always include a monitoring section!**

**What to monitor:**
```yaml
Model Performance:
  - Online metrics (CTR, conversion, revenue)
  - Offline metrics (AUC, accuracy)
  - Prediction distribution
  - Model drift

System Performance:
  - Latency (p50, p95, p99)
  - Throughput (QPS)
  - Error rate
  - Resource utilization

Data Quality:
  - Feature distribution
  - Missing values
  - Schema changes
  - Outliers

Business Metrics:
  - Revenue impact
  - User satisfaction
  - A/B test results
```

**Pro Tip:** Discuss when to alert and when to auto-rollback.

---

#### Tip 6: Consider the Full ML Lifecycle

**Don't just design the model!** Consider:

```
1. Data Collection
   - How do we collect training data?
   - Data quality checks?
   - Privacy compliance?

2. Feature Engineering
   - Online vs offline features?
   - Feature store?
   - Feature freshness?

3. Model Training
   - Training frequency?
   - Distributed training?
   - Hyperparameter tuning?

4. Model Evaluation
   - Offline metrics?
   - Online A/B testing?
   - Validation strategy?

5. Model Deployment
   - Blue-green deployment?
   - Canary releases?
   - Rollback strategy?

6. Monitoring & Maintenance
   - Performance tracking?
   - Drift detection?
   - Retraining triggers?
```

---

#### Tip 7: Use Numbers!

**Bad:** "We need a fast system"
**Good:** "We need p99 latency < 100ms to serve 10K QPS"

**Bad:** "We have a lot of data"
**Good:** "We have 100TB of training data with 1B examples"

**Bad:** "The model should be accurate"
**Good:** "We need 95% precision at 85% recall, improving on 90%/80% baseline"

**Numbers make your answer concrete and credible.**

---

## 8. Evaluation Rubric

### What Interviewers Grade You On

#### 1. Problem Understanding (20%)

**Excellent:**
- Asks insightful clarifying questions
- Identifies key constraints
- Understands business context
- Defines success metrics clearly

**Poor:**
- Jumps to solution immediately
- Misunderstands requirements
- Doesn't ask questions
- Vague success metrics

---

#### 2. System Design (30%)

**Excellent:**
- Complete end-to-end architecture
- Clear component separation
- Scalable design
- Considers data flow
- Includes monitoring

**Poor:**
- Missing critical components
- Unclear architecture
- Doesn't scale
- No data flow mentioned
- No monitoring

---

#### 3. ML Knowledge (25%)

**Excellent:**
- Appropriate algorithm choices
- Discusses multiple approaches
- Understands ML fundamentals
- Considers training and serving
- Mentions evaluation metrics

**Poor:**
- Wrong algorithm choice
- Only one approach
- Weak ML fundamentals
- Ignores training/serving split
- No metrics mentioned

---

#### 4. Trade-offs (15%)

**Excellent:**
- Explicitly discusses trade-offs
- Weighs pros and cons
- Makes justified decisions
- Considers multiple dimensions
- Relates to business impact

**Poor:**
- No trade-offs discussed
- Arbitrary decisions
- Doesn't justify choices
- One-dimensional thinking
- Ignores business context

---

#### 5. Communication (10%)

**Excellent:**
- Clear and structured
- Uses diagrams
- Good pacing
- Engages interviewer
- Summarizes key points

**Poor:**
- Disorganized rambling
- No visual aids
- Too fast or too slow
- Doesn't engage
- Misses key points

---

### Scoring Example

**Interview Question:** Design a spam detection system for email

**Candidate A (Strong):**

✅ **Problem Understanding:** Asked about email volume (10M/day), false positive tolerance (<0.1%), latency requirement (<1s)

✅ **System Design:** Drew complete architecture with feature extraction, model serving, feedback loop

✅ **ML Knowledge:** Discussed Naive Bayes baseline, then gradient boosting, explained why ensemble approach

✅ **Trade-offs:** Compared real-time vs batch processing, chose hybrid approach with justification

✅ **Communication:** Clear diagrams, engaged interviewer with questions

**Score: 85/100** → Strong Hire

---

**Candidate B (Weak):**

❌ **Problem Understanding:** Didn't ask about scale or requirements

❌ **System Design:** Vague architecture, missing components like monitoring

❌ **ML Knowledge:** Suggested "deep learning" without specifics

❌ **Trade-offs:** No discussion of alternatives or trade-offs

❌ **Communication:** Disorganized, no diagrams

**Score: 40/100** → No Hire

---

## 9. Common Mistakes

### Mistake 1: Jumping to Solution Too Quickly

**Bad:**
```
Interviewer: "Design a recommendation system"
Candidate: "We'll use collaborative filtering with matrix factorization..."
```

**Good:**
```
Interviewer: "Design a recommendation system"
Candidate: "Great! Let me ask a few clarifying questions first:
- How many users and items do we have?
- What's the latency requirement?
- Do we have explicit ratings or implicit feedback?
- What's the cold start situation?
..."
```

**Why it matters:** Shows structured thinking and avoids solving the wrong problem.

---

### Mistake 2: Over-Engineering

**Bad:**
"We'll use a distributed TensorFlow cluster with 100 GPUs, train a 10-billion parameter transformer model, deploy with Kubernetes across 20 regions..."

**Good:**
"Let's start with a simple baseline:
1. **Phase 1:** Popularity-based recommendations (no ML)
2. **Phase 2:** Collaborative filtering (simple ML)
3. **Phase 3:** Neural network if needed (complex ML)

Given our 10K users, Phase 2 should be sufficient. We can add complexity later if needed."

**Why it matters:** Real systems are built iteratively. Show practical thinking.

---

### Mistake 3: Ignoring Non-ML Components

**Bad:** Only talks about model architecture

**Good:** Discusses:
- Data collection and storage
- Feature engineering pipeline
- Model serving infrastructure
- Monitoring and alerts
- A/B testing framework
- Feedback loops

**Why it matters:** ML is <20% of the system. The rest is infrastructure.

---

### Mistake 4: No Numbers or Specifics

**Bad:**
- "We need a lot of storage"
- "The system should be fast"
- "We want high accuracy"

**Good:**
- "We need 100TB of storage for 1B training examples"
- "We need p99 latency < 100ms"
- "We want 95% precision at 85% recall"

**Why it matters:** Concrete numbers show you can think quantitatively.

---

### Mistake 5: Forgetting About Production

**Bad:** Focuses only on model accuracy

**Good:** Also discusses:
- Model deployment strategy
- A/B testing
- Monitoring and alerting
- Model drift detection
- Rollback procedures
- Cost optimization

**Why it matters:** Production ML is about reliability, not just accuracy.

---

### Mistake 6: Not Considering Edge Cases

**Bad:** Designs for happy path only

**Good:** Discusses:
- Cold start (new users/items)
- Sparse data
- Adversarial behavior
- System failures
- Data quality issues
- Bias and fairness

**Why it matters:** Real systems must handle edge cases gracefully.

---

### Mistake 7: Poor Time Management

**Bad Timeline:**
```
0-30 min: Talking about model architecture
30-45 min: Still on model details
45-60 min: Rushing through everything else
```

**Good Timeline:**
```
0-10 min: Clarification and requirements
10-25 min: High-level architecture
25-40 min: Component deep dives
40-55 min: Trade-offs and scaling
55-60 min: Wrap-up and Q&A
```

**Why it matters:** Shows ability to prioritize and structure discussion.

---

## 10. Practice Problems

### Problem Set 1: Recommendation Systems

**1. Design Netflix Movie Recommendations**
- 200M users, 10K movies
- Latency: <200ms
- Goal: Maximize watch time
- Challenge: Cold start for new users

**2. Design LinkedIn Job Recommendations**
- 800M users, 20M jobs
- Latency: <500ms
- Goal: Application rate
- Challenge: Job freshness (new jobs daily)

**3. Design Spotify Music Recommendations**
- 500M users, 80M songs
- Latency: <100ms
- Goal: Listening time
- Challenge: Music taste diversity

---

### Problem Set 2: Search & Ranking

**4. Design Google Search Ranking**
- 8B web pages
- Latency: <300ms
- Goal: User satisfaction (clicks, dwell time)
- Challenge: Adversarial SEO

**5. Design Amazon Product Search**
- 500M products
- Latency: <200ms
- Goal: Purchase conversion
- Challenge: Personalization + relevance

---

### Problem Set 3: Computer Vision

**6. Design Face Recognition System (Unlock Phone)**
- Latency: <500ms
- Goal: >99.9% accuracy
- Challenge: Security (false accepts <0.01%)

**7. Design Autonomous Vehicle Object Detection**
- Real-time (30 FPS)
- Latency: <33ms
- Goal: Detect cars, pedestrians, signs
- Challenge: Safety critical

---

### Problem Set 4: NLP

**8. Design Email Auto-Reply Suggestions**
- Latency: <200ms
- Goal: 30% suggestion acceptance rate
- Challenge: Personalization + context

**9. Design Chatbot for Customer Support**
- Latency: <1s
- Goal: 80% question resolution
- Challenge: Multi-turn dialogue

---

### Problem Set 5: Fraud/Anomaly Detection

**10. Design Credit Card Fraud Detection**
- Real-time (per transaction)
- Latency: <50ms
- Goal: >90% fraud detection, <2% false positives
- Challenge: Evolving fraud patterns

**11. Design Content Moderation System**
- 1M posts/hour
- Latency: <5s
- Goal: >95% harmful content detected
- Challenge: Multi-modal (text + image + video)

---

## 11. Resources & Further Reading

### Books
- **"Designing Machine Learning Systems"** by Chip Huyen
- **"Machine Learning System Design Interview"** by Ali Aminian
- **"Building Machine Learning Powered Applications"** by Emmanuel Ameisen

### Online Resources
- **EducativeML System Design Course
- **ByteByteGo** - System design fundamentals
- **ML@CMU** YouTube channel

### Real-World Systems (Read These!)
- **YouTube Recommendations** (Google AI Blog)
- **Amazon Product Recommendations** (AWS ML Blog)
- **Uber's ML Platform** (Uber Engineering Blog)
- **Netflix Recommendations** (Netflix Tech Blog)

### Practice Platforms
- **Pramp** - Mock interviews
- **Interviewing.io** - Real interviews with engineers
- **LeetCode** - System design discussion

---

## Final Tips for Success

### Before the Interview

1. **Study common ML systems**
   - Read about YouTube, Netflix, Amazon recommendations
   - Understand search ranking, fraud detection
   - Know the components and trade-offs

2. **Practice drawing architectures**
   - Use whiteboard or digital tools
   - Practice explaining while drawing
   - Get feedback from peers

3. **Review ML fundamentals**
   - Algorithms (when to use what)
   - Evaluation metrics
   - Training vs serving trade-offs

4. **Mock interviews**
   - Practice with friends
   - Record yourself
   - Time yourself (45-60 minutes)

### During the Interview

1. **Listen carefully**
   - Understand the problem
   - Don't interrupt
   - Take notes

2. **Think out loud**
   - Explain your reasoning
   - Don't go silent
   - Engage the interviewer

3. **Structure your answer**
   - Use the PROBLEM framework
   - Check time periodically
   - Prioritize important parts

4. **Be flexible**
   - Adapt to interviewer feedback
   - Explore alternatives if prompted
   - Don't be defensive

5. **Show production awareness**
   - Mention monitoring, testing, deployment
   - Discuss reliability and scalability
   - Consider costs

### After the Interview

1. **Reflect on performance**
   - What went well?
   - What could be better?
   - What surprised you?

2. **Follow up**
   - Send thank you email
   - Mention interesting discussion points
   - Show continued interest

3. **Keep practicing**
   - Learn from each interview
   - Study areas of weakness
   - Build real projects

---

## Summary

**ML System Design Success Formula:**

```
Success =
    Strong ML fundamentals +
    System design thinking +
    Production awareness +
    Clear communication +
    Structured approach +
    Practice, practice, practice!
```

**Remember:**
- ✅ Ask clarifying questions
- ✅ Draw clear architectures
- ✅ Discuss trade-offs
- ✅ Think end-to-end
- ✅ Consider scale and reliability
- ✅ Communicate clearly

**You've got this! Good luck! 🚀**

---

**Use this guide to:**
- Prepare for ML system design interviews
- Design production ML systems
- Learn from real-world examples
- Practice with structured framework

**Happy interviewing!** 🎯
