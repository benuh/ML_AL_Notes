# 💼 ML Interview Tips: From Application to Offer

**Get hired faster with proven interview strategies.** This guide covers everything from resume optimization to salary negotiation.

---

## 📋 Table of Contents

- [Interview Process Overview](#interview-process-overview)
- [Resume & Application](#resume--application)
- [Phone Screen](#phone-screen)
- [Technical Interviews](#technical-interviews)
- [ML System Design](#ml-system-design)
- [Behavioral Interviews](#behavioral-interviews)
- [Coding Challenges](#coding-challenges)
- [Take-Home Assignments](#take-home-assignments)
- [Culture Fit & Final Round](#culture-fit--final-round)
- [Salary Negotiation](#salary-negotiation)
- [Common Red Flags](#common-red-flags)
- [Interview Checklists](#interview-checklists)

---

## Interview Process Overview

### Typical ML Interview Loop

**Timeline: 4-8 weeks**

```
Application
    ↓
Recruiter Screen (30 min)
    ↓
Technical Phone Screen (45-60 min)
    ↓
Take-Home Assignment (4-8 hours)
    ↓
Onsite/Virtual Loop (4-6 hours)
├── ML Coding (60 min)
├── ML System Design (60 min)
├── ML Theory (45 min)
├── Behavioral (45 min)
└── Team/Culture Fit (30 min)
    ↓
Offer / Feedback
```

### Company-Specific Variations

**FAANG:**
- More rounds (5-7)
- LeetCode heavy
- Bar raiser round
- Strong emphasis on fundamentals

**Startups:**
- Faster process (2-3 weeks)
- Focus on practical skills
- May skip theory round
- Take-home more important

**Research Labs:**
- Paper discussion round
- Research presentation
- Theory deep dive
- PhD often required

---

## Resume & Application

### Resume Structure

**One-page format (unless >10 years experience):**

```
┌──────────────────────────────────────┐
│ NAME                                 │
│ Email | LinkedIn | GitHub | Portfolio│
├──────────────────────────────────────┤
│ SUMMARY (2-3 lines)                  │
│                                      │
├──────────────────────────────────────┤
│ TECHNICAL SKILLS                     │
│ ML/AI: [frameworks, algorithms]      │
│ Engineering: [languages, tools]      │
│ Data: [processing, visualization]    │
│                                      │
├──────────────────────────────────────┤
│ EXPERIENCE                           │
│ [Recent jobs with ML focus]          │
│                                      │
├──────────────────────────────────────┤
│ PROJECTS                             │
│ [3-4 impressive ML projects]         │
│                                      │
├──────────────────────────────────────┤
│ EDUCATION                            │
│ [Degree, relevant coursework]        │
└──────────────────────────────────────┘
```

### Writing Effective Bullets

**Formula:**
```
[Action Verb] [What you did] [Tech/Method] → [Measurable Impact]
```

**Examples:**

❌ **Weak bullets:**
- "Built machine learning models"
- "Worked on recommendation system"
- "Improved model accuracy"

✅ **Strong bullets:**
- "Developed fraud detection system using XGBoost and custom feature engineering, reducing false positives by 43% ($2.1M savings/year)"
- "Designed and deployed recommendation engine with collaborative filtering on AWS Lambda, increasing CTR by 23% (A/B test, p<0.01)"
- "Optimized model inference pipeline with ONNX Runtime and quantization, reducing latency from 450ms to 85ms (5.3x speedup)"

### Keywords to Include

**For ATS (Applicant Tracking Systems):**

Must-have keywords from job description:
- Programming languages: Python, SQL, R
- ML frameworks: PyTorch, TensorFlow, scikit-learn
- Tools: Git, Docker, Kubernetes, AWS/GCP
- Techniques: Deep Learning, NLP, Computer Vision
- Methods: Classification, Regression, Clustering
- Metrics: Accuracy, Precision, Recall, AUC

**Tip:** Mirror job description language

**Job says:** "Experience with natural language processing"
**Your resume:** "Built NLP pipeline using BERT for sentiment analysis..."

### GitHub Profile Optimization

**Must-haves:**
- [ ] Profile README with featured projects
- [ ] 3+ pinned repos with quality READMEs
- [ ] Green contribution graph (consistent activity)
- [ ] Profile photo and bio
- [ ] Links to portfolio/LinkedIn

**See:** [Portfolio Guide](./PORTFOLIO_GUIDE.md) for details

### Cover Letter (When to Write)

**Skip cover letter:**
- Applying to >20 jobs
- Role is clearly match
- Startup with no mention of cover letter

**Write cover letter:**
- Dream company
- Career transition
- Referred by employee
- Non-traditional background

**Keep it short (250 words):**
1. Why this company (20%)
2. Why you're qualified (60%)
3. Call to action (20%)

---

## Phone Screen

### What to Expect

**Duration:** 30-45 minutes

**Format:**
- 5 min: Introductions
- 10 min: Background discussion
- 15 min: Technical basics
- 5 min: Your questions

### Common Questions

**1. "Walk me through your resume"**

**Bad answer:**
> "I graduated from X, then worked at Y, now I'm here..."

**Good answer (3-minute pitch):**
```
"I'm an ML engineer with 2 years of experience building
production ML systems. Most recently, I led development of
a fraud detection system at [Company] that reduced fraud
losses by 43%.

My journey into ML started [brief origin story]. I've since
specialized in [your specialization: NLP/CV/recommender systems],
with particular strength in [skill: productionizing models/
distributed training/model optimization].

What excites me about [Target Company] is [specific reason
related to their work]. I'm particularly interested in
[specific project/technology they use]."
```

**2. "Why are you interested in this role?"**

❌ **Bad:** "I want to learn more about ML"
✅ **Good:** "I'm excited to work on [specific problem] at scale. My experience with [relevant project] would directly apply to [their use case]."

**3. "What's your current/expected salary?"**

**Don't answer directly!**

**Response:**
> "I'm more focused on finding the right role and team fit.
> I'm confident we can find a number that works if we're both
> excited about moving forward. What's the salary range for
> this position?"

### Technical Basics

**Be ready to explain:**
1. Core ML concepts (bias-variance, overfitting, cross-validation)
2. Algorithms you've used (how they work, pros/cons)
3. Metrics (when to use accuracy vs F1 vs AUC)
4. Your projects (deep technical details)

**Example questions:**
- "Explain the difference between L1 and L2 regularization"
- "When would you use precision vs recall?"
- "How do you handle imbalanced datasets?"
- "Walk me through a recent project - what challenges did you face?"

### Questions to Ask

**Good questions:**
- "What does a typical project lifecycle look like on this team?"
- "What ML stack does the team use? Any plans to adopt new tools?"
- "How is model performance measured and monitored?"
- "What's the biggest ML challenge the team is currently tackling?"
- "What opportunities are there for growth and learning?"

**Avoid:**
- Salary/benefits (save for later rounds)
- "What does your company do?" (Do your research!)
- Negative questions ("What's the worst part of working here?")

---

## Technical Interviews

### ML Coding Round

**Duration:** 45-60 minutes

**Format:** Implement ML algorithm or solve ML problem

**Common questions:**

**1. Implement Linear Regression from scratch**
```python
def linear_regression(X, y):
    """
    Implement linear regression using normal equation.

    Args:
        X: (n_samples, n_features)
        y: (n_samples,)

    Returns:
        weights: (n_features,)
    """
    # Your implementation here
    pass
```

**2. Implement K-Means clustering**
```python
def kmeans(X, k, max_iters=100):
    """
    Implement K-Means clustering.

    Args:
        X: (n_samples, n_features)
        k: number of clusters

    Returns:
        centroids: (k, n_features)
        labels: (n_samples,)
    """
    # Your implementation here
    pass
```

**3. Implement train/test split**
```python
def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split data into train and test sets.

    Args:
        X: features
        y: targets
        test_size: fraction for test set

    Returns:
        X_train, X_test, y_train, y_test
    """
    # Your implementation here
    pass
```

**Approach:**
1. **Clarify** (2 min): Ask questions, confirm understanding
2. **Plan** (3 min): Outline approach, discuss tradeoffs
3. **Implement** (30 min): Write code, explain as you go
4. **Test** (10 min): Walk through examples, edge cases
5. **Optimize** (5 min): Discuss improvements

**Key points:**
- Think out loud
- Write clean, readable code
- Test with examples
- Discuss time/space complexity

**See:** [ML Coding Interview Problems](../guides/interview-prep/ML_CODING_INTERVIEW_PROBLEMS.md)

### ML Theory Round

**Duration:** 45-60 minutes

**Format:** Deep dive into ML concepts

**Sample questions:**

**Q: Explain gradient descent**

**Strong answer:**
> "Gradient descent is an optimization algorithm that minimizes
> a loss function by iteratively moving in the direction of
> steepest descent.
>
> Mathematically: w_{t+1} = w_t - α * ∇L(w_t)
>
> Where:
> - w_t: current weights
> - α: learning rate
> - ∇L(w_t): gradient of loss with respect to weights
>
> It works by computing the derivative of the loss function,
> which tells us which direction to move weights to reduce loss.
>
> Key hyperparameter is learning rate:
> - Too large: overshoots minimum, diverges
> - Too small: slow convergence
>
> Variants include:
> - SGD: uses mini-batches, faster but noisy
> - Momentum: accumulates past gradients for smoother updates
> - Adam: adaptive learning rates per parameter
>
> In practice, I typically start with Adam (lr=0.001) as it's
> robust across many problems."

**Q: How do you handle imbalanced datasets?**

**Strong answer:**
> "Several approaches:
>
> 1. **Resampling:**
>    - Oversample minority class (SMOTE)
>    - Undersample majority class
>    - Pros: Simple, effective
>    - Cons: May overfit (oversample) or lose data (undersample)
>
> 2. **Class weights:**
>    - Weight loss function by inverse class frequency
>    - Most libraries support this (class_weight='balanced')
>    - Pros: No data modification
>
> 3. **Different metrics:**
>    - Don't use accuracy (misleading)
>    - Use F1-score, PR-AUC, or recall
>    - Optimize for business metric
>
> 4. **Ensemble methods:**
>    - Balanced Random Forest
>    - EasyEnsemble
>
> 5. **Anomaly detection:**
>    - If extreme imbalance (99:1), treat as anomaly detection
>    - One-class SVM, Isolation Forest
>
> My approach: Start with class weights (fastest), then try
> SMOTE if needed. Always use appropriate metrics (F1, not accuracy)."

**Common topics:**
- Bias-variance tradeoff
- Overfitting/underfitting
- Regularization (L1 vs L2)
- Cross-validation
- Evaluation metrics
- Optimization algorithms
- Neural network architectures
- Loss functions

---

## ML System Design

### The Framework

**Use PROBLEM approach:**

**P**roblem definition
**R**equirements (functional & non-functional)
**O**verview (high-level architecture)
**B**ottlenecks (identify and address)
**L**inks (data flow)
**E**valuation (metrics & monitoring)
**M**aintenance (updates & scaling)

### Example: Design a Recommendation System

**P - Problem (5 min)**
```
Clarifying questions:
- What are we recommending? (products, videos, etc.)
- How many users/items? (scale)
- Real-time or batch?
- Cold start problem? (new users/items)
```

**R - Requirements (5 min)**

**Functional:**
- Given user, return N recommendations
- Personalized to user preferences
- Diverse recommendations

**Non-functional:**
- Latency: <100ms p99
- Scale: 100M users, 1M items
- Availability: 99.9%

**O - Overview (10 min)**
```
┌─────────┐
│  User   │
└────┬────┘
     │
     ▼
┌─────────────────┐
│ API Gateway     │
└────┬────────────┘
     │
     ▼
┌──────────────────────────────┐
│ Recommendation Service       │
│                              │
│ 1. Candidate Generation      │
│    - Collaborative Filtering │
│    - Content-Based           │
│    - Popular Items           │
│                              │
│ 2. Ranking                   │
│    - ML Model (XGBoost)      │
│    - Features: user, item,   │
│      context                 │
│                              │
│ 3. Re-ranking               │
│    - Diversity               │
│    - Business rules          │
└──────────────────────────────┘
     │
     ▼
┌─────────────────┐
│  Return top-N   │
└─────────────────┘
```

**B - Bottlenecks (10 min)**
1. **Candidate generation too slow:**
   - Solution: Pre-compute candidates, cache
2. **Model inference latency:**
   - Solution: Model optimization (quantization), horizontal scaling
3. **Cold start:**
   - Solution: Content-based for new users, popularity-based

**L - Links (Data Flow) (5 min)**
```
User Activity
    ↓
Logging → Kafka → Data Lake
    ↓
Feature Engineering (Spark)
    ↓
Model Training (daily batch)
    ↓
Model Registry → Deploy to serving
    ↓
Prediction Cache (Redis)
```

**E - Evaluation (10 min)**

**Offline metrics:**
- Precision@K, Recall@K
- NDCG (ranking quality)
- Coverage (% of items recommended)

**Online metrics:**
- CTR (click-through rate)
- Conversion rate
- User engagement time

**A/B testing:**
- Control: current system
- Treatment: new model
- Statistical significance: p<0.05

**M - Maintenance (5 min)**
- **Model updates:** Retrain daily with new data
- **Monitoring:** Track CTR degradation, data drift
- **Scaling:** Horizontal scaling for serving, Spark for training
- **Fallback:** If model fails, serve popular items

**See:** [ML System Design Guide](../guides/interview-prep/ML_SYSTEM_DESIGN.md)

---

## Behavioral Interviews

### STAR Method

**S**ituation: Context
**T**ask: What you needed to do
**A**ction: What you did
**R**esult: Outcome (quantify!)

### Common Questions & Answers

**Q: "Tell me about a time you faced a difficult ML problem"**

**STAR answer:**

**Situation:**
> "At [Company], we built a fraud detection system that initially
> had 87% precision, but business required 95%+ to avoid
> false positives that hurt user experience."

**Task:**
> "I needed to improve precision while maintaining high recall
> (>80%) to catch actual fraud."

**Action:**
> "I took a three-pronged approach:
>
> 1. Feature engineering: Added velocity features (transactions
>    per hour, unusual patterns) and external data (device
>    fingerprinting)
>
> 2. Model ensemble: Stacked XGBoost, LightGBM, and neural network,
>    using meta-learner to combine predictions
>
> 3. Threshold optimization: Used precision-recall curve to find
>    optimal threshold for 95% precision
>
> I also set up A/B test to validate in production."

**Result:**
> "Achieved 96% precision at 82% recall, exceeding target.
> Reduced false positives by 60%, improving user satisfaction
> (NPS +12 points). System processed 500K transactions/day
> with <50ms latency."

**Q: "Tell me about a time you disagreed with a team member"**

**STAR answer:**

**Situation:**
> "On a recommendation project, my colleague wanted to use a
> complex deep learning model, while I thought XGBoost would
> be sufficient and much faster to deploy."

**Task:**
> "We needed to choose an approach that balanced accuracy with
> time-to-production (deadline in 2 weeks)."

**Action:**
> "Instead of arguing, I proposed we split a day:
> - Morning: I'd build XGBoost baseline
> - Afternoon: They'd build DL prototype
> - Then compare on hold-out set
>
> XGBoost: 0.72 AUC, 1 day to prod-ready
> DL model: 0.74 AUC, 1 week to prod-ready"

**Result:**
> "We chose XGBoost for V1, shipped on time. Later, we
> upgraded to DL model when we had more time. This approach
> worked well: ship fast, iterate. My colleague appreciated
> the data-driven decision process."

### Behavioral Topics to Prepare

**Leadership:**
- Led project from idea to production
- Mentored junior team member
- Influenced technical direction

**Conflict:**
- Disagreed with manager/colleague
- Received critical feedback
- Resolved team conflict

**Challenges:**
- Faced difficult technical problem
- Missed deadline
- Dealt with ambiguity

**Success:**
- Proud project
- Exceeded expectations
- Improved process/system

**Failure:**
- Mistake with consequences
- Failed project
- What you learned

**Prepare 2-3 stories for each category**

---

## Coding Challenges

### LeetCode Strategy

**For ML roles, focus on:**

**Easy (15 problems):**
- Arrays, strings, hash maps
- Two pointers, sliding window

**Medium (30 problems):**
- Trees, graphs
- Dynamic programming
- Binary search

**Hard (5 problems):**
- Advanced DP
- Graph algorithms

**ML-specific:**
- Matrix operations
- Sorting/searching
- Data structure design

**Study plan: 6 weeks**
```
Week 1-2: Easy (1-2 per day)
Week 3-4: Medium (1 per day)
Week 5: Hard (2-3 total)
Week 6: Review + mock interviews
```

### Coding Interview Tips

**1. Clarify before coding**
```
"Let me confirm:
- Input: array of integers, can be negative?
- Output: single integer?
- Constraints: array length? time/space limits?
- Edge cases: empty array? duplicates?"
```

**2. Think out loud**
```
"I'm thinking of two approaches:
1. Brute force: O(n²), simple but slow
2. Hash map: O(n), more complex but efficient

Let me go with approach 2..."
```

**3. Write clean code**
```python
# ✅ Good
def find_duplicates(nums: List[int]) -> List[int]:
    """
    Find duplicate numbers in array.

    Time: O(n), Space: O(n)
    """
    seen = set()
    duplicates = []

    for num in nums:
        if num in seen:
            duplicates.append(num)
        else:
            seen.add(num)

    return duplicates


# ❌ Bad
def f(a):
    s=set()
    d=[]
    for x in a:
        if x in s:d.append(x)
        else:s.add(x)
    return d
```

**4. Test with examples**
```
"Let me test with [1, 2, 2, 3]:
- i=0: num=1, seen={}, add 1
- i=1: num=2, seen={1}, add 2
- i=2: num=2, seen={1,2}, found duplicate!
- i=3: num=3, seen={1,2}, add 3
- return [2] ✓

Edge case: empty array []
- return [] ✓"
```

---

## Take-Home Assignments

### What Companies Look For

1. **Code quality** (30%)
   - Readable, well-organized
   - Docstrings, type hints
   - Tests

2. **Technical correctness** (30%)
   - Model works, achieves good performance
   - Proper evaluation methodology
   - No data leakage

3. **Communication** (20%)
   - Clear README
   - Explains approach and results
   - Documents decisions

4. **Completeness** (20%)
   - All requirements met
   - Goes beyond basics
   - Shows initiative

### Time Management

**Don't spend >6-8 hours**

**Time breakdown:**
```
1 hour:  Problem understanding, EDA
2 hours: Model development, experimentation
1 hour:  Evaluation, analysis
1 hour:  Code cleanup, documentation
1 hour:  README, presentation
```

**If you're taking longer:**
- Document what you'd do with more time
- Stop at minimum viable solution

### README Template

```markdown
# [Project Name]

## Summary
[2-3 sentences: What you built and key results]

## Approach
[Brief explanation of your method]

## Results
| Metric | Value |
|--------|-------|
| Accuracy | 92.3% |
| Precision | 89.1% |
| Recall | 94.2% |
| F1-Score | 91.6% |

Baseline (most frequent class): 65.2%

## Setup
```bash
pip install -r requirements.txt
python train.py
python evaluate.py
```

## Project Structure
[File tree with brief descriptions]

## Key Decisions
1. **Model choice:** XGBoost
   - Why: Handles mixed features well, fast training
   - Alternatives considered: Random Forest (worse performance),
     Neural Network (overkill for dataset size)

2. **Feature engineering:** [details]

3. **Evaluation:** 5-fold CV + hold-out test set

## What I'd Do With More Time
- [ ] Hyperparameter tuning (currently using defaults)
- [ ] Feature selection (recursive feature elimination)
- [ ] Ensemble with neural network
- [ ] Deploy as API
```

### Red Flags to Avoid

❌ No README or poor documentation
❌ Data leakage (scaling before split)
❌ No evaluation (just training accuracy)
❌ Messy notebooks with 1000+ lines
❌ No tests
❌ Hardcoded paths
❌ Missing requirements.txt
❌ Couldn't reproduce results

---

## Culture Fit & Final Round

### Questions They Ask

**"Why do you want to work here?"**

**Bad answer:**
> "I want to learn and grow" (too generic)

**Good answer:**
> "I'm excited about [specific project/tech]. I've been following
> your work on [blog post/paper], particularly [detail]. My
> experience with [your project] directly relates to [their use case].
> I'm also drawn to [company value] which aligns with my [value]."

**"Where do you see yourself in 5 years?"**

**Bad answer:**
> "I want to be a manager" (not always true)

**Good answer:**
> "I want to become a domain expert in [specialization: MLOps/NLP/CV].
> I'd love to lead projects end-to-end and mentor junior engineers.
> Whether that's as a Staff Engineer or Engineering Manager depends
> on where I can have the most impact."

### Questions to Ask Them

**About the role:**
- "What does success look like in this role in 6 months?"
- "What are the biggest challenges the team is facing?"
- "How is performance evaluated?"

**About the team:**
- "What's the team structure? How many ML engineers?"
- "How does the team collaborate with other teams (product, eng)?"
- "What's the on-call rotation like?"

**About growth:**
- "What learning and development opportunities are available?"
- "Is there a technical ladder? What does Staff/Principal look like?"
- "Are there opportunities to attend conferences or contribute to open source?"

**About culture:**
- "How would you describe the team culture?"
- "What do you like most about working here?"
- "How has the team changed in the past year?"

**Red flags if they can't answer:**
- Vague about responsibilities
- No clear success metrics
- Team/processes unclear
- Defensive responses

---

## Salary Negotiation

### Before the Offer

**Don't give numbers first!**

**Recruiter:** "What's your salary expectation?"

**You:** "I'm focused on finding the right fit. I'm confident we
can agree on compensation if we're both excited. What's the range
for this role?"

**If they insist:**
"Based on my research for [role] at [company type] in [location],
I'd expect $[market rate + 10%]. But I'm flexible based on the
total package."

### Market Research

**Resources:**
- levels.fyi (most reliable)
- Glassdoor
- Blind
- H1B database (for US)

**Typical ranges (2025, US):**

**Junior ML Engineer (0-2 years):**
- FAANG: $150-200K total comp
- Mid-size tech: $120-160K
- Startup: $100-140K + equity

**Mid-level ML Engineer (2-5 years):**
- FAANG: $200-300K total comp
- Mid-size tech: $150-220K
- Startup: $130-180K + equity

**Senior ML Engineer (5+ years):**
- FAANG: $300-500K total comp
- Mid-size tech: $200-300K
- Startup: $180-250K + equity

### After the Offer

**Don't accept immediately!**

**Steps:**

**1. Show enthusiasm (but don't commit)**
```
"Thank you! I'm very excited about this opportunity and the
team. Let me review the details and get back to you by [date]."
```

**2. Evaluate full package**
- Base salary
- Bonus/equity
- Benefits (health, 401k match, PTO)
- Growth opportunities
- Work-life balance
- Team/culture fit

**3. Negotiate (if warranted)**

**When to negotiate:**
- ✅ Offer below market rate
- ✅ You have competing offers
- ✅ Your skills are rare/in-demand
- ✅ You'd accept if number was higher

**When not to negotiate:**
- ❌ Offer is at/above market rate
- ❌ You're desperate for any job
- ❌ No competing offers and limited experience
- ❌ You'd accept regardless

**Negotiation script:**
```
"I'm very excited about joining the team! The offer is strong,
but I was hoping we could discuss the compensation.

Based on my research and the market rate for this role,
I was expecting closer to $[X]. Additionally, I have an offer
from [Company] at $[Y].

Is there flexibility on [salary/equity/bonus]?"
```

**What's negotiable:**
- Base salary (usually 5-10% room)
- Equity (can be substantial)
- Signing bonus (easier than base)
- Start date
- Remote/hybrid policy
- Learning budget
- PTO (hard at big companies)

**4. Get it in writing**

**Don't quit your job until you have:**
- Offer letter signed by company
- Background check passed
- Start date confirmed

---

## Common Red Flags

### During Interview

**🚩 Company red flags:**
- Can't explain what ML team does
- No clear ML strategy or roadmap
- "We're hiring our first ML engineer" (you'll be alone)
- Unrealistic expectations ("Build self-driving car in 3 months")
- No ML infrastructure (no data pipeline, no compute)
- Toxic team dynamics (interviewer badmouths colleagues)

**🚩 Role red flags:**
- Job description is vague or contradictory
- "We need someone to do ML and also frontend and also..." (no focus)
- No dedicated time for model development (100% maintenance)
- No access to data or APIs
- No support from leadership

**🚩 Interview process red flags:**
- Disorganized (reschedule multiple times, forget meetings)
- Disrespectful (late, unprepared, rude)
- Excessive rounds (>7 interviews)
- Take-home >8 hours
- Ghost you for weeks
- Ask for free consulting ("solve our real problem")

### After Offer

**🚩 Offer red flags:**
- Pressure to accept immediately ("offer expires in 24 hours")
- Significantly below market rate with no explanation
- Vague about equity details
- Won't provide written offer
- Require you to quit current job before accepting

---

## Interview Checklists

### Day Before Interview

**Technical prep:**
- [ ] Review your resume (be ready to discuss every bullet)
- [ ] Review company's ML work (blog, papers, products)
- [ ] Prepare 2-3 questions to ask
- [ ] Practice coding on whiteboard/screen share
- [ ] Review ML fundamentals flashcards

**Logistics:**
- [ ] Test video/audio setup (Zoom/Meet)
- [ ] Charge laptop, have backup charger
- [ ] Print copy of resume
- [ ] Set up quiet space
- [ ] Plan outfit (business casual)
- [ ] Get good sleep!

### Morning of Interview

- [ ] Light breakfast (avoid heavy food)
- [ ] Review your prepared stories (STAR method)
- [ ] Warm up with 1-2 easy LeetCode problems
- [ ] Arrive/log in 5-10 minutes early
- [ ] Have water nearby
- [ ] Have paper/pen for notes
- [ ] Silence phone

### During Interview

- [ ] Smile, be enthusiastic
- [ ] Think out loud
- [ ] Ask clarifying questions
- [ ] Admit when you don't know something
- [ ] Take notes on interviewer's questions/feedback
- [ ] Ask your prepared questions
- [ ] Thank them for their time

### After Interview

**Same day:**
- [ ] Send thank-you email (within 24 hours)
- [ ] Note what went well and what to improve
- [ ] Update interview tracker

**Thank-you email template:**
```
Subject: Thank you - [Your Name] - [Role]

Hi [Interviewer Name],

Thank you for taking the time to speak with me today about
the [Role] position. I really enjoyed learning about [specific
detail from conversation] and discussing [topic you discussed].

[Specific detail that shows you paid attention]: e.g., "Your
point about [X] really resonated with me, especially since I
faced a similar challenge in [your project]."

I'm very excited about the opportunity to contribute to
[team/project] and would love to move forward in the process.

Please let me know if you need any additional information.

Best regards,
[Your Name]
```

---

## Key Takeaways

**Interview success formula:**
1. **Preparation:** Study fundamentals, practice coding
2. **Communication:** Think out loud, explain clearly
3. **Enthusiasm:** Show genuine interest
4. **Questions:** Ask insightful questions
5. **Follow-up:** Send thank-you, stay organized

**Remember:**
- Interviews are two-way (you're evaluating them too)
- Rejection is normal (even pros get rejected)
- Practice improves performance
- Negotiate when appropriate
- Get everything in writing

**You got this!** 🚀

---

**Related guides:**
- [ML System Design](../guides/interview-prep/ML_SYSTEM_DESIGN.md) - Detailed framework
- [ML Coding Problems](../guides/interview-prep/ML_CODING_INTERVIEW_PROBLEMS.md) - Practice problems
- [ML Engineering Exercises](../guides/interview-prep/ML_ENGINEERING_EXERCISES.md) - Hands-on prep
- [Portfolio Guide](./PORTFOLIO_GUIDE.md) - Showcasing your work

---

*Last Updated: October 30, 2025*
*Part of: [Complete ML/AI Engineering Curriculum](../README.md)*
