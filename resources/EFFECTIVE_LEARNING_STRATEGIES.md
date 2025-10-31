# 🧠 Effective Learning Strategies for Machine Learning

**Learn smarter, not harder.** This guide covers evidence-based learning techniques to master ML faster and retain knowledge longer.

---

## 📋 Table of Contents

- [The Science of Learning](#the-science-of-learning)
- [Active Learning Techniques](#active-learning-techniques)
- [Spaced Repetition](#spaced-repetition)
- [The Feynman Technique](#the-feynman-technique)
- [Project-Based Learning](#project-based-learning)
- [Building Mental Models](#building-mental-models)
- [Effective Note-Taking](#effective-note-taking)
- [Debugging Your Learning](#debugging-your-learning)
- [Dealing with Overwhelm](#dealing-with-overwhelm)
- [Maintaining Motivation](#maintaining-motivation)
- [Study Schedules](#study-schedules)
- [Learning Resources Organization](#learning-resources-organization)

---

## The Science of Learning

### How Memory Works

**Three types of memory:**

1. **Working Memory** (7±2 items, seconds)
   - Like RAM in your computer
   - Where active thinking happens
   - Easily overloaded

2. **Short-term Memory** (days to weeks)
   - Temporary storage
   - Requires rehearsal to maintain
   - Most learning ends here (and is forgotten!)

3. **Long-term Memory** (years to lifetime)
   - Permanent storage
   - Requires consolidation
   - **Your goal:** Move knowledge here!

### The Forgetting Curve

**Without review:**
```
Retention
100% |█
     |  █
 80% |    █
     |      █
 60% |        █
     |          █
 40% |            █
     |              █
 20% |                █
     |__________________█___________
        1d  3d  7d  30d   → Time
```

**With spaced repetition:**
```
Retention
100% |█  █  █  █  █  █  █  █  █
     |
 80% |  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼
     |
 60% |
     |
     |_________________________________
        Reviews every X days → Time
```

**Key insight:** Review before you forget!

### Effective Learning Principles

1. **Active Recall** > Passive reading
2. **Spaced repetition** > Cramming
3. **Interleaving** > Blocked practice
4. **Elaboration** > Rote memorization
5. **Testing** > Re-reading

---

## Active Learning Techniques

### 1. The "Close the Book" Method

**Passive (ineffective):**
```
Read tutorial → Understand → Move to next section
❌ Feels like learning, but info doesn't stick
```

**Active (effective):**
```
Read section → Close book → Explain out loud → Try to code from memory
✅ Forces retrieval, strengthens memory
```

**Example:**
```python
# After reading about gradient descent:

# ❌ Passive: Copy code from tutorial
def gradient_descent(X, y, learning_rate, iterations):
    # ... copy paste ...

# ✅ Active: Close tutorial, code from memory
# 1. What does gradient descent do?
#    "It minimizes loss by updating weights"
# 2. What are the steps?
#    "Initialize weights, calculate gradient, update"
# 3. Now code it:

def gradient_descent_from_memory(X, y, lr=0.01, iters=1000):
    """I'm writing this from memory!"""
    weights = np.zeros(X.shape[1])

    for i in range(iters):
        # Predict
        predictions = X @ weights

        # Calculate error
        error = predictions - y

        # Gradient
        gradient = X.T @ error / len(y)

        # Update
        weights -= lr * gradient

    return weights

# Check against tutorial, identify gaps
```

### 2. Teach It to Learn It

**Principle:** If you can't explain it simply, you don't understand it.

**Method:**
1. Learn a concept
2. Explain it to someone (or rubber duck)
3. Identify gaps in your explanation
4. Re-learn those gaps
5. Repeat

**Example: Explaining backpropagation**

**First attempt (gaps identified):**
> "Backpropagation updates weights by... um... going backward?
> It uses gradients... somehow? And then... the weights get better?"
> ❌ Too vague, missing key details

**After re-learning:**
> "Backpropagation computes gradients of the loss function with
> respect to each weight by using the chain rule. Starting from
> the output layer, it calculates how much each weight contributed
> to the error, then propagates this information backward through
> the network. Each weight is updated proportionally to its
> gradient multiplied by the learning rate."
> ✅ Clear, specific, shows understanding

### 3. Practice Retrieval, Not Recognition

**Recognition (weak):**
```
Multiple choice: "Which of these is gradient descent?"
A) Batch normalization
B) Gradient descent ← Easy to recognize!
C) Dropout
D) Early stopping
```

**Retrieval (strong):**
```
Open-ended: "Explain how gradient descent works"
[Must generate answer from memory]
```

**Practice:**
- Use flashcards with open-ended questions
- Explain concepts without looking at notes
- Code implementations from scratch
- Write blog posts explaining concepts

### 4. Deliberate Practice

**Not all practice is equal.**

**❌ Ineffective practice:**
- Doing what you already know
- No immediate feedback
- No specific goals
- Comfort zone only

**✅ Deliberate practice:**
- Focus on weaknesses
- Immediate feedback
- Specific, measurable goals
- Slightly outside comfort zone

**Example:**
```
❌ Bad: "I'll build 5 image classifiers using transfer learning"
   (Repeating what you know)

✅ Good: "I'll implement ResNet from scratch to understand
         skip connections, then compare my implementation
         to PyTorch's version"
   (Targeted skill development)
```

---

## Spaced Repetition

### The System

**Review intervals:**
```
Learn → 1 day → 3 days → 7 days → 14 days → 30 days → 90 days
```

**If you forget, reset to 1 day.**

### Implementing for ML

**What to review:**
1. Core concepts (gradient descent, backprop, overfitting)
2. Math formulas (loss functions, optimizers)
3. Code patterns (train/test split, cross-validation)
4. Architecture details (ResNet, Transformers)

**Tool: Anki (flashcard app with spaced repetition)**

**Example cards:**

**Card 1: Core Concept**
```
Front: What is overfitting and how do you detect it?

Back:
- Definition: Model memorizes training data,
  poor generalization to new data
- Detection: Training accuracy >> Test accuracy
  (e.g., 99% train, 70% test)
- Solutions: Regularization, more data, simpler model,
  cross-validation
```

**Card 2: Code Pattern**
```
Front: Write code for k-fold cross-validation

Back:
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model, X, y,
    cv=5,  # 5 folds
    scoring='accuracy'
)

print(f"Mean: {scores.mean():.2%}")
print(f"Std: {scores.std():.2%}")
```

**Card 3: Math Formula**
```
Front: Write the formula for cross-entropy loss

Back:
L = -Σ y_i * log(ŷ_i)

Where:
- y_i: true label (one-hot encoded)
- ŷ_i: predicted probability
- Σ: sum over all classes

Intuition: Penalizes confident wrong predictions heavily
```

### Daily Review Routine

**10 minutes every morning:**
```
7:00 - 7:10 AM: Review Anki cards
- Old cards: Quick review
- New cards: Add 5 new concepts
- Failed cards: Re-learn immediately
```

**ROI:** 10 min/day = 60 hours/year of retention boost

---

## The Feynman Technique

**Named after physicist Richard Feynman, master explainer.**

### The 4 Steps

**Step 1: Choose a concept**
- Example: "Batch Normalization"

**Step 2: Explain it to a 12-year-old**
```
❌ Technical jargon:
"Batch normalization normalizes layer inputs by re-centering
 and re-scaling, reducing internal covariate shift"

✅ Simple explanation:
"Imagine a relay race. Each runner passes the baton at different
 speeds - some fast, some slow. This makes it hard for the next
 runner to coordinate. Batch normalization is like having a coach
 who makes sure every handoff happens at a consistent speed,
 making the race smoother and faster."
```

**Step 3: Identify gaps**
- Where did you get stuck?
- What words did you need to look up?
- What connections are unclear?

**Step 4: Review and simplify**
- Fill in the gaps
- Try explaining again
- Repeat until fluent

### Example: Explaining Gradient Descent

**First attempt:**
> "Gradient descent finds the minimum of a function by...
> taking steps? In the direction of... uh... steepest descent?
> It uses derivatives somehow..."

**Gaps identified:**
- What exactly is "steepest descent"?
- How do derivatives help?
- Why does this find minimum?

**After studying gaps:**
> "Imagine you're on a foggy mountain trying to reach the valley.
> You can't see far, but you can feel which direction slopes
> downward most steeply. You take a step in that direction, then
> reassess. Repeat until you can't go any lower.
>
> In ML, the 'mountain' is the loss function. The 'direction' is
> the gradient (derivative). Each 'step' is updating weights.
> The 'valley' is the minimum loss."

✅ **Test:** Can explain without jargon, using analogy

---

## Project-Based Learning

### Why Projects Beat Tutorials

**Tutorials:**
- Someone else's problems
- Someone else's solutions
- No decision-making
- False confidence

**Projects:**
- Your own problems
- Must figure out solutions
- Constant decision-making
- Real confidence

### The Project Progression

**Level 1: Guided (Week 1-4)**
```
Follow tutorial → Modify one thing → See what breaks → Fix it
```

**Level 2: Semi-guided (Week 5-8)**
```
Read problem → Find similar tutorial → Adapt to your problem
```

**Level 3: Independent (Week 9+)**
```
Define problem → Research approaches → Implement → Iterate
```

### Project Learning Cycle

```
1. Plan (10%)
   ↓
2. Implement (40%)
   ↓
3. Break (20%)          ← Intentionally break things!
   ↓
4. Debug (20%)          ← Real learning happens here
   ↓
5. Document (10%)       ← Solidify understanding
```

**Example: Breaking things to learn**
```python
# You've built a working sentiment classifier

# Now intentionally break it to understand:

# 1. What if I remove all preprocessing?
# Result: Accuracy drops from 87% to 62%
# Learning: Preprocessing is crucial!

# 2. What if I use only 10% of data?
# Result: Accuracy drops to 75%, high variance
# Learning: More data reduces overfitting

# 3. What if I remove the embedding layer?
# Result: Model won't converge
# Learning: Embeddings capture semantic meaning

# 4. What if I increase learning rate 10x?
# Result: Loss explodes (NaN)
# Learning: Learning rate is sensitive

# 5. What if I train for 100 epochs instead of 10?
# Result: Overfitting (train 95%, test 85%)
# Learning: Early stopping is necessary
```

### Learning from Failures

**Track what doesn't work:**
```markdown
## Experiment Log

### Attempt 1: Simple Logistic Regression
- Accuracy: 65%
- Problem: Can't capture non-linear patterns
- Learning: Need more complex model

### Attempt 2: Deep Neural Network (5 layers)
- Accuracy: Train 99%, Test 58%
- Problem: Overfitting
- Learning: Too complex for dataset size

### Attempt 3: Random Forest
- Accuracy: Train 87%, Test 83%
- Problem: Slow inference (200ms)
- Learning: Tree models are interpretable but slow

### Attempt 4: XGBoost with tuning
- Accuracy: Train 89%, Test 86%
- Inference: 50ms
- ✅ Success: Good balance of accuracy and speed
```

---

## Building Mental Models

### What Are Mental Models?

**Definition:** Frameworks for understanding how things work.

**Example: Overfitting**

**Weak mental model:**
> "Overfitting is bad. Use regularization."

**Strong mental model:**
> "Overfitting happens when model capacity exceeds dataset
> information. The model memorizes noise instead of learning
> patterns. It's like a student memorizing exam answers without
> understanding concepts - they fail on new questions.
>
> Solutions address either:
> - Reduce capacity: Simpler model, regularization, dropout
> - Increase data: More samples, data augmentation
> - Improve generalization: Cross-validation, ensemble methods"

### Building Strong Mental Models

**1. Understand at multiple levels:**
- **Intuition:** "What is this like?"
- **Mechanism:** "How does it work?"
- **Math:** "What's the formula?"
- **Code:** "How do I implement it?"
- **Application:** "When do I use it?"

**Example: Batch Normalization**

**Intuition:**
"Keeps layer inputs in a consistent range, like normalizing test scores"

**Mechanism:**
"For each batch, calculate mean and std, normalize, then scale and shift with learnable parameters"

**Math:**
```
BN(x) = γ * (x - μ) / √(σ² + ε) + β

Where:
- μ: batch mean
- σ²: batch variance
- γ, β: learnable parameters
- ε: small constant for stability
```

**Code:**
```python
def batch_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=0)
    var = x.var(axis=0)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta
```

**Application:**
"Use after convolutional/dense layers, before activation. Speeds up training, acts as regularization. Essential for deep networks (>10 layers)."

**2. Connect to existing knowledge:**
```
New concept: Attention mechanism

Connect to:
- Database queries (Query, Key, Value)
- Information retrieval (TF-IDF weighting)
- Human attention (focusing on relevant parts)
- Weighted average (soft selection)
```

**3. Test your model:**
- Can you predict behavior in new situations?
- Can you explain to others?
- Can you debug when things go wrong?

---

## Effective Note-Taking

### The Cornell Method for ML

**Format:**
```
┌─────────────────────────────────────────┐
│ Topic: Gradient Descent                 │
├───────────┬─────────────────────────────┤
│           │                             │
│  CUE      │     NOTES                   │
│  COLUMN   │                             │
│           │  Detailed notes here...     │
│  Key      │                             │
│  terms,   │                             │
│  questions│                             │
│           │                             │
│           │                             │
├───────────┴─────────────────────────────┤
│  SUMMARY                                │
│  Bottom section: 1-2 sentence summary   │
└─────────────────────────────────────────┘
```

**Example:**
```
┌─────────────────────────────────────────┐
│ Topic: Convolutional Neural Networks    │
├───────────┬─────────────────────────────┤
│ What is   │ CNN: Neural network with    │
│ a CNN?    │ convolutional layers        │
│           │                             │
│ Why use   │ Purpose: Extract spatial    │
│ CNN?      │ features from images        │
│           │                             │
│ Key parts │ Components:                 │
│           │ - Conv layer: Apply filters │
│           │ - Pooling: Downsample       │
│           │ - FC layer: Classification  │
│           │                             │
│ When to   │ Use for:                    │
│ use?      │ - Image classification      │
│           │ - Object detection          │
│           │ - Any spatial data          │
│           │                             │
│ Code?     │ PyTorch example:            │
│           │ nn.Conv2d(in, out, kernel)  │
│           │ nn.MaxPool2d(2)             │
├───────────┴─────────────────────────────┤
│ SUMMARY: CNNs use filters to detect     │
│ spatial patterns in images, reducing    │
│ parameters vs fully-connected networks  │
└─────────────────────────────────────────┘
```

### Digital Note System

**Tool: Obsidian / Notion / OneNote**

**Structure:**
```
ML_Notes/
├── 01_Fundamentals/
│   ├── Linear_Algebra.md
│   ├── Calculus.md
│   └── Probability.md
├── 02_Algorithms/
│   ├── Linear_Regression.md
│   ├── Decision_Trees.md
│   └── Neural_Networks.md
├── 03_Concepts/
│   ├── Overfitting.md
│   ├── Regularization.md
│   └── Cross_Validation.md
├── 04_Code_Snippets/
│   ├── Train_Test_Split.md
│   ├── Cross_Validation.md
│   └── Data_Preprocessing.md
└── 05_Projects/
    └── House_Price_Predictor.md
```

**Link concepts:**
```markdown
# Overfitting.md

## Definition
When model memorizes training data but can't generalize.

## Related Concepts
- [[Regularization]] - Solution to overfitting
- [[Cross_Validation]] - Technique to detect overfitting
- [[Bias_Variance_Tradeoff]] - Theoretical framework

## Causes
1. Model too complex
2. Too little data
3. Training too long

## Solutions
- Use [[Regularization]]
- Collect more data
- [[Early_Stopping]]
- [[Dropout]] (for neural networks)

## Code Example
See [[Cross_Validation]] notebook
```

### Visual Notes

**Draw diagrams:**
```
Linear Regression:

Input → [Weights] → Output
  X   ×     w    →   ŷ

Loss = (y - ŷ)²

Update: w = w - α * ∂L/∂w
```

**Use color coding:**
- 🟢 Green: Core concepts (must know)
- 🟡 Yellow: Important details
- 🔴 Red: Common mistakes
- 🔵 Blue: Code snippets

---

## Debugging Your Learning

### Signs You're Not Learning Effectively

**Warning signs:**

| Sign | Fix |
|------|-----|
| Can follow tutorial but can't code from scratch | More active recall practice |
| Understand in the moment, forget next day | Implement spaced repetition |
| Can't explain concepts simply | Use Feynman technique |
| Spend hours but no progress | Change learning method |
| Everything feels overwhelming | Break into smaller chunks |
| Afraid to start projects | Lower the bar (micro-projects) |

### The 80/20 of ML Learning

**20% of concepts → 80% of practical work:**

**Must-know (20%):**
- Linear algebra basics (vectors, matrices, dot products)
- Calculus basics (derivatives, chain rule)
- Gradient descent
- Train/test split, cross-validation
- Overfitting vs underfitting
- Linear regression, logistic regression
- Decision trees, random forests
- Neural networks basics
- Common metrics (accuracy, precision, recall, MSE, R²)

**Nice-to-know (80%):**
- Advanced optimizers (Adam details)
- Complex architectures
- Cutting-edge research
- Mathematical proofs
- Advanced statistics

**Focus on the 20% first.**

### Learning Audit

**Monthly check-in:**
```markdown
## Learning Audit - October 2025

### What did I learn?
- Implemented backpropagation from scratch
- Built end-to-end recommendation system
- Learned Docker basics for deployment

### What can I do now that I couldn't before?
- Can explain gradient descent to a beginner
- Can deploy ML model as API
- Can debug model performance issues

### What's still confusing?
- Attention mechanism details
- Optimal hyperparameter tuning strategy
- Production monitoring best practices

### Next month's focus:
- Deep dive into Transformers
- Build 2 portfolio projects
- Set up MLflow tracking
```

---

## Dealing with Overwhelm

### The Paradox of Choice

**Too many resources → Analysis paralysis**

**Solution: The Rule of One**
- ONE main course/book
- ONE project at a time
- ONE hour of focus (Pomodoro)
- ONE concept per day

### Information Overload

**Symptoms:**
- 50 browser tabs open
- 20 unfinished courses
- Constantly searching for "best" tutorial
- Feeling behind

**Cure:**
1. **Close all tabs** (bookmark if must)
2. **Pick ONE course** (any good one will do)
3. **Commit for 30 days** (no switching)
4. **Build, don't consume**

### When Stuck

**Unstuck protocol:**

**Step 1: Define the problem (5 min)**
```
What exactly am I trying to do?
What have I tried?
What error/issue am I seeing?
```

**Step 2: Rubber duck debug (10 min)**
Explain problem out loud to rubber duck (or pet)

**Step 3: Search strategically (15 min)**
```
Google: "[error message]" + "stack overflow"
GitHub Issues: Check library repo
Documentation: Re-read relevant section
```

**Step 4: Ask for help (after above steps)**
```
Post on:
- Stack Overflow
- r/learnmachinelearning
- Discord communities

Include:
- What you're trying to do
- What you've tried
- Minimal code example
- Error message
```

**Step 5: Move on (don't get stuck for hours)**
```
If stuck >2 hours:
- Mark as TODO
- Work on something else
- Return tomorrow with fresh mind
```

---

## Maintaining Motivation

### Why Motivation Fades

**The motivation curve:**
```
Motivation
   High  |█
         | █
         |  █
         |   █      ← "Valley of Despair"
         |    █
   Low   |     █_____________
         |─────────────────────→ Time
             Start    3mo   6mo
```

**Reality:** Everyone hits the valley. Pros keep going.

### Building Systems > Relying on Motivation

**❌ Motivation-dependent:**
"I'll study when I feel motivated"
- Result: Study 3 days, quit for 2 weeks, repeat

**✅ System-dependent:**
"I study every day 7-8 AM, motivated or not"
- Result: Consistent progress

### The 2-Minute Rule

**Problem:** "I don't feel like coding today"

**Solution:** "I'll just open VS Code and write one function"

**Result:** Once started, usually continue for 30+ min

**Key:** Lower the barrier to start

### Celebrate Small Wins

**Track progress visually:**
```markdown
## Streak Tracker

October 2025:
✅✅✅✅✅✅✅ Week 1: 7/7 days
✅✅✅❌✅✅✅ Week 2: 6/7 days
✅✅✅✅✅✅❌ Week 3: 6/7 days
✅✅✅✅✅✅✅ Week 4: 7/7 days

Total: 26/28 days (93%)
```

**Reward milestones:**
- Finished first project → Nice dinner
- 30-day streak → New book/course
- First job interview → Celebration

### Join a Community

**Why:**
- Accountability
- Support when stuck
- See others' progress (you're not alone)
- Networking

**Where:**
- Local ML meetups
- Discord servers
- r/learnmachinelearning
- Twitter ML community
- Study groups (find on Reddit/Discord)

---

## Study Schedules

### The Pomodoro Technique

**Structure:**
```
25 min: Focused work
 5 min: Break
25 min: Focused work
 5 min: Break
25 min: Focused work
 5 min: Break
25 min: Focused work
30 min: Long break
```

**Rules:**
- No distractions during 25 min (phone away, close Slack)
- Actually take breaks (walk, stretch, water)
- Track completed pomodoros

**Daily goal:** 8-12 pomodoros (3-5 hours actual work)

### Sample Daily Schedule

**For full-time learners (8-10 hours/day):**
```
7:00 - 7:30   Morning routine + Anki review
7:30 - 9:30   Deep work: Theory/reading
9:30 - 10:00  Break (walk, breakfast)
10:00 - 12:00 Deep work: Coding/implementation
12:00 - 13:00 Lunch + rest
13:00 - 15:00 Deep work: Project work
15:00 - 15:30 Break
15:30 - 17:00 Review, debugging, documentation
17:00 - 18:00 Light learning (videos, articles)
18:00+        Done for the day!
```

**For part-time learners (2-3 hours/day):**
```
6:00 - 6:30   Morning Anki review
Evening:
19:00 - 20:30 Deep work: Main learning/coding
20:30 - 21:00 Review and plan next day
```

### Weekly Structure

**Balanced approach:**
```
Monday:    New concept + exercises
Tuesday:   Project work
Wednesday: New concept + exercises
Thursday:  Project work
Friday:    Weekly review + blog post
Saturday:  Long project session
Sunday:    Rest / optional light learning
```

---

## Learning Resources Organization

### The Second Brain System

**Capture → Organize → Distill → Express**

**1. Capture:**
- Bookmark interesting articles → Pocket/Instapaper
- Save code snippets → Gist/notion
- Screenshot useful diagrams → Annotate and save

**2. Organize:**
```
ML_Knowledge_Base/
├── Inbox/              # Unsorted captures
├── Resources/          # Courses, books, articles
├── Notes/              # Processed notes
├── Projects/           # Project documentation
└── Archive/            # Old/completed material
```

**3. Distill:**
- Extract key insights
- Create flashcards
- Write summaries
- Draw diagrams

**4. Express:**
- Build projects
- Write blog posts
- Teach others
- Contribute to open source

### Resource Evaluation

**Before starting new resource, ask:**
- [ ] Does this fill a knowledge gap?
- [ ] Is it from reputable source?
- [ ] Do I have time to finish it?
- [ ] Will I build something with this?

**If 3+ are "No", skip it.**

---

## Action Plan

### Week 1: Set Up Systems
- [ ] Choose note-taking tool (Obsidian/Notion)
- [ ] Set up Anki for spaced repetition
- [ ] Create learning tracker (spreadsheet/app)
- [ ] Schedule daily learning time
- [ ] Join 2 ML communities

### Week 2: Implement Techniques
- [ ] Practice active recall (close book method)
- [ ] Create 10 Anki flashcards
- [ ] Use Feynman technique on 1 concept
- [ ] Start project learning log

### Week 3-4: Establish Habits
- [ ] Daily Anki reviews (10 min)
- [ ] Weekly learning audit
- [ ] Teach one concept to someone
- [ ] Track Pomodoro sessions

### Ongoing
- [ ] Monthly learning audit
- [ ] Update notes with new insights
- [ ] Share learnings publicly
- [ ] Adjust techniques based on what works

---

## Key Takeaways

**Most important principles:**
1. **Active > Passive** - Code, don't just watch
2. **Spaced > Cramming** - Review regularly
3. **Teach to learn** - Explain to others
4. **Systems > Motivation** - Build habits
5. **Break things** - Learn by debugging

**Remember:**
- Learning is a skill you can improve
- Consistency beats intensity
- Everyone struggles (you're not alone)
- Focus on progress, not perfection

---

**Related guides:**
- [Quick Start Guide](./QUICK_START_GUIDE.md) - Applying these techniques from day 1
- [Learning Schedule](./ML_LEARNING_SCHEDULE.md) - Structured study plans
- [Common ML Mistakes](./COMMON_ML_MISTAKES.md) - What not to do

---

*Last Updated: October 30, 2025*
*Part of: [Complete ML/AI Engineering Curriculum](../README.md)*
