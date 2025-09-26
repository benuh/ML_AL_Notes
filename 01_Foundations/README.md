# Module 01: Foundations of Machine Learning & AI

## 🎯 Learning Objectives
By the end of this module, you will:
- Understand what Machine Learning and AI are
- Know the different types of ML (supervised, unsupervised, reinforcement)
- Recognize common ML applications in daily life
- Understand the ML workflow and terminology

## 📚 Lessons

### 1.1 What is Machine Learning?
**Definition**: Machine Learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.

#### Traditional Programming vs. Machine Learning
```
Traditional Programming:
Data + Program → Output

Machine Learning:
Data + Output → Program (Model)
```

**Real-world Example**: Email spam detection
- Traditional: Write rules (if email contains "FREE MONEY" → spam)
- ML: Show examples of spam/not spam → model learns patterns

#### Key Concepts
- **Algorithm**: The method/recipe for learning
- **Model**: The result after training an algorithm on data
- **Training**: Process of teaching the algorithm using data
- **Prediction**: Using the trained model on new data

### 1.2 Types of Machine Learning

#### 🎯 Supervised Learning
**Definition**: Learning with labeled examples (input-output pairs)

**Examples**:
- Email classification (spam/not spam)
- Medical diagnosis (symptoms → disease)
- Stock price prediction (historical data → future price)

**Sub-types**:
- **Classification**: Predicting categories (spam/not spam)
- **Regression**: Predicting continuous values (house prices)

#### 🔍 Unsupervised Learning
**Definition**: Finding patterns in data without labels

**Examples**:
- Customer segmentation (grouping similar customers)
- Anomaly detection (finding unusual patterns)
- Recommendation systems (finding similar items)

**Sub-types**:
- **Clustering**: Grouping similar data points
- **Association**: Finding relationships between items
- **Dimensionality Reduction**: Simplifying complex data

#### 🎮 Reinforcement Learning
**Definition**: Learning through trial and error with rewards/penalties

**Examples**:
- Game playing (chess, Go, video games)
- Robot navigation
- Trading algorithms
- Chatbots

### 1.3 The Machine Learning Workflow

```
1. Problem Definition → 2. Data Collection → 3. Data Preparation
                ↓
6. Deployment ← 5. Model Evaluation ← 4. Model Training
```

#### Detailed Steps:
1. **Problem Definition**: What are we trying to solve?
2. **Data Collection**: Gather relevant data
3. **Data Preparation**: Clean and organize data
4. **Model Training**: Teach algorithm using data
5. **Model Evaluation**: Test how well it works
6. **Deployment**: Use model in real world

### 1.4 Common ML Applications

#### 🖼️ Computer Vision
- **Image Classification**: What's in this picture?
- **Object Detection**: Where are objects in this image?
- **Facial Recognition**: Who is this person?

#### 🗣️ Natural Language Processing (NLP)
- **Translation**: Google Translate
- **Chatbots**: Customer service, virtual assistants
- **Sentiment Analysis**: Is this review positive or negative?

#### 🎵 Recommendation Systems
- **Netflix**: What movies you might like
- **Spotify**: Music recommendations
- **Amazon**: Product suggestions

#### 🏥 Healthcare
- **Medical Imaging**: Detecting cancer in X-rays
- **Drug Discovery**: Finding new medicines
- **Personalized Treatment**: Tailored treatment plans

## 🎮 Interactive Demo: ML in Action

### Demo 1: Simple Classification
Let's build a simple model to classify fruits!

```python
# Simple fruit classifier based on size and color
def classify_fruit(size, color_red_value):
    """
    Simple rule-based classifier
    size: 1-10 (1=small, 10=large)
    color_red_value: 0-255 (0=no red, 255=very red)
    """
    if size > 7 and color_red_value > 150:
        return "Apple"
    elif size < 4 and color_red_value > 100:
        return "Cherry"
    elif size > 5 and color_red_value < 50:
        return "Lime"
    else:
        return "Unknown fruit"

# Test our classifier
print(classify_fruit(8, 200))  # Should output: Apple
print(classify_fruit(2, 180))  # Should output: Cherry
print(classify_fruit(6, 30))   # Should output: Lime
```

**Try it yourself**: What happens with size=5, color_red_value=100?

## 🧠 Key Terminology

| Term | Definition | Example |
|------|------------|---------|
| **Feature** | Input variable used to make predictions | Age, income, temperature |
| **Label/Target** | Output we're trying to predict | Price, category, yes/no |
| **Training Data** | Data used to teach the model | Historical sales data |
| **Test Data** | Data used to evaluate model performance | New, unseen data |
| **Overfitting** | Model memorizes training data too well | Model works great on training but fails on new data |
| **Underfitting** | Model is too simple to capture patterns | Model performs poorly even on training data |

## 📖 Sources & Further Reading

### Primary Sources:
1. **"Pattern Recognition and Machine Learning"** - Christopher Bishop
2. **"The Elements of Statistical Learning"** - Hastie, Tibshirani, Friedman
3. **"Introduction to Statistical Learning"** - James, Witten, Hastie, Tibshirani

### Online Resources:
- [Andrew Ng's Machine Learning Course](https://www.coursera.org/learn/machine-learning)
- [MIT 6.034 Artificial Intelligence](https://ocw.mit.edu/courses/6-034-artificial-intelligence-fall-2010/)
- [Stanford CS229 Machine Learning](http://cs229.stanford.edu/)

### Papers:
- "Computing Machinery and Intelligence" - Alan Turing (1950)
- "Some Studies in Machine Learning Using the Game of Checkers" - Arthur Samuel (1959)

## ✅ Self-Check Quiz

1. What's the difference between supervised and unsupervised learning?
2. Give an example of a classification problem vs. a regression problem
3. What are the 6 steps in the ML workflow?
4. Name 3 real-world applications of machine learning you use daily

## 🚀 Next Steps

Ready to dive deeper? Continue to [Module 02: Mathematics for ML](../02_Mathematics/README.md) where we'll cover the essential mathematical foundations!

---
*Estimated completion time: 2-3 hours*