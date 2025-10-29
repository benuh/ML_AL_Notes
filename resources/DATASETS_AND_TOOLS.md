# 📚 ML/AI Datasets and Tools Guide

## Comprehensive Resource for Data and Tools

This guide provides curated lists of datasets, tools, frameworks, and platforms for machine learning and AI development.

---

## 📋 Table of Contents

1. [Popular Datasets](#popular-datasets)
2. [ML Frameworks](#ml-frameworks)
3. [Development Tools](#development-tools)
4. [Cloud Platforms](#cloud-platforms)
5. [Data Processing Tools](#data-processing-tools)
6. [Visualization Tools](#visualization-tools)
7. [MLOps Tools](#mlops-tools)
8. [Learning Resources](#learning-resources)

---

## 📊 Popular Datasets

### Computer Vision

**MNIST** - Handwritten digits
```python
from torchvision import datasets, transforms

# Download MNIST
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

# Info
# Size: 60,000 training + 10,000 test
# Image size: 28x28 grayscale
# Classes: 10 (digits 0-9)
# Use case: Classification, beginner projects
```

**CIFAR-10/CIFAR-100**
```python
from torchvision import datasets

# CIFAR-10: 10 classes
cifar10 = datasets.CIFAR10(root='./data', train=True, download=True)

# CIFAR-100: 100 classes
cifar100 = datasets.CIFAR100(root='./data', train=True, download=True)

# Info
# Size: 50,000 training + 10,000 test (each)
# Image size: 32x32 RGB
# CIFAR-10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
# Use case: Image classification, CNN practice
```

**ImageNet**
```
# Info
# Size: 14 million images
# Classes: 20,000+ categories
# Image size: Variable
# Download: https://www.image-net.org/
# Use case: Large-scale classification, transfer learning

# Pre-trained models available in PyTorch/TensorFlow
from torchvision import models
resnet50 = models.resnet50(pretrained=True)
```

**COCO** - Common Objects in Context
```
# Info
# Size: 330K images
# Tasks: Object detection, segmentation, captioning
# Classes: 80 object categories
# Download: https://cocodataset.org/
# Use case: Object detection, instance segmentation

# Load with pycocotools
from pycocotools.coco import COCO

coco = COCO('annotations/instances_train2017.json')
cats = coco.loadCats(coco.getCatIds())
```

**CelebA** - Celebrity Faces
```
# Info
# Size: 200K celebrity images
# Attributes: 40 attribute annotations
# Tasks: Face detection, attribute recognition, landmark detection
# Download: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
# Use case: Face recognition, GANs
```

### Natural Language Processing

**IMDB Movie Reviews**
```python
from keras.datasets import imdb

# Load IMDB
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

# Info
# Size: 50,000 reviews (25K train, 25K test)
# Task: Sentiment classification (positive/negative)
# Use case: Text classification, sentiment analysis
```

**AG News**
```python
from datasets import load_dataset

# Load AG News
dataset = load_dataset('ag_news')

# Info
# Size: 120,000 training + 7,600 test
# Classes: 4 (World, Sports, Business, Sci/Tech)
# Use case: News classification
```

**SQuAD** - Stanford Question Answering Dataset
```python
from datasets import load_dataset

# Load SQuAD
squad = load_dataset('squad')

# Info
# Size: 100K+ question-answer pairs
# Task: Reading comprehension, QA
# Use case: Question answering systems
```

**WikiText-2/WikiText-103**
```python
from datasets import load_dataset

# Load WikiText
wikitext = load_dataset('wikitext', 'wikitext-2-raw-v1')

# Info
# Task: Language modeling
# Size: WikiText-2 (2M tokens), WikiText-103 (103M tokens)
# Use case: Language model pre-training
```

**Common Crawl**
```
# Info
# Size: Petabytes of web data
# Download: https://commoncrawl.org/
# Use case: Large-scale language model pre-training
# Note: Used to train GPT-3, LLaMA, etc.
```

### Structured Data

**Titanic**
```python
import pandas as pd

# Load from Kaggle
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

# Info
# Size: 891 rows
# Features: Age, Sex, Fare, Class, etc.
# Target: Survival (0/1)
# Use case: Binary classification, beginner project
```

**Housing Prices**
```python
from sklearn.datasets import fetch_california_housing

# Load California housing
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Info
# Size: 20,640 samples
# Features: 8 (median income, house age, etc.)
# Target: Median house value
# Use case: Regression, feature engineering
```

**Credit Card Fraud**
```
# Download from Kaggle
# https://www.kaggle.com/mlg-ulb/creditcardfraud

# Info
# Size: 284,807 transactions
# Imbalanced: 492 frauds (0.172%)
# Features: Time, Amount, V1-V28 (PCA components)
# Use case: Anomaly detection, imbalanced classification
```

**Adult Income**
```python
from sklearn.datasets import fetch_openml

# Load Adult dataset
adult = fetch_openml('adult', version=2, as_frame=True)
X, y = adult.data, adult.target

# Info
# Size: 48,842 samples
# Features: Age, education, occupation, etc.
# Target: Income >50K or <=50K
# Use case: Binary classification, fairness analysis
```

### Time Series

**Stock Market Data**
```python
import yfinance as yf

# Download stock data
stock = yf.download('AAPL', start='2020-01-01', end='2023-12-31')

# Info
# Data: OHLC (Open, High, Low, Close) + Volume
# Frequency: Daily, Hourly, etc.
# Use case: Time series forecasting, trading strategies
```

**Weather Data**
```python
# NOAA Climate Data
# https://www.ncdc.noaa.gov/

# Info
# Data: Temperature, precipitation, wind, etc.
# Frequency: Hourly, daily, monthly
# Use case: Weather forecasting, climate analysis
```

**Electricity Consumption**
```
# UCI Individual Household Electric Power Consumption
# https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption

# Info
# Size: 2 million measurements
# Period: 4 years
# Use case: Time series forecasting, energy prediction
```

### Audio

**LibriSpeech**
```
# Info
# Size: 1000 hours of speech
# Task: Speech recognition
# Download: https://www.openslr.org/12
# Use case: ASR, speech-to-text
```

**AudioSet**
```
# Info
# Size: 2 million YouTube videos
# Classes: 632 audio event classes
# Download: https://research.google.com/audioset/
# Use case: Audio classification, sound event detection
```

### Recommendation Systems

**MovieLens**
```python
from surprise import Dataset

# Load MovieLens
data = Dataset.load_builtin('ml-100k')

# Info
# Sizes: 100K, 1M, 10M, 20M, 25M ratings
# Data: User ratings for movies
# Download: https://grouplens.org/datasets/movielens/
# Use case: Collaborative filtering, recommender systems
```

**Amazon Reviews**
```
# Info
# Size: 233 million reviews
# Period: 1996-2018
# Download: https://nijianmo.github.io/amazon/index.html
# Use case: Recommendation, sentiment analysis
```

### Healthcare

**MIMIC-III**
```
# Info
# Data: ICU patient records
# Size: 60,000 ICU stays
# Access: Requires training certification
# Website: https://mimic.mit.edu/
# Use case: Clinical prediction, healthcare analytics
```

**Chest X-Ray Images**
```
# Info
# Size: 112,120 X-ray images
# Classes: Pneumonia, Normal
# Download: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
# Use case: Medical image classification
```

---

## 🔧 ML Frameworks

### Deep Learning

**PyTorch**
```python
import torch
import torch.nn as nn

# Why PyTorch?
# ✅ Pythonic and intuitive
# ✅ Dynamic computation graphs
# ✅ Great for research
# ✅ Strong community support
# ✅ TorchScript for production

# Installation
# pip install torch torchvision torchaudio

# Example
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Ecosystem:
# - torchvision: Computer vision
# - torchaudio: Audio processing
# - torchtext: NLP (deprecated, use HuggingFace)
# - PyTorch Lightning: High-level wrapper
```

**TensorFlow / Keras**
```python
import tensorflow as tf
from tensorflow import keras

# Why TensorFlow?
# ✅ Production-ready (TF Serving)
# ✅ Mobile deployment (TF Lite)
# ✅ Browser deployment (TensorFlow.js)
# ✅ TPU support
# ✅ Keras: Easy high-level API

# Installation
# pip install tensorflow

# Example
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Ecosystem:
# - TensorFlow Hub: Pre-trained models
# - TensorBoard: Visualization
# - TF Serving: Production serving
# - TF Lite: Mobile deployment
```

**JAX**
```python
import jax
import jax.numpy as jnp

# Why JAX?
# ✅ NumPy-like API
# ✅ Automatic differentiation
# ✅ JIT compilation
# ✅ Easy parallelization
# ✅ Functional programming

# Installation
# pip install jax jaxlib

# Example: Automatic differentiation
def f(x):
    return x ** 3 + 2 * x ** 2 - 5 * x + 3

grad_f = jax.grad(f)
print(grad_f(2.0))  # Derivative at x=2

# Ecosystem:
# - Flax: Neural network library
# - Haiku: Neural network framework
# - Optax: Optimization library
```

### Traditional ML

**scikit-learn**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Why scikit-learn?
# ✅ Consistent API
# ✅ Comprehensive algorithms
# ✅ Excellent documentation
# ✅ Integration with NumPy/pandas
# ✅ Production-ready

# Installation
# pip install scikit-learn

# Example
clf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(clf, X, y, cv=5)

# Algorithms:
# - Classification: LogisticRegression, SVC, RandomForest
# - Regression: LinearRegression, Ridge, GradientBoosting
# - Clustering: KMeans, DBSCAN, AgglomerativeClustering
# - Dimensionality Reduction: PCA, t-SNE, UMAP
# - Preprocessing: StandardScaler, LabelEncoder, OneHotEncoder
```

**XGBoost**
```python
import xgboost as xgb

# Why XGBoost?
# ✅ Best performance on tabular data
# ✅ Wins Kaggle competitions
# ✅ Fast training
# ✅ Feature importance
# ✅ Handles missing values

# Installation
# pip install xgboost

# Example
dtrain = xgb.DMatrix(X_train, label=y_train)
params = {'max_depth': 6, 'eta': 0.3, 'objective': 'binary:logistic'}
model = xgb.train(params, dtrain, num_boost_round=100)

# Alternatives:
# - LightGBM: Faster, uses histogram-based algorithms
# - CatBoost: Better with categorical features
```

**LightGBM**
```python
import lightgbm as lgb

# Why LightGBM?
# ✅ Faster than XGBoost
# ✅ Lower memory usage
# ✅ Better accuracy on large datasets
# ✅ Native categorical feature support

# Installation
# pip install lightgbm

# Example
train_data = lgb.Dataset(X_train, label=y_train)
params = {'objective': 'binary', 'metric': 'auc', 'boosting': 'gbdt'}
model = lgb.train(params, train_data, num_boost_round=100)
```

### Transformers & LLMs

**HuggingFace Transformers**
```python
from transformers import AutoTokenizer, AutoModel, pipeline

# Why Transformers?
# ✅ 100,000+ pre-trained models
# ✅ Easy fine-tuning
# ✅ Multi-task support
# ✅ Production-ready
# ✅ Great documentation

# Installation
# pip install transformers

# Example: Sentiment analysis
classifier = pipeline('sentiment-analysis')
result = classifier("I love this product!")

# Example: Load model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Use cases:
# - Text classification
# - Question answering
# - Translation
# - Summarization
# - Token classification (NER)
```

**LangChain**
```python
from langchain import OpenAI, LLMChain, PromptTemplate

# Why LangChain?
# ✅ LLM application framework
# ✅ Chain multiple operations
# ✅ Memory management
# ✅ Tool integration
# ✅ Agent creation

# Installation
# pip install langchain

# Example
llm = OpenAI(temperature=0.9)
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run("colorful socks")
```

---

## 🛠️ Development Tools

### IDEs & Editors

**Jupyter Notebook / JupyterLab**
```bash
# Installation
pip install jupyter
pip install jupyterlab

# Launch
jupyter notebook
jupyter lab

# Why Jupyter?
# ✅ Interactive development
# ✅ Visualizations inline
# ✅ Great for exploration
# ✅ Shareable notebooks
# ✅ Markdown support

# Extensions:
# - jupyterlab-git: Git integration
# - jupyterlab-execute-time: Show cell execution time
# - jupytext: Sync notebooks with .py files
```

**VS Code**
```
# Why VS Code?
# ✅ Free and open-source
# ✅ Rich extension ecosystem
# ✅ Integrated terminal
# ✅ Git integration
# ✅ Remote development

# Essential Extensions:
# - Python
# - Pylance
# - Jupyter
# - GitLens
# - Docker
# - Remote - SSH
```

**PyCharm**
```
# Why PyCharm?
# ✅ Professional Python IDE
# ✅ Excellent debugging
# ✅ Integrated testing
# ✅ Database tools
# ✅ Scientific mode

# Editions:
# - Community: Free
# - Professional: Paid (web frameworks, databases, remote development)
```

### Version Control

**Git & GitHub**
```bash
# Why Git?
# ✅ Industry standard
# ✅ Distributed version control
# ✅ Branching and merging
# ✅ Collaboration

# Essential commands
git init
git add .
git commit -m "message"
git push origin main
git pull
git branch feature-branch
git checkout feature-branch

# GitHub features:
# - Issues tracking
# - Pull requests
# - GitHub Actions (CI/CD)
# - GitHub Pages (hosting)
```

**DVC (Data Version Control)**
```bash
# Why DVC?
# ✅ Version large datasets
# ✅ Track ML experiments
# ✅ Reproducible pipelines
# ✅ Works with Git

# Installation
pip install dvc

# Usage
dvc init
dvc add data/large_dataset.csv
git add data/large_dataset.csv.dvc .gitignore
git commit -m "Add dataset"
dvc push  # Upload to remote storage
```

### Environment Management

**Conda**
```bash
# Why Conda?
# ✅ Manages Python versions
# ✅ Handles non-Python dependencies
# ✅ Cross-platform
# ✅ Popular in data science

# Create environment
conda create -n myenv python=3.9
conda activate myenv

# Install packages
conda install numpy pandas scikit-learn
conda install pytorch torchvision -c pytorch

# Export environment
conda env export > environment.yml

# Create from file
conda env create -f environment.yml
```

**venv / virtualenv**
```bash
# Why venv?
# ✅ Built-in (Python 3.3+)
# ✅ Lightweight
# ✅ Standard tool

# Create environment
python -m venv myenv

# Activate
source myenv/bin/activate  # Linux/Mac
myenv\Scripts\activate  # Windows

# Install packages
pip install -r requirements.txt

# Freeze dependencies
pip freeze > requirements.txt
```

**Docker**
```dockerfile
# Why Docker?
# ✅ Reproducible environments
# ✅ Isolate dependencies
# ✅ Easy deployment
# ✅ Version entire stack

# Example Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "train.py"]
```

---

## ☁️ Cloud Platforms

### Major Cloud Providers

**AWS (Amazon Web Services)**
```
ML Services:
✅ SageMaker: End-to-end ML platform
  - Notebook instances
  - Training jobs
  - Model deployment
  - Autopilot (AutoML)
  - Ground Truth (data labeling)

✅ Rekognition: Computer vision API
✅ Comprehend: NLP API
✅ Translate: Translation API
✅ Polly: Text-to-speech
✅ Transcribe: Speech-to-text

Compute:
- EC2: Virtual machines
- Lambda: Serverless functions
- ECS/EKS: Container orchestration

Storage:
- S3: Object storage
- EBS: Block storage
- EFS: File storage

Pricing: Pay-as-you-go
```

**Google Cloud Platform (GCP)**
```
ML Services:
✅ Vertex AI: Unified ML platform
  - Training
  - Prediction
  - Pipelines
  - Feature Store
  - Model monitoring

✅ Vision AI: Image analysis
✅ Natural Language AI: NLP
✅ Speech-to-Text / Text-to-Speech
✅ Translation AI
✅ AutoML: No-code ML

Compute:
- Compute Engine: VMs
- Cloud Functions: Serverless
- GKE: Kubernetes

Storage:
- Cloud Storage: Object storage
- Persistent Disk

Advantages:
- Best TPU access
- BigQuery integration
- Competitive pricing
```

**Microsoft Azure**
```
ML Services:
✅ Azure Machine Learning:
  - Automated ML
  - Designer (no-code)
  - Notebooks
  - Pipelines
  - Model management

✅ Cognitive Services:
  - Computer Vision
  - Language Understanding
  - Speech Services
  - OpenAI Service (GPT-4 access!)

Compute:
- Virtual Machines
- Azure Functions
- AKS: Kubernetes

Storage:
- Blob Storage
- Azure Files

Advantages:
- Enterprise integration
- OpenAI partnership
- Active Directory integration
```

### Specialized ML Platforms

**Databricks**
```
Why Databricks?
✅ Unified analytics platform
✅ Spark-based
✅ Collaborative notebooks
✅ MLflow integration
✅ Delta Lake (data versioning)

Features:
- Scalable data processing
- ML workflows
- Production ML deployment
- Unity Catalog (data governance)

Pricing: Based on compute units (DBUs)
```

**Paperspace Gradient**
```
Why Gradient?
✅ Free GPU access
✅ Jupyter notebooks
✅ One-click deployments
✅ Model versioning
✅ Collaborative workflows

Tiers:
- Free: Limited GPU hours
- Growth: More GPUs, more storage
- Pro: Dedicated resources
```

**Colab (Google Colaboratory)**
```python
# Why Colab?
# ✅ Free (with limitations)
# ✅ Free GPU/TPU
# ✅ No setup required
# ✅ Easy sharing
# ✅ Drive integration

# Access GPU
# Runtime → Change runtime type → GPU

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Limitations:
# - 12 hour session limit
# - Limited compute/memory
# - Can disconnect randomly
```

---

## 📊 Data Processing Tools

**pandas**
```python
import pandas as pd

# Why pandas?
# ✅ DataFrame abstraction
# ✅ Rich functionality
# ✅ SQL-like operations
# ✅ Great for structured data

# Common operations
df = pd.read_csv('data.csv')
df.head()
df.describe()
df.groupby('category').mean()
df.merge(other_df, on='key')
```

**NumPy**
```python
import numpy as np

# Why NumPy?
# ✅ Fast numerical computations
# ✅ Foundation for pandas, scikit-learn
# ✅ Broadcasting
# ✅ Linear algebra

# Common operations
arr = np.array([1, 2, 3, 4, 5])
arr.mean()
arr.std()
np.dot(matrix1, matrix2)
```

**Apache Spark (PySpark)**
```python
from pyspark.sql import SparkSession

# Why Spark?
# ✅ Distributed computing
# ✅ Handles big data
# ✅ Scalable
# ✅ SQL-like API

# Example
spark = SparkSession.builder.appName("ML").getOrCreate()
df = spark.read.csv("large_data.csv", header=True, inferSchema=True)
df.groupBy("category").count().show()
```

**Dask**
```python
import dask.dataframe as dd

# Why Dask?
# ✅ Parallel pandas
# ✅ Out-of-core computation
# ✅ Familiar pandas API
# ✅ Good for medium data

# Example
df = dd.read_csv('large_data_*.csv')
result = df.groupby('column').mean().compute()
```

**Polars**
```python
import polars as pl

# Why Polars?
# ✅ Faster than pandas
# ✅ Lower memory usage
# ✅ Lazy evaluation
# ✅ Better parallelization

# Example
df = pl.read_csv('data.csv')
result = df.groupby('category').agg(pl.col('value').mean())
```

---

## 📈 Visualization Tools

**Matplotlib**
```python
import matplotlib.pyplot as plt

# Why Matplotlib?
# ✅ Most popular
# ✅ Highly customizable
# ✅ Publication-quality plots
# ✅ Foundation for other libraries

# Example
plt.plot(x, y)
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('My Plot')
plt.show()
```

**Seaborn**
```python
import seaborn as sns

# Why Seaborn?
# ✅ Statistical visualizations
# ✅ Beautiful default styles
# ✅ Built on matplotlib
# ✅ Easy complex plots

# Example
sns.scatterplot(data=df, x='age', y='income', hue='gender')
sns.heatmap(correlation_matrix, annot=True)
```

**Plotly**
```python
import plotly.express as px

# Why Plotly?
# ✅ Interactive plots
# ✅ Web-based
# ✅ Great for dashboards
# ✅ Easy to use

# Example
fig = px.scatter(df, x='age', y='income', color='gender')
fig.show()
```

**Streamlit**
```python
import streamlit as st

# Why Streamlit?
# ✅ Quick ML app deployment
# ✅ Interactive widgets
# ✅ No frontend knowledge needed
# ✅ Easy sharing

# Example
st.title('My ML App')
uploaded_file = st.file_uploader("Upload CSV")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)

# Run: streamlit run app.py
```

---

## 🔄 MLOps Tools

**MLflow**
```python
import mlflow

# Why MLflow?
# ✅ Experiment tracking
# ✅ Model registry
# ✅ Model deployment
# ✅ Framework-agnostic

# Example
with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.sklearn.log_model(model, "model")
```

**Weights & Biases (wandb)**
```python
import wandb

# Why wandb?
# ✅ Beautiful dashboards
# ✅ Experiment tracking
# ✅ Hyperparameter sweeps
# ✅ Model versioning
# ✅ Collaboration features

# Example
wandb.init(project="my-project")
wandb.config.learning_rate = 0.01

for epoch in range(10):
    # Training...
    wandb.log({"loss": loss, "accuracy": acc})
```

**TensorBoard**
```python
from torch.utils.tensorboard import SummaryWriter

# Why TensorBoard?
# ✅ Free
# ✅ TensorFlow integration
# ✅ Visualization of training
# ✅ Graph visualization

# Example
writer = SummaryWriter('runs/experiment_1')

for epoch in range(10):
    # Training...
    writer.add_scalar('Loss/train', loss, epoch)
    writer.add_scalar('Accuracy/train', accuracy, epoch)

writer.close()

# View: tensorboard --logdir=runs
```

**Kubeflow**
```
Why Kubeflow?
✅ ML on Kubernetes
✅ End-to-end workflows
✅ Scalable pipelines
✅ Multi-framework support

Components:
- Notebooks: JupyterHub
- Pipelines: Workflow orchestration
- Training: Distributed training
- Serving: Model deployment
```

---

## 🎓 Learning Resources

### Online Courses

**Coursera**
- Andrew Ng's Machine Learning
- Deep Learning Specialization
- Natural Language Processing Specialization

**fast.ai**
- Practical Deep Learning for Coders
- Free, top-down approach

**DeepLearning.AI**
- Various specializations on Coursera
- High-quality content from Andrew Ng

**Udacity**
- Machine Learning Engineer Nanodegree
- Deep Learning Nanodegree

### Books

**Beginner**
- "Hands-On Machine Learning" by Aurélien Géron
- "Python Machine Learning" by Sebastian Raschka

**Advanced**
- "Deep Learning" by Goodfellow, Bengio, Courville
- "Pattern Recognition and Machine Learning" by Bishop

**Math**
- "Mathematics for Machine Learning" by Deisenroth, Faisal, Ong

### Websites & Blogs

**Papers & Research**
- arXiv.org: Pre-print server
- Papers with Code: Papers + implementations
- Distill.pub: Visual explanations

**Blogs**
- Towards Data Science (Medium)
- Google AI Blog
- OpenAI Blog
- DeepMind Blog

**Practice**
- Kaggle: Competitions + datasets
- LeetCode: Coding practice
- Project Euler: Math problems

---

## 🔗 Quick Links Summary

**Datasets**
- Kaggle: https://www.kaggle.com/datasets
- UCI ML Repository: https://archive.ics.uci.edu/
- HuggingFace Datasets: https://huggingface.co/datasets
- Google Dataset Search: https://datasetsearch.research.google.com/

**Model Hubs**
- HuggingFace: https://huggingface.co/models
- TensorFlow Hub: https://tfhub.dev/
- PyTorch Hub: https://pytorch.org/hub/
- ONNX Model Zoo: https://github.com/onnx/models

**Documentation**
- PyTorch: https://pytorch.org/docs/
- TensorFlow: https://www.tensorflow.org/api_docs
- scikit-learn: https://scikit-learn.org/stable/
- Pandas: https://pandas.pydata.org/docs/

**Communities**
- Reddit: r/MachineLearning, r/learnmachinelearning
- Discord: Various ML servers
- Twitter: #ML, #DeepLearning, #AI
- Stack Overflow: [machine-learning], [deep-learning]

---

*This guide is regularly updated. Bookmark and revisit for new tools and resources!*
