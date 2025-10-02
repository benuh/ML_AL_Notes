# 🚀 RAG Tutorial - Quick Start Guide

Get your RAG system running in 5 minutes!

## ⚡ Installation (2 minutes)

### Option 1: Quick Install (Recommended)
```bash
cd rag_tutorial
pip install -r requirements.txt
```

### Option 2: Minimal Install (Faster, ~500MB)
```bash
pip install sentence-transformers faiss-cpu chromadb nltk jupyter matplotlib seaborn pandas scikit-learn
```

### Option 3: Use Existing ML Environment
If you already have a machine learning environment:
```bash
pip install sentence-transformers faiss-cpu chromadb
```

---

## 🎯 Three Ways to Learn

### 1. 📓 Interactive Notebook (Recommended - 60 min)

**Full hands-on tutorial with code you can run:**

```bash
jupyter notebook RAG_Interactive_Lab.ipynb
```

**What you'll do:**
- ✅ Load real company documents
- ✅ Try 3 different chunking strategies
- ✅ Generate vector embeddings
- ✅ Build semantic search engine
- ✅ Create complete RAG system
- ✅ Test with realistic queries

**Perfect for:** Hands-on learners who want to understand by doing

---

### 2. 📖 Read the Documentation (20 min)

**Comprehensive guide with diagrams:**

```bash
open README.md  # or just view it on GitHub
```

**What's inside:**
- 🎯 What is RAG and why it matters
- 🔬 How RAG works (with flowcharts)
- 🏗️ System architecture
- 🎓 Key concepts explained
- 🌍 Real-world applications
- 📊 Performance tips

**Perfect for:** Understanding concepts before diving into code

---

### 3. 🎨 Visual Learning (10 min)

**Generate beautiful diagrams:**

```bash
cd rag_tutorial
python utils/visualizations.py
```

**Generates 5 diagrams:**
1. `rag_overview.png` - High-level RAG flow
2. `indexing_pipeline.png` - How documents are indexed
3. `query_pipeline.png` - How queries are processed
4. `chunking_strategies.png` - Comparing chunking methods
5. `embedding_similarity.png` - How semantic search works

**Perfect for:** Visual learners who want to see the big picture

---

## 🧪 Quick Test (3 minutes)

Test that everything works:

```python
# In Python or Jupyter notebook
from sentence_transformers import SentenceTransformer

# Load model (downloads ~80MB first time)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings
texts = ["remote work policy", "work from home"]
embeddings = model.encode(texts)

print(f"✅ Success! Generated embeddings with shape: {embeddings.shape}")
# Should see: (2, 384)
```

If this works, you're ready to go! 🎉

---

## 📚 Sample Documents Included

We've prepared realistic company documents for you:

1. **company_policies.txt** (3.7 KB)
   - Work hours, remote work, leave policies
   - Compensation and benefits
   - Code of conduct

2. **product_documentation.txt** (6.2 KB)
   - Installation instructions
   - Troubleshooting guides
   - Feature documentation

3. **sales_reports.txt** (8.5 KB)
   - Q4 2023 performance data
   - Regional breakdowns
   - Customer metrics

**Total:** ~18 KB of searchable content (perfect for learning!)

---

## 💡 Example Queries to Try

Once your RAG system is running, test these:

```python
queries = [
    "What is the remote work policy?",
    "How do I install the software on Mac?",
    "What were the Q4 sales numbers?",
    "Tell me about parental leave benefits",
    "How do I troubleshoot sync issues?",
    "What is the PTO policy for new employees?",
    "Which region had the best performance?"
]
```

---

## 🎓 Learning Path

### Beginner (1-2 hours)
1. Read README.md (20 min)
2. Run visualization script (5 min)
3. Complete notebook sections 1-5 (45 min)
4. Experiment with sample queries (10 min)

### Intermediate (3-4 hours)
1. Complete full notebook (90 min)
2. Try different chunking strategies (30 min)
3. Experiment with different embedding models (30 min)
4. Complete exercises at end of notebook (60 min)

### Advanced (Full day)
1. Complete intermediate path
2. Integrate with OpenAI/Anthropic API
3. Add your own documents
4. Implement hybrid search
5. Build evaluation framework
6. Deploy as API

---

## 🔧 Troubleshooting

### Issue: `ModuleNotFoundError`
```bash
# Solution: Install missing package
pip install <missing_package_name>
```

### Issue: `OSError: [Errno 28] No space left on device`
```bash
# Models download to ~/.cache/torch
# Free up space or set custom cache:
export TRANSFORMERS_CACHE=/path/to/custom/cache
```

### Issue: `OutOfMemoryError` when encoding
```python
# Solution: Process in smaller batches
embeddings = model.encode(texts, batch_size=8)  # Default is 32
```

### Issue: Jupyter notebook won't open
```bash
# Solution: Install/upgrade jupyter
pip install --upgrade jupyter notebook
```

### Issue: Diagrams not generating
```bash
# Solution: Install matplotlib
pip install matplotlib seaborn
```

---

## 📊 Performance Expectations

### On a typical laptop (8GB RAM):

| Task | Time | Notes |
|------|------|-------|
| Install dependencies | 2-5 min | First time only |
| Download embedding model | 1-2 min | First time only (~80MB) |
| Index 18KB documents | 5-10 sec | Our sample data |
| Single query search | <100ms | After indexing |
| Generate answer with LLM | 2-5 sec | If using API |

### Model sizes:

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| all-MiniLM-L6-v2 | 80MB | ⚡⚡⚡ Fast | ⭐⭐⭐ Good |
| all-mpnet-base-v2 | 420MB | ⚡⚡ Medium | ⭐⭐⭐⭐ Better |
| OpenAI ada-002 | API | ⚡ API call | ⭐⭐⭐⭐⭐ Best |

**Recommendation:** Start with `all-MiniLM-L6-v2` (default in tutorial)

---

## 🎯 Next Steps After Tutorial

1. **Add your own documents**
   ```bash
   # Just drop .txt files here:
   rag_tutorial/data/sample_documents/your_doc.txt
   ```

2. **Connect to real LLM**
   ```python
   # Get API key from OpenAI/Anthropic
   import openai
   openai.api_key = "your-key-here"
   ```

3. **Try different vector DBs**
   - FAISS: Fastest for local
   - ChromaDB: Best for persistence
   - Pinecone: Best for production

4. **Scale up**
   - Test with 100+ documents
   - Try different chunking strategies
   - Implement hybrid search

5. **Deploy**
   - Build FastAPI wrapper
   - Containerize with Docker
   - Deploy to cloud

---

## 📞 Getting Help

### Resources in this tutorial:
- `README.md` - Comprehensive guide
- `RAG_Interactive_Lab.ipynb` - Interactive code tutorial
- `data/sample_documents/` - Example documents
- `utils/visualizations.py` - Diagram generator

### External resources:
- [LangChain Docs](https://python.langchain.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss/wiki)

### Common issues:
Check the Troubleshooting section above!

---

## ✅ Checklist

Before starting, make sure you have:

- [ ] Python 3.8+ installed
- [ ] pip working
- [ ] At least 2GB free disk space
- [ ] Internet connection (for downloading models)
- [ ] (Optional) Jupyter installed
- [ ] (Optional) OpenAI/Anthropic API key

---

## 🎉 Ready to Start!

Choose your path:

**👨‍💻 Hands-on learner?**
```bash
jupyter notebook RAG_Interactive_Lab.ipynb
```

**📖 Theory first?**
```bash
open README.md
```

**🎨 Visual learner?**
```bash
python utils/visualizations.py
open diagrams/
```

**Let's build something amazing! 🚀**

---

## 💬 Feedback

Found this helpful? Have suggestions? Let us know!

**Happy learning!** 🎓
