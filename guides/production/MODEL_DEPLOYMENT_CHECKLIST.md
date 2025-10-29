# ✅ ML Model Deployment Checklist

**Production-Ready Deployment Guide**

> Last Updated: October 2025
> Use this checklist before deploying any ML model to production

---

## 📋 Quick Overview

Use this checklist to ensure your ML model is ready for production deployment. Each section contains critical items that must be verified before going live.

**Deployment Stages:**
1. Pre-Development
2. Development & Training
3. Pre-Deployment
4. Deployment
5. Post-Deployment

---

## 🎯 Stage 1: Pre-Development

### Business Requirements

- [ ] **Problem is clearly defined**
  - Business objective documented
  - Success metrics defined
  - Baseline performance established

- [ ] **ML is appropriate solution**
  - Problem requires prediction/pattern recognition
  - Data is available
  - Simpler heuristics won't work

- [ ] **Stakeholders aligned**
  - Expected performance communicated
  - Timeline agreed upon
  - Resources allocated

### Data Requirements

- [ ] **Sufficient data available**
  - Minimum 1000 samples per class (ideal: 10,000+)
  - Data quality assessed
  - Labeling strategy defined

- [ ] **Data access secured**
  - Legal approval obtained
  - Privacy requirements understood (GDPR, CCPA)
  - Data storage location determined

- [ ] **Data pipeline planned**
  - Data sources identified
  - Update frequency defined
  - Data versioning strategy chosen

---

## 🏗️ Stage 2: Development & Training

### Data Quality

- [ ] **Data validated**
  - Schema checked
  - Missing values analyzed
  - Outliers identified
  - Distribution examined

- [ ] **Data cleaning documented**
  - Cleaning steps recorded
  - Before/after statistics
  - Reproducible pipeline created

- [ ] **No data leakage**
  - Train/val/test split before any processing
  - No future information in features
  - Target encoding uses proper CV

### Model Development

- [ ] **Baseline established**
  - Simple model trained (e.g., logistic regression)
  - Random baseline measured
  - Human performance estimated (if applicable)

- [ ] **Model selection justified**
  - Multiple models compared
  - Trade-offs documented (accuracy vs. speed vs. interpretability)
  - Final model choice explained

- [ ] **Hyperparameters tuned**
  - Systematic search performed (grid/random/bayesian)
  - Cross-validation used
  - Best parameters documented

- [ ] **Model versioned**
  - Code in version control (git)
  - Models tracked (MLflow/Weights & Biases)
  - Experiments documented

### Training Process

- [ ] **Training reproducible**
  - Random seeds set
  - Environment documented (requirements.txt/environment.yml)
  - Training script automated

- [ ] **Training monitored**
  - Loss curves checked
  - Overfitting detected early
  - Learning rate appropriate

- [ ] **Checkpoints saved**
  - Best model saved
  - Multiple checkpoints kept
  - Easy to resume training

### Model Evaluation

- [ ] **Comprehensive evaluation**
  - Multiple metrics calculated
  - Confusion matrix analyzed
  - Performance per class checked

- [ ] **Test set properly used**
  - Held out until final evaluation
  - Representative of production data
  - Large enough for statistical significance

- [ ] **Edge cases tested**
  - Out-of-distribution examples
  - Adversarial examples
  - Boundary cases

- [ ] **Error analysis conducted**
  - Failure modes identified
  - Patterns in errors found
  - Improvements suggested

### Model Interpretability

- [ ] **Model is interpretable** (or explainable)
  - Feature importance calculated
  - SHAP/LIME values computed (for black-box models)
  - Predictions can be explained to stakeholders

- [ ] **Fairness evaluated**
  - Performance across demographic groups checked
  - Bias identified and mitigated
  - Ethical implications considered

---

## 🚀 Stage 3: Pre-Deployment

### Model Packaging

- [ ] **Model serialized correctly**
  - Saved in appropriate format (pkl, ONNX, TorchScript)
  - Saving/loading tested
  - Version number embedded

- [ ] **Dependencies documented**
  - requirements.txt created
  - Python version specified
  - System dependencies listed

- [ ] **Preprocessing bundled**
  - Same preprocessing for training and inference
  - Scalers/encoders saved with model
  - Pipeline validated

### API Development

- [ ] **API implemented**
  - REST API created (Flask/FastAPI)
  - Request/response schemas defined (Pydantic)
  - Error handling implemented

- [ ] **API documented**
  - Endpoint documentation (Swagger/OpenAPI)
  - Example requests provided
  - Rate limits specified

- [ ] **API tested**
  - Unit tests written
  - Integration tests passed
  - Load testing performed

### Containerization

- [ ] **Docker image created**
  - Dockerfile written
  - Multi-stage build used (for smaller image)
  - Image tested locally

- [ ] **Image optimized**
  - Minimal base image used
  - Layers optimized
  - Size reasonable (<1GB ideally)

- [ ] **Container security**
  - No secrets in image
  - Non-root user used
  - Vulnerabilities scanned

### Infrastructure

- [ ] **Compute resources determined**
  - CPU vs GPU decided
  - Memory requirements calculated
  - Scaling strategy planned

- [ ] **Deployment platform chosen**
  - Cloud provider selected (AWS/GCP/Azure)
  - Or on-premise infrastructure prepared
  - Costs estimated

- [ ] **Networking configured**
  - Endpoints defined
  - Load balancer setup
  - SSL certificates obtained

---

## 🔧 Stage 4: Deployment

### Pre-Deployment Testing

- [ ] **Staging environment tested**
  - Model deployed to staging
  - Integration with other services verified
  - Performance benchmarked

- [ ] **Load testing completed**
  - Expected traffic simulated
  - Latency under load measured
  - Auto-scaling tested

- [ ] **Canary deployment prepared**
  - Small percentage of traffic routed to new model
  - Rollback plan ready
  - Comparison metrics defined

### Monitoring Setup

- [ ] **Logging configured**
  - All predictions logged
  - Errors logged with context
  - Log rotation setup

- [ ] **Metrics tracked**
  - Request count
  - Latency (p50, p95, p99)
  - Error rate
  - Model-specific metrics (confidence, drift)

- [ ] **Alerting configured**
  - Critical failures alert immediately
  - Performance degradation monitored
  - On-call rotation defined

- [ ] **Dashboards created**
  - Real-time metrics visible
  - Historical trends tracked
  - Accessible to team

### Deployment Execution

- [ ] **Deployment automated**
  - CI/CD pipeline configured
  - Automated tests in pipeline
  - Manual approval gate for production

- [ ] **Gradual rollout**
  - Start with 5-10% traffic
  - Monitor for 24-48 hours
  - Gradually increase to 100%

- [ ] **Rollback tested**
  - Quick rollback mechanism available
  - Rollback tested in staging
  - Team trained on rollback procedure

---

## 📊 Stage 5: Post-Deployment

### Monitoring

- [ ] **Performance monitored continuously**
  - Accuracy/F1/AUC tracked (with ground truth when available)
  - Latency monitored
  - Resource usage tracked

- [ ] **Data drift detected**
  - Input distribution monitored
  - Statistical tests for drift (KS test, PSI)
  - Alerts on significant drift

- [ ] **Model drift detected**
  - Prediction distribution monitored
  - Performance degradation detected
  - Retraining triggered when needed

### Maintenance

- [ ] **Regular retraining scheduled**
  - Retraining frequency defined (weekly/monthly)
  - Automated or manual process
  - New data incorporated

- [ ] **Model registry updated**
  - All models versions tracked
  - Production model tagged
  - Deprecated models archived

- [ ] **Documentation maintained**
  - Model card created
  - Known issues documented
  - Updates logged

### Continuous Improvement

- [ ] **Feedback loop established**
  - User feedback collected
  - Errors analyzed
  - Model improvements prioritized

- [ ] **A/B testing**
  - New models tested against production
  - Statistical significance calculated
  - Winner gradually rolled out

- [ ] **Model performance reviewed**
  - Weekly/monthly review meetings
  - Metrics reviewed with stakeholders
  - Improvement areas identified

---

## 🔒 Security Checklist

### Data Security

- [ ] **Data encrypted**
  - At rest
  - In transit
  - In logs (PII removed)

- [ ] **Access controlled**
  - Role-based access control (RBAC)
  - Least privilege principle
  - Regular access audits

### Model Security

- [ ] **Model protected**
  - Model weights not publicly accessible
  - API authentication required
  - Rate limiting enabled

- [ ] **Adversarial robustness**
  - Adversarial examples tested
  - Input validation implemented
  - Anomaly detection added

### Compliance

- [ ] **Regulatory compliance**
  - GDPR compliance verified (if applicable)
  - CCPA compliance verified (if applicable)
  - Industry-specific regulations met (HIPAA, etc.)

- [ ] **Audit trail**
  - All predictions logged
  - Model changes tracked
  - Access logged

---

## 📈 Performance Checklist

### Latency

- [ ] **Latency requirements met**
  - P95 latency < target (e.g., 100ms)
  - P99 latency < 2x target
  - Timeout configured

- [ ] **Optimization applied**
  - Model quantized if needed
  - Batch inference used when possible
  - Caching implemented where appropriate

### Throughput

- [ ] **Throughput requirements met**
  - Requests per second tested
  - Auto-scaling configured
  - Rate limiting prevents overload

### Cost

- [ ] **Cost optimized**
  - Resource usage monitored
  - Auto-scaling prevents over-provisioning
  - Spot instances used where appropriate
  - Budget alerts configured

---

## 📝 Documentation Checklist

### Model Documentation

- [ ] **Model card created**
  - Model architecture described
  - Training data described
  - Performance metrics documented
  - Limitations listed
  - Intended use specified

- [ ] **Training documented**
  - Hyperparameters listed
  - Training time recorded
  - Compute resources used
  - Random seeds documented

### Deployment Documentation

- [ ] **Deployment guide written**
  - Step-by-step deployment instructions
  - Configuration parameters explained
  - Troubleshooting section included

- [ ] **API documentation complete**
  - All endpoints documented
  - Request/response examples provided
  - Error codes explained

### Runbooks Created

- [ ] **Incident response runbook**
  - Common issues and solutions
  - Escalation procedures
  - Contact information

- [ ] **Maintenance runbook**
  - Retraining procedures
  - Model update procedures
  - Rollback procedures

---

## 🎯 Quick Pre-Deployment Checklist

**Use this condensed checklist for final verification:**

### Critical Items (Must Have)

- [ ] Model performance meets business requirements
- [ ] No data leakage
- [ ] Reproducible training
- [ ] Comprehensive tests passing
- [ ] API working and documented
- [ ] Docker image built and tested
- [ ] Logging and monitoring setup
- [ ] Rollback plan ready
- [ ] Security reviewed
- [ ] Documentation complete

### Nice to Have

- [ ] A/B testing framework ready
- [ ] Feature flags for easy toggle
- [ ] Shadow mode deployment option
- [ ] Automated retraining pipeline
- [ ] Model explainability tools
- [ ] Cost optimization applied

---

## 🚨 Red Flags (Do Not Deploy If:)

❌ **Data Issues:**
- Data quality not verified
- Significant data leakage present
- Test set not representative

❌ **Model Issues:**
- Overfitting not addressed
- Performance below baseline
- Edge cases not tested

❌ **Infrastructure Issues:**
- No monitoring in place
- No rollback plan
- No error handling

❌ **Security Issues:**
- Credentials hardcoded
- No authentication
- Data not encrypted

❌ **Process Issues:**
- Not tested in staging
- No documentation
- Team not trained

---

## 📚 Templates and Tools

### Model Card Template

```markdown
# Model Card: [Model Name]

## Model Details
- **Model type:** [e.g., Random Forest Classifier]
- **Version:** [e.g., 1.0.0]
- **Date:** [YYYY-MM-DD]
- **Developers:** [Team/Individual]

## Intended Use
- **Primary use:** [Description]
- **Out-of-scope uses:** [What model should NOT be used for]

## Training Data
- **Dataset:** [Name and source]
- **Size:** [Number of samples]
- **Features:** [Number and types]
- **Date range:** [When data was collected]

## Evaluation Data
- **Dataset:** [Name]
- **Size:** [Number of samples]
- **Distribution:** [How it differs from training]

## Performance
- **Accuracy:** [Value]
- **Precision:** [Value]
- **Recall:** [Value]
- **F1 Score:** [Value]
- **AUC-ROC:** [Value]

## Limitations
- [List known limitations]
- [Edge cases where model fails]
- [Bias or fairness concerns]

## Ethical Considerations
- [Privacy considerations]
- [Fairness across groups]
- [Potential misuse]
```

### Deployment Runbook Template

```markdown
# Deployment Runbook: [Model Name]

## Pre-Deployment
1. Verify all tests passing
2. Check staging environment
3. Review monitoring dashboards
4. Inform stakeholders

## Deployment Steps
1. [Step-by-step instructions]
2. [With exact commands]
3. [And expected outputs]

## Verification
- [ ] Health check passing
- [ ] Sample predictions correct
- [ ] Latency acceptable
- [ ] Logs flowing

## Rollback
1. [Quick rollback steps]
2. [One command if possible]

## Contacts
- On-call: [Name, Phone]
- Escalation: [Name, Phone]
- Manager: [Name, Phone]
```

---

## 🎓 Best Practices Summary

### Do's ✅

- Start simple, iterate
- Test extensively
- Monitor everything
- Document thoroughly
- Automate deployment
- Plan for failure
- Get stakeholder buy-in

### Don'ts ❌

- Skip testing
- Deploy without monitoring
- Ignore security
- Forget documentation
- Make changes without version control
- Deploy on Friday (!)
- Skip staging environment

---

**Remember:** Production ML is 10% model training, 90% everything else!

Take your time with this checklist. A week spent on proper deployment saves months of firefighting.

---

*Last Updated: October 2025*
*For questions or suggestions, please open an issue*
