# ML Practical Templates & Checklists - Part 2

**Complete Collection of Production-Ready Templates for ML Projects (Continued)**

---

## 7. A/B Testing Plan Template

```markdown
# A/B Test Plan: [Test Name]

**Test ID:** [Unique identifier]
**Owner:** [Name]
**Start Date:** [YYYY-MM-DD]
**End Date:** [YYYY-MM-DD]
**Status:** [Planning / Running / Completed / Cancelled]

## Executive Summary
**Objective:** [One sentence describing what you're testing]
**Expected Impact:** [e.g., +5% conversion rate]
**Decision Criteria:** [What metric needs to improve by how much]
**Estimated Duration:** [X weeks]

## Hypothesis
### Problem Statement
[Describe the current problem or opportunity]

### Proposed Solution
[Describe the treatment/new model]

### Hypothesis Statement
**We believe that** [treatment]
**Will result in** [outcome]
**We will know this is true when** [measurable criteria]

**Example:**
- We believe that deploying the new recommendation model
- Will result in a 5% increase in click-through rate
- We will know this is true when CTR increases from 3.2% to 3.36% with p < 0.05

## Experiment Design

### Variants
| Variant | Description | Traffic % | Expected Users/Day |
|---------|-------------|-----------|---------------------|
| Control (A) | Current production model v2.1 | 50% | 50,000 |
| Treatment (B) | New model v3.0 | 50% | 50,000 |

### Randomization
- **Unit of Randomization:** [User ID / Session ID / Request ID]
- **Assignment Method:** [Hash-based / Random / Stratified]
- **Consistency:** [User sees same variant across sessions: Yes/No]

```python
def assign_variant(user_id, experiment_id, salt="exp_001"):
    """Deterministic assignment using hash"""
    hash_input = f"{user_id}_{experiment_id}_{salt}"
    hash_value = hashlib.md5(hash_input.encode()).hexdigest()
    hash_int = int(hash_value, 16)

    if hash_int % 100 < 50:
        return "control"
    else:
        return "treatment"
```

### Sample Size Calculation
```python
from statsmodels.stats.power import zt_ind_solve_power

# Parameters
baseline_rate = 0.032  # Current CTR: 3.2%
mde = 0.05  # Minimum Detectable Effect: 5% relative increase
alpha = 0.05  # Significance level
power = 0.80  # Statistical power

# Calculate required sample size per variant
effect_size = (baseline_rate * (1 + mde) - baseline_rate) / sqrt(baseline_rate * (1 - baseline_rate))
n_per_variant = zt_ind_solve_power(effect_size=effect_size, alpha=alpha, power=power)

print(f"Required sample size per variant: {n_per_variant:.0f}")
# Output: Required sample size per variant: 76,543
```

**Calculated Sample Size:** 76,543 users per variant
**Expected Daily Users:** 50,000 per variant
**Required Duration:** 2 days (minimum), running for 14 days for robustness

### Traffic Allocation
- **Phase 1 (Days 1-3):** 5% Control, 5% Treatment (safety check)
- **Phase 2 (Days 4-14):** 50% Control, 50% Treatment (full test)
- **Holdout Group:** 10% excluded from experiment (for long-term monitoring)

## Metrics

### Primary Metric
| Metric | Definition | Current Value | Target | MDE |
|--------|------------|---------------|--------|-----|
| Click-Through Rate (CTR) | `clicks / impressions` | 3.2% | 3.36% | +5% relative |

**Success Criteria:** CTR improvement with p-value < 0.05

### Secondary Metrics (Guardrail Metrics)
| Metric | Definition | Current Value | Acceptable Range |
|--------|------------|---------------|------------------|
| Conversion Rate | `purchases / users` | 1.8% | 1.7% - 2.0% |
| Revenue per User | `total_revenue / users` | $12.50 | > $12.00 |
| Page Load Time | `p95 latency` | 450ms | < 500ms |
| Error Rate | `errors / requests` | 0.05% | < 0.1% |

**Guardrail Conditions:**
- ❌ Stop experiment if any guardrail metric breached
- ⚠️ Investigate if secondary metrics show unexpected patterns

### Long-Term Metrics (Monitor post-launch)
- User retention (7-day, 30-day)
- Customer lifetime value (LTV)
- User satisfaction (NPS)

## Data Collection

### Events to Track
```python
# Event schema
{
    "event_type": "recommendation_impression",
    "timestamp": "2024-01-15T10:30:45Z",
    "user_id": "user_12345",
    "session_id": "sess_67890",
    "experiment_id": "rec_model_v3",
    "variant": "treatment",
    "recommendation_ids": ["item_1", "item_2", "item_3"],
    "position": [1, 2, 3],
    "context": {
        "page": "homepage",
        "device": "mobile"
    }
}

{
    "event_type": "recommendation_click",
    "timestamp": "2024-01-15T10:31:12Z",
    "user_id": "user_12345",
    "session_id": "sess_67890",
    "experiment_id": "rec_model_v3",
    "variant": "treatment",
    "recommendation_id": "item_2",
    "position": 2
}
```

### Data Quality Checks
- [ ] All users assigned to exactly one variant
- [ ] Assignment ratio correct (50/50 or as designed)
- [ ] No spillover between variants
- [ ] Complete event logging (no missing data)
- [ ] Consistent assignment across sessions

## Analysis Plan

### Statistical Test
- **Test Type:** Two-sample z-test for proportions
- **Significance Level:** α = 0.05
- **One-sided or Two-sided:** Two-sided
- **Multiple Testing Correction:** Bonferroni (if testing multiple metrics)

### Analysis Code
```python
from scipy import stats
import numpy as np

def analyze_experiment(control_data, treatment_data, metric='ctr'):
    """
    Analyze A/B test results
    """
    # Calculate metrics
    control_ctr = control_data['clicks'].sum() / control_data['impressions'].sum()
    treatment_ctr = treatment_data['clicks'].sum() / treatment_data['impressions'].sum()

    # Two-proportion z-test
    control_successes = control_data['clicks'].sum()
    treatment_successes = treatment_data['clicks'].sum()
    control_trials = control_data['impressions'].sum()
    treatment_trials = treatment_data['impressions'].sum()

    # Calculate pooled proportion
    pooled_p = (control_successes + treatment_successes) / (control_trials + treatment_trials)

    # Calculate standard error
    se = np.sqrt(pooled_p * (1 - pooled_p) * (1/control_trials + 1/treatment_trials))

    # Calculate z-score
    z_score = (treatment_ctr - control_ctr) / se

    # Calculate p-value (two-sided)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

    # Calculate confidence interval
    ci_margin = 1.96 * se
    ci_lower = (treatment_ctr - control_ctr) - ci_margin
    ci_upper = (treatment_ctr - control_ctr) + ci_margin

    # Calculate relative lift
    relative_lift = (treatment_ctr - control_ctr) / control_ctr

    return {
        'control_ctr': control_ctr,
        'treatment_ctr': treatment_ctr,
        'absolute_lift': treatment_ctr - control_ctr,
        'relative_lift': relative_lift,
        'p_value': p_value,
        'ci_95': (ci_lower, ci_upper),
        'statistically_significant': p_value < 0.05
    }
```

### Segmentation Analysis
Analyze results across key segments:
- Device type (mobile, desktop, tablet)
- User tenure (new, returning, power users)
- Geographic region
- Time of day / day of week

### Checks for Validity
- [ ] Sample Ratio Mismatch (SRM) check
- [ ] A/A test (if new infrastructure)
- [ ] Pre-experiment period comparison (check for pre-existing differences)
- [ ] Novelty effect check (performance over time)

```python
def check_sample_ratio_mismatch(control_count, treatment_count, expected_ratio=0.5):
    """
    Check if observed sample ratio matches expected ratio
    """
    total = control_count + treatment_count
    expected_control = total * expected_ratio
    expected_treatment = total * (1 - expected_ratio)

    # Chi-square test
    chi_square = (
        (control_count - expected_control)**2 / expected_control +
        (treatment_count - expected_treatment)**2 / expected_treatment
    )

    p_value = 1 - stats.chi2.cdf(chi_square, df=1)

    if p_value < 0.001:  # Very conservative threshold
        print(f"⚠️ WARNING: Sample Ratio Mismatch detected (p={p_value:.4f})")
        return False
    else:
        print(f"✅ Sample ratio looks good (p={p_value:.4f})")
        return True
```

## Decision Framework

### Decision Tree
```
If primary metric improved AND statistically significant AND guardrails OK:
    → Ship treatment to 100%

Else if primary metric improved BUT NOT statistically significant:
    If practical significance large AND business judgment positive:
        → Consider shipping (with caution)
    Else:
        → Continue experiment OR abandon

Else if primary metric neutral AND secondary metrics improved:
    → Business decision (consider shipping if strategic value)

Else if primary metric degraded OR guardrails breached:
    → Abandon treatment
```

### Decision Criteria
| Scenario | Primary Metric | Statistical Sig. | Guardrails | Decision |
|----------|----------------|------------------|------------|----------|
| 1 | ✅ Improved | ✅ Yes (p<0.05) | ✅ All OK | **Ship** |
| 2 | ✅ Improved | ❌ No (p>0.05) | ✅ All OK | Evaluate / Extend |
| 3 | ➖ Neutral | ✅ Yes | ✅ All OK | Business decision |
| 4 | ❌ Degraded | ✅ Yes (p<0.05) | ✅ All OK | **Don't ship** |
| 5 | ✅ Improved | ✅ Yes | ❌ Breached | **Don't ship** |

## Risks & Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Model latency spike | High | Low | Monitor p95 latency, kill switch ready |
| Sample ratio mismatch | High | Low | Daily SRM checks, investigate immediately |
| Novelty effect inflates results | Medium | Medium | Run for 2+ weeks, analyze weekly trends |
| Segment-specific degradation | Medium | Low | Detailed segment analysis before ship |
| Data logging failure | High | Low | Redundant logging, real-time monitoring |

### Kill Switch Criteria
Stop experiment immediately if:
- Error rate > 1% (20x normal)
- p95 latency > 1000ms (2x threshold)
- Revenue per user < $11.00 (10% drop)
- Significant user complaints (> 10 support tickets/day)

## Experiment Checklist

### Pre-Launch
- [ ] Hypothesis clearly defined
- [ ] Success metrics and guardrails defined
- [ ] Sample size calculated
- [ ] Randomization logic implemented and tested
- [ ] A/A test passed (if applicable)
- [ ] Data logging verified
- [ ] Monitoring dashboards ready
- [ ] Kill switch tested
- [ ] Stakeholder alignment on decision criteria
- [ ] Experiment plan reviewed and approved

### During Experiment
- [ ] Daily monitoring of key metrics
- [ ] Daily SRM check
- [ ] Weekly review of segment performance
- [ ] Alert on any guardrail breaches
- [ ] Document any incidents or anomalies

### Post-Experiment
- [ ] Statistical analysis completed
- [ ] Segment analysis completed
- [ ] Results validated by second analyst
- [ ] Decision documented with rationale
- [ ] Stakeholders notified
- [ ] If shipping: deployment plan created
- [ ] If not shipping: learnings documented

## Timeline

| Phase | Duration | Start Date | End Date | Activities |
|-------|----------|------------|----------|------------|
| Planning | 1 week | Jan 1 | Jan 7 | Define hypothesis, metrics, sample size |
| Development | 2 weeks | Jan 8 | Jan 21 | Implement treatment, logging, monitoring |
| QA & Testing | 1 week | Jan 22 | Jan 28 | A/A test, data validation |
| Ramp-up (5%) | 3 days | Jan 29 | Jan 31 | Safety check with small traffic |
| Full Test (50%) | 14 days | Feb 1 | Feb 14 | Main experiment period |
| Analysis | 3 days | Feb 15 | Feb 17 | Statistical analysis, segment analysis |
| Decision | 2 days | Feb 18 | Feb 19 | Review results, make go/no-go decision |
| Rollout (if shipping) | 1 week | Feb 20 | Feb 26 | Gradual rollout to 100% |

## Results (To be filled after experiment)

### Headline Results
| Metric | Control | Treatment | Lift | P-value | Sig? |
|--------|---------|-----------|------|---------|------|
| CTR (Primary) | 3.21% | 3.42% | +6.5% | 0.003 | ✅ |
| Conversion Rate | 1.79% | 1.84% | +2.8% | 0.12 | ❌ |
| Revenue/User | $12.48 | $12.91 | +3.4% | 0.045 | ✅ |
| p95 Latency | 447ms | 461ms | +3.1% | 0.21 | ❌ |

### Final Decision
**Decision:** [Ship / Don't Ship / Iterate]
**Rationale:** [Explanation]
**Approved by:** [Name] - [Date]
**Deployment Date:** [Date]

### Lessons Learned
1. [Learning 1]
2. [Learning 2]
3. [Learning 3]

### Follow-up Actions
- [ ] [Action 1]
- [ ] [Action 2]
```

---

## 8. Production Deployment Checklist

```markdown
# Production Deployment Checklist: [Model Name]

**Model:** [Model Name v3.0]
**Deployment Date:** [YYYY-MM-DD]
**Deployment Engineer:** [Name]
**Approval:** [PM Name, Tech Lead Name]

## Pre-Deployment Checklist

### 1. Model Validation
- [ ] Model performance meets success criteria on test set
- [ ] Model evaluated on recent production data (last 7 days)
- [ ] No significant performance degradation over time
- [ ] Error analysis completed and documented
- [ ] Model interpretability verified (SHAP values, feature importance)
- [ ] Fairness evaluation passed
- [ ] A/B test results positive (if applicable)

**Model Metrics:**
| Metric | Test Set | Recent Prod Data | Target | Status |
|--------|----------|------------------|--------|--------|
| AUC-ROC | 0.912 | 0.908 | > 0.85 | ✅ |
| Precision | 0.823 | 0.817 | > 0.75 | ✅ |
| Recall | 0.791 | 0.785 | > 0.70 | ✅ |

### 2. Model Artifacts
- [ ] Model file serialized and versioned
- [ ] Model file size acceptable (< 100MB target)
- [ ] Preprocessing pipeline included
- [ ] Feature transformations documented
- [ ] Model loaded successfully in test environment
- [ ] Model signature validated (input/output schema)
- [ ] Model checksum/hash calculated for integrity

**Model Info:**
- Version: [3.0.1]
- Size: [67 MB]
- Format: [pickle / ONNX / SavedModel]
- Location: [s3://ml-models/prod/rec-model-v3.0.1.pkl]
- Checksum: [SHA256:a3f5b2c...]

### 3. Code Review
- [ ] Inference code reviewed by 2+ engineers
- [ ] Unit tests written (>80% coverage)
- [ ] Integration tests written
- [ ] Load tests completed
- [ ] Error handling implemented
- [ ] Logging implemented
- [ ] No hardcoded credentials or secrets
- [ ] Code follows team style guide
- [ ] Code merged to main branch

**Test Coverage:** [87%]
**Load Test Results:** [1000 RPS sustained, p95 latency 45ms]

### 4. Infrastructure
- [ ] Compute resources provisioned (CPU/GPU/memory)
- [ ] Auto-scaling configured
- [ ] Load balancer configured
- [ ] Health check endpoint implemented
- [ ] Graceful shutdown implemented
- [ ] Resource limits set (CPU, memory, timeout)
- [ ] Separate staging environment available
- [ ] Production environment access controlled

**Infrastructure Specs:**
- Instance Type: [c5.2xlarge]
- Min/Max Instances: [2 / 10]
- CPU: [8 vCPUs]
- Memory: [16 GB]
- GPU: [None / Tesla T4]
- Timeout: [5 seconds]

### 5. Dependencies
- [ ] All dependencies listed in requirements.txt
- [ ] Dependencies pinned to specific versions
- [ ] No deprecated libraries
- [ ] Vulnerability scan passed
- [ ] License compatibility verified
- [ ] Containerized (Docker image built)
- [ ] Base image security scanned

**Key Dependencies:**
```
numpy==1.24.3
pandas==2.0.2
scikit-learn==1.3.0
xgboost==1.7.6
fastapi==0.100.0
pydantic==2.0.3
```

### 6. Data Pipeline
- [ ] Feature extraction pipeline tested
- [ ] Data schema validation implemented
- [ ] Handle missing/null values correctly
- [ ] Handle unexpected data types
- [ ] Feature drift monitoring in place
- [ ] Data quality checks in place
- [ ] Batch vs real-time features clearly separated
- [ ] Feature store integration tested (if applicable)

### 7. API / Integration
- [ ] API endpoint defined and documented
- [ ] Request/response schema validated
- [ ] Authentication/authorization implemented
- [ ] Rate limiting configured
- [ ] Request timeout configured
- [ ] Input validation implemented
- [ ] Error responses well-defined
- [ ] API versioning strategy defined
- [ ] Backward compatibility maintained

**API Specification:**
```python
POST /api/v3/predict
Authorization: Bearer <token>
Content-Type: application/json

Request:
{
    "user_id": "user_12345",
    "context": {
        "page": "homepage",
        "device": "mobile"
    },
    "num_recommendations": 10
}

Response:
{
    "user_id": "user_12345",
    "recommendations": [
        {"item_id": "item_789", "score": 0.92, "rank": 1},
        {"item_id": "item_456", "score": 0.87, "rank": 2}
    ],
    "model_version": "3.0.1",
    "latency_ms": 42
}
```

### 8. Monitoring & Observability
- [ ] Logging configured (INFO, WARN, ERROR levels)
- [ ] Structured logging implemented (JSON format)
- [ ] Metrics collection configured (Prometheus/CloudWatch)
- [ ] Dashboard created for key metrics
- [ ] Alerts configured for critical issues
- [ ] Distributed tracing enabled (if applicable)
- [ ] Log aggregation configured (ELK/Datadog)

**Metrics to Monitor:**
- Request rate (requests/second)
- Latency (p50, p95, p99)
- Error rate (%)
- Model prediction distribution
- Feature statistics (mean, std, nulls)
- Resource usage (CPU, memory, GPU)

**Alerts:**
| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| High error rate | Error rate > 1% | Critical | Page on-call |
| High latency | p95 > 500ms | High | Investigate |
| Low traffic | RPS < 10 | Medium | Check upstream |
| Model staleness | Model age > 30 days | Low | Trigger retrain |

### 9. Rollback Plan
- [ ] Previous model version available
- [ ] Rollback procedure documented
- [ ] Rollback can be executed in < 5 minutes
- [ ] Rollback tested in staging
- [ ] Feature flags configured for instant disable
- [ ] Database migrations reversible (if applicable)
- [ ] Rollback responsibility assigned

**Rollback Procedure:**
```bash
# Option 1: Feature flag (instant)
curl -X POST /api/admin/feature-flags/rec-model-v3 -d '{"enabled": false}'

# Option 2: Redeploy previous version
kubectl set image deployment/ml-service ml-service=ml-service:v2.1

# Option 3: Traffic shift (gradual)
kubectl patch service ml-service -p '{"spec":{"selector":{"version":"v2.1"}}}'
```

### 10. Security
- [ ] Model file encrypted at rest
- [ ] Secrets managed via secret manager (not in code)
- [ ] TLS/HTTPS enabled
- [ ] API authentication required
- [ ] Input sanitization implemented
- [ ] Rate limiting prevents abuse
- [ ] Network security groups configured
- [ ] Principle of least privilege (IAM roles)
- [ ] Security scan passed (SAST, DAST)

### 11. Compliance & Ethics
- [ ] Data privacy requirements met (GDPR, CCPA)
- [ ] PII handling compliant
- [ ] Model bias evaluated and acceptable
- [ ] Fairness metrics documented
- [ ] Explainability requirements met
- [ ] Model card created
- [ ] Legal/compliance review completed (if required)

### 12. Documentation
- [ ] Model documentation complete
- [ ] API documentation published
- [ ] Deployment runbook created
- [ ] Troubleshooting guide created
- [ ] Known issues documented
- [ ] On-call playbook updated
- [ ] Architecture diagram created
- [ ] Change log updated

### 13. Training & Communication
- [ ] Team trained on new model
- [ ] Stakeholders notified of deployment
- [ ] Customer-facing teams briefed (if user-visible changes)
- [ ] Release notes prepared
- [ ] Post-deployment review scheduled

## Deployment Execution

### Stage 1: Staging Deployment
- [ ] Deploy to staging environment
- [ ] Run smoke tests
- [ ] Verify API endpoints
- [ ] Check logs for errors
- [ ] Verify monitoring/alerts
- [ ] Load test in staging
- [ ] Staging sign-off obtained

**Staging Checklist:**
```bash
# Deploy to staging
kubectl apply -f k8s/staging/deployment.yaml

# Smoke tests
./tests/smoke_test.sh staging

# Load test
locust -f loadtest.py --host=https://staging.example.com --users=100 --spawn-rate=10
```

### Stage 2: Production Deployment (Canary)
- [ ] Deploy to 5% of production traffic
- [ ] Monitor for 1 hour
- [ ] Compare metrics (canary vs control)
- [ ] No errors or degradation
- [ ] Canary sign-off obtained

**Canary Metrics:**
| Metric | Control (v2.1) | Canary (v3.0) | Difference |
|--------|----------------|---------------|------------|
| p95 Latency | 445ms | 448ms | +0.7% ✅ |
| Error Rate | 0.05% | 0.06% | +0.01% ✅ |
| Prediction Quality | 0.856 AUC | 0.912 AUC | +6.5% ✅ |

### Stage 3: Gradual Rollout
- [ ] Increase to 25% traffic (monitor 24 hours)
- [ ] Increase to 50% traffic (monitor 24 hours)
- [ ] Increase to 100% traffic
- [ ] Monitor for 72 hours post-100%
- [ ] All metrics stable

**Rollout Schedule:**
| Date/Time | Traffic % | Monitor Duration | Go/No-Go |
|-----------|-----------|------------------|----------|
| Day 1, 10am | 5% | 1 hour | ✅ Go |
| Day 1, 2pm | 25% | 24 hours | ✅ Go |
| Day 2, 2pm | 50% | 24 hours | ✅ Go |
| Day 3, 2pm | 100% | 72 hours | ✅ Go |

### Stage 4: Validation
- [ ] All traffic on new version
- [ ] Metrics within expected ranges
- [ ] No increase in errors or latency
- [ ] Business metrics improved or stable
- [ ] Old version decommissioned (after 7 days)

## Post-Deployment

### Immediate (24 hours)
- [ ] Monitor dashboard every 4 hours
- [ ] Review logs for any errors
- [ ] Verify metrics collection
- [ ] Check alert configuration
- [ ] Update status page (if applicable)

### Short-term (1 week)
- [ ] Daily metrics review
- [ ] Compare pre/post deployment metrics
- [ ] Collect user feedback
- [ ] Monitor for edge cases
- [ ] Document any issues and resolutions

### Long-term (Ongoing)
- [ ] Weekly performance review
- [ ] Monthly model drift analysis
- [ ] Quarterly model retraining evaluation
- [ ] Track business impact
- [ ] Plan next iteration

## Sign-off

### Pre-Deployment Approval
- [ ] ML Engineer: [Name] - [Date] - ✅
- [ ] DevOps Engineer: [Name] - [Date] - ✅
- [ ] Product Manager: [Name] - [Date] - ✅
- [ ] Tech Lead: [Name] - [Date] - ✅

### Deployment Execution
- [ ] Staging Deployment: [Name] - [Date] - ✅
- [ ] Canary Deployment: [Name] - [Date] - ✅
- [ ] Full Rollout: [Name] - [Date] - ✅

### Post-Deployment Validation
- [ ] 24-hour Check: [Name] - [Date] - ✅
- [ ] 7-day Check: [Name] - [Date] - ✅
- [ ] Deployment Complete: [Name] - [Date] - ✅

## Rollback Log (if applicable)
| Date/Time | Reason | Action Taken | Result |
|-----------|--------|--------------|--------|
| - | - | - | - |

## Lessons Learned
1. [What went well]
2. [What could be improved]
3. [Unexpected challenges]
4. [Process improvements for next time]
```

---

## 9. Monitoring & Alerting Template

```markdown
# Monitoring & Alerting Plan: [Model Name]

**Model:** [Model Name v3.0]
**Owner:** [Team Name]
**Last Updated:** [YYYY-MM-DD]

## Monitoring Overview

### Monitoring Objectives
1. **Operational Health:** Ensure service is up and responsive
2. **Model Performance:** Track prediction quality and drift
3. **Business Impact:** Monitor key business metrics
4. **Cost Efficiency:** Track resource usage and costs

### Monitoring Stack
- **Metrics Collection:** Prometheus / CloudWatch
- **Visualization:** Grafana / CloudWatch Dashboards
- **Logging:** ELK Stack / CloudWatch Logs
- **Alerting:** PagerDuty / Opsgenie
- **Tracing:** Jaeger / X-Ray (optional)

## Metrics to Monitor

### 1. System Metrics (Operational)

#### Availability
```python
# SLO: 99.9% uptime (max 43 minutes downtime/month)
uptime_percentage = (successful_health_checks / total_health_checks) * 100
```

**Metrics:**
- Service uptime (%)
- Health check success rate
- HTTP 5xx error rate

**Dashboard Panels:**
- Uptime over time (30-day rolling)
- Current health status
- Error rate by endpoint

**Alerts:**
| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| Service Down | 3 consecutive failed health checks | Critical | Page on-call immediately |
| High 5xx Rate | Error rate > 1% for 5 min | High | Page on-call |
| Moderate 5xx | Error rate > 0.5% for 15 min | Medium | Slack alert |

#### Latency
```python
# SLO: p95 latency < 500ms, p99 < 1000ms
latency_p95 = percentile(response_times, 95)
latency_p99 = percentile(response_times, 99)
```

**Metrics:**
- p50, p95, p99 latency
- Average latency
- Latency by endpoint
- Latency by model version

**Dashboard Panels:**
- Latency percentiles over time
- Latency heatmap
- Latency distribution

**Alerts:**
| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| p99 Very High | p99 > 2000ms for 5 min | High | Investigate immediately |
| p95 High | p95 > 500ms for 15 min | Medium | Investigate |
| p50 Degraded | p50 > 200ms for 30 min | Low | Monitor |

#### Throughput
```python
# Track requests per second
requests_per_second = total_requests / time_window_seconds
```

**Metrics:**
- Requests per second (RPS)
- Requests per minute
- Traffic patterns by hour/day

**Dashboard Panels:**
- RPS over time
- Traffic heatmap (hour of day vs day of week)
- Comparison to historical baseline

**Alerts:**
| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| Traffic Spike | RPS > 2x normal for 10 min | Medium | Check for attack/legitimate surge |
| Traffic Drop | RPS < 50% normal for 15 min | High | Investigate upstream issues |

#### Resource Utilization
```python
# Track resource usage
cpu_utilization = (cpu_used / cpu_total) * 100
memory_utilization = (memory_used / memory_total) * 100
```

**Metrics:**
- CPU utilization (%)
- Memory utilization (%)
- GPU utilization (if applicable)
- Disk I/O
- Network I/O

**Dashboard Panels:**
- CPU/Memory usage over time
- Resource usage per instance
- Auto-scaling events

**Alerts:**
| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| High CPU | CPU > 90% for 10 min | High | Scale up or optimize |
| High Memory | Memory > 90% for 5 min | High | Scale up or investigate leak |
| OOM Kills | Out of memory events | Critical | Immediate action |

### 2. Model Performance Metrics

#### Prediction Quality
```python
# Track model predictions over time
def monitor_predictions():
    # Aggregate predictions
    positive_rate = predictions[predictions == 1].count() / len(predictions)
    avg_confidence = predictions_proba.mean()

    return {
        'positive_prediction_rate': positive_rate,
        'avg_confidence': avg_confidence
    }
```

**Metrics:**
- Prediction distribution (class balance)
- Average prediction confidence
- Prediction entropy
- Predictions by segment

**Dashboard Panels:**
- Prediction distribution over time
- Confidence score distribution
- Predictions per class
- Comparison to training distribution

**Alerts:**
| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| Prediction Shift | Positive rate changed > 20% | High | Investigate data drift |
| Low Confidence | Avg confidence < 0.6 | Medium | Check input quality |
| Extreme Predictions | >10% predictions at 0.0 or 1.0 | Medium | Investigate |

#### Feature Distribution Monitoring
```python
from scipy.stats import ks_2samp

def monitor_feature_drift(reference_data, current_data, feature):
    """Detect feature drift using KS test"""
    statistic, p_value = ks_2samp(
        reference_data[feature],
        current_data[feature]
    )

    if p_value < 0.05:
        return {
            'feature': feature,
            'drift_detected': True,
            'p_value': p_value,
            'statistic': statistic
        }
    return None
```

**Metrics:**
- Feature mean/std/min/max
- Feature null rate
- Feature value distribution
- KS statistic vs training data

**Dashboard Panels:**
- Feature statistics over time
- Top 10 drifted features
- Feature correlation changes

**Alerts:**
| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| Feature Drift | KS test p < 0.01 for important feature | High | Investigate & consider retrain |
| High Nulls | Null rate > 2x baseline | High | Check data pipeline |
| Out of Range | Values outside training range | Medium | Investigate data quality |

#### Ground Truth Monitoring (when available)
```python
# For cases where we get labels later
def calculate_online_metrics(predictions, ground_truth):
    """Calculate metrics on production predictions with delayed labels"""
    from sklearn.metrics import roc_auc_score, precision_score, recall_score

    return {
        'auc_roc': roc_auc_score(ground_truth, predictions),
        'precision': precision_score(ground_truth, predictions > 0.5),
        'recall': recall_score(ground_truth, predictions > 0.5)
    }
```

**Metrics:**
- Online AUC-ROC (when labels available)
- Online precision/recall
- Calibration error

**Dashboard Panels:**
- Metrics over time
- Comparison to offline evaluation
- Performance by segment

**Alerts:**
| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| Performance Drop | AUC drop > 0.05 from offline | Critical | Investigate immediately, consider rollback |
| Calibration Drift | ECE > 0.10 | High | Model needs recalibration |

### 3. Business Metrics

#### Key Business KPIs
```python
# Track business impact
click_through_rate = clicks / impressions
conversion_rate = conversions / users
revenue_per_user = total_revenue / total_users
```

**Metrics:**
- Click-through rate (CTR)
- Conversion rate
- Revenue per user
- Customer satisfaction (NPS, CSAT)
- User engagement metrics

**Dashboard Panels:**
- Business KPIs over time
- Comparison to pre-deployment baseline
- Impact attribution

**Alerts:**
| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| CTR Drop | CTR < baseline - 10% for 24h | High | Investigate model quality |
| Revenue Drop | Revenue/user < baseline - 5% | Critical | Escalate immediately |
| Conversion Drop | CVR < baseline - 10% | High | Check entire funnel |

### 4. Cost Metrics

#### Infrastructure Costs
```python
# Track costs
cost_per_request = total_infrastructure_cost / total_requests
cost_per_day = daily_compute_cost + daily_storage_cost
```

**Metrics:**
- Total monthly cost
- Cost per request
- Cost per prediction
- Compute vs storage costs
- Cost trend

**Dashboard Panels:**
- Daily/monthly costs
- Cost breakdown by resource type
- Cost efficiency (cost per 1000 requests)
- Budget vs actual

**Alerts:**
| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| Cost Spike | Daily cost > 1.5x baseline | Medium | Investigate resource usage |
| Budget Exceeded | Monthly cost > budget | High | Optimize or request increase |

## Dashboards

### Dashboard 1: Executive Summary
**Audience:** Leadership, Product Managers
**Refresh:** Every 5 minutes

**Panels:**
1. Service Health (Green/Yellow/Red indicator)
2. Key Business Metrics (CTR, Conversion, Revenue)
3. Request Volume (last 24h)
4. Error Rate (last 24h)
5. Cost Summary (MTD)

### Dashboard 2: Operational Health
**Audience:** On-call Engineers, SRE
**Refresh:** Every 1 minute

**Panels:**
1. Service Uptime (30-day)
2. Request Rate (last 4 hours)
3. Latency Percentiles (p50, p95, p99)
4. Error Rate by Type (4xx vs 5xx)
5. Resource Utilization (CPU, Memory)
6. Active Alerts
7. Recent Deployments

### Dashboard 3: Model Performance
**Audience:** ML Engineers, Data Scientists
**Refresh:** Every 5 minutes

**Panels:**
1. Prediction Distribution
2. Confidence Score Distribution
3. Top Features Statistics
4. Feature Drift Detection
5. Model Version Adoption
6. Online Metrics (if available)
7. Comparison to Offline Evaluation

### Dashboard 4: Business Impact
**Audience:** Product, Business Stakeholders
**Refresh:** Hourly

**Panels:**
1. Key Business KPIs (CTR, CVR, Revenue/User)
2. Pre/Post Deployment Comparison
3. Performance by Segment
4. User Engagement Metrics
5. A/B Test Results (if running)
6. Long-term Trends

## Alerting Strategy

### Alert Severity Levels

#### Critical (P0)
- **Definition:** Service completely down or major business impact
- **Response Time:** Immediate (< 5 minutes)
- **Notification:** Page on-call engineer + Slack #incidents
- **Escalation:** After 15 minutes, page manager

**Examples:**
- Service downtime
- Error rate > 10%
- Revenue drop > 20%

#### High (P1)
- **Definition:** Significant degradation or partial outage
- **Response Time:** < 15 minutes
- **Notification:** Slack #alerts + Email on-call
- **Escalation:** After 1 hour, escalate to P0

**Examples:**
- Error rate > 1%
- Latency > 2x SLA
- Performance drop > 5%

#### Medium (P2)
- **Definition:** Minor degradation, no immediate impact
- **Response Time:** < 1 hour during business hours
- **Notification:** Slack #alerts
- **Escalation:** None (but track in ticket)

**Examples:**
- Error rate > 0.5%
- Feature drift detected
- Resource utilization > 80%

#### Low (P3)
- **Definition:** Informational, no action required immediately
- **Response Time:** Next business day
- **Notification:** Email digest
- **Escalation:** None

**Examples:**
- Model age > 30 days
- Non-critical dependency update available

### Alert Configuration

```yaml
alerts:
  - name: HighErrorRate
    query: 'rate(http_requests_total{status=~"5.."}[5m]) > 0.01'
    for: 5m
    severity: critical
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value | humanizePercentage }}"
    actions:
      - type: pagerduty
        integration_key: ${PAGERDUTY_KEY}
      - type: slack
        channel: "#incidents"

  - name: HighLatency
    query: 'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 0.5'
    for: 15m
    severity: high
    annotations:
      summary: "p95 latency above SLA"
      description: "p95 latency is {{ $value }}s"
    actions:
      - type: slack
        channel: "#alerts"

  - name: FeatureDrift
    query: 'feature_drift_ks_statistic{feature="user_purchases_30d"} > 0.3'
    for: 1h
    severity: medium
    annotations:
      summary: "Feature drift detected"
      description: "{{ $labels.feature }} KS statistic: {{ $value }}"
    actions:
      - type: slack
        channel: "#ml-monitoring"
```

### Alert Best Practices
- [ ] Every alert is actionable
- [ ] Clear runbook for each alert
- [ ] No alert fatigue (< 5 alerts/day on average)
- [ ] Regular review and tuning of thresholds
- [ ] Auto-resolve when condition clears
- [ ] Include relevant context in alert message

## Logging Strategy

### Log Levels
- **DEBUG:** Detailed diagnostic information (disabled in prod)
- **INFO:** General informational messages
- **WARNING:** Warning messages (unexpected but handled)
- **ERROR:** Error messages (failures)
- **CRITICAL:** Critical errors (require immediate attention)

### Structured Logging Format
```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "level": "INFO",
  "service": "ml-recommendation-service",
  "version": "3.0.1",
  "trace_id": "abc123",
  "event": "prediction_made",
  "user_id": "user_12345",
  "model_version": "3.0.1",
  "latency_ms": 42,
  "prediction": {
    "top_item": "item_789",
    "score": 0.92,
    "num_recommendations": 10
  },
  "features": {
    "user_purchases_30d": 3,
    "avg_session_time": 180
  }
}
```

### What to Log

#### Request/Response
- Request ID
- User ID (hashed if PII)
- Timestamp
- Endpoint
- HTTP method
- Status code
- Latency

#### Model Predictions
- Model version
- Input features (sample or hash)
- Prediction output
- Confidence score
- Feature values (for drift monitoring)

#### Errors
- Error type
- Error message
- Stack trace
- Context (what was being processed)
- User impact

#### Performance
- Latency breakdown (feature extraction, inference, post-processing)
- Resource usage
- Cache hit/miss

### Log Retention
- **Real-time logs:** 7 days (hot storage)
- **Aggregated logs:** 90 days (warm storage)
- **Sampled logs:** 1 year (cold storage)
- **Audit logs:** 7 years (compliance)

## Monitoring Runbooks

### Runbook 1: High Error Rate
**Alert:** HighErrorRate triggered

**Steps:**
1. Check dashboard: What's the current error rate?
2. Check logs: What errors are occurring?
3. Check recent deployments: Was there a recent change?
4. Check dependencies: Are upstream services healthy?
5. If recent deployment: Consider rollback
6. If dependency issue: Escalate to owning team
7. If unknown: Gather logs and escalate

**Commands:**
```bash
# Check recent logs
kubectl logs -l app=ml-service --tail=100 | grep ERROR

# Check error distribution
curl http://metrics-server/api/errors/breakdown

# Rollback if needed
kubectl rollout undo deployment/ml-service
```

### Runbook 2: High Latency
**Alert:** HighLatency triggered

**Steps:**
1. Check current latency (p50, p95, p99)
2. Check traffic: Is there a traffic spike?
3. Check resource usage: CPU/Memory saturated?
4. Check database: Are queries slow?
5. If traffic spike: Scale up instances
6. If resource issue: Restart or scale up
7. If database slow: Optimize queries or scale DB

**Commands:**
```bash
# Check current latency
curl http://metrics-server/api/latency

# Scale up
kubectl scale deployment ml-service --replicas=10

# Check slow queries
kubectl logs -l app=ml-service | grep "query_time"
```

### Runbook 3: Feature Drift Detected
**Alert:** FeatureDrift triggered

**Steps:**
1. Identify which feature(s) drifted
2. Check data source: Is upstream data changed?
3. Analyze distribution: How much drift?
4. Check business context: Was there a real-world event?
5. If significant drift: Consider model retraining
6. If data quality issue: Fix upstream
7. Document findings

**Commands:**
```bash
# Get drift details
python scripts/analyze_drift.py --feature user_purchases_30d --days 7

# Compare distributions
python scripts/compare_distributions.py --baseline training_data.csv --current prod_last_7d.csv

# Trigger retraining pipeline (if needed)
airflow dags trigger model_retraining_pipeline
```

## Review & Maintenance

### Weekly Review
- [ ] Review all alerts triggered (root cause analysis)
- [ ] Check for alert fatigue (too many alerts)
- [ ] Review dashboard usage
- [ ] Update thresholds if needed

### Monthly Review
- [ ] Model performance trends
- [ ] Business impact review
- [ ] Cost analysis
- [ ] Capacity planning
- [ ] Update runbooks

### Quarterly Review
- [ ] Comprehensive monitoring audit
- [ ] Evaluate new monitoring tools
- [ ] Update SLOs/SLAs
- [ ] Training for on-call engineers
```

---

*This is Part 2 of the ML Practical Templates & Checklists guide.*

**Still to come in additional parts:**
- Incident Response Template
- ML System Design Template
- Interview Preparation Checklist

---

**Document Version:** 1.0
**Last Updated:** October 21, 2025
**Maintained by:** Benjamin Hu
