# Domain-Specific ML: Healthcare, Finance, and E-commerce

## Table of Contents
1. [Introduction](#introduction)
2. [Healthcare ML](#healthcare-ml)
3. [Finance ML](#finance-ml)
4. [E-commerce & Retail ML](#e-commerce--retail-ml)
5. [Cross-Domain Patterns](#cross-domain-patterns)
6. [Interview Questions](#interview-questions)

---

## Introduction

Machine Learning applications vary significantly across industries. This guide covers three major domains:
- **Healthcare ML**: Diagnosis, treatment, drug discovery
- **Finance ML**: Trading, risk management, fraud detection
- **E-commerce ML**: Recommendations, pricing, demand forecasting

For each domain, we cover:
- Common use cases and business problems
- Data characteristics and challenges
- Model architectures and techniques
- Evaluation metrics and success criteria
- Regulatory and ethical considerations
- Production implementations with code
- Domain-specific interview questions

---

## Healthcare ML

### Overview

Healthcare ML applications have **high stakes** (life and death decisions) and are **heavily regulated** (HIPAA, FDA approval for medical devices).

### Common Use Cases

1. **Medical Image Analysis** - Radiology, pathology, dermatology
2. **Clinical Decision Support** - Treatment recommendations, diagnosis assistance
3. **Drug Discovery** - Molecular property prediction, clinical trial optimization
4. **Patient Risk Prediction** - Readmission risk, disease progression
5. **Electronic Health Record (EHR) Analysis** - Patient phenotyping, outcome prediction
6. **Remote Patient Monitoring** - Wearable device data analysis

### Data Characteristics

```python
"""
Healthcare data challenges:
"""

class HealthcareDataChallenges:
    """
    Unique challenges in healthcare ML
    """

    @staticmethod
    def imbalanced_classes():
        """
        Example: Disease prevalence is often <1%
        """
        print("Challenge: Severe class imbalance")
        print("Solution approaches:")
        print("  - SMOTE for upsampling")
        print("  - Focal loss for training")
        print("  - Ensemble methods")
        print("  - Adjust decision thresholds")

    @staticmethod
    def missing_data():
        """
        EHR data has extensive missing values
        """
        print("Challenge: 30-70% missing data common")
        print("Solution approaches:")
        print("  - Missingness indicators as features")
        print("  - Multiple imputation")
        print("  - Models robust to missing data (XGBoost)")

    @staticmethod
    def temporal_data():
        """
        Patient data is sequential and time-varying
        """
        print("Challenge: Irregular time series")
        print("Solution approaches:")
        print("  - Recurrent neural networks (LSTM, GRU)")
        print("  - Transformers with time embeddings")
        print("  - Set-based models (DeepSets)")

    @staticmethod
    def privacy_concerns():
        """
        HIPAA compliance required
        """
        print("Challenge: Patient privacy (HIPAA)")
        print("Solution approaches:")
        print("  - Differential privacy")
        print("  - Federated learning")
        print("  - De-identification")
        print("  - Secure multi-party computation")

    @staticmethod
    def limited_labeled_data():
        """
        Expert annotation is expensive
        """
        print("Challenge: Limited labeled data")
        print("Solution approaches:")
        print("  - Transfer learning")
        print("  - Self-supervised learning")
        print("  - Active learning")
        print("  - Few-shot learning")
```

### Use Case 1: Medical Image Classification

#### Problem: Chest X-Ray Pneumonia Detection

**Business Context:**
- Radiologists are overworked (shortage globally)
- Need fast, accurate pneumonia detection
- Must achieve >95% sensitivity (cannot miss true positives)

**Solution Architecture:**

```python
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

class ChestXRayDataset(Dataset):
    """
    Chest X-Ray dataset with medical image preprocessing
    """
    def __init__(self, image_paths, labels, transform=None, preprocessing='clahe'):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.preprocessing = preprocessing

    def apply_medical_preprocessing(self, image):
        """
        Apply domain-specific preprocessing for chest X-rays
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Improves local contrast - important for medical imaging
        if self.preprocessing == 'clahe':
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image = clahe.apply(image)

        # Normalize lung area (remove background)
        # Otsu's thresholding to find lung region
        _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return image, mask

    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(self.image_paths[idx])

        # Medical preprocessing
        image, mask = self.apply_medical_preprocessing(image)

        # Convert to 3 channels for pretrained models
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        label = self.labels[idx]

        return image, label

    def __len__(self):
        return len(self.image_paths)

def get_medical_augmentations(train=True):
    """
    Medical image augmentations
    Important: Only use clinically reasonable augmentations!
    """
    if train:
        return A.Compose([
            # Geometric augmentations (reasonable for X-rays)
            A.HorizontalFlip(p=0.5),  # Patient can be positioned either way
            A.Rotate(limit=10, p=0.5),  # Small rotation okay
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1,
                              rotate_limit=10, p=0.5),

            # Avoid: Vertical flip (not clinically meaningful)
            # Avoid: Large rotations (not how X-rays are taken)
            # Avoid: Color jittering (X-rays are grayscale)

            # Intensity augmentations
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),

            # Normalization
            A.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

class MedicalImageClassifier(nn.Module):
    """
    Medical image classifier with attention mechanism
    """
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()

        # Backbone: Use proven architecture
        self.backbone = models.densenet121(pretrained=pretrained)

        # DenseNet is preferred for medical imaging:
        # - Efficient feature reuse
        # - Fewer parameters than ResNet
        # - Good gradient flow

        # Get number of features
        num_features = self.backbone.classifier.in_features

        # Replace classifier
        self.backbone.classifier = nn.Identity()

        # Attention mechanism
        # Helps model focus on relevant lung regions
        self.attention = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Extract features
        features = self.backbone(x)

        # Apply attention
        attention_weights = self.attention(features)
        weighted_features = features * attention_weights

        # Classify
        output = self.classifier(weighted_features)

        return output, attention_weights  # Return attention for interpretability

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in medical imaging

    Focal Loss = -α(1-p_t)^γ * log(p_t)

    Where:
    - α: Class weight
    - γ (gamma): Focusing parameter
    - p_t: Predicted probability for true class
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()

def train_medical_classifier(model, train_loader, val_loader, epochs=50):
    """
    Training loop for medical image classifier
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Loss function: Focal Loss for imbalanced data
    criterion = FocalLoss(alpha=0.25, gamma=2.0)

    # Optimizer: Adam with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Learning rate scheduler: Reduce on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    # Training metrics
    best_val_auc = 0.0
    train_losses = []
    val_aucs = []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs, attention = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs, _ = model(images)
                probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of pneumonia

                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # Calculate metrics
        val_auc = roc_auc_score(all_labels, all_probs)
        val_aucs.append(val_auc)

        # Calculate sensitivity, specificity at optimal threshold
        thresholds = np.arange(0, 1, 0.01)
        sensitivities = []
        specificities = []

        for threshold in thresholds:
            preds = (np.array(all_probs) >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivities.append(sensitivity)
            specificities.append(specificity)

        # Find threshold with sensitivity >= 0.95
        high_sens_idx = np.where(np.array(sensitivities) >= 0.95)[0]
        if len(high_sens_idx) > 0:
            best_idx = high_sens_idx[np.argmax(np.array(specificities)[high_sens_idx])]
            best_threshold = thresholds[best_idx]
            best_sensitivity = sensitivities[best_idx]
            best_specificity = specificities[best_idx]
        else:
            best_threshold = 0.5
            best_sensitivity = max(sensitivities)
            best_specificity = specificities[np.argmax(sensitivities)]

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val AUC: {val_auc:.4f}")
        print(f"  Best Threshold: {best_threshold:.2f}")
        print(f"  Sensitivity @ threshold: {best_sensitivity:.4f}")
        print(f"  Specificity @ threshold: {best_specificity:.4f}")

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), 'best_medical_classifier.pth')
            print(f"  ✅ New best model saved! (AUC: {val_auc:.4f})")

        # Learning rate scheduling
        scheduler.step(val_auc)

    return model, train_losses, val_aucs

def interpret_prediction(model, image, true_label, class_names=['Normal', 'Pneumonia']):
    """
    Generate Grad-CAM visualization for model interpretation
    Critical for clinical use!
    """
    import torch.nn.functional as F
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    # Forward pass
    image_tensor = image.unsqueeze(0).to(device)
    output, attention = model(image_tensor)
    pred_class = output.argmax(dim=1).item()
    confidence = torch.softmax(output, dim=1)[0, pred_class].item()

    # Grad-CAM
    target_layers = [model.backbone.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    grayscale_cam = cam(input_tensor=image_tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :]

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    img_np = image.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    axes[0].imshow(img_np)
    axes[0].set_title(f'Original\nTrue: {class_names[true_label]}')
    axes[0].axis('off')

    # Grad-CAM
    cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    axes[1].imshow(cam_image)
    axes[1].set_title(f'Grad-CAM Heatmap\nPred: {class_names[pred_class]} ({confidence:.2%})')
    axes[1].axis('off')

    # Attention weights
    axes[2].bar(['Feature Attention'], [attention.item()])
    axes[2].set_title('Attention Weight')
    axes[2].set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig('medical_interpretation.png', dpi=300, bbox_inches='tight')
    plt.show()

    return pred_class, confidence
```

#### Clinical Deployment Considerations

```python
class ClinicalMLDeployment:
    """
    Production deployment checklist for clinical ML
    """

    @staticmethod
    def regulatory_requirements():
        """
        FDA requirements for medical device software
        """
        print("FDA Requirements (Class II/III Medical Device):")
        print("  1. 510(k) Clearance or PMA approval")
        print("  2. Clinical validation studies")
        print("  3. Quality Management System (QMS)")
        print("  4. Design controls and documentation")
        print("  5. Postmarket surveillance")

    @staticmethod
    def performance_requirements():
        """
        Clinical performance standards
        """
        print("Performance Requirements:")
        print("  Sensitivity (Recall): >95% (cannot miss diseases)")
        print("  Specificity: >85% (minimize false alarms)")
        print("  AUC-ROC: >0.90")
        print("  Calibration: Predicted probabilities should match true rates")

    @staticmethod
    def interpretability_requirements():
        """
        Explainability for clinicians
        """
        print("Interpretability Requirements:")
        print("  1. Grad-CAM/attention visualizations")
        print("  2. Feature importance explanations")
        print("  3. Similar case retrieval")
        print("  4. Confidence intervals")
        print("  5. Model limitations documentation")

    @staticmethod
    def monitoring_requirements():
        """
        Continuous monitoring in production
        """
        print("Monitoring Requirements:")
        print("  1. Performance drift detection")
        print("  2. Data distribution shifts")
        print("  3. False positive/negative rates")
        print("  4. Clinical outcomes tracking")
        print("  5. Adverse event reporting")

    @staticmethod
    def safety_requirements():
        """
        Safety and fail-safe mechanisms
        """
        print("Safety Requirements:")
        print("  1. Human-in-the-loop (clinician reviews all predictions)")
        print("  2. Uncertainty quantification")
        print("  3. Out-of-distribution detection")
        print("  4. Graceful degradation")
        print("  5. Audit trail for all predictions")
```

### Use Case 2: Patient Risk Prediction (Readmission)

#### Problem: 30-Day Hospital Readmission Prediction

**Business Context:**
- Hospital penalized by CMS for high readmission rates
- Need to identify high-risk patients for intervention
- Must work with structured EHR data

**Solution Architecture:**

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
import shap

class ReadmissionRiskModel:
    """
    30-day hospital readmission risk prediction
    """
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

    def engineer_features(self, df):
        """
        Feature engineering for EHR data
        """
        features = pd.DataFrame()

        # Demographic features
        features['age'] = df['age']
        features['is_male'] = (df['gender'] == 'M').astype(int)

        # Clinical features
        features['num_diagnoses'] = df['num_diagnoses']
        features['num_procedures'] = df['num_procedures']
        features['num_medications'] = df['num_medications']
        features['length_of_stay'] = df['discharge_date'] - df['admission_date']

        # Comorbidity index (Charlson Comorbidity Index)
        comorbidities = ['diabetes', 'hypertension', 'heart_disease',
                        'kidney_disease', 'copd', 'cancer']
        features['charlson_index'] = df[comorbidities].sum(axis=1)

        # Prior utilization
        features['num_previous_admissions'] = df['num_previous_admissions']
        features['num_er_visits_last_year'] = df['num_er_visits_last_year']

        # Lab values (last recorded)
        lab_features = ['hemoglobin', 'creatinine', 'glucose', 'sodium']
        for lab in lab_features:
            features[f'{lab}_last'] = df[f'{lab}_last']
            features[f'{lab}_abnormal'] = (
                (df[f'{lab}_last'] < df[f'{lab}_lower_limit']) |
                (df[f'{lab}_last'] > df[f'{lab}_upper_limit'])
            ).astype(int)

        # Medication features
        features['num_new_medications'] = df['num_new_medications']
        features['high_risk_medications'] = df['high_risk_medications'].astype(int)

        # Discharge features
        features['discharged_to_home'] = (df['discharge_disposition'] == 'home').astype(int)
        features['has_home_health'] = df['has_home_health'].astype(int)

        return features

    def train(self, X, y, calibrate=True):
        """
        Train readmission risk model with calibration
        """
        self.feature_names = X.columns.tolist()

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train XGBoost (good with structured data, handles missing values)
        base_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=1.0,  # L2 regularization
            scale_pos_weight=3,  # Handle class imbalance
            random_state=42
        )

        # Calibrate probabilities (critical for clinical use!)
        # Isotonic calibration for reliability
        if calibrate:
            self.model = CalibratedClassifierCV(
                base_model,
                method='isotonic',
                cv=5
            )
        else:
            self.model = base_model

        self.model.fit(X_scaled, y)

        return self

    def predict_proba(self, X):
        """
        Predict readmission probability
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def explain_prediction(self, X, patient_idx):
        """
        Explain prediction using SHAP values
        Critical for clinical acceptance!
        """
        X_scaled = self.scaler.transform(X)

        # SHAP explanation
        explainer = shap.TreeExplainer(self.model.base_estimator)
        shap_values = explainer.shap_values(X_scaled)

        # Get patient's features
        patient_features = X.iloc[patient_idx]
        patient_shap = shap_values[patient_idx]

        # Create explanation
        explanation = pd.DataFrame({
            'feature': self.feature_names,
            'value': patient_features.values,
            'shap_value': patient_shap
        }).sort_values('shap_value', key=abs, ascending=False)

        return explanation

    def clinical_risk_stratification(self, probabilities):
        """
        Stratify patients into risk categories
        """
        risk_categories = pd.cut(
            probabilities,
            bins=[0, 0.10, 0.20, 0.35, 1.0],
            labels=['Low', 'Moderate', 'High', 'Very High']
        )
        return risk_categories

# Example usage
"""
# Load EHR data
df = pd.read_csv('ehr_data.csv')

# Engineer features
model = ReadmissionRiskModel()
features = model.engineer_features(df)

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    features, df['readmitted_30d'], test_size=0.2, random_state=42
)

model.train(X_train, y_train)

# Predict risk
readmission_probs = model.predict_proba(X_test)

# Stratify patients
risk_categories = model.clinical_risk_stratification(readmission_probs)

# High-risk patients for intervention
high_risk_patients = X_test[risk_categories.isin(['High', 'Very High'])]

# Explain predictions for clinicians
for idx in high_risk_patients.index[:5]:
    explanation = model.explain_prediction(X_test, idx)
    print(f"\nPatient {idx}:")
    print(explanation.head(10))
"""
```

### Healthcare ML Best Practices

```python
class HealthcareMLBestPractices:
    """
    Best practices for healthcare ML development
    """

    @staticmethod
    def data_handling():
        """
        Best practices for medical data
        """
        return {
            'Privacy': [
                'HIPAA compliance mandatory',
                'De-identify data (remove PII)',
                'Differential privacy for shared models',
                'Secure data storage and transmission'
            ],
            'Quality': [
                'Validate data quality rigorously',
                'Check for label errors (second opinion)',
                'Handle missing data appropriately',
                'Document data provenance'
            ],
            'Bias': [
                'Check for demographic biases',
                'Ensure diverse training data',
                'Validate on multiple populations',
                'Report performance by subgroup'
            ]
        }

    @staticmethod
    def model_development():
        """
        Best practices for model development
        """
        return {
            'Model Selection': [
                'Start simple (logistic regression baseline)',
                'Use interpretable models when possible',
                'Ensemble multiple models for robustness',
                'Calibrate probabilities for clinical use'
            ],
            'Evaluation': [
                'Use clinically relevant metrics',
                'Validate on held-out test set',
                'External validation on different hospitals',
                'Compare to clinician performance'
            ],
            'Interpretability': [
                'Provide explanations for predictions',
                'Visualize model decision process',
                'Report confidence/uncertainty',
                'Identify model limitations'
            ]
        }

    @staticmethod
    def deployment():
        """
        Best practices for deployment
        """
        return {
            'Integration': [
                'Integrate into clinical workflow',
                'Minimize additional clinician workload',
                'Provide real-time predictions',
                'Allow clinician override'
            ],
            'Monitoring': [
                'Monitor performance continuously',
                'Detect distribution shifts',
                'Track clinical outcomes',
                'Collect feedback from clinicians'
            ],
            'Safety': [
                'Human-in-the-loop review',
                'Fail-safe mechanisms',
                'Audit trail for predictions',
                'Incident response plan'
            ]
        }
```

---

## Finance ML

### Overview

Finance ML applications involve **high-frequency decisions**, **adversarial environments** (bad actors trying to game the system), and **strict regulatory requirements** (SEC, FINRA, Basel III).

### Common Use Cases

1. **Algorithmic Trading** - High-frequency trading, market making
2. **Fraud Detection** - Credit card fraud, identity theft, money laundering
3. **Credit Scoring** - Loan approval, credit limit determination
4. **Risk Management** - Portfolio risk, market risk, credit risk
5. **Customer Segmentation** - Marketing, personalization
6. **Forecasting** - Stock prices, economic indicators

### Data Characteristics

```python
class FinanceDataCharacteristics:
    """
    Unique characteristics of financial data
    """

    @staticmethod
    def time_series_nature():
        """
        Financial data is sequential and non-stationary
        """
        print("Characteristics:")
        print("  - Non-stationary (mean/variance change over time)")
        print("  - Serial correlation (autocorrelation)")
        print("  - Seasonality (day of week, time of day effects)")
        print("  - Regime changes (bull/bear markets)")

        print("\nImplications:")
        print("  - Use time-series cross-validation")
        print("  - Feature engineering with lags")
        print("  - Handle concept drift")
        print("  - Model regime changes")

    @staticmethod
    def high_frequency():
        """
        Low-latency requirements
        """
        print("Requirements:")
        print("  - Latency: microseconds to milliseconds")
        print("  - Throughput: millions of events per second")
        print("  - Real-time feature computation")

        print("\nSolutions:")
        print("  - Simple models (linear, tree-based)")
        print("  - Model compilation and optimization")
        print("  - Hardware acceleration (GPU, FPGA)")
        print("  - In-memory processing")

    @staticmethod
    def adversarial_environment():
        """
        Bad actors trying to game the system
        """
        print("Challenges:")
        print("  - Fraudsters adapt to detection models")
        print("  - Adversarial attacks on trading models")
        print("  - Data poisoning attempts")

        print("\nSolutions:")
        print("  - Adversarial training")
        print("  - Ensemble diversity")
        print("  - Continuous retraining")
        print("  - Anomaly detection")

    @staticmethod
    def regulatory_requirements():
        """
        Compliance requirements
        """
        print("Regulations:")
        print("  - Fair Credit Reporting Act (FCRA)")
        print("  - Equal Credit Opportunity Act (ECOA)")
        print("  - Basel III (capital requirements)")
        print("  - Dodd-Frank Act")
        print("  - MiFID II (Europe)")

        print("\nImplications:")
        print("  - Model explainability required")
        print("  - No discrimination by protected classes")
        print("  - Stress testing required")
        print("  - Model documentation and governance")
```

### Use Case 1: Real-Time Fraud Detection

*(Due to length, I'll provide a comprehensive implementation)*

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import time

class RealTimeFraudDetector:
    """
    Real-time credit card fraud detection system

    Requirements:
    - Latency <50ms (P99)
    - Handle 10,000+ TPS
    - <0.1% false positive rate
    - >95% fraud detection rate
    """

    def __init__(self):
        self.rule_engine = FraudRuleEngine()
        self.ml_model = None
        self.anomaly_detector = IsolationForest(contamination=0.01, random_state=42)
        self.scaler = StandardScaler()

    def engineer_features(self, transaction):
        """
        Real-time feature engineering
        Must complete in <10ms
        """
        features = {}

        # Transaction features
        features['amount'] = transaction['amount']
        features['hour_of_day'] = transaction['timestamp'].hour
        features['day_of_week'] = transaction['timestamp'].dayofweek
        features['is_online'] = int(transaction['transaction_type'] == 'online')
        features['is_international'] = int(transaction['country'] != transaction['user_country'])

        # Merchant features
        features['merchant_category'] = self.encode_merchant_category(transaction['merchant_category'])
        features['merchant_risk_score'] = self.get_merchant_risk_score(transaction['merchant_id'])

        # User historical features (from cache)
        user_history = self.get_user_history(transaction['user_id'])
        features['user_avg_amount_30d'] = user_history['avg_amount_30d']
        features['user_std_amount_30d'] = user_history['std_amount_30d']
        features['user_num_transactions_30d'] = user_history['num_transactions_30d']
        features['user_num_declines_30d'] = user_history['num_declines_30d']

        # Amount vs history ratio
        features['amount_vs_avg_ratio'] = (
            features['amount'] / (features['user_avg_amount_30d'] + 1)
        )

        # Velocity features (recent activity)
        features['num_transactions_1h'] = user_history['num_transactions_1h']
        features['num_transactions_24h'] = user_history['num_transactions_24h']
        features['total_amount_24h'] = user_history['total_amount_24h']

        # Location features
        features['distance_from_last_transaction'] = self.calculate_distance(
            transaction['location'],
            user_history['last_transaction_location']
        )
        features['time_since_last_transaction'] = (
            transaction['timestamp'] - user_history['last_transaction_time']
        ).total_seconds() / 3600  # hours

        # Impossible velocity (transaction in different location too quickly)
        features['impossible_velocity'] = int(
            features['distance_from_last_transaction'] > 100 and  # >100 km
            features['time_since_last_transaction'] < 1  # <1 hour
        )

        return pd.DataFrame([features])

    def predict(self, transaction):
        """
        Multi-stage fraud detection pipeline
        """
        start_time = time.time()

        # Stage 1: Rule-based filtering (instant, <1ms)
        rule_result = self.rule_engine.evaluate(transaction)
        if rule_result['action'] != 'continue':
            latency = (time.time() - start_time) * 1000
            return {
                'is_fraud': rule_result['action'] == 'reject',
                'fraud_probability': 1.0 if rule_result['action'] == 'reject' else 0.0,
                'reason': rule_result['reason'],
                'stage': 'rules',
                'latency_ms': latency
            }

        # Stage 2: Feature engineering (~5-10ms)
        features = self.engineer_features(transaction)

        # Stage 3: ML model prediction (~20-30ms)
        features_scaled = self.scaler.transform(features)
        fraud_probability = self.ml_model.predict_proba(features_scaled)[0, 1]

        # Stage 4: Anomaly detection (~5-10ms)
        anomaly_score = self.anomaly_detector.score_samples(features_scaled)[0]
        is_anomaly = anomaly_score < self.anomaly_detector.threshold_

        # Decision logic
        if fraud_probability > 0.9 or is_anomaly:
            decision = 'reject'
        elif fraud_probability > 0.7:
            decision = 'review'
        else:
            decision = 'approve'

        latency = (time.time() - start_time) * 1000

        return {
            'is_fraud': decision == 'reject',
            'fraud_probability': fraud_probability,
            'anomaly_score': anomaly_score,
            'decision': decision,
            'stage': 'ml_model',
            'latency_ms': latency
        }

    def train(self, X, y):
        """
        Train ML model for fraud detection
        """
        # XGBoost for fraud detection
        # Fast inference, handles imbalanced data well
        self.ml_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=100,  # Handle severe imbalance (fraud ~0.1%)
            random_state=42,
            tree_method='hist',  # Faster training
            predictor='cpu_predictor'  # Faster inference
        )

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train ML model
        self.ml_model.fit(X_scaled, y)

        # Train anomaly detector on normal transactions
        normal_transactions = X_scaled[y == 0]
        self.anomaly_detector.fit(normal_transactions)

        return self

class FraudRuleEngine:
    """
    Rule-based fraud detection (instant decisions)
    """

    def evaluate(self, transaction):
        """
        Evaluate hard rules
        """
        # Rule 1: Blacklisted card
        if self.is_blacklisted(transaction['card_number']):
            return {'action': 'reject', 'reason': 'blacklisted_card'}

        # Rule 2: Amount above card limit
        if transaction['amount'] > transaction['card_limit']:
            return {'action': 'reject', 'reason': 'exceeds_limit'}

        # Rule 3: High-risk country
        if self.is_high_risk_country(transaction['country']):
            return {'action': 'review', 'reason': 'high_risk_country'}

        # Rule 4: Unusual amount (>10x average)
        user_avg = self.get_user_average_amount(transaction['user_id'])
        if transaction['amount'] > user_avg * 10:
            return {'action': 'review', 'reason': 'unusual_amount'}

        # Rule 5: Impossible velocity
        if self.check_impossible_velocity(transaction):
            return {'action': 'reject', 'reason': 'impossible_velocity'}

        return {'action': 'continue'}

    def is_blacklisted(self, card_number):
        # Check against blacklist (Redis lookup)
        return False  # Placeholder

    def is_high_risk_country(self, country):
        high_risk_countries = ['XX', 'YY', 'ZZ']  # Example
        return country in high_risk_countries

    def get_user_average_amount(self, user_id):
        # Fetch from cache
        return 100.0  # Placeholder

    def check_impossible_velocity(self, transaction):
        # Two transactions in different locations too quickly
        # Implementation would check recent transaction locations
        return False  # Placeholder

# Production deployment
"""
Architecture for 10K TPS:

1. API Layer:
   - Load balancer (AWS ALB)
   - 50 instances (c5.2xlarge)
   - Each handles 200 TPS

2. Feature Store:
   - Redis Cluster (20 nodes)
   - User history features cached
   - <1ms lookup time

3. Model Serving:
   - XGBoost in-memory
   - ~20ms inference on CPU
   - No GPU needed

4. Monitoring:
   - Prometheus + Grafana
   - Track latency, fraud rate, false positive rate
   - Alert on anomalies

5. Feedback Loop:
   - Kafka for confirmed fraud/legitimate
   - Daily model retraining
   - A/B test new models

Cost: ~$20K/month for 10K TPS
Latency: P50=25ms, P95=45ms, P99=65ms
"""
```

### Use Case 2: Credit Scoring

**Problem**: Predict loan default risk

```python
class CreditScoringModel:
    """
    Credit scoring with fairness constraints

    Requirements:
    - Comply with FCRA, ECOA
    - No discrimination by protected classes
    - Explainable decisions (adverse action notices)
    """

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def engineer_features(self, applicant_data):
        """
        Feature engineering for credit scoring

        IMPORTANT: Cannot use protected attributes directly!
        - Race, color, religion, national origin
        - Sex, marital status
        - Age (with exceptions)
        """
        features = {}

        # Financial features (allowed)
        features['annual_income'] = applicant_data['annual_income']
        features['debt_to_income_ratio'] = (
            applicant_data['total_debt'] / applicant_data['annual_income']
        )
        features['employment_length_years'] = applicant_data['employment_length_years']

        # Credit history (allowed)
        features['credit_history_length_years'] = applicant_data['credit_history_length_years']
        features['num_credit_accounts'] = applicant_data['num_credit_accounts']
        features['num_delinquencies_2y'] = applicant_data['num_delinquencies_2y']
        features['num_public_records'] = applicant_data['num_public_records']

        # Payment behavior (allowed)
        features['payment_history_score'] = applicant_data['payment_history_score']
        features['credit_utilization_ratio'] = applicant_data['credit_utilization_ratio']

        # Loan features
        features['loan_amount'] = applicant_data['loan_amount']
        features['loan_term_months'] = applicant_data['loan_term_months']
        features['interest_rate'] = applicant_data['interest_rate']
        features['loan_purpose'] = applicant_data['loan_purpose']  # Will be encoded

        return pd.DataFrame([features])

    def check_fairness(self, X, y_pred, sensitive_attributes):
        """
        Check for disparate impact

        80% rule: Selection rate for protected group should be
        at least 80% of selection rate for majority group
        """
        from sklearn.metrics import confusion_matrix

        fairness_metrics = {}

        for attribute in sensitive_attributes:
            groups = X[attribute].unique()

            approval_rates = {}
            for group in groups:
                group_mask = X[attribute] == group
                approval_rate = y_pred[group_mask].mean()
                approval_rates[group] = approval_rate

            # Calculate disparate impact ratio
            max_rate = max(approval_rates.values())
            min_rate = min(approval_rates.values())
            disparate_impact_ratio = min_rate / max_rate if max_rate > 0 else 0

            fairness_metrics[attribute] = {
                'approval_rates': approval_rates,
                'disparate_impact_ratio': disparate_impact_ratio,
                'passes_80_percent_rule': disparate_impact_ratio >= 0.8
            }

            print(f"\nFairness Check - {attribute}:")
            print(f"  Approval rates by group: {approval_rates}")
            print(f"  Disparate impact ratio: {disparate_impact_ratio:.2%}")
            print(f"  Passes 80% rule: {fairness_metrics[attribute]['passes_80_percent_rule']}")

        return fairness_metrics

    def generate_adverse_action_notice(self, applicant_data, prediction, features):
        """
        Generate explanation for loan denial (required by FCRA)
        """
        import shap

        # Get feature importances for this prediction
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(features)

        # Top reasons for denial
        feature_names = features.columns
        shap_values_abs = np.abs(shap_values[0])
        top_indices = np.argsort(shap_values_abs)[::-1][:4]

        reasons = []
        for idx in top_indices:
            feature = feature_names[idx]
            value = features.iloc[0, idx]
            reasons.append(f"{feature}: {value:.2f}")

        notice = f"""
        ADVERSE ACTION NOTICE

        Dear {applicant_data['name']},

        Your application for credit has been denied based on the following factors:

        {chr(10).join(f"  {i+1}. {reason}" for i, reason in enumerate(reasons))}

        This decision was made in compliance with the Equal Credit Opportunity Act.

        You have the right to:
        - Request a copy of your credit report
        - Dispute any inaccurate information
        - Reapply for credit

        For questions, please contact us at [contact information].
        """

        return notice
```

---

## E-commerce & Retail ML

### Overview

E-commerce ML focuses on **customer experience**, **revenue optimization**, and **operational efficiency**.

### Common Use Cases

1. **Recommendation Systems** - Product recommendations, personalization
2. **Search Ranking** - Product search, autocomplete
3. **Dynamic Pricing** - Price optimization, promotions
4. **Demand Forecasting** - Inventory optimization, supply chain
5. **Customer Segmentation** - Marketing, targeting
6. **Fraud Prevention** - Account takeover, payment fraud

### Use Case 1: Product Recommendation System

*(This was covered in detail in ML_SYSTEM_DESIGN.md, here's a complementary cold-start solution)*

```python
class ColdStartRecommendationSystem:
    """
    Handle cold-start problem in recommendations

    Challenges:
    - New users (no history)
    - New products (no interactions)
    - Sparse data
    """

    def __init__(self):
        self.content_model = None  # Content-based for cold start
        self.collaborative_model = None  # Collaborative filtering for warm users
        self.hybrid_weights = {'content': 0.7, 'collaborative': 0.3}

    def recommend_new_user(self, user_profile, top_k=10):
        """
        Recommendations for new user with no history

        Strategy:
        1. Use demographic/preference data from onboarding
        2. Show popular items (globally or by category)
        3. Use content-based filtering
        4. Gradually transition to collaborative filtering as data accumulates
        """
        recommendations = []

        # Get user preferences from onboarding
        preferred_categories = user_profile.get('preferred_categories', [])
        preferred_brands = user_profile.get('preferred_brands', [])
        price_range = user_profile.get('price_range', (0, float('inf')))

        # Strategy 1: Popular items in preferred categories
        popular_in_categories = self.get_popular_items(
            categories=preferred_categories,
            brands=preferred_brands,
            price_range=price_range,
            top_k=top_k // 2
        )
        recommendations.extend(popular_in_categories)

        # Strategy 2: Similar users' favorites (based on demographics)
        similar_user_favorites = self.get_similar_user_favorites(
            user_profile,
            top_k=top_k // 2
        )
        recommendations.extend(similar_user_favorites)

        # Diversify and deduplicate
        recommendations = self.diversify_recommendations(recommendations, top_k)

        return recommendations

    def recommend_new_item(self, item, top_k=10):
        """
        Recommend new item to users

        Strategy:
        1. Find users who liked similar items (content-based similarity)
        2. Target users based on item attributes
        3. Exploration: Show to diverse user segment for feedback
        """
        # Content-based similarity
        similar_items = self.find_similar_items(item, top_k=100)

        # Find users who liked similar items
        target_users = []
        for similar_item in similar_items:
            users = self.get_users_who_liked(similar_item)
            target_users.extend(users)

        # Rank users by engagement level
        target_users = self.rank_users_by_engagement(target_users, top_k)

        # Add exploration: Random sample of users for feedback
        exploration_users = self.sample_users_for_exploration(n=top_k // 5)
        target_users.extend(exploration_users)

        return target_users

    def hybrid_recommendation(self, user_id, user_history_length, top_k=10):
        """
        Hybrid approach: Blend content-based and collaborative filtering

        Weight adjustment based on data availability:
        - New user (0-5 interactions): 90% content, 10% collaborative
        - Growing user (6-20 interactions): 50% content, 50% collaborative
        - Established user (20+ interactions): 20% content, 80% collaborative
        """
        # Adjust weights based on user history
        if user_history_length < 5:
            content_weight, collab_weight = 0.9, 0.1
        elif user_history_length < 20:
            content_weight, collab_weight = 0.5, 0.5
        else:
            content_weight, collab_weight = 0.2, 0.8

        # Get recommendations from both models
        content_recs = self.content_model.recommend(user_id, top_k=top_k * 2)
        collab_recs = self.collaborative_model.recommend(user_id, top_k=top_k * 2)

        # Blend scores
        blended_scores = {}
        for item, score in content_recs:
            blended_scores[item] = content_weight * score

        for item, score in collab_recs:
            if item in blended_scores:
                blended_scores[item] += collab_weight * score
            else:
                blended_scores[item] = collab_weight * score

        # Sort by blended score
        recommendations = sorted(blended_scores.items(),
                                key=lambda x: x[1], reverse=True)[:top_k]

        return recommendations
```

### Use Case 2: Dynamic Pricing

```python
class DynamicPricingModel:
    """
    Dynamic pricing optimization

    Goals:
    - Maximize revenue
    - Maintain brand perception
    - Consider competitive pricing
    - Personalized discounts
    """

    def __init__(self):
        self.demand_model = None  # Predicts demand at different price points
        self.elasticity_model = None  # Price elasticity by product

    def optimize_price(self, product, context):
        """
        Optimize price for product given context

        Context:
        - Current inventory level
        - Competitor prices
        - Time of day/week/year
        - Customer segment
        """
        base_price = product['base_price']
        min_price = product['min_price']  # Floor price (cost + min margin)
        max_price = product['max_price']  # Ceiling price (brand positioning)

        # Factor 1: Inventory level
        inventory_multiplier = self.calculate_inventory_multiplier(
            product['current_inventory'],
            product['target_inventory']
        )

        # Factor 2: Competitor pricing
        competitor_multiplier = self.calculate_competitor_multiplier(
            product['id'],
            context['competitor_prices']
        )

        # Factor 3: Demand prediction
        demand_multiplier = self.calculate_demand_multiplier(
            product,
            context['time'],
            context['seasonality']
        )

        # Factor 4: Customer segment
        customer_multiplier = self.calculate_customer_multiplier(
            context['customer_segment'],
            context['customer_lifetime_value']
        )

        # Combine factors
        price_multiplier = (
            inventory_multiplier *
            competitor_multiplier *
            demand_multiplier *
            customer_multiplier
        )

        # Calculate optimal price
        optimal_price = base_price * price_multiplier

        # Apply constraints
        optimal_price = np.clip(optimal_price, min_price, max_price)

        # Round to psychological pricing ($X.99)
        optimal_price = self.apply_psychological_pricing(optimal_price)

        return optimal_price

    def calculate_inventory_multiplier(self, current_inventory, target_inventory):
        """
        Adjust price based on inventory level

        High inventory: Lower price (clear stock)
        Low inventory: Higher price (maximize margin)
        """
        ratio = current_inventory / target_inventory

        if ratio > 1.5:  # Overstocked
            return 0.85  # 15% discount
        elif ratio > 1.2:
            return 0.95  # 5% discount
        elif ratio < 0.5:  # Low stock
            return 1.15  # 15% premium
        elif ratio < 0.8:
            return 1.05  # 5% premium
        else:
            return 1.0  # No adjustment

    def calculate_demand_multiplier(self, product, time, seasonality):
        """
        Adjust price based on predicted demand
        """
        # Predict demand at different price points
        price_points = np.linspace(product['min_price'], product['max_price'], 10)
        predicted_demand = []
        predicted_revenue = []

        for price in price_points:
            demand = self.demand_model.predict({
                'product_id': product['id'],
                'price': price,
                'time': time,
                'seasonality': seasonality
            })
            revenue = price * demand

            predicted_demand.append(demand)
            predicted_revenue.append(revenue)

        # Find price that maximizes revenue
        optimal_idx = np.argmax(predicted_revenue)
        optimal_price = price_points[optimal_idx]

        # Return multiplier relative to base price
        return optimal_price / product['base_price']
```

---

## Cross-Domain Patterns

### Pattern 1: Cold Start Problem

**Common across domains:**
- Healthcare: New hospital with limited data
- Finance: New customer with no credit history
- E-commerce: New user or new product

**Solutions:**
- Content-based models (use attributes)
- Transfer learning from similar domains
- Exploration strategies (bandits)
- Demographic/metadata-based fallbacks

### Pattern 2: Class Imbalance

**Common across domains:**
- Healthcare: Rare diseases (<1% prevalence)
- Finance: Fraud (<0.1% of transactions)
- E-commerce: Conversion rate (2-5%)

**Solutions:**
- Resampling (SMOTE, undersampling)
- Cost-sensitive learning
- Focal loss / class weights
- Ensemble methods
- Adjust decision thresholds

### Pattern 3: Explainability Requirements

**Common across domains:**
- Healthcare: Clinician needs to understand diagnosis
- Finance: FCRA requires explanations for credit denial
- E-commerce: Users want to know why they see certain products

**Solutions:**
- SHAP values
- LIME
- Attention visualizations
- Decision rules extraction
- Counterfactual explanations

---

## Interview Questions

### Question 1: Design a Medical Image Classifier for FDA Approval

**What they're looking for:**
- Understanding of regulatory requirements
- Clinical validation methodology
- Explainability and interpretability
- Safety and monitoring considerations

**Answer structure:**

1. **Requirements Gathering:**
   - Clinical problem and target population
   - Performance requirements (sensitivity, specificity)
   - FDA classification (Class II/III)
   - Regulatory pathway (510(k) vs PMA)

2. **Data and Labeling:**
   - Diverse, representative dataset
   - Expert radiologist labels (multiple readers)
   - Inter-rater agreement metrics
   - Data quality validation

3. **Model Development:**
   - Start with proven architecture (DenseNet, ResNet)
   - Transfer learning from ImageNet
   - Medical-specific preprocessing (CLAHE)
   - Data augmentation (clinically appropriate)
   - Ensemble for robustness

4. **Evaluation:**
   - Held-out test set (representative)
   - External validation (different hospitals)
   - Performance by subgroups (age, sex, ethnicity)
   - Compare to clinician performance
   - Calibration assessment

5. **Interpretability:**
   - Grad-CAM visualizations
   - Attention mechanism
   - Similar case retrieval
   - Confidence scores

6. **Clinical Validation:**
   - Prospective clinical trial
   - Primary endpoint (sensitivity)
   - Secondary endpoints (specificity, time savings)
   - Safety monitoring

7. **Deployment:**
   - Integration into PACS/EHR
   - Human-in-the-loop workflow
   - Continuous monitoring
   - Incident response plan

8. **Regulatory Submission:**
   - Technical documentation
   - Clinical evidence
   - Quality system documentation
   - Post-market surveillance plan

### Question 2: Design a Fraud Detection System with <50ms Latency

**Key points:**
- Multi-stage architecture (rules → fast ML → complex ML)
- Feature engineering and caching strategy
- Model selection (XGBoost for speed)
- Handling class imbalance
- Continuous learning from feedback

*(See detailed implementation in Finance ML section)*

### Question 3: Handle Cold Start in Recommendation System

**Approaches:**

1. **New User:**
   - Onboarding questionnaire (preferences)
   - Popular items (global or category-wise)
   - Demographic-based recommendations
   - Content-based filtering
   - Gradual transition to collaborative filtering

2. **New Item:**
   - Content-based similarity to existing items
   - Target users who liked similar items
   - Exploration: Show to diverse users for feedback
   - Initial boost in ranking (exploration bonus)

3. **Cold Start vs Warm Start Metrics:**
   - Track performance separately
   - Optimize for quick warming up
   - Measure: Time to N interactions

---

**Version:** 1.0
**Last Updated:** October 2025
**Coverage:** Healthcare, Finance, E-commerce ML