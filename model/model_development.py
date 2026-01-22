# =====================================================
# Breast Cancer Prediction Model Development
# Educational Purpose Only
# =====================================================

import numpy as np
import pandas as pd
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------------------------------
# 1. Load Breast Cancer Wisconsin Dataset
# -----------------------------------------------------
data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)
df['diagnosis'] = data.target   # 0 = Benign, 1 = Malignant

print("Dataset loaded successfully")
print("Dataset shape:", df.shape)

# -----------------------------------------------------
# 2. Feature Selection (Choose 5 features)
# -----------------------------------------------------
selected_features = [
    'mean radius',
    'mean texture',
    'mean perimeter',
    'mean area',
    'mean concavity'
]

X = df[selected_features]
y = df['diagnosis']

print("\nSelected Features:")
print(selected_features)

# -----------------------------------------------------
# 3. Handle Missing Values (if any)
# -----------------------------------------------------
X = X.fillna(X.mean())

# -----------------------------------------------------
# 4. Split Dataset
# -----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------------------
# 5. Feature Scaling (MANDATORY)
# -----------------------------------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------------------
# 6. Train Model (Logistic Regression)
# -----------------------------------------------------
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

print("\nModel training completed")

# -----------------------------------------------------
# 7. Model Evaluation
# -----------------------------------------------------
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

# -----------------------------------------------------
# 8. Save Model and Scaler
# -----------------------------------------------------
joblib.dump(model, "breast_cancer_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel and scaler saved successfully")

# -----------------------------------------------------
# 9. Reload Model & Test Prediction (No Retraining)
# -----------------------------------------------------
loaded_model = joblib.load("breast_cancer_model.pkl")
loaded_scaler = joblib.load("scaler.pkl")

sample_input = [[14.5, 20.0, 95.0, 700.0, 0.15]]
sample_scaled = loaded_scaler.transform(sample_input)

prediction = loaded_model.predict(sample_scaled)

print("\nSample Prediction Test:")
print("Prediction Result:",
      "Malignant" if prediction[0] == 1 else "Benign")

# -----------------------------------------------------
# END OF SCRIPT
# -----------------------------------------------------
