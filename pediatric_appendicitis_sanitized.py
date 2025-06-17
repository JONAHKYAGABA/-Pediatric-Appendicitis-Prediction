# Pediatric Appendicitis Prediction Script (Detailed & Sanitized Version)

# --- Library Imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, classification_report, confusion_matrix
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# --- Data Loading ---
data_path = "appendicitis.csv"  # Ensure this file exists in the same directory as the script
df = pd.read_csv(data_path)

print("Initial dataset shape:", df.shape)

# --- Data Cleaning ---
print("\nChecking for missing values:")
print(df.isnull().sum())

# Drop rows with any missing values
df.dropna(inplace=True)
print("Dataset shape after dropping missing values:", df.shape)

# --- Feature Engineering ---
print("\nPreview of cleaned dataset:")
print(df.head())

# Define feature matrix X and target vector y
X = df.drop(columns=["Appendicitis"])
y = df["Appendicitis"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Feature Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Define Models ---
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# --- Model Evaluation ---
results = []
plt.figure(figsize=(10, 6))

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    results.append((name, auc, f1))

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")

    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# --- Plot ROC Curve ---
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison of Classifiers")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Performance Summary Table ---
result_df = pd.DataFrame(results, columns=["Model", "AUC", "F1 Score"])
print("\nModel Performance Summary:")
print(result_df.sort_values(by="AUC", ascending=False))

# ✅ Sanitized: No access tokens, no hardcoded credentials, and no shell commands present
