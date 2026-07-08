import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score
from imblearn.over_sampling import SMOTE

# 1. Load the Dataset
# (Using the standard Kaggle dataset: 'creditcard.csv')
print("Loading dataset...")
df = pd.read_csv('../data/creditcard.csv')

# 2. Separate Features and Target Variable
# 'Class' column: 0 = Legitimate, 1 = Fraudulent
X = df.drop(columns=['Class', 'Time']) # Dropping Time as it is just an elapsed counter
y = df['Class']

# 3. Stratified Train-Test Split 
# Ensures both train and test sets have the same ratio of fraud transactions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 4. Feature Scaling (Scaling 'Amount')
scaler = StandardScaler()
# Fit on training data and transform both sets
X_train['Amount'] = scaler.fit_transform(X_train['Amount'].values.reshape(-1, 1))
X_test['Amount'] = scaler.transform(X_test['Amount'].values.reshape(-1, 1))

print(f"Original training shape: {X_train.shape}")
print(f"Original class distribution: \n{y_train.value_counts()}")

# 5. Handle Imbalance using SMOTE (Oversampling the minority fraud class)
print("\nApplying SMOTE to balance the dataset...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"Balanced training shape: {X_train_balanced.shape}")
print(f"Balanced class distribution: \n{y_train_balanced.value_counts()}")

# 6. Model Training (Random Forest)
print("\nTraining Random Forest Classifier (this may take a minute)...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_balanced, y_train_balanced)

# 7. Model Evaluation
print("\nEvaluating Model Performance...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Print Evaluation Metrics
print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# Calculate Area Under the Precision-Recall Curve (AUPRC)
auprc = average_precision_score(y_test, y_pred_proba)
print(f"Area Under the Precision-Recall Curve (AUPRC): {auprc:.4f}")