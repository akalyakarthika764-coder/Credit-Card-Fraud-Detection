# src/preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_and_preprocess_data(data_path):
    print("[-] Loading dataset...")
    df = pd.read_csv(data_path)
    
    # Separate features and target (Dropping 'Time' as it's just an elapsed counter)
    X = df.drop(columns=['Class', 'Time'])
    y = df['Class']
    
    # Stratified split to keep the fraud percentage consistent across splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale the 'Amount' column to match the PCA features scale
    scaler = StandardScaler()
    X_train['Amount'] = scaler.fit_transform(X_train['Amount'].values.reshape(-1, 1))
    X_test['Amount'] = scaler.transform(X_test['Amount'].values.reshape(-1, 1))
    
    return X_train, X_test, y_train, y_test

def apply_smote(X_train, y_train):
    print("[-] Balancing dataset with SMOTE...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled