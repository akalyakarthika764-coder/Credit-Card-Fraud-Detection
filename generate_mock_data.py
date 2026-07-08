# generate_mock_data.py
import os
import numpy as np
import pandas as pd

def create_mock_creditcard_csv():
    print("[-] Generating synthetic credit card dataset...")
    
    # Define directory and file path relative to the root folder
    data_dir = './data'
    file_path = os.path.join(data_dir, 'creditcard.csv')
    
    # Create the data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    np.random.seed(42)
    num_records = 5000  # Small scale for fast testing
    
    # 1. Generate 'Time' (Sequential seconds elapsed)
    time_col = np.sort(np.random.randint(0, 172792, size=num_records))
    
    # 2. Generate PCA features V1 to V28 (Normally distributed)
    v_features = {f'V{i}': np.random.normal(loc=0.0, scale=1.0, size=num_records) for i in range(1, 29)}
    
    # 3. Generate 'Amount' (Varying transaction costs)
    amount_col = np.random.exponential(scale=88.0, size=num_records)
    
    # 4. Generate highly imbalanced 'Class' labels (approx 0.2% fraud)
    # 0 = Legit, 1 = Fraud
    class_col = np.random.choice([0, 1], size=num_records, p=[0.998, 0.002])
    
    # Combine everything into a dictionary
    data = {
        'Time': time_col,
        **v_features,
        'Amount': amount_col,
        'Class': class_col
    }
    
    # Build DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    
    print(f"[+] Successfully generated mock dataset at: {file_path}")
    print(f"Total Transactions: {len(df)}")
    print(f"Class Distribution:\n{df['Class'].value_counts()}")

if __name__ == '__main__':
    create_mock_creditcard_csv()