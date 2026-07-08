# Credit Card Fraud Detection Pipeline

An end-to-end Machine Learning pipeline designed to detect fraudulent credit card transactions. This project implements advanced handling for highly imbalanced data structures using synthetic data generation techniques paired with ensemble classification methods.

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Machine Learning:** Scikit-Learn
* **Imbalance Handling:** Imbalanced-Learn (SMOTE)
* **Data Processing:** Pandas, NumPy
* **Visualization:** Seaborn, Matplotlib

---

## 📁 Project Structure

```text
credit_card_fraud_detection/
│
├── data/
│   └── creditcard.csv             # Transaction dataset
│
├── notebooks/
│   └── exploratory_analysis.ipynb # EDA & feature relationships
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py           # Standard scaling & data balancing
│   └── train.py                   # Model training and metric logging
│
├── models/
│   └── fraud_model.pkl            # Trained, serialized model file
│
├── generate_mock_data.py          # Synthetic data simulation script
└── requirements.txt               # Project dependency versions
