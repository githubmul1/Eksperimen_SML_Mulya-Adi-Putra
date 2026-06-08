import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import logging

# Konfigurasi Logging
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info("Memulai proses...")

# Load dataset
logging.info("Memuat dataset preprocessed...")
df = pd.read_csv('credit_risk_preprocessed.csv')

# Pisahkan fitur dan target
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# Split train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mengaktifkan fitur autolog MLflow
mlflow.autolog()

# Memulai proses training
with mlflow.start_run(run_name="RandomForest_Basic_Run"):
    logging.info("Memulai pelatihan model Random Forest...")
    
    # Inisialisasi model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Latih model
    rf_model.fit(X_train, y_train)
    
    logging.info("Pelatihan selesai! Artefak dan metrik berhasil dicatat.")