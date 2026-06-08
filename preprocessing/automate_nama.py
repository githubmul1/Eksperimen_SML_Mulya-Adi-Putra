import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def run_preprocessing(raw_data_path, output_path):
    print("Memulai proses otomatisasi preprocessing...")
    
    # 1. Load Data
    df = pd.read_csv(raw_data_path)

    # 2. Cleaning (Hapus duplikat dan outlier)
    df = df.drop_duplicates()
    df = df[df['person_age'] <= 100]
    df = df[df['person_emp_length'] <= 100]
    
    # 3. Handle Missing Values
    df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
    df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())

    # 4. Encoding Kategorikal
    categorical_cols = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # 5. Split Fitur & Target
    X = df_encoded.drop('loan_status', axis=1)
    y = df_encoded['loan_status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 6. Scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

    # 7. Menggabungkan data siap latih dan menyimpannya
    train_data_final = pd.concat([X_train_scaled, y_train.reset_index(drop=True)], axis=1)
    
    # Memastikan folder tujuan ada, lalu simpan file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    train_data_final.to_csv(output_path, index=False)
    
    print(f"Preprocessing selesai! Data siap latih disimpan di: {output_path}")
    return train_data_final

# Blok ini akan tereksekusi saat file dijalankan di terminal
if __name__ == "__main__":
    # Tentukan lokasi file mentah dan lokasi tujuan simpan
    RAW_DATA = 'credit_risk_raw/credit_risk_dataset.csv'
    PROCESSED_DATA = 'preprocessing/credit_risk_preprocessed.csv'
    
    # Jalankan fungsi
    run_preprocessing(RAW_DATA, PROCESSED_DATA)