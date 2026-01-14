import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
from data_splitter import split_dataset
from models import run_models, train_and_evaluate_model
from visualizations import plot_confusion_matrices, plot_feature_importance
from tuning import tune_models
from oversampling import apply_smote
from predictor import predict_new_data

def load_and_preprocess_data(file_path):
    print(f"Incarcare date din {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Eroare: Fisierul nu a fost gasit.")
        return None, None, None, None

    initial_rows = len(df)
    df = df.drop_duplicates()
    if len(df) < initial_rows:
        print(f"S-au eliminat {initial_rows - len(df)} duplicate.")

    cols_to_drop_for_realism = [
        # 'error_rate_percent',
        'p95_latency_ms',
        # 'network_latency_ms',
        # 'avg_latency_ms',
        'circuit_breaker_open',
        # 'db_connection_pool_exhausted',
        'retry_storm_detected'
    ]

    for col in cols_to_drop_for_realism:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"Coloana '{col}' eliminata pentru a creste realismul predictiilor.")

    target_col = 'system_state'
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # One-Hot Encoding pentru variabilele de intrare categorice
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # 3. Codificare Target
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col])
    print(f"Clase target mapate: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # 4. Normalizare (StandardScaler) pentru variabilele numerice
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Separare X si y
    X = df.drop(columns=[target_col])
    y = df[target_col]

    print("Preprocesare finalizata cu succes.")
    return X, y, scaler, le, numerical_cols

import sys


if __name__ == "__main__":
    class Tee(object):
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush() # Ensure immediate write
        def flush(self):
            for f in self.files:
                f.flush()

    f = open('execution_log.txt', 'w')
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, f)

    try:
        file_path = 'dataset/distributed_system_architecture_stress_dataset.csv'
        X, y, scaler, le, numerical_cols = load_and_preprocess_data(file_path)
        
        if X is not None:
            print(X.head())

            X_train, X_test, X_pred, y_train, y_test, y_pred = split_dataset(X, y)
            
            print("\nDupa splitare:")
            print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
            print(f"X_pred shape: {X_pred.shape}, y_pred shape: {y_pred.shape}")

            X_train_res, y_train_res = apply_smote(X_train, y_train)

            PERFORM_TUNING = True

            models_results = {}

            if PERFORM_TUNING:
                print("Se executa optimizarea hiperparametrilor...")
                best_estimators = tune_models(X_train_res, y_train_res)
                
                print("\nEvaluare modele optimizate pe setul de Test:")
                for name, model in best_estimators.items():
                    model_result = train_and_evaluate_model(model, X_train_res, y_train_res, X_test, y_test, name)
                    models_results[name] = model_result
            else:
                models_results = run_models(X_train_res, y_train_res, X_test, y_test)

            # Vizualizari
            print("\nGenerare grafice...")
            class_names = le.classes_
            plot_confusion_matrices(models_results, y_test, class_names)
            plot_feature_importance(models_results, X.columns)
            print("Grafice generate.")

            # Predicție finală și Export
            predict_new_data(models_results, X_pred, y_pred, scaler, le, numerical_cols)

    finally:
        sys.stdout = original_stdout
        f.close()
