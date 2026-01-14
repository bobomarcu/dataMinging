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
    """
    Incarca setul de date, realizeaza sanitizarea, curatarea si normalizarea datelor.
    """
    print(f"Incarcare date din {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Eroare: Fisierul nu a fost gasit.")
        return None, None, None, None

    # initial_rows = len(df)
    # df = df.drop_duplicates()
    # if len(df) < initial_rows:
    #     print(f"S-au eliminat {initial_rows - len(df)} duplicate.")

    # Identificam si eliminam coloanele care pot cauza "target leakage" sau overfitting excesiv.
    # Acestea sunt adesea indicatori directi ai unei probleme, nu cauze subtile.
    cols_to_drop_for_realism = [
        # 'error_rate_percent',
        'p95_latency_ms',
        # 'network_latency_ms', # Poate fi un indicator foarte puternic
        # 'avg_latency_ms',     # La fel ca mai sus
        'circuit_breaker_open',
        # 'db_connection_pool_exhausted',
        'retry_storm_detected'
    ]

    for col in cols_to_drop_for_realism:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"Coloana '{col}' eliminata pentru a creste realismul predictiilor.")

    # 2. Codificare Variabile Categorice (Features)
    # Identificam coloanele categorice si numerice
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

# ... imports ramase ...

if __name__ == "__main__":
    # Redirectam output-ul catre un fisier, dar pastram si consola
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
        # ... restul codului din main ...
        X, y, scaler, le, numerical_cols = load_and_preprocess_data(file_path)
        
        if X is not None:
            # ... (tot codul existent) ...
            print(X.head())

            # Utilizam functia de splitare a datelor
            X_train, X_test, X_pred, y_train, y_test, y_pred = split_dataset(X, y)
            
            print("\nDupa splitare:")
            print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
            print(f"X_pred shape: {X_pred.shape}, y_pred shape: {y_pred.shape}")

            # Aplicam SMOTE pentru a echilibra setul de antrenament
            X_train_res, y_train_res = apply_smote(X_train, y_train)

            # Opțiune: Tuning sau Rulare standard?
            # Pentru demo, activam tuning-ul.
            PERFORM_TUNING = True

            models_results = {}

            if PERFORM_TUNING:
                print("Se executa optimizarea hiperparametrilor...")
                # Tuning se face pe setul de training echilibrat cu SMOTE
                best_estimators = tune_models(X_train_res, y_train_res) 
                
                # Re-antrenam (daca e cazul) sau doar evaluam modelele gasite pe tot setul de test
                print("\nEvaluare modele optimizate pe setul de Test:")
                for name, model in best_estimators.items():
                    # Nota: GridSearchCV returneaza modelul deja antrenat pe setul de tuning (subset).
                    # Ideal este sa il re-antrenam pe tot X_train_res inainte de testare finala.
                    model_result = train_and_evaluate_model(model, X_train_res, y_train_res, X_test, y_test, name)
                    models_results[name] = model_result
            else:
                # Rulare cu parametri default pe setul de training echilibrat cu SMOTE
                models_results = run_models(X_train_res, y_train_res, X_test, y_test)

            # Vizualizari
            print("\nGenerare grafice...")
            class_names = le.classes_ # Obtinem numele reale ale claselor (Healthy, etc.)
            plot_confusion_matrices(models_results, y_test, class_names)
            plot_feature_importance(models_results, X.columns)
            print("Grafice generate.")

            # Predicție finală și Export
            predict_new_data(models_results, X_pred, y_pred, scaler, le, numerical_cols)

    finally:
        sys.stdout = original_stdout
        f.close()
