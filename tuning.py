from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd

def tune_models(X_train, y_train):
    print(f"\n{'='*20} Start Tuning Hiperparametri (Grid Search) {'='*20}")
    
    SAMPLE_SIZE = 30000
    if len(X_train) > SAMPLE_SIZE:
        print(f"Dataset mare ({len(X_train)}). Se utilizeaza un subset de {SAMPLE_SIZE} pentru tuning rapid...")
        X_tune = X_train[:SAMPLE_SIZE]
        y_tune = y_train[:SAMPLE_SIZE]
    else:
        X_tune = X_train
        y_tune = y_train

    best_models = {}

    # 1. Random Forest Tuning
    print("\n--- Tuning Random Forest ---")
    rf_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
    rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='recall_macro', n_jobs=-1, verbose=1)
    rf_grid.fit(X_tune, y_tune)
    print(f"Cei mai buni parametri RF: {rf_grid.best_params_}")
    best_models['RandomForest'] = rf_grid.best_estimator_

    # 2. Gradient Boosting Tuning
    print("\n--- Tuning Gradient Boosting ---")
    gb_params = {
        'n_estimators': [100, 500],
        'learning_rate': [0.05, 0.1, 0.2]
    }
    gb = GradientBoostingClassifier(random_state=42)
    gb_grid = GridSearchCV(gb, gb_params, cv=3, scoring='recall_macro', n_jobs=-1, verbose=1)
    gb_grid.fit(X_tune, y_tune)
    print(f"Cei mai buni parametri GB: {gb_grid.best_params_}")
    best_models['GradientBoosting'] = gb_grid.best_estimator_

    # 3. Logistic Regression Tuning
    print("\n--- Tuning Logistic Regression ---")
    lr_params = {
        'C': [0.1, 1, 10],
        'solver': ['lbfgs']
    }
    lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1, class_weight='balanced')
    lr_grid = GridSearchCV(lr, lr_params, cv=3, scoring='recall_macro', n_jobs=-1, verbose=1)
    lr_grid.fit(X_tune, y_tune)
    print(f"Cei mai buni parametri LR: {lr_grid.best_params_}")
    best_models['LogisticRegression'] = lr_grid.best_estimator_

    return best_models
