import learn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name="Model"):
    """
    Antreneaza un model si afiseaza metricile de performanta.
    """
    print(f"\n{'='*20} Antrenare {model_name} {'='*20}")
    
    # Antrenare
    model.fit(X_train, y_train)
    
    # Predictie pe setul de test
    y_pred = model.predict(X_test)
    
    # Evaluare
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acuratete {model_name}: {accuracy:.4f}")
    
    print("\nRaport de Clasificare:")
    print(classification_report(y_test, y_pred))
    
    return model, y_pred

def run_models(X_train, y_train, X_test, y_test):
    """
    Ruleaza cei 3 algoritmi specificati: Random Forest, Gradient Boosting, Logistic Regression
    """
    models = {}
    
    # 1. Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    models['RandomForest'] = train_and_evaluate_model(rf_model, X_train, y_train, X_test, y_test, "Random Forest")
    
    # 2. Gradient Boosting
    # Gradient Boosting nu suporta direct 'class_weight'. Pentru imbalance, se recomanda
    # tehnici de over/undersampling sau librarii precum XGBoost/LightGBM.
    gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    models['GradientBoosting'] = train_and_evaluate_model(gb_model, X_train, y_train, X_test, y_test, "Gradient Boosting")
    
    # 3. Logistic Regression
    # max_iter crescut pentru convergenta pe dataset-uri mari
    lr_model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1, class_weight='balanced')
    models['LogisticRegression'] = train_and_evaluate_model(lr_model, X_train, y_train, X_test, y_test, "Logistic Regression")
    
    return models
