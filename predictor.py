import pandas as pd
import numpy as np
import os
from sklearn.metrics import recall_score, accuracy_score
from visualizations import plot_prediction_scatter

def predict_new_data(models_results, X_pred, y_pred, feature_scaler, label_encoder, numerical_cols, save_path="predictions.csv"):
    """
    Selecteaza cel mai bun model si face predictii pe setul de date noi (X_pred).
    Salveaza rezultatele intr-un CSV pentru analiza.
    """
    print(f"\n{'='*20} Predictie pe Date Noi (Simulare Productie) {'='*20}")
    
    best_model_name = None
    best_score = -1
    best_model = None

    # 1. Alegem cel mai bun model pe baza Recall Macro
    for name, (model, y_test_pred) in models_results.items():
        y_pred_new = model.predict(X_pred)
        current_score = recall_score(y_pred, y_pred_new, average='macro')
        print(f"Model: {name} - Recall Macro pe setul de predictie: {current_score:.4f}")
        
        if current_score > best_score:
            best_score = current_score
            best_model_name = name
            best_model = model

    print(f"\n>>> Modelul castigator selectat: {best_model_name} (Recall: {best_score:.4f})")

    # 2. Generam predictiile finale cu modelul castigator
    final_predictions = best_model.predict(X_pred)
    
    # 3. Reconstruim un DataFrame lizibil
    # Copiem X_pred pentru a nu modifica originalul
    df_results = X_pred.copy()
    
    # Inversam scalarea DOAR pentru coloanele numerice
    if numerical_cols:
        try:
            df_results[numerical_cols] = feature_scaler.inverse_transform(df_results[numerical_cols])
        except Exception as e:
            print(f"Atentie: Nu s-a putut inversa scalarea complet ({e}). Se folosesc valorile scalate.")
    
    # Adaugam valorile reale si cele prezise
    df_results['Actual_State'] = label_encoder.inverse_transform(y_pred)
    df_results['Predicted_State'] = label_encoder.inverse_transform(final_predictions)
    
    # Marcam unde a gresit modelul
    df_results['Correct_Prediction'] = df_results['Actual_State'] == df_results['Predicted_State']

    # 4. Salvare
    df_results.to_csv(save_path, index=False)
    print(f"Rezultatele predictiei au fost salvate in: {save_path}")
    
    # 5. Generare Scatter Plot Comparativ (Actual vs Predicted)
    if os.path.exists(save_path):
        print(f"Citire date din {save_path} pentru generarea Scatter Plot Comparativ...")
        df_loaded = pd.read_csv(save_path)
        
        # Identificam coloanele de features
        metadata_cols = ['Actual_State', 'Predicted_State', 'Correct_Prediction']
        feature_cols = [c for c in df_loaded.columns if c not in metadata_cols]
        
        X_plot = df_loaded[feature_cols]
        y_actual_plot = df_loaded['Actual_State']
        y_predicted_plot = df_loaded['Predicted_State']
        
        # Apelam noua functie care deseneaza ambele grafice
        plot_prediction_scatter(X_plot, y_actual_plot, y_predicted_plot)

    
    # Afisam cateva exemple de erori critice
    critical_errors = df_results[
        (df_results['Actual_State'].isin(['total_outage', 'cascading_failure'])) & 
        (df_results['Correct_Prediction'] == False)
    ]
    
    if not critical_errors.empty:
        print(f"\nAtentie! {len(critical_errors)} erori critice detectate (Failed/Outage ratate):")
        print(critical_errors[['Actual_State', 'Predicted_State']].head())
    else:
        print("\nExcelent! Nicio eroare critica detectata pe setul de predictie.")

    return df_results
