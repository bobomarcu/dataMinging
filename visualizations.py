import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_confusion_matrices(models_results, y_test, class_names):
    """
    Genereaza matricea de confuzie pentru fiecare model din dictionarul models_results.
    
    models_results: dict {nume_model: (model_instanta, y_pred)}
    y_test: valorile reale
    class_names: lista cu numele claselor (ex: ['Healthy', 'Stressed', 'Failed'])
    """
    n_models = len(models_results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    
    if n_models == 1:
        axes = [axes]

    for ax, (name, (model, y_pred)) in zip(axes, models_results.items()):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=class_names, yticklabels=class_names)
        ax.set_title(f'Confusion Matrix - {name}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    plt.tight_layout()
    plt.show()

def plot_feature_importance(models_results, feature_names):
    """
    Afiseaza top 10 cele mai importante feature-uri pentru modelele bazate pe arbori (Random Forest, Gradient Boosting).
    """
    for name, (model, _) in models_results.items():
        # Verificam daca modelul are atributul feature_importances_
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(10, 6))
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:10] # Top 10
            
            plt.title(f'Top 10 Feature Importance - {name}')
            plt.bar(range(len(indices)), importances[indices], align='center')
            plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
