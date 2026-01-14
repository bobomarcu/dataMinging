import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import os

def ensure_output_dir(directory="output_images"):
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_confusion_matrices(models_results, y_test, class_names, save_dir="output_images"):
    ensure_output_dir(save_dir)
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
    filename = os.path.join(save_dir, "confusion_matrices.png")
    plt.savefig(filename)
    print(f"Matricea de confuzie salvata in: {filename}")
    plt.close()

def plot_feature_importance(models_results, feature_names, save_dir="output_images"):
    ensure_output_dir(save_dir)
    
    for name, (model, _) in models_results.items():
        importances = None
        metric_name = "Importance"
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            metric_name = "Feature Importance"
            
        elif hasattr(model, 'coef_'):
            importances = np.mean(np.abs(model.coef_), axis=0)
            metric_name = "Mean Abs Coefficient"
            
        if importances is not None:
            plt.figure(figsize=(10, 6))
            indices = np.argsort(importances)[::-1][:10] # Top 10
            
            plt.title(f'Top 10 {metric_name} - {name}')
            plt.bar(range(len(indices)), importances[indices], align='center')
            plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
            plt.tight_layout()
            
            filename = os.path.join(save_dir, f"feature_importance_{name}.png")
            plt.savefig(filename)
            print(f"Grafic salvat pentru {name} in: {filename}")
            plt.close()
        else:
            print(f"Modelul {name} nu suporta vizualizarea importantei feature-urilor.")

def plot_prediction_scatter(X, y_actual, y_pred, save_dir="output_images"):
    ensure_output_dir(save_dir)
    
    print("Se calculeaza PCA (2 componente) pe datele din CSV...")
    
    X_numeric = X.select_dtypes(include=[np.number])
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_numeric)
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    all_classes = np.unique(np.concatenate([y_actual, y_pred]))
    colors = sns.color_palette("husl", len(all_classes))
    color_map = dict(zip(all_classes, colors))
    
    def draw_scatter(ax, y_data, title):
        for class_label in all_classes:
            mask = y_data == class_label
            if np.sum(mask) > 0:
                ax.scatter(
                    X_pca[mask, 0], 
                    X_pca[mask, 1], 
                    label=f"{class_label} ({np.sum(mask)})",
                    color=color_map[class_label],
                    alpha=0.6,
                    edgecolor='w',
                    linewidth=0.5,
                    s=40
                )
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.legend(title='State', loc='best')
        ax.grid(True, linestyle='--', alpha=0.3)

    draw_scatter(axes[0], y_actual, "ACTUAL State (Ground Truth)")
    
    draw_scatter(axes[1], y_pred, "PREDICTED State (Model Output)")
    
    plt.tight_layout()
    filename = os.path.join(save_dir, "prediction_scatter_comparison.png")
    plt.savefig(filename)
    print(f"Graficul Comparativ (Actual vs Predicted) salvat in: {filename}")
    plt.close()