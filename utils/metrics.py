"""
Utilitaires pour le calcul et la visualisation de métriques
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score
)
import tensorflow as tf
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

import config


def calculate_metrics(y_true, y_pred, class_names=None):
    """Calcule toutes les métriques importantes"""
    if class_names is None:
        class_names = config.CLASSES

    # Métriques globales
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )

    # Métriques par classe
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )

    metrics = {
        'global': {
            'precision': precision,
            'recall': recall,
            'f1': f1
        },
        'per_class': {}
    }

    for i, class_name in enumerate(class_names):
        metrics['per_class'][class_name] = {
            'precision': precision_per_class[i],
            'recall': recall_per_class[i],
            'f1': f1_per_class[i],
            'support': int(support_per_class[i]) if i < len(support_per_class) else 0
        }

    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """Génère et sauvegarde la matrice de confusion"""
    if class_names is None:
        class_names = config.CLASSES

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Matrice de confusion')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Matrice de confusion sauvegardée: {save_path}")

    plt.close()

    return cm


def plot_training_history(history, save_path=None):
    """Visualise l'historique d'entraînement"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
    axes[0, 0].set_title('Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train')
    axes[0, 1].plot(history.history['val_loss'], label='Validation')
    axes[0, 1].set_title('Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Precision
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], label='Train')
        axes[1, 0].plot(history.history['val_precision'], label='Validation')
        axes[1, 0].set_title('Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

    # Recall
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'], label='Train')
        axes[1, 1].plot(history.history['val_recall'], label='Validation')
        axes[1, 1].set_title('Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Historique d'entraînement sauvegardé: {save_path}")

    plt.close()


def print_classification_report(y_true, y_pred, class_names=None):
    """Affiche le rapport de classification détaillé"""
    if class_names is None:
        class_names = config.CLASSES

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )

    print("\n" + "=" * 60)
    print("RAPPORT DE CLASSIFICATION")
    print("=" * 60)
    print(report)

    # Vérifier si le recall global atteint l'objectif
    metrics = calculate_metrics(y_true, y_pred, class_names)
    global_recall = metrics['global']['recall']

    print(f"\nRecall global: {global_recall:.4f}")
    if global_recall >= config.TARGET_RECALL:
        print(f"✓ Objectif de recall atteint ({config.TARGET_RECALL})")
    else:
        print(f"✗ Objectif de recall non atteint ({config.TARGET_RECALL})")
        print(f"  Écart: {config.TARGET_RECALL - global_recall:.4f}")

    return report


def get_misclassified_samples(y_true, y_pred, file_paths, top_n=10):
    """Identifie les échantillons mal classifiés"""
    misclassified_indices = np.where(y_true != y_pred)[0]

    results = []
    for idx in misclassified_indices[:top_n]:
        results.append({
            'file': file_paths[idx],
            'true_class': config.CLASSES[y_true[idx]],
            'predicted_class': config.CLASSES[y_pred[idx]]
        })

    return results