import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.tree import plot_tree

# Tema Ayarları
sns.set_theme(style="whitegrid")

def print_performance_summary(model_name, y_true, y_pred, y_prob, cv_scores=None):
    """
    Modelin performans raporunu ve temel metriklerini basar.
    """
    print("\n" + "="*45)
    print(f"{model_name.upper()} PERFORMANS ÖZETİ")
    print(f"Test Seti Doğruluğu: %{accuracy_score(y_true, y_pred)*100:.1f}")
    
    if cv_scores is not None:
        print(f"5-Fold CV Ortalaması: %{cv_scores.mean()*100:.1f}")
    
    print(f"AUC Değeri: {roc_auc_score(y_true, y_prob):.3f}")
    print("-" * 45)
    print(classification_report(y_true, y_pred))

def plot_confusion_matrices(models_data, output_path=None):
    """
    Birden fazla model için karışıklık matrislerini yan yana çizer.
    """
    fig, ax = plt.subplots(1, len(models_data), figsize=(14, 5))
    
    for i, (name, y_true, y_pred) in enumerate(models_data):
        sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax[i])
        acc = accuracy_score(y_true, y_pred) * 100
        ax[i].set_title(f'{name} Matrisi\n(Doğruluk: %{acc:.1f})')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(os.path.join(output_path, 'confusion_matrices.png'))
        print(f"Matris grafiği kaydedildi: {output_path}/confusion_matrices.png")
    plt.show()

def plot_roc_curves(curves_data, output_path=None):
    """
    Modellerin ROC eğrilerini karşılaştırmalı olarak çizer.
    """
    plt.figure(figsize=(8, 6))
    
    for y_true, y_prob, name, color in curves_data:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', color=color, lw=2)
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.title('ROC Eğrileri Karşılaştırması')
    plt.legend()
    
    if output_path:
        plt.savefig(os.path.join(output_path, 'roc_curves.png'))
        print(f"ROC eğrileri grafiği kaydedildi: {output_path}/roc_curves.png")
    plt.show()

def plot_decision_tree_structure(dt_model, feature_names, output_path=None):
    """
    Karar Ağacı yapısını görselleştirir.
    """
    plt.figure(figsize=(20, 10))
    plot_tree(dt_model, 
              feature_names=feature_names, 
              class_names=['Yok', 'Var'], 
              filled=True, 
              rounded=True)
    plt.title('Karar Ağacı Görselleştirmesi (Maks Derinlik 4)')
    
    if output_path:
        plt.savefig(os.path.join(output_path, 'decision_tree.png'))
        print(f"Karar ağacı yapısı kaydedildi: {output_path}/decision_tree.png")
    plt.show()
