import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    roc_auc_score, 
    roc_curve,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.preprocessing import StandardScaler
import warnings
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Stil ayarları
plt.style.use('seaborn-v0_8-muted')
sns.set_theme(style="whitegrid")
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """UCI Heart Disease Cleveland veri setini yükler ve temizler."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    df = pd.read_csv(url, names=column_names)
    
    # Eksik değerleri temizleme
    df = df.replace('?', np.nan)
    numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca', 'thal']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Eksik satırları çıkart (genellikle 6 satır)
    df_clean = df.dropna().copy()
    
    # Hedef değişkeni ikili (binary) sınıfa indirgeme (0: Yok, 1: Var)
    df_clean['target'] = df_clean['target'].apply(lambda x: 0 if x == 0 else 1)
    
    return df_clean

def perform_eda(df, output_dir):
    """Keşifsel Veri Analizi (EDA) görselleştirmelerini oluşturur."""
    print("[*] EDA görselleştirmeleri hazırlanıyor...")
    
    # 1. Temel Betimsel İstatistikler
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    
    # Target Dağılımı
    sns.countplot(x='target', data=df, ax=axes[0, 0], palette=['#2ecc71', '#e74c3c'])
    axes[0, 0].set_title('Hedef Değişken (Hastalık Var/Yok)')
    
    # Yaş Dağılımı
    sns.histplot(df['age'], kde=True, ax=axes[0, 1], color='#3498db')
    axes[0, 1].set_title('Yaş Dağılımı')
    
    # Cinsiyet Dağılımı
    sns.countplot(x='sex', data=df, ax=axes[0, 2], palette=['#ff9ff3', '#54a0ff'])
    axes[0, 2].set_xticklabels(['Kadın', 'Erkek'])
    axes[0, 2].set_title('Cinsiyet Dağılımı')
    
    # Kan Basıncı
    sns.histplot(df['trestbps'], kde=True, ax=axes[1, 0], color='#9b59b6')
    axes[1, 0].set_title('Dinlenme Kan Basıncı')
    
    # Kolesterol
    sns.histplot(df['chol'], kde=True, ax=axes[1, 1], color='#f39c12')
    axes[1, 1].set_title('Serum Kolesterol')
    
    # Max Kalp Hızı
    sns.histplot(df['thalach'], kde=True, ax=axes[1, 2], color='#1abc9c')
    axes[1, 2].set_title('Maksimum Kalp Hızı')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'eda_stats.png', dpi=150)
    
    # 2. Korelasyon Matrisi
    plt.figure(figsize=(12, 10))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='RdBu_r', center=0)
    plt.title('Korelasyon Matrisi (Üçgen Maske)')
    plt.savefig(output_dir / 'correlation_matrix.png', dpi=150)
    
    # 3. Kutu Grafikleri (Outlier ve Dağılım Analizi)
    cols_to_plot = ['age', 'chol', 'thalach', 'oldpeak']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(cols_to_plot):
        sns.boxplot(x='target', y=col, data=df, ax=axes[i], palette='Set2')
        axes[i].set_title(f'{col.capitalize()} vs Target')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'boxplots.png', dpi=150)

def train_and_evaluate(df, output_dir):
    """Modelleri eğitir, test eder ve sonuçları raporlar."""
    print("[*] Model eğitim süreci başlıyor...")
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Veri bölme
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Ölçeklendirme
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Lojistik Regresyon
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    lr_preds = lr_model.predict(X_test_scaled)
    lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    # 2. Karar Ağacı
    dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt_model.fit(X_train, y_train) # Tree-based modeller genellikle ölçeklendirme gerektirmez
    dt_preds = dt_model.predict(X_test)
    dt_probs = dt_model.predict_proba(X_test)[:, 1]
    
    # Model Performans Karşılaştırması Görselleştirme
    metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
    
    lr_metrics = [
        accuracy_score(y_test, lr_preds),
        precision_score(y_test, lr_preds),
        recall_score(y_test, lr_preds),
        f1_score(y_test, lr_preds),
        roc_auc_score(y_test, lr_probs)
    ]
    
    dt_metrics = [
        accuracy_score(y_test, dt_preds),
        precision_score(y_test, dt_preds),
        recall_score(y_test, dt_preds),
        f1_score(y_test, dt_preds),
        roc_auc_score(y_test, dt_probs)
    ]
    
    # Görselleştirme: Model Karşılaştırması
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar plot
    x = np.arange(len(metrics_list))
    width = 0.35
    axes[0].bar(x - width/2, lr_metrics, width, label='LR', color='#3498db', alpha=0.8)
    axes[0].bar(x + width/2, dt_metrics, width, label='DT', color='#e74c3c', alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_list)
    axes[0].set_title('Performans Metrikleri Karşılaştırması')
    axes[0].legend()
    
    # ROC Curves
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
    fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_probs)
    axes[1].plot(fpr_lr, tpr_lr, label=f'LR (AUC={lr_metrics[4]:.3f})', lw=2)
    axes[1].plot(fpr_dt, tpr_dt, label=f'DT (AUC={dt_metrics[4]:.3f})', lw=2)
    axes[1].plot([0, 1], [0, 1], 'k--')
    axes[1].set_title('ROC Eğrileri')
    axes[1].legend()
    
    plt.savefig(output_dir / 'model_evaluation.png', dpi=150)
    
    # Özellik Önemleri (DT)
    plt.figure(figsize=(10, 6))
    feat_importances = pd.Series(dt_model.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh', color='#2ecc71')
    plt.title('Karar Ağacı - En Önemli 10 Özellik')
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png', dpi=150)
    
    # Karar Ağacı Görselleştirmesi
    plt.figure(figsize=(20, 10))
    plot_tree(dt_model, feature_names=X.columns, class_names=['Yok', 'Var'], filled=True, rounded=True)
    plt.title('Karar Ağacı Yapısı')
    plt.savefig(output_dir / 'decision_tree_structure.png', dpi=150)
    
    print("\n--- Model Performans Raporu (Lojistik Regresyon) ---")
    print(classification_report(y_test, lr_preds))
    print("\n--- Model Performans Raporu (Karar Ağacı) ---")
    print(classification_report(y_test, dt_preds))

def main():
    # Çıktı klasörü yönetimi
    output_dir = Path('./output/heart_disease_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Veri hazırlığı
    data = load_and_preprocess_data()
    print(f"[+] Veri seti yüklendi ve temizlendi. Örnek sayısı: {len(data)}")
    
    # Analiz ve Modelleme
    perform_eda(data, output_dir)
    train_and_evaluate(data, output_dir)
    
    print(f"\n[!] Tüm analiz tamamlandı. Görseller '{output_dir}' klasörüne kaydedildi.")

if __name__ == "__main__":
    main()
