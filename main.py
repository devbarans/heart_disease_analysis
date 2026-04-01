import os
from data_loader import load_heart_disease_data
from processing import prepare_data
from models import train_logistic_regression, train_decision_tree, perform_cross_validation
from evaluation import (
    print_performance_summary, 
    plot_confusion_matrices, 
    plot_roc_curves, 
    plot_decision_tree_structure
)

def main():
    # --- PROJE AYARLARI ---
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Çıktı klasörü oluşturuldu: {OUTPUT_DIR}")

    # 1. Veri Yükleme
    df = load_heart_disease_data()
    if df is None:
        return

    # 2. Veri Ön İşleme
    # Stratified split ve StandardScaler uygulaması
    prepared_data = prepare_data(df)
    
    X_train_scaled = prepared_data['X_train_scaled']
    X_test_scaled = prepared_data['X_test_scaled']
    X_train_orig = prepared_data['X_train_orig']
    X_test_orig = prepared_data['X_test_orig']
    y_train = prepared_data['y_train']
    y_test = prepared_data['y_test']
    scaler = prepared_data['scaler']

    # 3. Model Eğitimi
    # Lojistik Regresyon (Ölçeklenmiş veri ile)
    lr_model = train_logistic_regression(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]

    # Karar Ağacı (Ham veri ile - Karar ağaçları ölçeklendirmeye duyarlı değildir)
    dt_model = train_decision_tree(X_train_orig, y_train)
    dt_pred = dt_model.predict(X_test_orig)
    dt_prob = dt_model.predict_proba(X_test_orig)[:, 1]

    # 4. Çapraz Doğrulama (Hoca istediği için Lojistik Regresyon üzerinden yapıyoruz)
    # Tüm özellikleri ölçeklendirip çapraz doğrulama yapalım
    X_scaled = scaler.fit_transform(df.drop('target', axis=1))
    cv_sonuclar = perform_cross_validation(lr_model, X_scaled, df['target'])

    # 5. Görselleştirme ve Değerlendirme
    
    # Performans Özetlerini Yazdırıyoruz
    print_performance_summary("Lojistik Regresyon", y_test, lr_pred, lr_prob, cv_sonuclar)
    print_performance_summary("Karar Ağacı", y_test, dt_pred, dt_prob)

    # Grafik Çizimleri ve Kaydı
    # Karışıklık Matrisleri
    plot_confusion_matrices([
        ('Lojistik Regresyon', y_test, lr_pred),
        ('Karar Ağacı', y_test, dt_pred)
    ], output_path=OUTPUT_DIR)

    # ROC Eğrileri
    plot_roc_curves([
        (y_test, lr_prob, 'Lojistik Regresyon', 'blue'),
        (y_test, dt_prob, 'Karar Ağacı', 'red')
    ], output_path=OUTPUT_DIR)

    # Karar Ağacı Yapısı
    plot_decision_tree_structure(
        dt_model, 
        feature_names=X_train_orig.columns, 
        output_path=OUTPUT_DIR
    )

if __name__ == "__main__":
    main()