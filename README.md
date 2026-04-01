# Kalp Hastalığı Risk Analizi ve Tahmini

Bu proje, UCI Cleveland veri setini kullanarak bireylerin kalp hastalığı riskini tahmin etmek amacıyla geliştirilmiştir. Çalışmada iki farklı makine öğrenmesi modeli (Lojistik Regresyon ve Karar Ağaçları) kullanılarak performans karşılaştırması yapılmış ve sonuçlar görselleştirilmiştir.

## Projenin Amacı

Projenin temel amacı, klinik veriler üzerinden kalp hastalığı varlığını %80 ve üzeri doğruluk oranlarıyla tahmin edebilen, sürdürülebilir ve modüler bir analiz boru hattı (pipeline) oluşturmaktır. Kod yapısı gereksiz karmaşıklıktan arındırılarak endüstri standartlarına uygun şekilde modüllere ayrılmıştır.

## Kullanılan Veri Seti

Çalışmada kullanılan veriler, UCI Machine Learning Repository üzerinden sağlanan "Cleveland Heart Disease" veri setidir. Veri seti üzerinde şu işlemler gerçekleştirilmiştir:
- Eksik verilerin (missing values) temizlenmesi.
- Hedef değişkenin (sağlıklı/hasta) ikili sınıflandırma (binary classification) formatına dönüştürülmesi.
- Sayısal özelliklerin standartlaştırılması (StandardScaler).

## Proje Yapısı

Kodun okunabilirliğini ve yönetilebilirliğini artırmak için proje aşağıdaki modüler yapıda kurgulanmıştır:

- **data_loader.py**: Verinin internetten çekilmesi ve temel temizlik işlemlerinin yapılması.
- **processing.py**: Verilerin eğitim ve test seti olarak bölünmesi ve ölçeklendirilmesi.
- **models.py**: Lojistik Regresyon ve Karar Ağacı modellerinin eğitimi ve çapraz doğrulama işlemleri.
- **evaluation.py**: Model performanslarının raporlanması ve grafiklerin (ROC Eğrisi, Karışıklık Matrisi) çizilmesi.
- **main.py**: Tüm sürecin yönetimini sağlayan ana dosya.

## Kurulum ve Çalıştırma

Gerekli kütüphaneleri yüklemek için:

```bash
pip install -r requirements.txt
