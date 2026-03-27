#  Kalp Hastalığı Risk Analizi ve Tahminlemesi

Bu proje, UCI Cleveland veri setini kullanarak kalp hastalığı risk faktörlerini analiz etmeyi ve makine öğrenmesi modelleri (Lojistik Regresyon ve Karar Ağaçları) ile hastalık varlığını tahminlemeyi amaçlayan kapsamlı bir veri bilimi çalışmasıdır.

---

##  Proje Genel Bakış

Kalp hastalıkları, dünya genelinde en önemli sağlık sorunlarından biridir. Bu çalışmada, hastaların demografik bilgileri, kan analizleri ve elektrokardiyografi sonuçları kullanılarak bir analiz boru hattı (pipeline) oluşturulmuştur.

###  Öne Çıkan Özellikler
- **Otomatik Veri Temizleme:** Eksik değerlerin (missing values) tespiti ve uygun dönüşümlerin yapılması.
- **Keşifsel Veri Analizi (EDA):** Verinin yapısını anlamak için gelişmiş görselleştirmeler.
- **Model Karşılaştırması:** Lineer (Lojistik Regresyon) ve Non-lineer (Karar Ağacı) modellerin performans analizi.
- **Detaylı Raporlama:** ROC-AUC, F1-Score ve Karmaşıklık Matrisi (Confusion Matrix) üzerinden değerlendirme.

---

##  Veri Seti Hakkında

Proje, UCI Machine Learning Repository'de bulunan **Heart Disease Cleveland** veri setini temel alır.

| Özellik | Açıklama |
| :--- | :--- |
| `age` | Yaş |
| `sex` | Cinsiyet (1: Erkek, 0: Kadın) |
| `cp` | Göğüs ağrısı tipi |
| `trestbps` | Dinlenme kan basıncı |
| `chol` | Serum kolesterol (mg/dl) |
| `fbs` | Açlık kan şekeri > 120 mg/dl (1: Doğru, 0: Yanlış) |
| `thalach` | Elde edilen maksimum kalp hızı |
| `target` | Kalp hastalığı durumu (0: Yok, 1: Var) |

---

##  Keşifsel Veri Analizi (EDA)

Veri setindeki demografik ve klinik değişkenlerin dağılımları ile hedef değişken arasındaki ilişkiler aşağıda görselleştirilmiştir.

````carousel
![EDA İstatistikleri](output/heart_disease_analysis/eda_stats.png)
<!-- slide -->
![Korelasyon Matrisi](output/heart_disease_analysis/correlation_matrix.png)
<!-- slide -->
![Aykırı Değer Analizi](output/heart_disease_analysis/boxplots.png)
````

> [!NOTE]
> **Gözlem:** Korelasyon matrisinde `thalach` (maksimum kalp hızı) ile `target` (hastalık) arasında negatif bir korelasyon gözlemlenirken, `age` ve `oldpeak` özelliklerinin hastalıkla pozitif ilişkili olduğu görülmektedir.

---

##  Modelleme ve Performans

Proje kapsamında iki farklı algoritma eğitilmiş ve test edilmiştir.

### 1. Performans Karşılaştırması
Modellerin Accuracy, Precision, Recall ve F1 skorları ile ROC eğrileri karşılaştırılmıştır.

![Model Değerlendirme](output/heart_disease_analysis/model_evaluation.png)

### 2. Özellik Önemleri
Karar Ağacı modeli tarafından belirlenen, tahminde en etkili olan ilk 10 özellik:

![Özellik Önemi](output/heart_disease_analysis/feature_importance.png)

### 3. Karar Ağacı Yapısı
Modelin karar verme sürecini temsil eden hiyerarşik yapı:

![Karar Ağacı Yapısı](output/heart_disease_analysis/decision_tree_structure.png)

---

## 🛠️ Kurulum ve Çalıştırma

Projeyi yerel bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

1. **Depoyu klonlayın:**
   ```bash
   git clone <repository-url>
   cd heart-disease-analysis
   ```

2. **Gerekli paketleri yükleyin:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Analizi başlatın:**
   ```bash
   python main.py
   ```

---

##  Sonuçlar

Yapılan analizler sonucunda:
- **Lojistik Regresyon**, genel doğruluk (Accuracy) ve ROC-AUC skorlarında Karar Ağacı modeline göre daha stabil bir performans sergilemiştir.
- **Karar Ağacı** modeli, verinin hiyerarşik yapısını anlamak ve hangi kriterlerin (örneğin `thal` veya `cp`) daha kritik olduğunu yorumlamak için değerli içgörüler sunmuştur.
- Klinik bulgular arasında **maksimum kalp hızı** ve **göğüs ağrısı tipi**, kalp hastalığı tahmininde en belirleyici faktörler olarak öne çıkmaktadır.

---

> [!TIP]
> Bu analiz bir temel teşkil etmektedir. Daha gelişmiş sonuçlar için **Random Forest** veya **XGBoost** gibi topluluk (ensemble) öğrenme yöntemleri denenebilir.
