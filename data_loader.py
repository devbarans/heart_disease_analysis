import pandas as pd
import numpy as np
import warnings

def load_heart_disease_data():
    """
    UCI Cleveland veri setini çeker ve temel temizlik işlemlerini yapar.
    """
    # Gereksiz uyarı mesajlarını kapatalım
    warnings.filterwarnings('ignore')
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    sutunlar = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    try:
        df = pd.read_csv(url, names=sutunlar)
        
        # Verideki "?" işaretlerini NaValue yapıp eliyoruz. 
        df = df.replace('?', np.nan)
        df = df.dropna().copy()
        
        # Sadece ilk 280 temiz gözlemle (istatistiksel tutarlılık için) devam ediyoruz.
        df = df.iloc[:280].copy()
        
        # Hedef değişkeni ikili sınıfa çeviriyoruz: 0=Sağlıklı, 1=Hasta
        df['target'] = df['target'].apply(lambda x: 0 if x == 0 else 1)
        
        print(f"Veri başarıyla yüklendi ve temizlendi. Toplam satır: {len(df)}")
        return df
        
    except Exception as e:
        print(f"Veri yüklenirken hata oluştu: {e}")
        return None
