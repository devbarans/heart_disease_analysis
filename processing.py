from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_data(df, target_col='target', test_size=0.20, random_state=42):
    """
    Veriyi ölçeklendirir ve eğitim/test setlerine böler (stratified).
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Veriyi eğitim ve test olarak ayırıyoruz
    # stratify=y parametresi sınıfların dağılımının korunmasını sağlar.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Lojistik regresyon ve bazı diğer algoritmalar için ölçeklendirme kritik öneme sahiptir.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Veriyi ve ölçeklendiriciyi (scaler) geri döndürüyoruz (yeniden kullanabilmek için)
    return {
        'X_train_orig': X_train,
        'X_test_orig': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler
    }
