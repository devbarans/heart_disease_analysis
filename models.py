from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

def train_logistic_regression(X_train, y_train):
    """
    Lojistik Regresyon modelini eğitir.
    """
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    return lr

def train_decision_tree(X_train, y_train, max_depth=4, random_state=42):
    """
    Karar Ağacı (Decision Tree) modelini eğitir (overfitting'i engellemek için max_depth ile sınırlandırılmıştır).
    """
    dt = DecisionTreeClassifier(
        max_depth=max_depth, 
        random_state=random_state, 
        criterion='gini'
    )
    dt.fit(X_train, y_train)
    return dt

def perform_cross_validation(model, X, y, cv=5):
    """
    Modelin sağlamlığını 5-katlı çapraz doğrulama ile test eder.
    """
    cv_scores = cross_val_score(model, X, y, cv=cv)
    return cv_scores
