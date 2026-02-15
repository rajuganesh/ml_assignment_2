from sklearn.ensemble import RandomForestClassifier
from evaluate import fetch_metrics


def get_random_forest_model(X_train_scaled, y_train):
    model= RandomForestClassifier(n_estimators=100,random_state=42)
    model.fit(X_train_scaled, y_train)
    return model

