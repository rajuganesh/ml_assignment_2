from sklearn.neighbors import KNeighborsClassifier 
from evaluate import fetch_metrics


def get_knn_classifier_model(X_train_scaled, y_train):
    model=KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_scaled, y_train)
    return model


