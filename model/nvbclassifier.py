from sklearn.naive_bayes import GaussianNB
from evaluate import fetch_metrics


def get_naive_bayes_classifier_model(X_train_scaled, y_train):
    model = GaussianNB()
    model.fit(X_train_scaled, y_train)
    return model

