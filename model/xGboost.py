
from xgboost import XGBClassifier
from evaluate import fetch_metrics



def get_xgboost_model(X_train_scaled, y_train):
    model=XGBClassifier(eval_metric='logloss',random_state=42)
    model.fit(X_train_scaled, y_train)
    return model

