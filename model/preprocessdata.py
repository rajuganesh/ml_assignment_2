import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

def preprocess_and_scale_data():
    # 1. Load the split datasets
    train_df = pd.read_csv('./data/Train.csv')
    test_df = pd.read_csv('./data/Test.csv')
    #eval_df = pd.read_csv('Evaluation.csv')

    # Target variable name for the UCI Default dataset
    target = 'default.payment.next.month'

    # 2. Separate Features and Target
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    #X_eval = eval_df.drop(columns=[target])
    #y_eval = eval_df[target]

    # 3. Initialize and Fit Scaler ONLY on Training Data
    # This ensures no information from test/eval sets leaks into the training process
    scaler = StandardScaler()
    scaler.fit(X_train)

    # 4. Transform all datasets using the parameters from X_train
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    #X_eval_scaled = scaler.transform(X_eval)

    # 5. Save the scaler to be used later in your Streamlit App
    # Per Step 3 of the assignment, this should go into your /model/ folder
    joblib.dump(scaler, 'model/pkl/scaler.pkl')
    
    print("Scaling complete. Scaler saved to 'model/scaler.pkl'")
    return X_train,X_test,X_train_scaled, X_test_scaled, y_train, y_test,scaler

# --- Method for Real-time Evaluation (for app.py) ---
def scale_test_data(test_df_value):
    """
    Method to scale evaluation.csv exactly like Train.csv 
    for the Streamlit upload feature.
    """
    # Load the saved scaler
    loaded_scaler = joblib.load('model/pkl/scaler.pkl')
    
    # Load new evaluation data
    #new_data = pd.read_csv(input_csv_path)
    
    # If target exists in evaluation file, drop it for scaling
    target = 'default.payment.next.month'
    #if target in new_data.columns:
    y_test=test_df_value[target]
    x_test = test_df_value.drop(columns=[target])
        
    # Scale data
    X_test_scaled = loaded_scaler.transform(x_test)
    return X_test_scaled,y_test