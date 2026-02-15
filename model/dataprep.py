import pandas as pd
from sklearn.model_selection import train_test_split
import kagglehub
import os 


os.makedirs("./data/", exist_ok=True)

# 1. Download the dataset from Kaggle
# URL: https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset
path = kagglehub.dataset_download("uciml/default-of-credit-card-clients-dataset")

# Load the file (usually downloaded as a CSV in this specific Kaggle repo)
# Adjust filename if necessary based on the downloaded folder content
df = pd.read_csv(f"{path}/UCI_Credit_Card.csv")

# 2. Basic Cleaning (Mandatory for Assignment 2)
# Drop ID column to ensure clean feature space 
df.drop('ID', axis=1, inplace=True)

# 3. Data Splitting Logic
# First split: Separate Train (65%) from the rest (35%)
train_df, temp_df = train_test_split(
    df, 
    test_size=0.35, 
    random_state=42, 
    stratify=df['default.payment.next.month']
)

# Second split: Divide the remaining 35% into Test (20%) and Evaluation (15%)
# (20/35 ≈ 0.57 of the remaining data)
test_df, eval_df = train_test_split(
    temp_df, 
    test_size=(15/35), 
    random_state=42, 
    stratify=temp_df['default.payment.next.month']
)

# 4. Save to CSV files
train_df.to_csv('./data/Train.csv', index=False)       # 65% for model training [cite: 32]
test_df.to_csv('./data/Test.csv', index=False)         # 20% for model testing/metrics [cite: 40]
eval_df.to_csv('./data/Evaluation.csv', index=False)   # 15% for post-creation upload 

print(f"Data Split Complete:")
print(f"Train.csv: {train_df.shape}")
print(f"Test.csv: {test_df.shape}")
print(f"Evaluation.csv: {eval_df.shape}")