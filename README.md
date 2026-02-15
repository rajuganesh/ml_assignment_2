# Credit Card Default Prediction - ML Assignment 2

## a. Problem Statement

The objective of this project is to develop and implement multiple machine learning classification models to predict whether a credit card client will default on their payment next month. By comparing six different algorithms, we identify which model provides the most reliable predictions based on a specific set of evaluation metrics, including MCC and AUC scores.

## b. Dataset Description 

* **Source:** UCI Machine Learning Repository / Kaggle.


* **Instance Size:** 30,000 observations (Meets the requirement of >500).


* **Feature Size:** 24 attributes (Meets the requirement of >12).


* **Target Variable:** `default.payment.next.month` (Binary: 1 = yes, 0 = no).

### Features List

1. **ID:** Client ID.
2. **LIMIT_BAL:** Amount of given credit.
3. **SEX:** Gender (1=male, 2=female).
4. **EDUCATION:** (1=grad school, 2=university, 3=high school, 4=others).
5. **MARRIAGE:** Marital status (1=married, 2=single, 3=others).
6. **AGE:** Age in years.
7. **PAY_0 to PAY_6:** Repayment status from April to September 2005.
8. **BILL_AMT1 to BILL_AMT6:** Bill statement amount from April to September 2005.
9. **PAY_AMT1 to PAY_AMT6:** Previous payment amount from April to September 2005.

## c. Models Used and Comparison Table 

The following six models were implemented and evaluated on the same dataset:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| **Logistic Regression** | 0.81 | 0.71 | 0.68 | 0.24 | 0.35 | 0.32 |
| **Decision Tree** | 0.73 | 0.61 | 0.39 | 0.41 | 0.40 | 0.22 |
| **k-Nearest Neighbor** | 0.79 | 0.70 | 0.54 | 0.35 | 0.42 | 0.31 |
| **Naive Bayes** | 0.75 | 0.73 | 0.45 | 0.57 | 0.50 | 0.34 |
| **Random Forest (Ensemble)** | 0.81 | 0.76 | 0.63 | 0.36 | 0.46 | 0.38 |
| **XGBoost (Ensemble)** | 0.81 | 0.76 | 0.61 | 0.36 | 0.46 | 0.37 |

## d. Performance Observations 

Based on the metrics calculated, the following observations were noted:

| ML Model Name | Observation about model performance |
| --- | --- |
| **Logistic Regression** | High accuracy (81%) but poor recall (24%), indicating it struggles to identify actual defaulters. |
| **Decision Tree** | Lowest AUC (0.61), suggesting it is the least effective at distinguishing between classes for this data. |
| **kNN** | Balanced performance but sensitive to the scale of the credit balance and bill amount features. |
| **Naive Bayes** | Highest Recall (57%), making it the most "cautious" model for detecting defaults despite lower precision. |
| **Random Forest** | Strong overall performer with the highest MCC (0.38), indicating good correlation in predictions. |
| **XGBoost** | Highly efficient with AUC (0.76) tied for highest; provides consistent results comparable to Random Forest. |

## e. Screenshots 

Included in the final submission PDF per guidelines.

1. **BITS Virtual Lab Execution:** 

![Alt text](image.png)

2. **Streamlit App Interface:** 

![Alt text](image2.png)



Final Submission Checklist 

* [X] GitHub Repository Link works.


* [X] Streamlit App Link opens and is interactive.


* [X] `requirements.txt` included with necessary dependencies.


* [X] `model/` folder contains all source code and saved models.


* [X] README.md follows the mandatory structure and is included in the PDF.

---
