# credit_card_fraud_detection
Fraud detection is one of the oldest and most commercially critical applications of machine learning. Visa and Mastercard process over 700 million transactions per day. Even a fraud rate of 0.1% translates to hundreds of thousands of fraudulent transactions — each carrying real financial loss
# Credit Card Fraud Detection Using Machine Learning

## Overview

This project focuses on detecting fraudulent credit card transactions using Machine Learning techniques. Due to the highly imbalanced nature of fraud datasets, traditional accuracy metrics can be misleading. Therefore, this project evaluates models using precision, recall, F1-score, confusion matrices, Precision-Recall curves, and threshold tuning.

## Objectives

* Explore and preprocess transaction data.
* Analyze class imbalance between legitimate and fraudulent transactions.
* Train and evaluate machine learning models.
* Improve fraud detection using SMOTE oversampling.
* Optimize decision thresholds to maximize fraud recall.
* Compare model performance using business-relevant metrics.

## Dataset

The dataset contains anonymized credit card transactions with:

* Feature columns representing transaction characteristics.
* A target column (`Class`):

  * `0` = Legitimate Transaction
  * `1` = Fraudulent Transaction

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Imbalanced-learn (SMOTE)

## Workflow

### 1. Data Preprocessing

* Loaded dataset using Pandas.
* Checked missing values.
* Removed null records.
* Examined descriptive statistics.
* Visualized feature distributions.

### 2. Exploratory Data Analysis

* Analyzed class distribution.
* Identified severe class imbalance.
* Visualized transaction characteristics using histograms and plots.

### 3. Train-Test Split

The dataset was divided into:

* 80% Training Data
* 20% Testing Data

Stratified sampling was used to preserve class proportions.

### 4. Model Training

The following models were trained and evaluated:

* Decision Tree Classifier
* XGBoost Classifier

Both scaled and unscaled feature versions were tested.

### 5. Model Evaluation

Performance metrics included:

* Accuracy Score
* Confusion Matrix
* Precision
* Recall
* F1-Score
* Classification Report

Special emphasis was placed on Fraud Recall (Class = 1).

### 6. Handling Class Imbalance with SMOTE

SMOTE (Synthetic Minority Oversampling Technique) was applied only to the training data.

Benefits:

* Balances class distribution.
* Generates synthetic fraud examples.
* Helps models learn minority-class patterns more effectively.

The model was retrained using the SMOTE-balanced dataset and evaluated on the untouched test set.

### 7. Precision-Recall Analysis

Fraud probabilities were obtained using:

```python
predict_proba(X_test)[:, 1]
```

A Precision-Recall curve was plotted to study the trade-off between:

* Catching more frauds (Recall)
* Reducing false alarms (Precision)

Average Precision (AP) score was also calculated.

### 8. Threshold Optimization

Instead of using the default classification threshold of 0.5, multiple thresholds were evaluated.

The selected threshold:

* Achieves at least 90% fraud recall.
* Maximizes precision among qualifying thresholds.

Metrics reported:

* Precision
* Recall
* F1-Score
* Chosen Threshold

## Key Findings

* Accuracy alone is insufficient for fraud detection due to class imbalance.
* Fraud Recall is a critical business metric.
* SMOTE generally improves fraud detection performance.
* Precision-Recall analysis provides better insight than accuracy.
* Threshold tuning enables customization based on business objectives.

## Business Impact

### False Negatives (Missed Fraud)

* Financial losses
* Chargebacks
* Customer dissatisfaction
* Compliance risks

### False Positives (Flagged Legitimate Transactions)

* Customer inconvenience
* Lost sales opportunities
* Increased support workload

The final operating threshold should be determined jointly by data science, fraud operations, risk management, product teams, and business stakeholders.

## Future Improvements

* Hyperparameter tuning using GridSearchCV.
* Ensemble learning techniques.
* Cost-sensitive learning approaches.
* Real-time fraud detection pipelines.
* Deep learning models for anomaly detection.

## Conclusion

This project demonstrates a complete fraud detection workflow, from data preprocessing and model training to class imbalance handling and threshold optimization. The results highlight the importance of evaluating fraud detection systems using recall-focused metrics and business-driven decision thresholds rather than relying solely on accuracy.
