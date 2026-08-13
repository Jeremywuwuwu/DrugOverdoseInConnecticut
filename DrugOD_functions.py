#Capstone Project Wu ALY 6140: Connecticut Drug Overdose Analysis
#Functions for data cleaning, visualization, and predictive modeling

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)

#Function 1
def clean_fentanyl_column(df):
    df['Fentanyl'] = df['Fentanyl'].replace({
        '1-A': 1,
        '1 POPS': 1,
        '1 (PTCH)': 1
    })
    df['Fentanyl'] = df['Fentanyl'].astype(int)
    return df

#Function 2
def plot_drug_correlation_heatmap(df, drug_cols):
    drug_corr = df[drug_cols].corr()
    mask = np.triu(np.ones_like(drug_corr, dtype=bool), k=1)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(drug_corr, annot=True, fmt='.2f', cmap='coolwarm', 
                square=True, mask=mask)
    plt.title('Drug Co-occurrence Correlation')
    plt.tight_layout()
    return plt.gcf()

#Function 3
def prepare_model_data(df, target_col, feature_cols):
    f = df[target_col].copy()
    feat = df[feature_cols].copy()
    
    categorical_cols = ['Sex', 'Race', 'DeathCounty']
    
    cols_to_encode = [col for col in categorical_cols if col in feat.columns]
    feat = pd.get_dummies(feat, columns=cols_to_encode, drop_first=True)
    
    f_train, f_test, feat_train, feat_test = train_test_split(
        f, feat, test_size=0.2, random_state=42, stratify=f
    )
    
    return f_train, f_test, feat_train, feat_test

#Function 4
def train_random_forest_model(feat_train, f_train):
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=10,
        class_weight='balanced'
    )
    
    rf_model.fit(feat_train, f_train)
    return rf_model

#Function 5
def evaluate_model_performance(model, feat_test, f_test):
    f_pred = model.predict(feat_test)
    f_pred_proba = model.predict_proba(feat_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(f_test, f_pred),
        'precision': precision_score(f_test, f_pred),
        'recall': recall_score(f_test, f_pred),
        'f1_score': f1_score(f_test, f_pred),
        'roc_auc': roc_auc_score(f_test, f_pred_proba),
        'confusion_matrix': confusion_matrix(f_test, f_pred)
    }
    
    return metrics
