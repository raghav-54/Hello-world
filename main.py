#!/usr/bin/env python3
"""
Credit Card Fraud Detection - Main Script
Simple version for beginners
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data():
    """Load and preprocess the credit card dataset"""
    try:
        # Load data
        data = pd.read_csv('data/creditcard.csv')
        print(f"Dataset loaded: {data.shape}")
        
        # Check for missing values
        print(f"Missing values: {data.isnull().sum().sum()}")
        
        # Separate features and target
        X = data.drop('Class', axis=1)
        y = data['Class']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        return X_scaled, y, scaler
        
    except FileNotFoundError:
        print("❌ Dataset not found! Please download from Kaggle and place in data/creditcard.csv")
        return None, None, None

def train_models(X_train, y_train):
    """Train different ML models"""
    models = {}
    
    print("🤖 Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    
    print("🤖 Training Logistic Regression...")
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    models['Logistic Regression'] = lr
    
    return models

def evaluate_models(models, X_test, y_test):
    """Evaluate all models"""
    results = {}
    
    for name, model in models.items():
        print(f"\n📊 Evaluating {name}...")
        
        # Predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store results
        results[name] = {
            'accuracy': accuracy,
            'model': model,
            'predictions': y_pred
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()
    
    return results

def main():
    print("🔍 Credit Card Fraud Detection System")
    print("=" * 50)
    
    # Load and preprocess data
    X, y, scaler = load_and_preprocess_data()
    
    if X is None:
        return
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Fraud ratio: {y.sum()/len(y):.4f}")
    
    # Balance the training data
    print("\n⚖️ Balancing dataset with SMOTE...")
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
    print(f"Balanced training set: {X_balanced.shape}")
    
    # Train models
    print("\n🚀 Training models...")
    models = train_models(X_balanced, y_balanced)
    
    # Evaluate models
    results = evaluate_models(models, X_test, y_test)
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_model = results[best_model_name]['model']
    
    print(f"\n🏆 Best model: {best_model_name}")
    print(f"Best accuracy: {results[best_model_name]['accuracy']:.4f}")
    
    # Save the best model
    joblib.dump(best_model, 'fraud_detection_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("\n💾 Models saved!")
    
    # Demo prediction
    print("\n🎯 Testing with sample transaction...")
    sample = X_test.iloc[0:1]
    prediction = best_model.predict(sample)[0]
    probability = best_model.predict_proba(sample)[0][1]
    
    print(f"Prediction: {'FRAUD' if prediction == 1 else 'NORMAL'}")
    print(f"Fraud probability: {probability:.4f}")

if __name__ == "__main__":
    main()
