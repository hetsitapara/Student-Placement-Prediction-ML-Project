import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

def main():
    print("=" * 60)
    print("Starting ML Model Training Pipeline...")
    print("=" * 60)
    
    # 1. Load Dataset
    data_path = 'data/placement.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please run data generation first.")
    
    df = pd.read_csv(data_path)
    print(f"Loaded dataset from {data_path} with {df.shape[0]} rows and {df.shape[1]} columns.")
    
    # 2. Split Features (X) and Target (y)
    X = df.drop(columns=['placed'])
    y = df['placed']
    
    # 3. Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Data split successfully:")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Testing samples: {X_test.shape[0]}")
    
    # 4. Feature Preprocessing (Standard Scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Feature scaling completed.")
    
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Saved Scaler to 'models/scaler.pkl'")
    
    # 5. Define Models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100)
    }
    
    # 6. Train and Evaluate
    results = {}
    best_accuracy = 0
    best_model_name = None
    best_model_obj = None
    
    print("\nTraining and evaluating models:")
    print("-" * 50)
    
    for name, model in models.items():
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        }
        
        print(f"{name}:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print("-" * 50)
        
        # Track best model based on Accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
            best_model_obj = model
            
    # 7. Print Summary
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE COMPARISON SUMMARY")
    print("=" * 60)
    summary_df = pd.DataFrame(results).T
    print(summary_df.to_string())
    print("=" * 60)
    
    # 8. Save Best Model
    print(f"\nBest Model identified: {best_model_name} with Accuracy: {best_accuracy:.4f}")
    model_save_path = 'models/best_model.pkl'
    joblib.dump(best_model_obj, model_save_path)
    print(f"Saved Best Model to '{model_save_path}'")
    print("Pipeline successfully completed!")

if __name__ == '__main__':
    main()
