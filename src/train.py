"""
Script d'entraînement du modèle de prédiction de résistance du béton
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn

import joblib
import os
from datetime import datetime

def load_data(data_path):
    """Charge les données nettoyées"""
    print(f"Chargement des données depuis {data_path}")
    df = pd.read_csv(data_path)
    print(f"Données chargées : {df.shape}")
    return df


def prepare_data(df, test_size=0.2, random_state=42):
    """Prépare les données pour l'entraînement"""
    print("\nPréparation des données...")
    
    # Séparer features et target
    X = df.drop('strength', axis=1)
    y = df['strength']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Train: {X_train_scaled.shape[0]} exemples")
    print(f"Test: {X_test_scaled.shape[0]} exemples")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_model(X_train, y_train, n_estimators=100):
    """Entraîne un modèle Random Forest"""
    print(f"\nEntraînement du modèle (n_estimators={n_estimators})...")
    
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    
    print("Entraînement terminé !")
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Évalue le modèle et retourne les métriques"""
    print("\nÉvaluation du modèle...")
    
    # Prédictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Métriques
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    print(f"Train R² : {train_r2:.4f}")
    print(f"Test R²  : {test_r2:.4f}")
    print(f"RMSE     : {test_rmse:.2f} MPa")
    print(f"MAE      : {test_mae:.2f} MPa")
    
    return {
        "train_r2": train_r2,
        "test_r2": test_r2,
        "test_rmse": test_rmse,
        "test_mae": test_mae
    }

if __name__ == "__main__":
    # 1. Charger
    df = load_data("data/processed/concrete_clean.csv")
    
    # 2. Préparer
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    
    # 3. Entraîner
    model = train_model(X_train, y_train, n_estimators=100)
    
    # 4. Évaluer
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    print("\n Pipeline terminé !")


def save_model(model, scaler, metrics):
    """Sauvegarde le modèle localement"""
    version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_dir = f"models/{version}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Sauvegarder
    joblib.dump(model, f"{model_dir}/model.pkl")
    joblib.dump(scaler, f"{model_dir}/scaler.pkl")
    
    # Sauvegarder métriques
    import json
    with open(f"{model_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n Modèle sauvegardé dans {model_dir}/")
    return version


if __name__ == "__main__":
    # Définir l'expérience MLflow
    mlflow.set_experiment("concrete_strength_prediction")
    
    # Démarrer un run MLflow
    with mlflow.start_run():
        
        # 1. Charger
        df = load_data("data/processed/concrete_clean.csv")
        
        # 2. Préparer
        X_train, X_test, y_train, y_test, scaler = prepare_data(df)
        
        # Logger les paramètres
        mlflow.log_param("n_estimators", 120)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("n_samples", len(df))
        
        # 3. Entraîner
        model = train_model(X_train, y_train, n_estimators=120)
        
        # 4. Évaluer
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        
        # Logger les métriques
        mlflow.log_metric("train_r2", metrics["train_r2"])
        mlflow.log_metric("test_r2", metrics["test_r2"])
        mlflow.log_metric("test_rmse", metrics["test_rmse"])
        mlflow.log_metric("test_mae", metrics["test_mae"])
        
        # Logger le modèle
        mlflow.sklearn.log_model(model, "model")
        
        print("\n Pipeline terminé et loggé dans MLflow !")

        # 5. Sauvegarder
        version = save_model(model, scaler, metrics)
        mlflow.log_param("version", version)
