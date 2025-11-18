from fastapi import FastAPI
import joblib
import os
from pydantic import BaseModel
import pandas as pd

class ConcreteInput(BaseModel):
    """Schéma des données d'entrée"""
    cement: float
    blast_furnace_slag: float
    fly_ash: float
    water: float
    superplasticizer: float
    coarse_aggregate: float
    fine_aggregate: float
    age: int
    
    class Config:
        schema_extra = {
            "example": {
                "cement": 540.0,
                "blast_furnace_slag": 0.0,
                "fly_ash": 0.0,
                "water": 162.0,
                "superplasticizer": 2.5,
                "coarse_aggregate": 1040.0,
                "fine_aggregate": 676.0,
                "age": 28
            }
        }

# Créer l'application FastAPI
app = FastAPI(title="Concrete Strength API")

# Variables globales pour le modèle et le scaler
model = None
scaler = None

@app.on_event("startup")
def load_model():
    """Charge le modèle et le scaler au démarrage de l'API"""
    global model, scaler

    # Trouver la dernière version du modèle
    model_dir = "models/v20251118_094354"  # Exemple de version, à adapter selon le contexte
    model_path = os.path.join(model_dir, "model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    # charger le modèle et le scaler
    print("Chargement du modèle et du scaler...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Modèle et scaler chargés avec succès.")

@app.get("/")
def home():
    """Page d'accueil"""
    return {
        "message": "API de prédiction de résistance du béton",
        "version" : "1.0.0",
        "endpoints": {
            "/" : "Page d'accueil",
            "/predict" : "Faire une prédiction de résistance du béton",
            "/health" : "Vérifier que l'API fonctionne"
        }}

@app.get("/health")
def health():
    """Vérifier que l'API fonctionne"""
    model_loaded = model is not None and scaler is not None
    status = "ok" if model_loaded else "error"
    return {"status": status, "model_loaded": model_loaded}

@app.post("/predict")
def predict(data: ConcreteInput):
    """Prédire la résistance du béton"""
    
    # IMPORTANT : Respecter l'ordre des colonnes de l'entraînement
    input_df = pd.DataFrame([{
        "cement": data.cement,
        "blast_furnace_slag": data.blast_furnace_slag,
        "fly_ash": data.fly_ash,
        "water": data.water,
        "superplasticizer": data.superplasticizer,
        "coarse_aggregate": data.coarse_aggregate,
        "fine_aggregate": data.fine_aggregate,
        "age": data.age
    }])
    
    # Afficher pour debug
    print("Colonnes API:", input_df.columns.tolist())
    print("Valeurs:", input_df.values[0])
    
    # Normaliser
    input_scaled = scaler.transform(input_df)
    print("Valeurs normalisées:", input_scaled[0])
    
    # Prédire
    prediction = model.predict(input_scaled)[0]
    
    return {
        "predicted_strength_mpa": round(prediction, 2),
        "input": data.dict()
    }