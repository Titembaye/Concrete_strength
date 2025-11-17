import joblib
import numpy as np
import pandas as pd

class ConcretePredictor:
    """Classe pour faire des prédictions de résistance du béton"""
    
    def __init__(self, model_path, scaler_path):
        """Charge le modèle et le scaler"""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
    
    def predict(self, data):
        """
        Fait une prédiction
        
        Args:
            data: dict ou DataFrame avec les features
        
        Returns:
            float: résistance prédite en MPa
        """
        # Convertir en DataFrame si c'est un dict
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Normaliser
        data_scaled = self.scaler.transform(data)
        
        # Prédire
        prediction = self.model.predict(data_scaled)
        
        return prediction[0]
    

# Exemple d'utilisation
if __name__ == "__main__":
    # Charger le prédicteur
    predictor = ConcretePredictor(
        model_path="models/v1.0/random_forest_model.pkl",
        scaler_path="models/v1.0/scaler.pkl"
    )
    
    # Exemple de données
    sample = {
        'cement': 540.0,
        'blast_furnace_slag': 0.0,
        'fly_ash': 0.0,
        'water': 162.0,
        'superplasticizer': 2.5,
        'coarse_aggregate': 1040.0,
        'fine_aggregate': 676.0,
        'age': 28
    }
    
    # Prédire
    strength = predictor.predict(sample)
    print(f"Résistance prédite : {strength:.2f} MPa")
