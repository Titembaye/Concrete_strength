# Concrete Strength Prediction - MLOps Project

Projet de prédiction de la résistance à la compression du béton utilisant des techniques de Machine Learning et des pratiques MLOps.

**API en production** : https://concrete-strength-cbj0.onrender.com

---

## Objectif

Créer un système de machine learning qui prédit la **résistance à la compression du béton** (en MPa) à partir de sa composition et de son âge.

### Avantages
- Prédiction immédiate vs 28 jours d'attente pour les tests physiques
- Réduction des tests destructifs coûteux
- Optimisation des formules de béton
- Réduction de l'empreinte carbone

---

## Résultats

| Modèle | R² | RMSE | MAE |
|--------|-----|------|-----|
| Régression Linéaire | 0.628 | 9.8 MPa | 7.7 MPa |
| **Random Forest** | **0.884** | **5.5 MPa** | **3.7 MPa** |

Le modèle Random Forest explique **88.4%** de la variance avec une erreur moyenne de **3.7 MPa**.

---

## Dataset

**Source** : UCI Machine Learning Repository - Concrete Compressive Strength Dataset

- **1030 exemples**
- **8 features** : Cement, Blast Furnace Slag, Fly Ash, Water, Superplasticizer, Coarse Aggregate, Fine Aggregate, Age
- **1 target** : Strength (résistance en MPa)

---

## Installation

### Prérequis
- Python 3.11+
- Docker
- Git

### Étapes

```bash
# 1. Cloner le repository
git clone https://github.com/Titembaye/Concrete_strength.git
cd Concrete_strength

# 2. Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate     # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Récupérer les données et modèles (DVC)
dvc pull
```

---

## Utilisation

### Lancer l'API localement

```bash
uvicorn src.api:app --reload
```

Accès : http://localhost:8000

### Réentraîner le modèle

```bash
python src/train.py
```

---

## Stack MLOps

- **ML** : scikit-learn, pandas, numpy
- **Experiment Tracking** : MLflow
- **Data Versioning** : DVC
- **Monitoring** : Evidently AI, Prometheus, Grafana
- **API** : FastAPI
- **Containerisation** : Docker
- **CI/CD** : GitHub Actions
- **Déploiement** : Render

---

## CI/CD Pipeline

Workflow automatisé avec **GitHub Actions** :
1. Tests automatiques
2. Build de l'image Docker
3. Push sur Docker Hub
4. Déploiement automatique sur Render

---

## Ressources

- [Dataset UCI](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [DVC Documentation](https://dvc.org/doc)

---

## Auteur

**Donald TITEMBAYE** - Projet d'apprentissage MLOps

[GitHub](https://github.com/Titembaye/Concrete_strength) | [API Live](https://concrete-strength-cbj0.onrender.com)
