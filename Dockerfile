# Image de base Python
FROM python:3.11-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers requirements
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

# Exposer le port
EXPOSE 8000

# Commande de démarrage
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]