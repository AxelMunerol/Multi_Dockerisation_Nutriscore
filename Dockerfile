# Utiliser une image Python officielle
FROM python:3.10-slim

WORKDIR /app

# Copier les fichiers de l'application Flask et le requirements.txt
COPY requirements.txt .
COPY . .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port 5000 pour l'API Flask
EXPOSE 5000

CMD ["python", "run.py"]
