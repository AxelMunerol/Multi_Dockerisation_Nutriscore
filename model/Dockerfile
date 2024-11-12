# Utiliser une image Python officielle
FROM python:3.10-slim

# Définir le répertoire de travail (en utilisant le format Linux)
WORKDIR /model

# Copier les fichiers nécessaires
COPY model.pkl .
COPY scaler.pkl .

# Commande à exécuter (optionnel, ici vous pouvez laisser une commande pour garder le conteneur actif)
