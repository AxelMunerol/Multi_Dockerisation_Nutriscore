# Utiliser une image Python officielle
FROM python:3.10-slim

# Définir le répertoire de travail (en utilisant le format Linux)
WORKDIR /model

# Copier les fichiers nécessaires
COPY requirements.txt .
COPY model.pkl .

# Installer les dépendances depuis le fichier requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Commande à exécuter (optionnel, ici vous pouvez laisser une commande pour garder le conteneur actif)
CMD tail -f /dev/null
