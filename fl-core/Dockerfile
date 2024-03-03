FROM python:3.11-slim AS base  

WORKDIR /app

# Installation des dépendances 
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip  
RUN pip install -r requirements.txt 

# Copie du code source 
COPY . .

# Définition du point d'entrée
CMD ["python", "main.py"]