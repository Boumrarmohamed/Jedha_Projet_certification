Projet Getaround - Déploiement sur Hugging Face
📌 Description

Projet réalisé dans le cadre de la certification Data Scientist.
Il comprend :

Un notebook d’analyse exploratoire des données de Getaround.

Un modèle de prédiction du prix journalier des véhicules.

Un dashboard Streamlit interactif pour visualiser les insights.

Une API FastAPI déployée sur Hugging Face avec un endpoint /predict.


🚀 Déploiements en ligne

🌐 Dashboard (Streamlit) : https://huggingface.co/spaces/BOUMRAR/Getaround-delay-dashboard

⚡ API (FastAPI) : (https://BOUMRAR-Getaround-api.hf.space)

Documentation interactive : https://BOUMRAR-Getaround-api.hf.space/docs

💻 Installation locale
1. Clone du projet
git clone https://github.com/BOUMRAR/Jedha_Deployment_Huggingface.git
cd Jedha_Deployment_Huggingface

2. Créer un environnement virtuel
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1

# Mac/Linux
source .venv/bin/activate

3. Installer les dépendances
pip install -r requirements.txt

📊 Lancer le Dashboard Streamlit en local
cd Streamlit_dashboard
streamlit run Dashboard.py

➡️ Le Dashboard sera disponible à l’adresse : http://localhost:8501

⚡ Lancer l’API FastAPI en local
cd API_predict
uvicorn main:app --reload

➡️ L’API sera disponible à l’adresse : http://127.0.0.1:8000

🔮 Exemple d’appel à l’API /predict
Requête curl :
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"input": [["Citroën", 0.27, 0.36, "diesel", "red", "SUV", 1, 1, 1, 1, 1, 0, 1]]}'

Réponse attendue :
{"prediction": [120.02]}

📦 Déploiement avec Docker

Le projet utilise un Dockerfile pour automatiser le déploiement sur Hugging Face Spaces.

Exemple de Dockerfile pour l’API FastAPI :

FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./main.py /code/
COPY ./getaround_pricing_model.pkl /code/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

ℹ️ Important : Hugging Face impose d’écouter sur le port 7860.
Une fois le Dockerfile et le code pushés, Hugging Face build automatiquement l’image et lance l’API ou le dashboard.

👤 Auteur

Projet réalisé par Mohamed BOUMRAR dans le cadre de la certification Data Scientist chez Jedha.


