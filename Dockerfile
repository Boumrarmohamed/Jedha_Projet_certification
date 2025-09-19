FROM python:3.9

WORKDIR /code

# Copier les dépendances
COPY ./requirements.txt /code/requirements.txt

# Installer les dépendances
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copier tout le contenu de ton dashboard
COPY ./Streamlit_dashboard /code/

# Commande pour lancer Streamlit
CMD ["streamlit", "run", "Dashboard.py", "--server.port=7860", "--server.address=0.0.0.0"]
