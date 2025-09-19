import pandas as pd
import joblib
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Union

# Initialisation FastAPI
app = FastAPI(
    title="GetAround Pricing API",
    description="API de prédiction de prix pour GetAround",
    version="1.0"
)

# Modèle de données pour input
class PredictionInput(BaseModel):
    input: List[List[Union[float, str]]]

# Chargement du modèle ML
try:
    model = joblib.load("getaround_pricing_model.pkl")
    print("Modèle chargé avec succès")
except Exception as e:
    print(f"Erreur lors du chargement du modèle: {e}")
    model = None

# Colonnes exactes attendues par le modèle
model_columns = [
    'model_key', 'mileage', 'engine_power', 'fuel', 'paint_color',
    'car_type', 'private_parking_available', 'has_gps',
    'has_air_conditioning', 'automatic_car', 'has_getaround_connect',
    'winter_tires', 'has_speed_regulator'
]

# Colonnes catégorielles à forcer en string
categorical_columns = [
    'model_key', 'fuel', 'paint_color', 'car_type',
    'private_parking_available', 'has_gps', 'has_air_conditioning',
    'automatic_car', 'has_getaround_connect', 'winter_tires', 'has_speed_regulator'
]

# Route racine
@app.get("/")
def root():
    return {"message": "GetAround Pricing API"}

# Endpoint de prédiction
@app.post("/predict")
def predict(data: PredictionInput):
    if model is None:
        return {"error": "Modèle non disponible"}

    try:
        input_data = []
        for row in data.input:
            # Compléter automatiquement les colonnes manquantes
            if len(row) < len(model_columns):
                row = row + [0] * (len(model_columns) - len(row))
            input_data.append(row)
        
        # Créer DataFrame
        df = pd.DataFrame(input_data, columns=model_columns)

        # Convertir les colonnes catégorielles en str
        for col in categorical_columns:
            df[col] = df[col].astype(str)

        # Prédiction
        predictions = model.predict(df)
        predictions_list = [round(float(p), 2) for p in predictions]

        return {"prediction": predictions_list}

    except Exception as e:
        return {"error": str(e)}

# Documentation HTML
@app.get("/docs", response_class=HTMLResponse)
def documentation():
    return """
    <h1>GetAround API Documentation</h1>
    <h2>GET /</h2>
    <p>Page d'accueil de l'API</p>
    <h2>POST /predict</h2>
    <p>Input JSON:</p>
    <code>{"input": [["Citroën", 0.27, 0.36, "diesel", "red", "SUV", 1, 1, 1, 1, 1, 0, 1]]}</code>
    <p>Output JSON:</p>
    <code>{"prediction": [123.45]}</code>
    <h3>Exemple curl Windows PowerShell:</h3>
    <code>curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{\"input\": [[\"Citroën\", 0.27, 0.36, \"diesel\", \"red\", \"SUV\", 1, 1, 1, 1, 1, 0, 1]]}'</code>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
