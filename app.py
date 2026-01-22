import os
import pickle
import logging
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Titanic Survival Analytics")

# Templates
templates = Jinja2Templates(directory="templates")

# Load model
MODEL_PATH = 'titanicmodel.pkl'

def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Model loaded successfully from {MODEL_PATH}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    else:
        logger.warning(f"Model file {MODEL_PATH} not found.")
        return None

model = load_model()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction_text": None})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    Pclass: int = Form(...),
    Sex: int = Form(...),
    Age: float = Form(...),
    SibSp: int = Form(...),
    Parch: int = Form(...),
    Fare: float = Form(...)
):
    if model is None:
        logger.error("Prediction attempted but model is not loaded.")
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "prediction_text": "Error: Maritime prediction model is currently offline."}
        )
    
    try:
        # Prepare input data
        input_data = {
            'Pclass': Pclass,
            'Sex': Sex,
            'Age': Age,
            'SibSp': SibSp,
            'Parch': Parch,
            'Fare': Fare
        }
        logger.info(f"Prediction requested with data: {input_data}")
        
        input_df = pd.DataFrame([input_data])
        
        # Execute prediction
        prediction = model.predict(input_df)[0]
        survival_prob = model.predict_proba(input_df)[0][1] * 100
        
        status = 'Survived' if prediction == 1 else 'Did Not Survive'
        prediction_text = f"The analysis suggests the passenger likely {status} with a {survival_prob:.1f}% confidence interval."
        logger.info(f"Analysis result: {status} ({survival_prob:.1f}%)")
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        prediction_text = f"An anomaly occurred during data synthesis: {str(e)}"
    
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "prediction_text": prediction_text}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
