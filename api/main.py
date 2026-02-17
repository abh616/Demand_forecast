from fastapi import FastAPI
from pydantic import BaseModel
from src.inference import predict

app = FastAPI()

class ForecastRequest(BaseModel):
    product_name: str
    location: str
    platform: str
    category: str

@app.get("/")
def home():
    return {"message": "Demand Forecast API running"}

@app.post("/predict")
def get_prediction(data: ForecastRequest):
    result = predict(data.dict())
    return {
    "model_version": "rf_v1",
    "predicted_units_sold": result
}

