from fastapi import FastAPI
from models import Model
from classes import IrisInput
from predict import Predict
import torch

app = FastAPI()

loaded_model = None

loaded_model = Model() 
loaded_model.load_state_dict(torch.load('./iris_model.pth'))
loaded_model.eval() 

@app.post("/predict")
async def predict(input_data: IrisInput):
    if loaded_model is None:
        return {"error": "Model is not loaded."}
    
    flower = Predict(loaded_model, input_data)
    
    return {"predicted_class": flower}
