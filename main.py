import torch
from fastapi import FastAPI

from classes import IrisInput
from predict import Predict

# from models import Model

app = FastAPI()

loaded_model = None

# loaded_model = Model()
# loaded_model.load_state_dict(torch.load('./saved_models/iris_model.pth'))

loaded_model = torch.jit.load("./saved_models/traced_model.pth")
loaded_model.eval()


@app.post("/predict")
async def predict(input_data: IrisInput):
    if loaded_model is None:
        return {"error": "Model is not loaded."}

    flower = Predict(loaded_model, input_data)

    return {"predicted_class": flower}
