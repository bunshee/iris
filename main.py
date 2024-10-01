from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as f

app = FastAPI()

class Model(nn.Module):
    def __init__(self, in_feat=4, h1=12, h2=12, h3=7, out_feat=3):
        super().__init__()
        self.layr1 = nn.Linear(in_feat, h1)
        self.layr2 = nn.Linear(h1, h2)
        self.layr3 = nn.Linear(h2, h3)
        self.outlayr = nn.Linear(h3, out_feat)

    def forward(self, inpts):
        out1 = f.relu(self.layr1(inpts))
        out2 = f.relu(self.layr2(out1))
        out3 = f.relu(self.layr3(out2))
        return self.outlayr(out3)


loaded_model = Model() 
loaded_model.load_state_dict(torch.load('./iris_model.pth'))
loaded_model.eval() 

class IrisInput(BaseModel):
    SepalLengthCm: float
    SepalWidthCm: float
    PetalLengthCm: float
    PetalWidthCm: float

@app.post("/predict")
async def predict(input_data: IrisInput):
    test_inputs = torch.tensor([[input_data.SepalLengthCm, 
                                 input_data.SepalWidthCm, 
                                 input_data.PetalLengthCm, 
                                 input_data.PetalWidthCm]])
    test_outputs = loaded_model(test_inputs)
    _, predicted = torch.max(test_outputs, 1)
    
    return {"predicted_class": predicted.item()}

