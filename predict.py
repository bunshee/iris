import torch
import json

with open('classes.json', 'r') as file:
    iris = json.load(file)



def Predict(model, input_data):
    test_inputs = torch.tensor([[input_data.SepalLengthCm, 
                                 input_data.SepalWidthCm, 
                                 input_data.PetalLengthCm, 
                                 input_data.PetalWidthCm]], 
                                dtype=torch.float32)  

    with torch.no_grad():  
        test_outputs = model(test_inputs)

    _, predicted = torch.max(test_outputs, 1)

    return iris[str(predicted.item())]
