import torch

iris = {0: "Iris Setosa", 1: "Iris Versicolour", 2: "Iris Virginica"}

def Predict(model, input_data):
    test_inputs = torch.tensor([[input_data.SepalLengthCm, 
                                 input_data.SepalWidthCm, 
                                 input_data.PetalLengthCm, 
                                 input_data.PetalWidthCm]], 
                                dtype=torch.float32)  

    with torch.no_grad():  
        test_outputs = model(test_inputs)

    _, predicted = torch.max(test_outputs, 1)

    return iris[predicted.item()]
