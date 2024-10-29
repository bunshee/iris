The classification api could be used navigating to "/predict" route, sending sepal annd petal width and length data in body json format, example: data = {
    "SepalLengthCm": 5.1,
    "SepalWidthCm": 3.5,
    "PetalLengthCm": 1.4,
    "PetalWidthCm": 0.2
}

and the prediction will in the form of a string specifying the flower type.



# To install and use this project:
1- Clone/Download Project
2- In the project folder create new virtual environemnt using **python3 -m venv <your-venv-name>**
3- Install requirements using **pip install -r requirements.txt**
4- To execute this project use **uvicorn main:app**
