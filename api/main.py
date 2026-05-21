from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("models/model.pkl")

@app.get("/")
def home():
    return {"message": "API Running"}

@app.post("/predict")
def predict(data: list):

    array = np.array(data).reshape(1, -1)

    prediction = model.predict(array)

    return {"prediction": int(prediction[0])}