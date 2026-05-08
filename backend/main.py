from fastapi import FastAPI
from schemas import NewsRequest
from database import prediction_collection
import pickle
from datetime import datetime

app = FastAPI()

# Load trained model
model = pickle.load(open("../models/model.pkl", "rb"))

# Load vectorizer
vectorizer = pickle.load(open("../models/vectorizer.pkl", "rb"))


@app.get("/")
def home():
    return {
        "message": "Fake News Detection API Running"
    }


@app.post("/predict")
def predict_news(request: NewsRequest):

    text = request.text

    # Convert text to vector
    vector = vectorizer.transform([text])

    # Predict
    prediction = model.predict(vector)[0]

    # Convert prediction to label
    result = "Real News" if prediction == 1 else "Fake News"

    # Store in MongoDB
    prediction_collection.insert_one({
        "news": text,
        "prediction": result,
        "created_at": datetime.utcnow()
    })

    return {
        "prediction": result
    }