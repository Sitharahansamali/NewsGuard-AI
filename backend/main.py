from fastapi import FastAPI
from nltk import probability
from schemas import NewsRequest
from database import prediction_collection
import pickle
from datetime import datetime

from url_extractor import extract_news_from_url
from credibility import get_domain
from credibility import check_source_credibility

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
def predict_news(text: str):

    vector = vectorizer.transform([text])

    prediction = model.predict(vector)[0]

    probabilities = model.predict_proba(vector)[0]

    confidence = float(max(probabilities))

    if prediction == 1:
        final_prediction = "Real News"
    else:
        final_prediction = "Fake News"

    return {
        "prediction": final_prediction,
        "confidence": confidence
    }

@app.post("/predict_url")
def predict_url(url: str):

    article = extract_news_from_url(url)

    text = article["text"]

    vector = vectorizer.transform([text])

    prediction = model.predict(vector)[0]

    probabilities = model.predict_proba(vector)[0]

    confidence = float(max(probabilities))

    ml_score = confidence * 100

    domain = get_domain(url)

    credibility = check_source_credibility(domain)

    credibility_score = credibility["score"]

    # HYBRID FINAL SCORE

    final_score = (
        (ml_score * 0.6)
        +
        (credibility_score * 0.4)
    )

    # FINAL DECISION

    if final_score >= 70:
        final_prediction = "Real News"

    elif final_score >= 50:
        final_prediction = "Suspicious"

    else:
        final_prediction = "Fake News"

    return {

        "title": article["title"],

        "source": domain,

        "ml_confidence": round(ml_score, 2),

        "credibility_score": credibility_score,

        "final_score": round(final_score, 2),

        "prediction": final_prediction,

        "credibility_level": credibility["level"]
    }