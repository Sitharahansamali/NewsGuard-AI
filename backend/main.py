from fastapi import FastAPI
from schemas import NewsRequest
from database import prediction_collection

from datetime import datetime

import mlflow
import mlflow.sklearn

from url_extractor import extract_news_from_url
from credibility import get_domain
from credibility import check_source_credibility


app = FastAPI()


# ============================================================
# MLflow Configuration
# ============================================================

mlflow.set_tracking_uri("sqlite:///"
"../mlflow.db")

MODEL_NAME = "NewsGuard-Fake-News-Classifier"
MODEL_ALIAS = "champion"

MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"


# ============================================================
# Load Champion Model
# ============================================================

print("Loading MLflow champion model...")

model = mlflow.sklearn.load_model(MODEL_URI)

print("Champion model loaded successfully!")


# ============================================================
# Helper function
# ============================================================

def predict_text(text: str):

    prediction = model.predict([text])[0]

    probabilities = model.predict_proba([text])[0]

    confidence = float(max(probabilities))

    if prediction == 1:
        final_prediction = "Real News"
    else:
        final_prediction = "Fake News"

    return final_prediction, confidence


# ============================================================
# Home
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Fake News Detection API Running",
        "model": MODEL_NAME,
        "model_alias": MODEL_ALIAS
    }


# ============================================================
# Text Prediction
# ============================================================

@app.post("/predict")
def predict_news(text: str):

    final_prediction, confidence = predict_text(text)

    # Save prediction to database
    prediction_collection.insert_one({
        "text": text,
        "prediction": final_prediction,
        "confidence": confidence,
        "timestamp": datetime.utcnow()
    })

    return {
        "prediction": final_prediction,
        "confidence": confidence
    }


# ============================================================
# URL Prediction
# ============================================================

@app.post("/predict_url")
def predict_url(url: str):

    article = extract_news_from_url(url)

    text = article["text"]

    final_prediction_ml, confidence = predict_text(text)

    ml_score = confidence * 100

    domain = get_domain(url)

    credibility = check_source_credibility(domain)

    credibility_score = credibility["score"]

    # ========================================================
    # HYBRID FINAL SCORE
    # ========================================================

    final_score = (
        (ml_score * 0.6)
        +
        (credibility_score * 0.4)
    )

    # ========================================================
    # FINAL DECISION
    # ========================================================

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