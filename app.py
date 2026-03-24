
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="AG News Classifier", version="1.0")

vectorizer = joblib.load("tfidf_vectorizer.pkl")
classifier = joblib.load("nb_classifier.pkl")

LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

class ArticleInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    predicted_class: int
    label: str
    confidence: float

@app.get("/")
def root():
    return {"message": "AG News Classifier API. POST to /predict with a JSON body: {text: '...'}"}

@app.post("/predict", response_model=PredictionOutput)
def predict(article: ArticleInput):
    vec = vectorizer.transform([article.text])
    pred = classifier.predict(vec)[0]
    proba = classifier.predict_proba(vec)[0]
    confidence = round(float(np.max(proba)), 4)
    return {
        "predicted_class": int(pred),
        "label": LABELS[int(pred)],
        "confidence": confidence
    }
