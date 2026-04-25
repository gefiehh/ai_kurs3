import spacy
import joblib
import numpy as np

nlp = spacy.load("ru_core_news_md")
embedding_model = joblib.load("embedding_model.pkl")

def sentence_vector(text):
    doc = nlp(text.lower())
    return doc.vector

def predict_intent(text):
    vector = sentence_vector(text)
    vector = vector.reshape(1, -1)          
    intent = embedding_model.predict(vector)[0]
    proba = embedding_model.predict_proba(vector)[0]
    confidence = np.max(proba)
    return intent, confidence