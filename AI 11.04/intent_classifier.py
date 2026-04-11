import joblib
import spacy

nlp = spacy.load("ru_core_news_sm")
pipeline = joblib.load("intent_pipeline.pkl")  

def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def predict_intent(text):
    processed = preprocess(text)
    intent = pipeline.predict([processed])[0]
    proba = pipeline.predict_proba([processed])[0]
    confidence = max(proba)
    return intent, confidence