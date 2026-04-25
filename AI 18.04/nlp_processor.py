import spacy
from weather_api import get_weather

nlp = spacy.load("ru_core_news_sm")

WEATHER_WORDS = [
    "погода",
    "температура",
    "дождь",
    "снег",
    "ветер",
    "жара",
    "холод",
    "холодный",
    "теплый",
    "жаркий"
]


def extract_city(text):
    doc = nlp(text)

    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC"]:
            return " ".join([token.lemma_ for token in ent]).title()

    return None


def is_weather_request(text):
    doc = nlp(text)

    for token in doc:
        if token.lemma_ in WEATHER_WORDS:
            return True

    return False


def process_nlp_query(text):
    doc = nlp(text)

    city = extract_city(text)

    if city and is_weather_request(text):
        return get_weather(city)

    return None