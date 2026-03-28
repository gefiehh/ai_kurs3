import re
import string
from datetime import datetime
import spacy
import joblib
import numpy as np
from handlers import (
    set_name,
    handle_greeting,
    handle_farewell,
    handle_howareyou,
    handle_time,
    handle_addition
)
from logger import init_db, get_user, save_user, log_message_to_db
from patterns import special_patterns
from weather_api import get_weather
from nlp_processor import extract_city

# Загружаем эмбеддинги
nlp = spacy.load("ru_core_news_md")
embedding_model = joblib.load("embedding_model.pkl")

class DialogState:
    START = "start"
    WAIT_CITY = "wait_city"

user_states = {}
user_context = {}

def get_state(user_id):
    return user_states.get(user_id, DialogState.START)

def set_state(user_id, state):
    user_states[user_id] = state

def clear_context(user_id):
    user_context.pop(user_id, None)
    user_states.pop(user_id, None)

def handle_weather_dialog(user_id, text, bot, city_from_pattern=None):
    state = get_state(user_id)
    if state == DialogState.START:
        if city_from_pattern:
            weather = get_weather(city_from_pattern)
            if bot.name:
                weather = f"{bot.name}, {weather}"
            clear_context(user_id)
            return weather
        else:
            set_state(user_id, DialogState.WAIT_CITY)
            return "В каком городе интересует погода?"
    elif state == DialogState.WAIT_CITY:
        city = text.strip()
        weather = get_weather(city)
        if bot.name:
            weather = f"{bot.name}, {weather}"
        clear_context(user_id)
        set_state(user_id, DialogState.START)
        return weather
    return "Что-то пошло не так в диалоге о погоде…"

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

class ChatBot:
    def __init__(self):
        self.name = None
        self.user_id = 1
        self.special_patterns = special_patterns
        self.load_user()

    def load_user(self):
        saved_name = get_user(self.user_id)
        if saved_name:
            self.name = saved_name

    def _old_patterns_fallback(self, original_message):
        message_without_punct = original_message.translate(
            str.maketrans('', '', string.punctuation)
        )
        for pattern, handler_name in self.special_patterns:
            match = pattern.search(original_message)
            if match:
                handler = globals()[handler_name]
                return handler(match, self)
        return None

    def process_message(self, message):
        original = message.strip()
        user_id = self.user_id

        state = get_state(user_id)

        # Приоритет состояний
        if state == DialogState.WAIT_CITY:
            city = original.strip()
            weather = get_weather(city)
            if self.name:
                weather = f"{self.name}, {weather}"
            clear_context(user_id)
            set_state(user_id, DialogState.START)
            return weather

        # Word Embeddings классификация
        intent, conf = predict_intent(original)

        if conf < 0.40: 
            fallback = self._old_patterns_fallback(original)
            if fallback:
                return fallback
            return "Извини, я не очень уверен… Попробуй перефразировать :)"

        if intent == "greeting":
            return handle_greeting(None, self)
        elif intent == "farewell":
            return handle_farewell(None, self)
        elif intent == "howareyou":
            return handle_howareyou(None, self)
        elif intent == "time":
            return handle_time(None, self)
        elif intent == "set_name":
            return "Скажи «меня зовут Имя», я запомню"
        elif intent == "addition":
            for pat, h in self.special_patterns:
                match = pat.search(original)
                if match:
                    return globals()[h](match, self)
            return "Напиши пример сложения, например 15 + 7"
        elif intent == "weather":
            city = extract_city(original)
            if city:
                return handle_weather_dialog(user_id, original, self, city_from_pattern=city)
            else:
                set_state(user_id, DialogState.WAIT_CITY)
                return "В каком городе показать погоду?"
        else:
            fallback = self._old_patterns_fallback(original)
            if fallback:
                return fallback
            return "Я не понял запрос. Попробуй ещё раз!"

    def run_console_chat(self):
        while True:
            user_input = input("\nВы: ").strip()
            if not user_input:
                continue
            response = self.process_message(user_input)
            print(f"Бот: {response}")
            log_message_to_db(user_input, response, self.name)

if __name__ == "__main__":
    init_db()
    bot = ChatBot()
    bot.run_console_chat()