import re
import string
from datetime import datetime
import joblib
import spacy
from handlers import (
    set_name,
    handle_greeting,
    handle_farewell,
    handle_howareyou,
    handle_time,
    handle_addition
)
from logger import init_db, get_user, save_user, log_message_to_db
from patterns import special_patterns, normal_patterns
from weather_api import get_weather
from nlp_processor import extract_city

nlp = spacy.load("ru_core_news_sm")
pipeline = joblib.load("intent_pipeline.pkl")

class DialogState:
    START = "start"
    WAIT_CITY = "wait_city"

user_states = {}
user_context = {}

def get_state(user_id):
    return user_states.get(user_id, DialogState.START)

def set_state(user_id, state):
    user_states[user_id] = state

def get_context(user_id, key, default=None):
    return user_context.get(user_id, {}).get(key, default)

def set_context(user_id, key, value):
    if user_id not in user_context:
        user_context[user_id] = {}
    user_context[user_id][key] = value

def clear_context(user_id):
    user_context.pop(user_id, None)
    user_states.pop(user_id, None)

def handle_weather_dialog(user_id, text, bot, city_from_pattern=None):
    state = get_state(user_id)
    text_lower = text.lower().strip()
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

class ChatBot:
    def __init__(self):
        self.name = None
        self.user_id = 1
        self.special_patterns = special_patterns
        self.normal_patterns = normal_patterns
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
        for pattern, handler_name in self.normal_patterns:
            match = pattern.search(message_without_punct)
            if match:
                if handler_name == "handle_weather":
                    city = match.group(1).strip()
                    return handle_weather_dialog(self.user_id, original_message, self, city_from_pattern=city)
                else:
                    handler = globals()[handler_name]
                    return handler(match, self)
        return None

    def process_message(self, message):
        original = message.strip()
        user_id = self.user_id
        text_lower = original.lower()

        state = get_state(user_id)

        # Сначала обрабатываем состояния 
        if state == DialogState.WAIT_CITY:
            city = original.strip()  # просто берём текст как город
            weather = get_weather(city)
            if self.name:
                weather = f"{self.name}, {weather}"
            clear_context(user_id)
            set_state(user_id, DialogState.START)
            return weather

        intent, conf = predict_intent(original)

        if conf < 0.35: 
            fallback = self._old_patterns_fallback(original)
            if fallback:
                return fallback
            return "Извини, я не очень уверен в запросе… Попробуй перефразировать :)"

        # Остальная обработка интентов
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
        elif intent == "unknown":
            fallback = self._old_patterns_fallback(original)
            if fallback:
                return fallback
            return "Я не понял запрос. Попробуй ещё раз!"

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