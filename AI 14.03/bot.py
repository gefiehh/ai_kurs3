import re
import string
from datetime import datetime

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
from nlp_processor import process_nlp_query
from weather_api import get_weather

class DialogState:
    START     = "start"
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

    def process_message(self, message):
        original_message = message.lower().strip()
        user_id = self.user_id

        # NLP 
        nlp_response = process_nlp_query(original_message)
        if nlp_response:
            return nlp_response

        for pattern, handler_name in self.special_patterns:
            match = pattern.search(original_message)
            if match:
                handler = globals()[handler_name]
                return handler(match, self)

        message_without_punct = original_message.translate(
            str.maketrans('', '', string.punctuation)
        )

        for pattern, handler_name in self.normal_patterns:
            match = pattern.search(message_without_punct)
            if match:
                if handler_name == "handle_weather":
                    city = match.group(1).strip()
                    return handle_weather_dialog(user_id, original_message, self, city_from_pattern=city)
                else:
                    handler = globals()[handler_name]
                    return handler(match, self)

        state = get_state(user_id)
        if state != DialogState.START:
            return handle_weather_dialog(user_id, original_message, self)

        weather_keywords = [
            "погода", "какая погода", "погоду", "температура",
            "сколько градусов", "жарко", "холодно", "дождь", "снег"
        ]
        if any(kw in original_message for kw in weather_keywords):
            return handle_weather_dialog(user_id, original_message, self)

        return "Я не понимаю запрос."

    def run_console_chat(self):
        while True:
            user_input = input("\nВы: ").strip()

            response = self.process_message(user_input)
            print(f"Бот: {response}")

            log_message_to_db(user_input, response, self.name)


if __name__ == "__main__":
    init_db()
    bot = ChatBot()
    bot.run_console_chat()