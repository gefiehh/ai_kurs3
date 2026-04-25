import torch
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from handlers import set_name, handle_greeting, handle_farewell, handle_howareyou, handle_addition
from logger import init_db, get_user, save_user, log_message_to_db
from skills.time_skill import get_time
from skills.date_skill import get_date
from skills.help_skill import get_help
from skills.smalltalk_skill import get_smalltalk_response
from skills.thanks_skill import get_thanks_response
from skills.weather_skill import handle_weather
from weather_api import get_weather
from nlp_processor import extract_city
from voice_service import VoiceService  # Импорт голосового сервиса

MODEL_PATH = "./intent_bert_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

intent_list = ["greeting", "farewell", "howareyou", "time", "date", "set_name", "addition", "weather", "help", "smalltalk", "thanks", "unknown"]
id2label = {i: label for i, label in enumerate(intent_list)}

nlp = spacy.load("ru_core_news_sm")

class DialogState:
    START = "start"
    WAIT_CITY = "wait_city"

user_states = {}
last_intent = {}

def get_state(user_id):
    return user_states.get(user_id, DialogState.START)

def set_state(user_id, state):
    user_states[user_id] = state

def clear_context(user_id):
    user_states.pop(user_id, None)
    last_intent.pop(user_id, None)

def predict_intent(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    confidence = torch.softmax(outputs.logits, dim=1)[0][predicted_class].item()
    return id2label[predicted_class], confidence

class ChatBot:
    def __init__(self):
        self.name = None
        self.user_id = 1
        self.load_user()
        
        # Инициализация голосового сервиса (VITS)
        print("Загрузка голосового движка VITS...")
        self.voice = VoiceService(language="ru")
        
        if self.voice.is_enabled():
            print("Голосовой движок VITS активен")
        else:
            print("Голосовой движок недоступен, бот будет работать только с текстом")
    
    def load_user(self):
        saved_name = get_user(self.user_id)
        if saved_name:
            self.name = saved_name
    
    def process_message(self, message: str):
        original = message.strip()
        user_id = self.user_id

        if not original:
            return "Напиши что-нибудь :)"

        state = get_state(user_id)

        if state == DialogState.WAIT_CITY:
            weather = get_weather(original)
            if self.name:
                weather = f"{self.name}, {weather}"
            clear_context(user_id)
            # Озвучиваем ответ
            self.voice.speak(weather)
            return weather

        intent, confidence = predict_intent(original)

        if confidence < 0.35:
            response = self._rule_based_fallback(original)
            # Озвучиваем ответ
            self.voice.speak(response)
            return response

        # Маршрутизация по интентам
        if intent == "greeting":
            response = handle_greeting(None, self)
        elif intent == "farewell":
            response = handle_farewell(None, self)
        elif intent == "howareyou":
            response = handle_howareyou(None, self)
        elif intent == "time":
            response = f"Сейчас {get_time()}"
        elif intent == "date":
            response = f"Сегодня {get_date()}"
        elif intent == "help":
            response = get_help(self.name)
        elif intent == "smalltalk":
            response = get_smalltalk_response(original)
        elif intent == "thanks":
            response = get_thanks_response()
        elif intent == "set_name":
            response = "Скажи «меня зовут Имя», я запомню"
        elif intent == "addition":
            response = handle_addition(None, self) if "+" in original else "Напиши пример: 15 + 7"
        elif intent == "weather":
            result = handle_weather(original, self.name)
            if result:
                response = result
            else:
                set_state(user_id, DialogState.WAIT_CITY)
                response = "В каком городе показать погоду?"
        else:
            response = self._rule_based_fallback(original)
        
        # Озвучиваем ответ асинхронно
        self.voice.speak(response)
        return response
    
    def _rule_based_fallback(self, text):
        text_lower = text.lower()
        if any(word in text_lower for word in ["привет", "здравствуй", "хай", "ку"]):
            return handle_greeting(None, self)
        if any(word in text_lower for word in ["пока", "до свидания"]):
            return handle_farewell(None, self)
        if any(word in text_lower for word in ["дела", "как ты", "как сам", "настроение"]):
            return handle_howareyou(None, self)
        return "Я не понял запрос. Попробуй ещё раз!"
    
    def run_console_chat(self):
        while True:
            user_input = input("Вы: ").strip()
            if user_input.lower() in ["выход", "quit", "exit"]:
                response = "До свидания!"
                print(f"Бот: {response}")
                self.voice.speak(response)
                break
            
            response = self.process_message(user_input)
            print(f"Бот: {response}")
            log_message_to_db(user_input, response, self.name)


if __name__ == "__main__":
    init_db()
    bot = ChatBot()
    bot.run_console_chat()