import re
import string
from handlers import set_name, handle_greeting, handle_farewell, handle_howareyou, handle_time, handle_weather, handle_addition
from logger import init_db, get_user, save_user, log_message_to_db
from patterns import special_patterns, normal_patterns

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
        
        for pattern, handler_name in self.special_patterns:
            match = pattern.search(original_message)
            if match:
                handler = globals()[handler_name]
                return handler(match, self)
        
        message_without_punct = original_message.translate(str.maketrans('', '', string.punctuation))
        
        for pattern, handler_name in self.normal_patterns:
            match = pattern.search(message_without_punct)
            if match:
                handler = globals()[handler_name]
                return handler(match, self)
        
        return "Я не понимаю запрос."
    
    def run_console_chat(self):
        while True:
            user_input = input("\n").strip()
            
            response = self.process_message(user_input)
            print(f"{response}")
            
            log_message_to_db(user_input, response, self.name)


if __name__ == "__main__":
    init_db()
    bot = ChatBot()
    bot.run_console_chat()