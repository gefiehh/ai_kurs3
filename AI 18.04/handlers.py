from datetime import datetime
from weather_api import get_weather
from logger import save_user

def set_name(match, bot):
    bot.name = match.group(1)
    save_user(bot.user_id, bot.name)
    return f"Приятно познакомиться, {bot.name}! Я запомнил ваше имя."

def handle_greeting(match, bot):
    if bot.name:
        return f"Здравствуйте, {bot.name}! Чем могу помочь?"
    return "Здравствуйте! Чем могу помочь?"

def handle_farewell(match, bot):
    if bot.name:
        return f"До свидания, {bot.name}!"
    return "До свидания!"

def handle_howareyou(match, bot):
    if bot.name:
        return f"Все хорошо, {bot.name}! Чем могу помочь?"
    return "Все хорошо!"

def handle_time(match, bot):
    now = datetime.now()
    time_str = now.strftime("%H:%M")
    date_str = now.strftime("%d.%m.%Y")
    
    if bot.name:
        return f"{bot.name}, сейчас {time_str} ({date_str})"
    return f"Сейчас {time_str} ({date_str})"

def handle_weather(match, bot):
    city = match.group(1).strip()
    return get_weather(city)

def handle_addition(match, bot):
    a = float(match.group(1))
    b = float(match.group(2))
    return f"Результат: {a + b}"