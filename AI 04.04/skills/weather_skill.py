from nlp_processor import extract_city
from weather_api import get_weather

def handle_weather(text: str, bot_name: str = None) -> str:
    """Обработка всех weather-запросов"""
    city = extract_city(text)
    
    if city:
        weather_info = get_weather(city)
        return f"{bot_name}, {weather_info}" if bot_name else weather_info
    else:
        # Если город не найден — переходим в режим ожидания
        return None 