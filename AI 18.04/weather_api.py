import requests

API_KEY = "04900391277fd6be201642cda037dfba"

def get_weather(city):

    city = city.strip()

    url = "http://api.weatherstack.com/current"

    params = {
        "access_key": API_KEY,
        "query": city,
        "units": "m"
    }

    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
    except requests.RequestException:
        return "Ошибка получения данных о погоде."

    try:
        data = response.json()
        
        if "error" in data:
            return f"Ошибка: {data['error']['info']}"
        
        temperature = data["current"]["temperature"]
        weather_desc = data["current"]["weather_descriptions"][0]
        wind_speed = data["current"]["wind_speed"]
        humidity = data["current"]["humidity"]
        
        return (f"Погода в городе {city}:\n"
                f"Температура: {temperature}°C\n"
                f"Описание: {weather_desc}\n"
                f"Скорость ветра: {wind_speed} м/с\n"
                f"Влажность: {humidity}%")
    except (KeyError, ValueError, IndexError) as e:
        return f"Ошибка обработки данных о погоде: {str(e)}"