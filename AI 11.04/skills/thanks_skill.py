def get_thanks_response() -> str:
    """Ответ на благодарность"""
    responses = [
        "Пожалуйста! Рад помочь",
        "Не за что!",
        "Всегда пожалуйста!",
        "Обращайся в любое время"
    ]
    import random
    return random.choice(responses)