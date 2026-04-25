def get_smalltalk_response(text: str) -> str:
    """Простой smalltalk обработчик"""
    text_lower = text.lower()
    
    if "дела" in text_lower or "сам" in text_lower:
        return "Всё отлично, спасибо! А у тебя как дела?"
    elif "настроение" in text_lower:
        return "Настроение прекрасное! А у тебя?"
    elif "как ты" in text_lower:
        return "Я в полном порядке! Рад тебя слышать."
    else:
        return "Хорошо, спасибо! Расскажи, как у тебя сегодня?"