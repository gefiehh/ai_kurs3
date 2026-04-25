from datetime import datetime

def get_time() -> str:
    """Возвращает текущее время в формате ЧЧ:ММ"""
    return datetime.now().strftime("%H:%M")