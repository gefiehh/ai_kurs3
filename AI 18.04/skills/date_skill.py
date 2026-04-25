from datetime import datetime

def get_date() -> str:
    """Возвращает текущую дату в формате ДД.ММ.ГГГГ"""
    return datetime.now().strftime("%d.%m.%Y")