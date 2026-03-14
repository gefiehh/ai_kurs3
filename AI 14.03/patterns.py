import re

special_patterns = [
    (re.compile(r"(\d+)\s*\+\s*(\d+)"), "handle_addition"),
]

normal_patterns = [
    (re.compile(r"меня зовут ([а-яА-Яa-zA-Z]+)", re.IGNORECASE), "set_name"),
    (re.compile(r"^(привет|здравствуй|добрый день)$", re.IGNORECASE), "handle_greeting"),
    (re.compile(r"^(пока|до свидания)$", re.IGNORECASE), "handle_farewell"),
    (re.compile(r"погода в ([а-яА-Яa-zA-Z\- ]+)", re.IGNORECASE), "handle_weather"),
    (re.compile(r"как (у тебя )?дела", re.IGNORECASE), "handle_howareyou"),
    (re.compile(r"(сколько|какое) (сейчас )?время", re.IGNORECASE), "handle_time"),
]