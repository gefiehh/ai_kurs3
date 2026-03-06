import sqlite3
from datetime import datetime

def init_db():
    conn = sqlite3.connect("bot.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            name TEXT
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            user_message TEXT,
            bot_response TEXT,
            user_name TEXT
        )
    """)

    conn.commit()
    conn.close()

def save_user(user_id, name):
    conn = sqlite3.connect("bot.db")
    cursor = conn.cursor()

    cursor.execute("INSERT OR REPLACE INTO users (user_id, name) VALUES (?, ?)",
                   (user_id, name))

    conn.commit()
    conn.close()

def get_user(user_id):
    conn = sqlite3.connect("bot.db")
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM users WHERE user_id=?", (user_id,))
    result = cursor.fetchone()
    conn.close()

    return result[0] if result else None

def log_message_to_db(user_message, bot_response, user_name=None):
    conn = sqlite3.connect("bot.db")
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO messages (timestamp, user_message, bot_response, user_name)
        VALUES (?, ?, ?, ?)
    """, (datetime.now(), user_message, bot_response, user_name))
    
    conn.commit()
    conn.close()