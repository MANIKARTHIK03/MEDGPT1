import sqlite3
import os

DB_PATH = "chat_memory.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS chats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user TEXT,
        message TEXT
    )""")
    conn.commit()
    conn.close()

def save_message(user, message):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO chats (user, message) VALUES (?, ?)", (user, message))
    conn.commit()
    conn.close()

def load_history():
    if not os.path.exists(DB_PATH):
        return []
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT user, message FROM chats ORDER BY id")
    data = c.fetchall()
    conn.close()
    return data
