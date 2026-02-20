import tkinter as tk
from tkinter import scrolledtext, simpledialog
import datetime
import hashlib
import sqlite3
from llama_cpp import Llama
from offline_agent_cli import OfflineAgent

agent = OfflineAgent()


# —————————————————————————————
# Локальная установка модели
# —————————————————————————————
MODEL_PATH = "models/llama-2-7b-chat.Q4_K_M.gguf"
llm = Llama(model_path=MODEL_PATH)

# —————————————————————————————
# Инициализация базы SQLite
# —————————————————————————————
conn = sqlite3.connect("offline_agent.db")
cursor = conn.cursor()


def prompt_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_cached_response(session_key, phash):
    row = cursor.execute(
        "SELECT response FROM caches WHERE session_key=? AND prompt_hash=?",
        (session_key, phash)
    ).fetchone()
    return row[0] if row else None


def cache_response(session_key, phash, bot_text):
    cursor.execute(
        "INSERT OR IGNORE INTO caches(session_key, prompt_hash, response) VALUES(?,?,?)",
        (session_key, phash, bot_text)
    )
    conn.commit()


def log_dialogue(session_key, user_text, bot_text, rating):
    cursor.execute(
        "INSERT INTO dialogues(session_key, user_text, bot_text, rating) VALUES(?,?,?,?)",
        (session_key, user_text, bot_text, rating),
    )
    conn.commit()


# —————————————————————————————
# Основное приложение GUI
# —————————————————————————————

class ChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Offline LLM Chat")
        self.session_key = "default"

        # виджет истории переписки
        self.history_txt = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=20)
        self.history_txt.pack(padx=10, pady=10)

        # ввод пользователя
        self.entry_txt = tk.Entry(root, width=70)
        self.entry_txt.pack(side=tk.LEFT, padx=10, pady=5)
        self.entry_txt.bind("<Return>", self.send_message)

        # кнопка отправки
        self.send_button = tk.Button(root, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.LEFT, padx=5)

    def send_message(self, event=None):
        user_text = self.entry_txt.get().strip()
        if not user_text:
            return

        # выводим в историю
        self.history_txt.insert(tk.END, f"You: {user_text}\n")
        self.entry_txt.delete(0, tk.END)

        # готовим prompt
        history = self.history_txt.get("1.0", tk.END)
        full_prompt = history + f"User: {user_text}"
        ph = prompt_hash(full_prompt)

        # ищем в кеше
        cached = get_cached_response(self.session_key, ph)
        if cached:
            bot_response = cached
        else:
            # генерируем ответ с помощью локальной модели
            out = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=256,
            )
            bot_response = out["choices"][0]["message"]["content"].strip()
            cache_response(self.session_key, ph, bot_response)

        self.history_txt.insert(tk.END, f"Bot: {bot_response}\n")

        # запрашиваем рейтинг
        self.ask_rating(user_text, bot_response)

    def ask_rating(self, user_text, bot_response):
        # простое диалоговое окно оценок
        rating = simpledialog.askinteger(
            "Rate Response", "Оцените ответ (0–5):", minvalue=0, maxvalue=5
        )
        log_dialogue(self.session_key, user_text, bot_response, rating)
        self.history_txt.insert(tk.END, f"(Rated: {rating})\n\n")
        self.history_txt.see(tk.END)


if __name__ == "__main__":
    root = tk.Tk()
    app = ChatApp(root)
    root.mainloop()
    conn.close()
