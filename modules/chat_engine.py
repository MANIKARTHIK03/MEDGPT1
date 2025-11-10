# modules/chat_engine.py

from openai import OpenAI
import os
from dotenv import load_dotenv

# Load your API key from .env or system environment
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ask_medgpt(prompt):
    """
    Ask the MedGPT model a medical question and return the text response.
    Ensures the function always returns a string even if an error occurs.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful and educational medical assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ MedGPT error: {str(e)}"
