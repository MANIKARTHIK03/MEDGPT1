from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_prescription(symptoms):
    prompt = f"""
    You are a medical assistant. A patient describes these symptoms:
    "{symptoms}"

    Please:
    1. Suggest the most likely common illness (non-emergency).
    2. Recommend safe, over-the-counter medicines with dosage and duration.
    3. Present everything in a clear prescription-style format.
    4. Include a short note: "Consult a qualified doctor before using any medication."

    Return a clean, readable text.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()
