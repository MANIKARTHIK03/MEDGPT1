import os
import io
import easyocr
from PIL import Image, ImageEnhance, ImageFilter
from dotenv import load_dotenv
from openai import OpenAI

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def preprocess_image(img_file):
    """Preprocess image for better OCR accuracy."""
    image = Image.open(img_file).convert("L")  # Convert to grayscale
    image = image.filter(ImageFilter.SHARPEN)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)  # Increase contrast
    return image

def extract_text_from_image(img_file):
    """
    Extract text using EasyOCR, with improved preprocessing.
    Falls back to pytesseract if available (for local use).
    """
    try:
        # Preprocess image
        processed_image = preprocess_image(img_file)
        image_bytes = io.BytesIO()
        processed_image.save(image_bytes, format="PNG")
        image_bytes = image_bytes.getvalue()

        # Try EasyOCR
        reader = easyocr.Reader(['en'], gpu=False)
        results = reader.readtext(image_bytes, detail=0)
        text = " ".join(results).strip()

        # If EasyOCR fails or detects too little text, try Tesseract fallback (only locally)
        if len(text) < 5:
            try:
                import pytesseract
                pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
                text = pytesseract.image_to_string(processed_image).strip()
            except Exception:
                pass

        # Clean and normalize text
        text = " ".join(text.split())

        if not text:
            return "⚠️ Could not detect valid text from the image. Please try again with a clearer photo."

        return text
    except Exception as e:
        return f"⚠️ Error reading image: {e}"

def analyze_medicine_info(text):
    """Use AI to analyze and describe the detected medicine."""
    if not text or text.startswith("⚠️"):
        return "⚠️ Could not detect valid text from the image. Please try again with a better photo."

    prompt = f"""
    You are a medical assistant AI. Based on the detected text from a medicine image,
    identify the medicine and provide:
    - Its full name and category
    - Its uses and dosage form
    - Common side effects
    - Key precautions or warnings (if known)

    Detected text: {text}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error during AI analysis: {e}"
