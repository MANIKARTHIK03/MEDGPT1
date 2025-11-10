import streamlit as st
import pandas as pd
import base64
import tempfile
import os
import time
from gtts import gTTS
from deep_translator import GoogleTranslator
from io import BytesIO
from streamlit_mic_recorder import mic_recorder

# Local modules (MedGPT logic)
from modules.chat_engine import ask_medgpt
from modules.eda import load_data, show_basic_info
from modules.viz import show_visualizations
from modules.prediction import train_predict
from modules.report_generator import generate_report
import warnings
try:
    from pydub import AudioSegment
except (ImportError, ModuleNotFoundError):
    warnings.warn("⚠️ pyaudioop not available, skipping audio optimization")
    AudioSegment = None

import warnings
try:
    import speech_recognition as sr
except (ImportError, ModuleNotFoundError) as e:
    warnings.warn(f"⚠️ SpeechRecognition unavailable ({e}). Voice input disabled.")
    sr = None

if sr is None:
    st.warning("🎤 Voice input is not supported on this environment (Python 3.13).")
else:
    # existing mic_recorder / recognizer logic here



# ---------------- FFmpeg setup ---------------- #
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg-2025-11-06-git-222127418b-full_build\bin"
AudioSegment.converter = r"C:\ffmpeg-2025-11-06-git-222127418b-full_build\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\ffmpeg-2025-11-06-git-222127418b-full_build\bin\ffprobe.exe"


# ---------------- Page Setup ---------------- #
st.set_page_config(page_title="🩺 MedGPT - AI Healthcare Assistant", layout="wide")

# ✅ Background Function (Dark Top → Light Bottom)
def add_background_image(image_path):
    abs_path = os.path.abspath(image_path)
    if not os.path.exists(abs_path):
        st.error(f"❌ Background image not found: {abs_path}")
        return

    with open(abs_path, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()

    st.markdown(
        f"""
        <style>
        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
            background: none !important;
        }}

        /* Background with dark-to-light fade */
        html::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background:
                linear-gradient(to bottom, rgba(0, 0, 0, 0.4) 0%, rgba(255, 255, 255, 0.6) 60%, rgba(255, 255, 255, 0.9) 100%),
                url("data:image/png;base64,{encoded}") center center / cover no-repeat fixed;
            filter: blur(7px) brightness(0.9);
            z-index: -1;
        }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #CAF0F8 0%, #ADE8F4 100%) !important;
            backdrop-filter: blur(6px);
        }}

        .stMarkdown, .stTextInput, .stSelectbox, .stChatInput, .stDataFrame {{
            background: rgba(255, 255, 255, 0.85);
            border-radius: 10px;
            backdrop-filter: blur(8px);
            padding: 10px;
        }}

        h1, h2, h3, h4, h5, h6 {{
            color: #023E8A !important;
            font-weight: 700;
        }}

        .stButton>button {{
            background-color: #00B487 !important;
            color: white !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            padding: 0.6em 1.2em !important;
            transition: 0.2s ease;
        }}
        .stButton>button:hover {{
            background-color: #00956E !important;
            transform: scale(1.05);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ✅ Call background image
add_background_image("bp.png")

# ---------- SIDEBAR NAVIGATION ---------- #
st.sidebar.title("🩺 MedGPT Navigation")
page = st.sidebar.radio(
    "Go to:",
    [
        "💬 Chatbot",
        "🧠 Symptom Checker",
        "💊 Drug Interaction Checker",
        "📈 Health Dashboard",
        "🖼️ Face Analyzer",
        "⚖️ BMI & Health Insights",
        "📂 Data Analysis",
        "📊 Visualization",
        "🤖 Prediction",
        "🧾 Prescription Assistant",
        "💊 Medicine Identifier",
    ],
)


# 🌐 Language Selection
st.sidebar.markdown("### 🌍 Language Settings")

language_options = {
    "English": "en",
    "Telugu": "te",
    "Hindi": "hi",
    "Tamil": "ta",
    "Kannada": "kn"
}

text_lang = st.sidebar.selectbox(
    "Choose Text Output Language:",
    options=list(language_options.keys()),
    index=0
)

voice_lang = st.sidebar.selectbox(
    "Choose Voice Output Language:",
    options=list(language_options.keys()),
    index=0
)

st.title("🩺 MedGPT - Smart Healthcare Assistant")

# ---------------- Helper Functions ---------------- #
def safe_translate(text, target_lang, retries=3):
    for attempt in range(retries):
        try:
            return GoogleTranslator(source='auto', target=target_lang).translate(text)
        except Exception:
            time.sleep(1.5)
    return text

def chunked_translate(text, target_lang):
    max_chunk = 4000
    chunks = [text[i:i + max_chunk] for i in range(0, len(text), max_chunk)]
    return " ".join(safe_translate(c, target_lang) for c in chunks)

# ---------------- Chatbot Page ---------------- #
if page == "💬 Chatbot":
    st.write("Ask me anything about health or medical insights (educational use only).")

    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_voice_input" not in st.session_state:
        st.session_state.last_voice_input = None

    st.subheader("🎤 Voice Input (Speak Your Question)")
    audio_bytes = mic_recorder(start_prompt="🎙️ Start Talking", stop_prompt="⏹️ Stop Recording", key="voice_input")

    user_input = None
    recognizer = sr.Recognizer()

    if audio_bytes and audio_bytes != st.session_state.last_voice_input:
        st.session_state.last_voice_input = audio_bytes
        audio_data_bytes = audio_bytes["bytes"] if isinstance(audio_bytes, dict) else audio_bytes

        try:
            audio = AudioSegment.from_file(BytesIO(audio_data_bytes))
        except Exception as e:
            st.error(f"⚠️ Could not decode audio: {e}")
            st.stop()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            audio.export(temp_wav.name, format="wav")
            temp_wav_path = temp_wav.name

        with sr.AudioFile(temp_wav_path) as source:
            audio_data = recognizer.record(source)
            try:
                user_input = recognizer.recognize_google(audio_data)
                st.success(f"🗣️ You said: {user_input}")
            except sr.UnknownValueError:
                st.error("❌ Sorry, I couldn’t understand your voice.")
                user_input = None

    text_input = st.chat_input("Type your medical question here...")
    if text_input:
        user_input = text_input
        st.session_state.last_voice_input = None

    if user_input:
        with st.spinner("Thinking..."):
            reply_raw = ask_medgpt(user_input)
            reply = "".join(reply_raw) if hasattr(reply_raw, "__iter__") else str(reply_raw)

        target_lang_code = language_options[text_lang]
        reply_translated = chunked_translate(reply, target_lang_code)
        final_reply = f"**🩺 {text_lang} Output:** {reply_translated}"

        st.session_state.history.append(("You", user_input))
        st.session_state.history.append(("MedGPT", final_reply))

        try:
            st.markdown(f"#### 🔊 {voice_lang} Voice Output:")
            clean_voice_text = reply_translated.replace("*", "").replace("🩺", "")
            voice_code = language_options[voice_lang]
            tts = gTTS(clean_voice_text, lang=voice_code)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as audio_file:
                tts.save(audio_file.name)
                st.audio(audio_file.name, format="audio/mp3")
        except Exception as e:
            st.warning(f"🎧 Could not generate voice: {e}")

    for sender, msg in st.session_state.history:
        st.chat_message("user" if sender == "You" else "assistant").markdown(msg)

    if st.button("🧾 Download Chat as PDF"):
        chat_history = "\n".join([f"{s}: {m}" for s, m in st.session_state.history])
        pdf_path = generate_report(chat_history)
        with open(pdf_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        st.markdown(
            f'<a href="data:application/octet-stream;base64,{b64}" '
            'download="MedGPT_Chat_Report.pdf">📥 Click here to download</a>',
            unsafe_allow_html=True,
        )

# ---------- SYMPTOM CHECKER ---------- #
elif page == "🧠 Symptom Checker":
    st.title("🧠 Symptom Checker")
    st.write("Enter your symptoms to find possible matching conditions (educational use only).")

    import numpy as np

    try:
        df = pd.read_csv("symptom_disease.csv")
    except Exception:
        st.error("⚠️ symptom_disease.csv not found.")
        st.stop()

    # Normalize columns
    df.columns = [c.strip().lower() for c in df.columns]
    if "diseases" not in df.columns:
        st.error("⚠️ CSV must have a 'diseases' column as the first column.")
        st.stop()

    # Symptom columns (all except 'diseases')
    symptom_cols = [c for c in df.columns if c != "diseases"]

    # Get user input
    user_input = st.text_area("Enter symptoms (comma separated)", placeholder="e.g. fever, cough, chest pain")

    if st.button("Analyze Symptoms"):
        start_time = time.time()

        if not user_input.strip():
            st.warning("Please enter some symptoms.")
        else:
            user_symptoms = [s.strip().lower() for s in user_input.split(",") if s.strip()]

            # Check valid/invalid symptoms
            valid_symptoms = [s for s in user_symptoms if s in symptom_cols]
            invalid_symptoms = [s for s in user_symptoms if s not in symptom_cols]

            if not valid_symptoms:
                st.error("❌ None of the entered symptoms exist in the dataset.")
                st.info(f"Example valid symptoms: {', '.join(symptom_cols[:10])} ...")
            else:
                # Convert dataset to NumPy for speed
                symptom_matrix = df[valid_symptoms].to_numpy(dtype=np.int16)
                total_symptoms = df[symptom_cols].sum(axis=1).to_numpy(dtype=np.int16)
                matches = symptom_matrix.sum(axis=1)
                scores = matches / np.maximum(total_symptoms, 1)

                # Filter and sort top results
                mask = matches > 0
                if np.any(mask):
                    top_indices = np.argsort(scores[mask])[::-1][:10]
                    filtered = df.loc[mask].iloc[top_indices]
                    filtered_scores = scores[mask][top_indices]
                    filtered_matches = matches[mask][top_indices]

                    st.subheader("🔍 Possible Diseases")
                    for i, (disease, score, match_count) in enumerate(zip(filtered["diseases"], filtered_scores, filtered_matches), start=1):
                        st.write(f"{i}. **{disease}** — match score: {score:.2f} ({int(match_count)} symptom{'s' if match_count>1 else ''} matched)")
                else:
                    st.info("No matches found. Try entering more symptoms.")

            if invalid_symptoms:
                st.warning(f"⚠️ Unknown symptoms: {', '.join(invalid_symptoms)}")

        end_time = time.time()
        st.caption(f"⏱️ Analyzed in {end_time - start_time:.2f} seconds.")

# ---------- DRUG INTERACTION CHECKER ---------- #
# ---------------- DRUG INTERACTION CHECKER ---------------- #
elif page == "💊 Drug Interaction Checker":
    import pandas as pd
    import requests
    import time
    import json

    st.header("💊 Drug Interaction Checker")
    st.write("Check for known interactions between medicines (educational use only).")

    # --- Load local backup drug data (JSON fallback) ---
    backup_data = []
    try:
        with open("drug_backup.json", "r") as f:
            backup_data = json.load(f)
            st.success("✅ Local backup drug data loaded successfully.")
    except Exception:
        st.warning("⚠️ Backup drug interaction data not found. Only CSV/API will be used.")

    # --- Load CSV data (optional upload) ---
    uploaded_csv = st.file_uploader("Upload local drug interaction CSV (optional)", type=["csv"])
    if uploaded_csv:
        df = pd.read_csv(uploaded_csv)
        st.success("✅ Local CSV loaded successfully.")
    else:
        try:
            df = pd.read_csv("drug_interactions.csv")  # Default CSV file
            st.success("✅ Local CSV backup data loaded successfully.")
        except Exception:
            df = None
            st.warning("⚠️ No local CSV found. Online API or JSON fallback will be used.")

    # --- Validate CSV structure ---
    if df is not None:
        columns = df.columns.str.lower().tolist()
        if not any(col.startswith("drug") for col in columns):
            st.error("⚠️ CSV must contain drug_a, drug_b, severity_notes columns.")
            df = None
        else:
            st.success(f"✅ CSV columns detected: {', '.join(columns)}")

    # --- Function: Get RXCUI ID from RxNav API ---
    def get_rxnav_rxcui(drug_name):
        """Get RXCUI (drug ID) from RxNav API, retry once if needed."""
        try:
            response = requests.get(
                f"https://rxnav.nlm.nih.gov/REST/rxcui.json?name={drug_name}",
                timeout=8
            )
            if response.status_code == 200:
                data = response.json()
                ids = data.get("idGroup", {}).get("rxnormId", [])
                if ids:
                    return ids[0]
            # Retry with lowercase
            time.sleep(0.5)
            response = requests.get(
                f"https://rxnav.nlm.nih.gov/REST/rxcui.json?name={drug_name.lower().strip()}",
                timeout=8
            )
            data = response.json()
            return data.get("idGroup", {}).get("rxnormId", [None])[0]
        except Exception:
            return None

    # --- Function: Check interactions using RxNav API ---
    def check_interaction_online(drug_a, drug_b):
        """Check interaction between two drugs using RxNav API."""
        try:
            rxcui_a = get_rxnav_rxcui(drug_a)
            rxcui_b = get_rxnav_rxcui(drug_b)

            if not rxcui_a or not rxcui_b:
                st.info(f"🌐 Could not find online data for {drug_a} + {drug_b}. Using local sources instead.")
                return None

            url = f"https://rxnav.nlm.nih.gov/REST/interaction/list.json?rxcuis={rxcui_a}+{rxcui_b}"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                groups = data.get("fullInteractionTypeGroup", [])
                if groups:
                    results = []
                    for g in groups:
                        for interaction in g.get("fullInteractionType", []):
                            for pair in interaction.get("interactionPair", []):
                                desc = pair.get("description", "")
                                results.append(desc)
                    return results if results else None
                else:
                    return None
            else:
                st.info(f"🌐 Online API returned no info for {drug_a} + {drug_b}.")
                return None

        except requests.exceptions.Timeout:
            st.info("⏱️ Online API took too long to respond. Using local sources instead.")
            return None
        except Exception:
            st.info(f"⚠️ Could not retrieve online data for {drug_a} + {drug_b}.")
            return None

    # --- User Input ---
    drug_input = st.text_input("Enter drug names (comma separated):", placeholder="Paracetamol, Alcohol")

    if st.button("Check Interactions"):
        if not drug_input.strip():
            st.warning("⚠️ Please enter at least one drug name.")
        else:
            drugs = [d.strip().title() for d in drug_input.split(",") if d.strip()]
            checked_pairs = set()
            found_local = False
            found_api = False
            found_backup = False

            # --- Step 1: Local CSV Dataset ---
            if df is not None:
                for a in drugs:
                    for b in drugs:
                        if a != b and (a, b) not in checked_pairs and (b, a) not in checked_pairs:
                            checked_pairs.add((a, b))
                            subset = df[
                                ((df["drug_a"].str.lower() == a.lower()) &
                                 (df["drug_b"].str.lower() == b.lower())) |
                                ((df["drug_a"].str.lower() == b.lower()) &
                                 (df["drug_b"].str.lower() == a.lower()))
                            ]
                            if not subset.empty:
                                found_local = True
                                info = subset.iloc[0].get("severity_notes", "No details provided.")
                                st.warning(f"⚠️ {a} + {b} — {info}")

            # --- Step 2: Online API Check ---
            st.info("🌐 Checking online medical databases...")
            for a in drugs:
                for b in drugs:
                    if a != b and (a, b) not in checked_pairs and (b, a) not in checked_pairs:
                        checked_pairs.add((a, b))
                        interactions = check_interaction_online(a, b)
                        if interactions:
                            found_api = True
                            for desc in interactions:
                                st.warning(f"⚠️ {a} + {b}: {desc}")

            # --- Step 3: Local JSON Backup Fallback ---
            if not found_api and backup_data:
                for i in range(len(drugs)):
                    for j in range(i + 1, len(drugs)):
                        a, b = drugs[i].lower(), drugs[j].lower()
                        for entry in backup_data:
                            if ({a, b} == {entry["drug_a"].lower(), entry["drug_b"].lower()}):
                                st.warning(f"💊 **{a.title()} + {b.title()}** — {entry['description']}")
                                found_backup = True
                if not found_backup:
                    st.info("✅ No interactions found in local backup data.")

            # --- Step 4: Summary ---
            if not found_local and not found_api and not found_backup:
                st.success("✅ No major interactions found in online or local databases.")

            if checked_pairs:
                st.caption(f"🔍 Drugs checked: {', '.join([f'{a} + {b}' for a, b in checked_pairs])}")

# ---------------- HEALTH DASHBOARD (Auto-Fix Columns) ---------------- #
elif page == "📈 Health Dashboard":
    import plotly.express as px
    import pandas as pd

    st.title("📈 Personal Health Dashboard")
    st.write("Upload your health data or view your vital trends below.")

    uploaded = st.file_uploader("📤 Upload your health_data.csv", type=["csv"])

    # Load the dataset
    if uploaded:
        df = pd.read_csv(uploaded)
        st.success("✅ File uploaded successfully.")
    else:
        try:
            df = pd.read_csv("health_data.csv")
            st.success("✅ Loaded health_data.csv from your MedGPT folder.")
        except Exception:
            df = None
            st.warning("⚠️ Please upload a valid health_data.csv file.")

    if df is not None:
        # Normalize column names
        df.columns = df.columns.str.lower().str.strip()

        # --- Automatic column name mapping ---
        rename_map = {}
        for col in df.columns:
            if "systolic" in col and "bp" in col:
                rename_map[col] = "systolic"
            elif col == "systolic":
                rename_map[col] = "systolic"
            elif "diastolic" in col and "bp" in col:
                rename_map[col] = "diastolic"
            elif col == "diastolic":
                rename_map[col] = "diastolic"
            elif "heart" in col and "rate" in col:
                rename_map[col] = "heart_rate"
            elif "timestamp" in col or "time" in col or "date" in col:
                rename_map[col] = "date"
            elif "oxygen" in col:
                rename_map[col] = "oxygen %"
            elif "temp" in col:
                rename_map[col] = "body temp"

        df.rename(columns=rename_map, inplace=True)

        # Convert date if available
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # If still no systolic/diastolic, show sample of columns
        if "systolic" not in df.columns or "diastolic" not in df.columns:
            st.warning("⚠️ Blood pressure columns not detected. Found columns:")
            st.write(list(df.columns))

        # Limit rows for performance
        df = df.tail(500).reset_index(drop=True)

        # Show preview
        st.subheader("📊 Data Preview (Last 10 Records)")
        st.dataframe(df.tail(10))

        # Safely calculate averages
        avg_systolic = round(df["systolic"].mean(), 1) if "systolic" in df.columns else "N/A"
        avg_diastolic = round(df["diastolic"].mean(), 1) if "diastolic" in df.columns else "N/A"
        avg_hr = round(df["heart_rate"].mean(), 1) if "heart_rate" in df.columns else "N/A"

        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Systolic", f"{avg_systolic} mmHg")
        col2.metric("Avg Diastolic", f"{avg_diastolic} mmHg")
        col3.metric("Avg Heart Rate", f"{avg_hr} bpm")

        st.markdown("---")

        # Charts (only draw if columns exist)
        if "systolic" in df.columns and "diastolic" in df.columns:
            st.subheader("📈 Blood Pressure Trend")
            fig_bp = px.line(
                df, x="date", y=["systolic", "diastolic"],
                markers=True, labels={"value": "mmHg", "variable": "Type"},
                title="Blood Pressure (Systolic/Diastolic)"
            )
            st.plotly_chart(fig_bp, use_container_width=True)

        if "heart_rate" in df.columns:
            st.subheader("❤️ Heart Rate Trend")
            fig_hr = px.line(df, x="date", y="heart_rate",
                             markers=True, title="Heart Rate (BPM)")
            st.plotly_chart(fig_hr, use_container_width=True)

        if "oxygen %" in df.columns:
            st.subheader("🫁 Oxygen Level")
            fig_o2 = px.line(df, x="date", y="oxygen %",
                             markers=True, title="Oxygen Level (%)")
            st.plotly_chart(fig_o2, use_container_width=True)

        if "body temp" in df.columns:
            st.subheader("🌡️ Body Temperature")
            fig_temp = px.line(df, x="date", y="body temp",
                               markers=True, title="Body Temperature (°C)")
            st.plotly_chart(fig_temp, use_container_width=True)

        st.markdown("---")
        st.success("✅ Dashboard loaded successfully.")
    else:
        st.info("Please upload a valid health data file to visualize metrics.")


# ---------- FACE ANALYZER (DeepFace + OpenCV - Standalone) ----------
elif page == "🖼️ Face Analyzer":
    import io
    import numpy as np
    import streamlit as st
    from PIL import Image
    import cv2

    st.title("🧠 AI Face Analyzer")
    st.write(
        "Upload or capture an image to analyze **Age**, **Emotion**, and **Skin Tone** using AI — no dataset or CSV file required."
    )

    # --- Image Input ---
    col1, col2 = st.columns(2)
    with col1:
        cam_img = st.camera_input("📸 Capture Your Face")
    with col2:
        upload_img = st.file_uploader("📁 Or Upload an Image", type=["png", "jpg", "jpeg"])

    image_obj = None
    if cam_img:
        image_bytes = cam_img.getvalue()
        image_obj = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    elif upload_img:
        image_obj = Image.open(upload_img).convert("RGB")

    # --- Try Loading DeepFace ---
    deepface_available = False
    try:
        from deepface import DeepFace
        deepface_available = True
    except Exception:
        deepface_available = False

    # --- Helper Functions ---
    def pil_to_cv2(pil_img):
        arr = np.array(pil_img)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    def detect_face(pil_img):
        """Detect the largest face in the image using OpenCV"""
        cv_img = pil_to_cv2(pil_img)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face = cv_img[y:y + h, x:x + w]
        return Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

    def get_skin_tone(pil_img):
        """Estimate skin tone brightness and average RGB"""
        arr = np.array(pil_img)
        h, w, _ = arr.shape
        patch = arr[h // 3:2 * h // 3, w // 3:2 * w // 3]
        mean_rgb = patch.mean(axis=(0, 1)).astype(int)
        brightness = int(0.2126 * mean_rgb[0] + 0.7152 * mean_rgb[1] + 0.0722 * mean_rgb[2])
        return {"RGB": mean_rgb.tolist(), "Brightness": brightness}

    # --- Main Analyzer Logic ---
    if image_obj is None:
        st.info("📷 Please upload or capture a face image to start the analysis.")
    else:
        st.image(image_obj, caption="Analyzing This Image", use_container_width=True)
        st.markdown("---")

        result_data = {}
        face_crop = detect_face(image_obj)

        if face_crop:
            st.image(face_crop, caption="Detected Face", width=240)
        else:
            st.warning("⚠️ No face detected. Try a clearer, front-facing photo.")

        # --- DeepFace AI Analysis ---
        if deepface_available:
            try:
                with st.spinner("Analyzing face using DeepFace AI..."):
                    result = DeepFace.analyze(
                        np.array(image_obj),
                        actions=["age", "emotion"],
                        enforce_detection=False,
                        detector_backend="retinaface"
                    )

                    # Extract DeepFace Results
                    result_data["Age"] = int(result.get("age", 0))
                    result_data["Emotion"] = result.get("dominant_emotion", "Neutral").capitalize()

                    st.success("✅ DeepFace AI Analysis Complete!")
                    st.write(f"**Estimated Age:** {result_data['Age']} years")
                    st.write(f"**Dominant Emotion:** {result_data['Emotion']}")
            except Exception as e:
                st.warning(f"⚠️ DeepFace analysis failed: {e}")
        else:
            st.error("DeepFace not installed. Run `pip install deepface` to enable AI analysis.")

        # --- Skin Tone Analysis (Always Available) ---
        if face_crop:
            with st.spinner("Analyzing skin tone..."):
                tone = get_skin_tone(face_crop)
                result_data["Skin RGB"] = tone["RGB"]
                result_data["Brightness"] = tone["Brightness"]

                if tone["Brightness"] < 70:
                    tone_label = "Dark"
                elif tone["Brightness"] < 140:
                    tone_label = "Medium"
                else:
                    tone_label = "Fair"

                st.write(f"**Skin Tone:** {tone_label}")
                st.write(f"**Average RGB:** {tone['RGB']} | **Brightness:** {tone['Brightness']}")

        # --- Summary Output ---
        st.markdown("---")
        st.subheader("🧾 Summary")
        st.json(result_data)

        st.caption("⚠️ This AI-based analysis is for educational and demo purposes only — not for medical use.")


# ---------------- BMI & HEALTH INSIGHTS PAGE ---------------- #
elif page == "⚖️ BMI & Health Insights":
    import math
    import plotly.graph_objects as go

    st.title("⚖️ BMI & Health Insights")
    st.write("Enter your basic health details below to calculate BMI and get instant health recommendations.")

    with st.form("bmi_form"):
        name = st.text_input("👤 Name", placeholder="Enter your full name")
        age = st.number_input("🎂 Age", min_value=1, max_value=120, step=1, help="Enter your age in years")
        gender = st.selectbox("⚧️ Gender", ["Male", "Female", "Other"], help="Select your gender")
        height = st.number_input("📏 Height (cm)", min_value=50, max_value=250, step=1, help="Enter your height in centimeters")
        weight = st.number_input("⚖️ Weight (kg)", min_value=10, max_value=300, step=1, help="Enter your weight in kilograms")
        heart_rate = st.number_input("❤️ Heart Rate (bpm)", min_value=30, max_value=200, step=1, help="Enter your resting heart rate in beats per minute")
        systolic = st.number_input("🩸 Systolic BP", min_value=80, max_value=250, step=1, help="Enter the top number of your blood pressure (Systolic)")
        diastolic = st.number_input("💧 Diastolic BP", min_value=40, max_value=150, step=1, help="Enter the bottom number of your blood pressure (Diastolic)")
        oxygen = st.number_input("🫁 Oxygen (%)", min_value=70, max_value=100, step=1, help="Enter your oxygen saturation percentage")
        temp = st.number_input("🌡️ Body Temperature (°C)", min_value=30.0, max_value=45.0, step=0.1, help="Enter your current body temperature in Celsius")
        submitted = st.form_submit_button("🔍 Analyze")

    if submitted:
        if height <= 0 or weight <= 0:
            st.error("Please enter valid height and weight values.")
        else:
            # Calculate BMI
            height_m = height / 100
            bmi = round(weight / (height_m ** 2), 1)

            # BMI category
            if bmi < 18.5:
                category = "Underweight"
                status_color = "yellow"
            elif 18.5 <= bmi < 24.9:
                category = "Normal"
                status_color = "green"
            elif 25 <= bmi < 29.9:
                category = "Overweight"
                status_color = "orange"
            else:
                category = "Obese"
                status_color = "red"

            # Show BMI Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=bmi,
                delta={'reference': 22, 'increasing': {'color': 'red'}},
                gauge={
                    'axis': {'range': [10, 40]},
                    'bar': {'color': status_color},
                    'steps': [
                        {'range': [10, 18.5], 'color': "yellow"},
                        {'range': [18.5, 24.9], 'color': "green"},
                        {'range': [25, 29.9], 'color': "orange"},
                        {'range': [30, 40], 'color': "red"}
                    ],
                },
                title={'text': "BMI Index"}
            ))
            st.plotly_chart(fig, use_container_width=True)

            st.success(f"✅ Your BMI is **{bmi}**, which is categorized as **{category}**.")

            # --- Health Recommendations ---
            recommendations = []

            # BMI-based
            if category == "Underweight":
                recommendations.append("Eat nutrient-rich foods and increase calorie intake gradually.")
            elif category == "Normal":
                recommendations.append("Maintain your healthy diet and regular exercise.")
            elif category == "Overweight":
                recommendations.append("Consider light cardio and balanced meals to reduce BMI.")
            elif category == "Obese":
                recommendations.append("Consult a dietitian; consider regular exercise and calorie control.")

            # Heart rate
            if heart_rate > 100:
                recommendations.append("Heart rate is elevated — try relaxation or check stress levels.")
            elif heart_rate < 60:
                recommendations.append("Heart rate is quite low — check if you feel dizzy or weak.")

            # Blood pressure
            if systolic > 130 or diastolic > 85:
                recommendations.append("Your blood pressure is slightly high — reduce salt and caffeine intake.")
            elif systolic < 100 or diastolic < 60:
                recommendations.append("BP is on the lower side — ensure proper hydration.")

            # Oxygen
            if oxygen < 94:
                recommendations.append("Oxygen level is low — breathe deeply or seek medical help if persistent.")

            # Temperature
            if temp > 37.5:
                recommendations.append("You might have a mild fever — rest and stay hydrated.")

            st.markdown("### 💬 AI-Style Health Insights")
            for i, rec in enumerate(recommendations, start=1):
                st.write(f"**{i}.** {rec}")

            # Final summary card
            st.markdown("---")
            st.subheader("🩺 Summary")
            st.write(f"**Name:** {name if name else 'N/A'}")
            st.write(f"**Age:** {age} years | **Gender:** {gender}")
            st.write(f"**BMI:** {bmi} ({category})")
            st.write(f"**Heart Rate:** {heart_rate} bpm | **BP:** {systolic}/{diastolic} mmHg")
            st.write(f"**Oxygen:** {oxygen}% | **Temp:** {temp}°C")
            st.caption("⚠️ These are AI-generated wellness insights. For medical advice, consult a doctor.")



# ---------------- Data Analysis Page ---------------- #
elif page == "📂 Data Analysis":
    uploaded_file = st.file_uploader("Upload your medical dataset (CSV)", type=["csv"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            show_basic_info(df)

# ---------------- Visualization Page ---------------- #
elif page == "📊 Visualization":
    uploaded_file = st.file_uploader("Upload CSV for visualization", type=["csv"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            show_visualizations(df)

# ---------------- Prediction Page ---------------- #
elif page == "🤖 Prediction":
    uploaded_file = st.file_uploader("Upload CSV for prediction", type=["csv"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            train_predict(df)

# ---------------- Medicine Identifier Page ---------------- #
elif page == "💊 Medicine Identifier":
    st.subheader("💊 Upload a Tablet or Medicine Image")
    uploaded_img = st.file_uploader("Upload image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_img:
        from modules.medicine_identifier import extract_text_from_image, analyze_medicine_info
        from modules.email_sender import send_medicine_email

        st.image(uploaded_img, caption="Uploaded Image", use_column_width=True)
        with st.spinner("🔍 Reading image..."):
            text = extract_text_from_image(uploaded_img)
            st.write(f"**Detected Text:** {text}")

        with st.spinner("💬 Analyzing medicine info..."):
            info = analyze_medicine_info(text)
            st.success("✅ Medicine Information:")
            st.write(info)

        st.subheader("📧 Send This Report to Your Email")
        user_email = st.text_input("Enter your email address:")

        if st.button("Send Email"):
            if user_email:
                uploaded_img.seek(0)
                image_bytes = uploaded_img.read()
                status = send_medicine_email(
                    to_email=user_email,
                    detected_text=text,
                    description=info,
                    image_bytes=image_bytes,
                    image_filename=uploaded_img.name,
                )
                if status is True:
                    st.success(f"✅ Report sent successfully to **{user_email}**")
                else:
                    st.error(status)
            else:
                st.warning("⚠️ Please enter your email address before sending.")

# ---------------- Prescription Assistant Page ---------------- #
elif page == "🧾 Prescription Assistant":
    st.subheader("🧾 Describe Your Symptoms to Get an AI Prescription (Educational Only)")
    patient_name = st.text_input("👤 Enter Patient Name:")
    age = st.number_input("🎂 Enter Age:", min_value=0, max_value=120, step=1)
    gender = st.selectbox("⚧️ Select Gender:", ["Select gender", "Male", "Female", "Other"])
    symptoms = st.text_area("🩺 Describe your symptoms:")
    user_email = st.text_input("📧 Enter your email:")

    if st.button("Generate Prescription"):
        from modules.prescription_assistant import generate_prescription
        from modules.email_sender import send_prescription

        full_description = f"Name: {patient_name}\nAge: {age}\nGender: {gender}\nSymptoms: {symptoms}\n"
        with st.spinner("Analyzing symptoms..."):
            result = generate_prescription(full_description)
            st.success("✅ AI-generated Prescription:")
            st.write(result)

        pdf_path = generate_report(
            f"{full_description}\n--- AI Prescription ---\n{result}\n"
        )

        if user_email:
            with st.spinner("📧 Sending prescription to your email..."):
                status = send_prescription(user_email, pdf_path)
                st.success(status)

    st.markdown(
        """
        ⚠️ **Disclaimer:** This prescription is AI-generated for *educational purposes only*.
        Always consult a licensed doctor before taking any medication.
        """
    )
