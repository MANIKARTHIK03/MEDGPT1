import speech_recognition as sr

print("🔍 Listing available microphones...")
for i, name in enumerate(sr.Microphone.list_microphone_names()):
    print(f"{i}: {name}")

try:
    mic_index = int(input("\n🎙️ Enter the mic index you want to test: "))
    with sr.Microphone(device_index=mic_index) as source:
        print(f"🎧 Testing microphone {mic_index} ... please say something!")
        audio = sr.Recognizer().listen(source, timeout=5, phrase_time_limit=5)
    print("✅ Microphone works!")
except Exception as e:
    print(f"❌ Error testing microphone: {e}")
