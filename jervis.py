import queue
import sys
import time
from datetime import datetime

import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel
import pyttsx3
import requests

# ---------------------------
#  CONFIG
# ---------------------------
SAMPLE_RATE = 16000        # 16 kHz audio
FRAME_DURATION = 30        # ms per frame (10, 20, or 30 allowed)
CHANNELS = 1
WAKE_WORDS = ["jarvis", "jervis"]
MODEL_SIZE = "small"       # "tiny", "base", "small", "medium" etc.

# ---------------------------
#  TEXT TO SPEECH (pyttsx3)
# ---------------------------
engine = pyttsx3.init()
engine.setProperty("rate", 170)  # speaking speed


def speak(text: str):
    """Jarvis bolne ke liye function."""
    print(f"Jarvis 🧠:", text)
    engine.say(text)
    engine.runAndWait()


# ---------------------------
#  SPEECH-TO-TEXT MODEL
# ---------------------------
print("Loading speech-to-text model (faster-whisper)...")
stt_model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
print("Model loaded ✅")

# ---------------------------
#  AUDIO STREAM + VAD
# ---------------------------
audio_queue = queue.Queue()
vad = webrtcvad.Vad(2)  # 0–3 (3 = most aggressive)


def audio_callback(indata, frames, time_info, status):
    """Mic se aane wala data queue me daalna."""
    if status:
        print(status, file=sys.stderr)

    # stereo ko mono me convert
    if indata.ndim > 1 and indata.shape[1] > 1:
        indata = np.mean(indata, axis=1, keepdims=True)

    audio_queue.put(bytes(indata))


def record_voice_command(timeout=8):
    """
    VAD use karke voice segment capture karega.
    timeout = max seconds ek command ke liye.
    """
    speak("Listening...")
    frames = []
    start_time = time.time()
    voiced_frames = 0
    silent_frames = 0

    frame_size = int(SAMPLE_RATE * FRAME_DURATION / 1000)

    while True:
        try:
            frame = audio_queue.get(timeout=timeout)
        except queue.Empty:
            break

        if len(frame) < frame_size * 2:
            continue

        # sirf exact frame size ka part VAD ko do
        chunk = frame[: frame_size * 2]
        is_speech = vad.is_speech(chunk, SAMPLE_RATE)

        if is_speech:
            voiced_frames += 1
            silent_frames = 0
            frames.append(frame)
        else:
            silent_frames += 1
            frames.append(frame)

        # agar kaafi der se silence hai
        if voiced_frames > 5 and silent_frames > 10:
            break

        # global timeout
        if time.time() - start_time > timeout:
            break

    if not frames:
        return None

    audio_bytes = b"".join(frames)
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return audio_np


# ---------------------------
#  TRANSCRIBE
# ---------------------------
def transcribe_audio(audio_np: np.ndarray) -> str:
    """Audio ko text me convert kare."""
    segments, info = stt_model.transcribe(audio_np, beam_size=1)
    text = ""
    for seg in segments:
        text += seg.text + " "
    return text.strip().lower()


# ---------------------------
#  COMMANDS
# ---------------------------
def get_time():
    now = datetime.now()
    return now.strftime("%I:%M %p")


def google_search_short(query: str) -> str:
    """
    Simple web search style response.
    (Yeh sirf demo hai, direct Google API nahi.)
    """
    try:
        url = "https://duckduckgo.com/?q=" + query.replace(" ", "+")
        return f"Here is what I found about {query}. You can open: {url}"
    except Exception:
        return "I tried to search but something went wrong."


def handle_command(text: str) -> str:
    """User ka command samajh ke reply generate kare."""

    # wake-word normalise
    if any(wake in text for wake in WAKE_WORDS):
        for w in WAKE_WORDS:
            text = text.replace(w, "")
        text = text.strip()

    if text == "":
        return "Yes, I am listening."

    # basic commands
    if "time" in text or "samay" in text or "baj" in text:
        return f"The time is {get_time()}."

    if "hello" in text or "hi" in text:
        return "Hello Manik, how can I help you?"

    if "how are you" in text:
        return "I am just code, but I am running perfectly fine!"

    if "your name" in text or "who are you" in text:
        return "My name is Jarvis, your Python voice assistant."

    if "search" in text or "google" in text:
        query = (
            text.replace("search", "")
            .replace("on google", "")
            .replace("google", "")
            .strip()
        )
        if not query:
            return "What should I search for?"
        return google_search_short(query)

    if "shutdown" in text or "band ho ja" in text or "exit" in text:
        return "Okay, shutting down. Bye!"

    # default
    return f"You said: {text}. I have not been taught this command yet."


# ---------------------------
#  MAIN LOOP
# ---------------------------
def main():
    speak("Jarvis online. Say 'Jarvis' and then your command.")

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=int(SAMPLE_RATE * FRAME_DURATION / 1000),
        dtype="int16",
        channels=CHANNELS,
        callback=audio_callback,
    ):
        while True:
            audio_np = record_voice_command(timeout=10)
            if audio_np is None:
                print("No voice detected.")
                continue

            try:
                text = transcribe_audio(audio_np)
            except Exception as e:
                print("Transcription error:", e)
                speak("Sorry, I could not understand.")
                continue

            print("You said 🎤:", text)

            if not any(w in text for w in WAKE_WORDS):
                print("Wake word not detected, ignoring...")
                continue

            reply = handle_command(text)
            speak(reply)

            if "shutdown" in text or "band ho ja" in text or "exit" in text:
                break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting Jarvis.")
