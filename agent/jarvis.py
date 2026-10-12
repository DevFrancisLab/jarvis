
import os
import time
import wave
import tempfile
import collections
import threading

import numpy as np
import noisereduce as nr
import requests
import sounddevice as sd
import torch

from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad, VADIterator
from windows_tts import generate_speech, play_speech, stop_speech


load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "").rstrip("/")
OLLAMA_USERNAME = os.getenv("OLLAMA_USERNAME", "")
OLLAMA_PASSWORD = os.getenv("OLLAMA_PASSWORD", "")

MODEL_NAME = "llama3.2:3b"

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512
FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE

CALIBRATION_SECONDS = 0.8
START_TIMEOUT_SECONDS = 12
MIN_SPEECH_SECONDS = 0.45
MAX_RECORD_SECONDS = 20
PRE_ROLL_FRAMES = 8

# Voice interruption detection
INTERRUPT_PROBABILITY = 0.80
INTERRUPT_CONFIRM_SECONDS = 0.35
PLAYBACK_START_GRACE_SECONDS = 0.60

MAX_HISTORY_MESSAGES = 20

SYSTEM_PROMPT = """
You are Jarvis, the user's personal AI voice assistant.

IDENTITY:
- Your name is Jarvis.
- You are an AI assistant powered by the Llama language model.
- Do not claim another company created you unless that is established.
- If asked who you are, explain that you are Jarvis, a personal assistant
  built around an AI language model.

CONVERSATION:
- Use the recent conversation provided to you.
- Understand follow-up questions using earlier messages.
- Resolve references such as "it", "they", and "that" from context.
- Do not claim to remember conversations absent from this context.

RESPONSE STYLE:
- Be natural, helpful, and concise because answers are spoken aloud.
- Avoid Markdown and unnecessary formatting.
- Do not invent facts. Admit uncertainty when necessary.
- Do not claim to have performed actions you have not performed.
"""


def make_vad_iterator(vad_model):
    return VADIterator(
        vad_model,
        sampling_rate=SAMPLE_RATE,
        threshold=0.65,
        min_silence_duration_ms=800,
        speech_pad_ms=120,
    )


def speech_probability(vad_model, frame):
    """Return Silero's speech probability for one audio frame."""
    tensor = torch.from_numpy(
        frame.astype(np.float32) / 32768.0
    )

    with torch.inference_mode():
        result = vad_model(tensor, SAMPLE_RATE)

    return float(result.item())


def clean_audio(recorded_frames, noise_frames=None):
    """Convert recorded frames to float audio and reduce background noise."""
    if not recorded_frames:
        return None

    audio_int16 = np.concatenate(recorded_frames).astype(np.int16)
    audio = audio_int16.astype(np.float32) / 32768.0

    # Noise reduction needs a sufficiently long reference signal.
    if noise_frames:
        noise_int16 = np.concatenate(noise_frames).astype(np.int16)
        noise_profile = noise_int16.astype(np.float32) / 32768.0
    else:
        noise_profile = np.array([], dtype=np.float32)

    try:
        if len(audio) >= 2048 and len(noise_profile) >= 2048:
            cleaned = nr.reduce_noise(
                y=audio,
                sr=SAMPLE_RATE,
                y_noise=noise_profile,
                stationary=True,
                prop_decrease=0.75,
            )
        else:
            cleaned = audio

    except Exception as exc:
        print(f"Noise reduction skipped: {exc}")
        cleaned = audio

    return np.clip(cleaned, -1.0, 1.0)


def listen(vad_model):
    """Listen for a complete user request."""
    print("\nListening. Start speaking when you're ready...")

    noise_frames = []
    pre_roll = collections.deque(maxlen=PRE_ROLL_FRAMES)
    recorded_frames = []

    speech_started = False
    speech_start_time = None
    listen_start = time.monotonic()

    vad = make_vad_iterator(vad_model)

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=FRAME_SAMPLES,
        ) as stream:

            calibration_start = time.monotonic()

            while (
                time.monotonic() - calibration_start
                < CALIBRATION_SECONDS
            ):
                data, _ = stream.read(FRAME_SAMPLES)
                noise_frames.append(data[:, 0].copy())

            print("Noise calibrated. Speak now...")

            while True:
                now = time.monotonic()

                if (
                    not speech_started
                    and now - listen_start
                    > START_TIMEOUT_SECONDS + CALIBRATION_SECONDS
                ):
                    print("No speech detected. Try again.")
                    return None

                data, _ = stream.read(FRAME_SAMPLES)
                frame = data[:, 0].copy()
                pre_roll.append(frame)

                tensor = torch.from_numpy(
                    frame.astype(np.float32) / 32768.0
                )

                event = vad(tensor)

                if not speech_started:
                    if event and "start" in event:
                        speech_started = True
                        speech_start_time = now
                        recorded_frames = list(pre_roll)
                        print("Speech detected. Listening...")
                else:
                    recorded_frames.append(frame)
                    elapsed = now - speech_start_time

                    if (
                        event
                        and "end" in event
                        and elapsed >= MIN_SPEECH_SECONDS
                    ):
                        print("End of speech detected.")
                        break

                    if elapsed >= MAX_RECORD_SECONDS:
                        print("Maximum recording time reached.")
                        break

    except Exception as exc:
        print(f"Microphone or VAD error: {exc}")
        return None

    return clean_audio(recorded_frames, noise_frames)


def monitor_for_interruption(vad_model, playback_done, interrupted):
    """
    Monitor the microphone while Jarvis speaks.

    This function only detects an interruption. It deliberately does not
    attempt to transcribe the user's words while Jarvis is playing audio.
    The main thread starts a fresh listening session after playback stops.
    """
    playback_start = time.monotonic()
    consecutive_speech_frames = 0

    required_frames = max(
        1,
        int(INTERRUPT_CONFIRM_SECONDS / FRAME_SECONDS),
    )

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=FRAME_SAMPLES,
        ) as stream:

            print("Interruption listener active.")

            while not playback_done.is_set():
                data, _ = stream.read(FRAME_SAMPLES)
                frame = data[:, 0].copy()
                now = time.monotonic()

                # Ignore initial audio output and speaker startup noise.
                if (
                    now - playback_start
                    < PLAYBACK_START_GRACE_SECONDS
                ):
                    continue

                try:
                    probability = speech_probability(
                        vad_model,
                        frame,
                    )
                except Exception as exc:
                    print(f"Interruption detection error: {exc}")
                    continue

                if probability >= INTERRUPT_PROBABILITY:
                    consecutive_speech_frames += 1
                else:
                    consecutive_speech_frames = max(
                        0,
                        consecutive_speech_frames - 1,
                    )

                if consecutive_speech_frames >= required_frames:
                    interrupted.set()

                    print(
                        "\nVoice detected. Stopping Jarvis..."
                    )

                    # Stop playback immediately.
                    stop_speech()
                    break

    except Exception as exc:
        if not playback_done.is_set():
            print(f"Interruption listener error: {exc}")


def play_with_barge_in(audio_path, vad_model):
    """
    Play a response while monitoring for voice interruption.

    Returns True when an interruption is detected. The caller then
    opens a fresh microphone session to record the complete request.
    """
    playback_done = threading.Event()
    interrupted = threading.Event()

    def playback_worker():
        try:
            play_speech(audio_path)
        except Exception as exc:
            print(f"Playback error: {exc}")
        finally:
            playback_done.set()

    player = threading.Thread(
        target=playback_worker,
        name="jarvis-playback",
        daemon=True,
    )

    listener = threading.Thread(
        target=monitor_for_interruption,
        args=(vad_model, playback_done, interrupted),
        name="jarvis-interruption-listener",
        daemon=True,
    )

    player.start()
    listener.start()

    try:
        while player.is_alive() and not interrupted.is_set():
            time.sleep(0.03)

    except KeyboardInterrupt:
        stop_speech()
        playback_done.set()
        raise

    if interrupted.is_set():
        stop_speech()

    # Wait until playback and microphone monitoring have released their
    # resources before starting another microphone session.
    player.join(timeout=3)
    listener.join(timeout=5)

    if listener.is_alive():
        print("Warning: microphone monitor did not stop cleanly.")

    return interrupted.is_set()


def transcribe_audio(whisper, audio):
    """Save audio temporarily and transcribe it with Faster Whisper."""
    if audio is None or len(audio) == 0:
        return ""

    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False,
    ) as temp_file:
        audio_path = temp_file.name

    try:
        audio_int16 = np.clip(
            audio * 32767,
            -32768,
            32767,
        ).astype(np.int16)

        with wave.open(audio_path, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(audio_int16.tobytes())

        segments, _ = whisper.transcribe(
            audio_path,
            language="en",
            beam_size=3,
            vad_filter=True,
        )

        return " ".join(
            segment.text.strip()
            for segment in segments
        ).strip()

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


def ask_jarvis(question, conversation_history):
    """Send the question and recent conversation to remote Ollama."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *conversation_history,
        {"role": "user", "content": question},
    ]

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        auth=HTTPBasicAuth(
            OLLAMA_USERNAME,
            OLLAMA_PASSWORD,
        ),
        json={
            "model": MODEL_NAME,
            "messages": messages,
            "stream": False,
        },
        timeout=120,
    )

    response.raise_for_status()

    answer = response.json()["message"]["content"].strip()

    conversation_history.extend([
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ])

    if len(conversation_history) > MAX_HISTORY_MESSAGES:
        del conversation_history[
            :len(conversation_history) - MAX_HISTORY_MESSAGES
        ]

    return answer


def main():
    if not OLLAMA_URL or not OLLAMA_USERNAME or not OLLAMA_PASSWORD:
        raise RuntimeError(
            "Missing Ollama configuration. Check your local .env file."
        )

    print("Loading Silero VAD...")
    vad_model = load_silero_vad()

    print("Loading Whisper base model on CPU...")
    whisper = WhisperModel(
        "base",
        device="cpu",
        compute_type="int8",
    )

    conversation_history = []

    print("\nJarvis is ready.")
    print("Voice interruption detection enabled.")
    print("Speak while Jarvis is talking to stop it.")
    print("Type reset to clear memory, or q to quit.")

    while True:
        try:
            command = input(
                "\n[Enter = listen | reset = clear memory | q = quit] "
            ).strip().lower()

            if command == "q":
                print("Jarvis shutting down. Goodbye!")
                break

            if command == "reset":
                conversation_history.clear()
                print("Conversation memory cleared.")
                continue

            audio = listen(vad_model)

            if audio is None:
                continue

            print("Transcribing audio...")
            question = transcribe_audio(whisper, audio)

            if not question:
                print("I couldn't understand that. Please try again.")
                continue

            while question:
                print(f"\nYou: {question}")
                print("Jarvis is thinking...")

                answer = ask_jarvis(
                    question,
                    conversation_history,
                )

                print(f"\nJarvis: {answer}")
                print("Speaking...")

                audio_path = generate_speech(answer)
                print(f"Reply saved to {audio_path}")

                was_interrupted = play_with_barge_in(
                    audio_path,
                    vad_model,
                )

                if not was_interrupted:
                    print("Finished speaking.")
                    break

                # The playback-monitoring stream has now been released.
                # Record the actual request in a clean, separate session.
                print(
                    "\nJarvis stopped. "
                    "Please say your complete request now."
                )

                interruption_audio = listen(vad_model)

                if interruption_audio is None:
                    print("No request captured. Returning to standby.")
                    break

                print("Transcribing your new request...")
                question = transcribe_audio(
                    whisper,
                    interruption_audio,
                )

                if not question:
                    print(
                        "I couldn't understand your request. "
                        "Returning to standby."
                    )
                    break

                # Continue the conversation with the newly transcribed
                # request and preserve the existing conversation history.

        except requests.RequestException as exc:
            print(f"Could not reach the remote AI: {exc}")
            print(
                "Check that your Kaggle notebook and ngrok tunnel "
                "are running."
            )

        except KeyboardInterrupt:
            stop_speech()
            print("\nJarvis stopped.")
            break

        except Exception as exc:
            print(f"Something went wrong: {exc}")


if __name__ == "__main__":
    main()