import wave
import sounddevice as sd
from faster_whisper import WhisperModel

OUTPUT_FILE = "microphone-test.wav"
SAMPLE_RATE = 16000
DURATION = 5

print("Speak clearly after recording starts...")
audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="int16",
)
sd.wait()

with wave.open(OUTPUT_FILE, "wb") as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(SAMPLE_RATE)
    wav.writeframes(audio.tobytes())

print(f"\nAudio saved to {OUTPUT_FILE}")
print("Loading Whisper base model...")

model = WhisperModel("base", device="cpu", compute_type="int8")
segments, info = model.transcribe(
    OUTPUT_FILE,
    beam_size=3,
    vad_filter=True,
)

transcript = " ".join(segment.text.strip() for segment in segments)

print(f"\nDetected language: {info.language}")
print("You said:")
print(transcript if transcript else "(No speech detected)")
