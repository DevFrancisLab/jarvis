from pathlib import Path
from piper import PiperVoice

model_path = Path("en_US-lessac-medium.onnx")
output_path = Path("jarvis-test.wav")

if not model_path.exists():
    raise FileNotFoundError(
        f"Voice model not found: {model_path.resolve()}. "
        "Run the Piper voice download command first."
    )

voice = PiperVoice.load(str(model_path))

with output_path.open("wb") as wav_file:
    voice.synthesize_wav(
        "Hello! I am Jarvis, your local voice assistant. How can I help you today?",
        wav_file,
    )

print(f"Speech generated: {output_path.resolve()}")
