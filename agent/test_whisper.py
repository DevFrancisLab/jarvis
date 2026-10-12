from faster_whisper import WhisperModel

audio_file = "jarvis-output.wav"

print("Loading Whisper base model...")
model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8",
)

print("Transcribing audio...")
segments, info = model.transcribe(
    audio_file,
    beam_size=3,
    vad_filter=True,
)

print(f"Detected language: {info.language}")
print("Transcript:")

transcript = " ".join(segment.text.strip() for segment in segments)
print(transcript if transcript else "(No speech detected)")
