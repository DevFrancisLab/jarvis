import sounddevice as sd

print("Available audio devices:")
print(sd.query_devices())

print("\nDefault input device:")
print(sd.query_devices(kind="input"))

print("\nRecording 5 seconds...")
audio = sd.rec(
    int(5 * 16000),
    samplerate=16000,
    channels=1,
    dtype="float32",
)
sd.wait()

print("Recording complete.")
print("Samples captured:", len(audio))
