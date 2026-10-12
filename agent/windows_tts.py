
import subprocess
import winsound
from pathlib import Path


def generate_speech(
    text: str,
    output_path: str = "jarvis-output.wav",
) -> Path:
    """Generate a WAV file using Windows built-in speech synthesis."""
    output = Path(output_path).resolve()
    safe_text = text.replace("'", "''")
    safe_output = str(output).replace("'", "''")

    script = f"""
Add-Type -AssemblyName System.Speech
$speech = New-Object System.Speech.Synthesis.SpeechSynthesizer

try {{
    $speech.SelectVoice('Microsoft Zira Desktop')
    $speech.Rate = 0
    $speech.Volume = 100
    $speech.SetOutputToWaveFile('{safe_output}')
    $speech.Speak('{safe_text}')
}} finally {{
    $speech.Dispose()
}}
"""

    subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", script],
        check=True,
    )

    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError("Windows TTS did not generate an audio file.")

    return output


def play_speech(output_path: str | Path) -> None:
    """Play a WAV file synchronously."""
    winsound.PlaySound(str(output_path), winsound.SND_FILENAME)


def stop_speech() -> None:
    """Stop currently playing WAV audio."""
    winsound.PlaySound(None, winsound.SND_PURGE)


def speak(text: str, output_path: str = "jarvis-output.wav") -> Path:
    """Generate and play speech. Suitable for simple tests."""
    output = generate_speech(text, output_path)
    play_speech(output)
    return output


if __name__ == "__main__":
    result = speak(
        "Hello! I am Jarvis. You can now test my voice output."
    )
    print(f"Speech played successfully: {result}")