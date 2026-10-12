# Jarvis

Jarvis is a personal, voice-enabled AI assistant for Windows. You speak into the microphone, the app transcribes what you said, a remote [Ollama](https://ollama.com/) server answers with the `llama3.2:3b` model, and Windows reads the reply aloud.

Speech capture, voice activity detection, transcription, and playback run on your PC. The language model does not. This repository is the Windows client. It does not contain the Kaggle notebook or the command that starts the ngrok tunnel.

## Features

### Working in `agent/jarvis.py`

- **Microphone input.** Audio is recorded from the default input device at 16 kHz, mono, 16-bit, in 512-sample frames (`sounddevice`).
- **Voice activity detection.** [Silero VAD](https://github.com/snakers4/silero-vad) decides when speech starts and ends. The start threshold is `0.65`, trailing silence is 800 ms, and 120 ms of padding is kept around speech. About 0.25 seconds of audio before the start is kept (8 frames).
- **Noise calibration.** The first 0.8 seconds of each listen are stored as a noise profile. `noisereduce` then reduces stationary noise when both the utterance and the profile are long enough (at least 2048 samples). If they are shorter, or if noise reduction fails, the original audio is used.
- **Speech-to-text.** [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) `base` runs on CPU with `int8` compute. Transcription is forced to English (`language="en"`), with beam size 3 and Whisper's own VAD filter.
- **Remote LLM.** Each turn is sent to `POST {OLLAMA_BASE_URL}/api/chat` with HTTP Basic authentication. The model name is hardcoded as `llama3.2:3b`. Responses are not streamed (`stream: false`). The request timeout is 120 seconds.
- **Conversation history.** The system prompt plus up to 20 recent user and assistant messages are sent on every request. History lives in memory for the current process. Typing `reset` clears it. Quitting discards it.
- **Text-to-speech.** `agent/windows_tts.py` uses Windows `System.Speech` and the **Microsoft Zira Desktop** voice (rate 0, volume 100). Speech is written to `jarvis-output.wav` in the working directory and played with `winsound`.
- **Voice interruption.** While the reply plays, a second thread listens. After a 0.6 second grace period, sustained speech (Silero probability at least 0.80 for about 0.35 seconds) stops playback immediately.

### Partial: barge-in

Interruption detection and stop are implemented. Capturing the words spoken *over* the reply is not.

When an interruption is detected, Jarvis stops talking and asks you to say the full request again. A new microphone session then records that second utterance. Speech that overlapped playback is discarded on purpose: the interruption listener does not transcribe it. There is no acoustic echo cancellation, so speaker audio can also trip the detector.

### Present in the repo, not used by the assistant

| Piece | Status |
| --- | --- |
| `agent/test_piper.py` and `en_US-lessac-medium.onnx.json` | Standalone Piper TTS experiment. `jarvis.py` never calls Piper. The ONNX weights are not in git. |
| `livekit-agents` and `livekit-plugins-openai` | Declared dependencies. No project script imports them. |
| `agent/src/agent/__init__.py` | Package entry point. `uv run agent` prints `Hello from agent!` and does not start Jarvis. |
| `agent/requirements.txt` | Dependency list kept beside the project. Installation uses `uv` and `uv.lock`, not this file. |

## Architecture

```mermaid
flowchart LR
  mic[Microphone] --> capture[sounddevice 16 kHz]
  capture --> vad[Silero VAD]
  vad --> denoise[noisereduce]
  denoise --> stt[Faster Whisper base]
  stt --> history[In-memory history]
  history --> ollama["POST /api/chat llama3.2:3b"]
  ollama --> tts[Windows Zira TTS]
  tts --> speakers[Speakers]
  speakers --> interrupt[Mic monitor during playback]
  interrupt --> stop[Stop WAV playback]
```

1. You press Enter. Jarvis calibrates the room noise, then waits up to 12 seconds for speech.
2. Silero marks the start and end of the utterance. Recording stops after at least 0.45 seconds of speech followed by silence, or at 20 seconds.
3. The utterance is written to a temporary WAV file, transcribed, and the temp file is deleted.
4. The transcript, the system prompt, and recent history are posted to the remote Ollama chat API.
5. The reply text is synthesized to `jarvis-output.wav` and played. The interruption monitor runs at the same time.
6. If you interrupt, playback stops and Jarvis records a new request. History from earlier turns is kept.

### Remote Ollama and ngrok

Jarvis does not start Ollama, Kaggle, or ngrok. On a connection error it tells you to check that the Kaggle notebook and ngrok tunnel are running.

The client expects:

- `OLLAMA_BASE_URL` to be the public base URL of that tunnel, with no path and no trailing slash.
- Every request to use HTTP Basic Auth (`OLLAMA_USERNAME` / `OLLAMA_PASSWORD`). That is how the authenticated tunnel is consumed. The process that opens the tunnel is not in this repository.
- The chat call to be `POST {OLLAMA_BASE_URL}/api/chat` with body `{"model": "llama3.2:3b", "messages": [...], "stream": false}`.
- The answer to be `response["message"]["content"]`.

Diagnostic scripts also call `GET {OLLAMA_BASE_URL}` and `GET {OLLAMA_BASE_URL}/api/tags`.

The model name is fixed in `agent/jarvis.py` and `agent/test_ollama_chat.py`. Changing models means editing those files. There is no model environment variable.

## Technology stack

Used by the running assistant:

- Python 3.12 (see Prerequisites)
- [uv](https://docs.astral.sh/uv/) for the environment and lockfile
- `sounddevice` for microphone capture
- `numpy` for audio buffers (installed transitively)
- `silero-vad` and `torch` for voice activity detection
- `noisereduce` for stationary noise reduction
- `faster-whisper` for speech-to-text (`av` / PyAV is declared because Whisper decodes audio with it)
- `requests` for the Ollama HTTP API
- `python-dotenv` to load `agent/.env`
- Windows PowerShell `System.Speech` and `winsound` for synthesis and playback
- Remote Ollama serving `llama3.2:3b`, reached through an authenticated ngrok URL

Installed for experiments or future work, and unused by `jarvis.py`:

- `piper-tts`
- `livekit-agents`
- `livekit-plugins-openai`

## Prerequisites

- **Python 3.12.** `agent/.python-version` pins `3.12`. `agent/pyproject.toml` allows `>=3.12`.
- **uv.** The lockfile and project metadata are set up for `uv sync`.
- **Windows.** Text-to-speech shells out to `powershell.exe`, uses `System.Speech`, and plays audio with `winsound`. The **Microsoft Zira Desktop** voice must be installed.
- **A microphone** that Windows exposes as the default input device, and **speakers or headphones** for playback.
- **A running remote Ollama server** for `llama3.2:3b`, published at the URL in `OLLAMA_BASE_URL` and protected with the username and password in `.env`. The client error text expects that server to be a Kaggle notebook behind ngrok. Those server-side steps are not in this repo.
- Network access on first launch so Faster Whisper can download the `base` weights and Silero can download its VAD weights, if they are not already cached.

## Project structure

```text
jarvis/
├── .gitignore                          # ignores .env, WAV, ONNX, virtualenvs
├── README.md
└── agent/                              # uv project; run everything from here
    ├── .env.example                    # placeholders only
    ├── .python-version                 # 3.12
    ├── pyproject.toml                  # dependencies and the unused `agent` script
    ├── uv.lock
    ├── requirements.txt                # parallel dependency list; uv.lock is what `uv sync` installs
    ├── README.md                       # empty file referenced by pyproject.toml
    ├── jarvis.py                       # the assistant
    ├── windows_tts.py                  # Windows TTS used by jarvis.py
    ├── diagnose_ollama.py              # GET the tunnel URL and print a short response preview
    ├── test_ollama.py                  # GET /api/tags
    ├── test_ollama_chat.py             # one llama3.2:3b chat request
    ├── test_microphone.py              # list devices and record 5 seconds
    ├── test_live_speech.py             # record 5 seconds and transcribe
    ├── test_whisper.py                 # transcribe an existing jarvis-output.wav
    ├── test_piper.py                   # experimental Piper synthesis
    ├── en_US-lessac-medium.onnx.json   # Piper voice config (en_US, lessac, medium, 22050 Hz)
    └── src/agent/__init__.py           # placeholder console script
```

These paths are local and gitignored when present: `agent/.env`, `agent/.venv/`, `*.wav`, and `*.onnx` (including `en_US-lessac-medium.onnx`).

## Installation

From PowerShell, clone the repository and enter the Python project directory. `uv sync` creates `agent/.venv` from `uv.lock` and `.python-version`. You do not need to create the virtual environment by hand.

```powershell
git clone https://github.com/DevFrancisLab/jarvis.git
cd jarvis\agent
uv sync
```

If `uv` is not installed:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Open a new terminal after installing uv, then run `uv sync` from `agent`.

`uv run` uses `agent/.venv` without activating it. To activate the environment in the current PowerShell session:

```powershell
.\.venv\Scripts\Activate.ps1
```

If activation is blocked by the execution policy, allow scripts for this session only and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Configuration

The assistant reads three variables from the environment. `load_dotenv()` looks for `.env` starting beside `jarvis.py`, so the file belongs in `agent/.env`.

| Variable | Required | Purpose |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | Yes | Tunnel base URL. Trailing slashes are stripped. Do not include `/api/chat`. |
| `OLLAMA_USERNAME` | Yes | HTTP Basic Auth user expected by the tunnel. |
| `OLLAMA_PASSWORD` | Yes | HTTP Basic Auth password expected by the tunnel. |

`jarvis.py` refuses to start if any of the three is missing. `test_ollama.py` and `diagnose_ollama.py` raise a `KeyError` instead if a variable is unset.

Create the local file from the example and edit the placeholders:

```powershell
cd jarvis\agent
Copy-Item .env.example .env
notepad .env
```

`agent/.env.example`:

```dotenv
# Copy this file to agent/.env and replace every placeholder.
# Never commit .env. Never put real credentials in this example.

# Base URL of the remote Ollama server. No trailing slash, and do not append /api/chat.
# The running client expects this to be the public URL of an authenticated ngrok tunnel.
OLLAMA_BASE_URL=https://your-tunnel.ngrok-free.app

# HTTP Basic Auth credentials required by that tunnel.
OLLAMA_USERNAME=your-username
OLLAMA_PASSWORD=your-password
```

## Ollama setup

The language model is **`llama3.2:3b`**, served by a remote Ollama process. Jarvis calls:

```text
POST {OLLAMA_BASE_URL}/api/chat
Authorization: Basic <OLLAMA_USERNAME:OLLAMA_PASSWORD>
Content-Type: application/json

{
  "model": "llama3.2:3b",
  "messages": [system prompt, recent history, current user turn],
  "stream": false
}
```

`GET {OLLAMA_BASE_URL}/api/tags` is how `test_ollama.py` checks that the same URL lists models. `diagnose_ollama.py` requests the base URL itself and prints the HTTP status, content type, and the first 300 characters. Use that when the body is an ngrok HTML page instead of the Ollama API.

This repository does not include:

- the Kaggle notebook that runs Ollama
- the ngrok command that opens the tunnel
- an ngrok authtoken or basic-auth flag

The client only knows that the tunnel must already be up, must speak the Ollama HTTP API, and must accept the username and password from `.env`. When the notebook or tunnel stops, or when ngrok issues a new URL, update `OLLAMA_BASE_URL` and start the remote side again. Confirm the exact notebook and `ngrok` invocation outside this repo before treating any server command as part of the project.

A chat smoke test, after `.env` is filled in:

```powershell
cd jarvis\agent
uv run python test_ollama.py
uv run python test_ollama_chat.py
```

`test_ollama.py` should list `llama3.2:3b` among the models. `test_ollama_chat.py` sends “Say hello. You are Jarvis.” and prints the reply.

## Running Jarvis

From `agent`, with the remote tunnel running:

```powershell
cd jarvis\agent
uv run python jarvis.py
```

If the virtual environment is already activated, `python jarvis.py` is the same entry point. Do not use `uv run agent`. That command runs the placeholder package script.

On startup Jarvis loads Silero VAD and the Whisper `base` model, then waits at this prompt:

```text
[Enter = listen | reset = clear memory | q = quit]
```

| Input | What happens |
| --- | --- |
| Enter | Calibrate noise, listen, transcribe, ask Ollama, and speak the reply. |
| `reset` | Clear conversation memory. |
| `q` | Stop and exit. |
| Ctrl+C | Stop playback, if any, and exit. |

After “Noise calibrated. Speak now…”, talk in a normal voice and pause when you are done. Jarvis prints `You:` and `Jarvis:` before it speaks. To interrupt a spoken reply, talk for a fraction of a second after playback has started, then wait for “Please say your complete request now.” and repeat the full request.

The first launch can take a while while Whisper and Silero weights download. Later launches still load both models before the prompt appears.

## Testing and troubleshooting

Run these from `agent` with `uv run python <script>`. They are manual diagnostic scripts, not a pytest suite.

| Script | What it checks |
| --- | --- |
| `test_microphone.py` | Lists PortAudio devices, records 5 seconds at 16 kHz, prints the sample count. It does not save or play audio. |
| `test_live_speech.py` | Records 5 seconds to `microphone-test.wav` and transcribes it with Whisper `base`. |
| `test_whisper.py` | Transcribes an existing `jarvis-output.wav`. It does not record. Run a TTS test or a Jarvis reply first. |
| `windows_tts.py` | Speaks a fixed English sentence through Microsoft Zira and writes `jarvis-output.wav`. |
| `diagnose_ollama.py` | GET the base URL with Basic Auth. Shows status, content type, and a short body preview. |
| `test_ollama.py` | GET `/api/tags` and lists model names. |
| `test_ollama_chat.py` | One non-streaming chat call to `llama3.2:3b`. |
| `test_piper.py` | Experimental. Synthesizes `jarvis-test.wav` with Piper if `en_US-lessac-medium.onnx` is in `agent`. |

### Microphone access

`test_microphone.py` should show your input device and a non-zero sample count. If Jarvis prints `Microphone or VAD error`, check that an input device exists, that Windows microphone privacy allows desktop apps, and that another app is not holding the device exclusively. Jarvis always uses the default input. It has no device-selection flag. `test_microphone.py` is how you see which device that is.

### Ollama connectivity

If startup says `Missing Ollama configuration. Check your local .env file.`, one of the three variables is empty or `.env` is not in `agent`.

If a turn prints `Could not reach the remote AI`, the Kaggle notebook or ngrok tunnel is down, the URL is wrong, or the request exceeded 120 seconds. Run `diagnose_ollama.py`, then `test_ollama.py`. A JSON body with a `models` array means the API is reachable. An HTML content type usually means the URL is an ngrok error or interstitial page rather than Ollama.

`test_ollama.py` prints the HTTP status on failure. `401` means the username or password does not match the tunnel. A successful tags list that does not include `llama3.2:3b` means the chat call will fail until that model is present on the server.

### Expired ngrok URLs

The client stores no previous URL and does not refresh it. Restarting the tunnel can issue a new URL. Put the current HTTPS base URL in `OLLAMA_BASE_URL`, save `.env`, and start Jarvis again. Confirm with `uv run python test_ollama.py` before a voice session.

### Speech recognition

`No speech detected. Try again.` means silence lasted through the 0.8 second calibration plus the 12 second start window.

`I couldn't understand that. Please try again.` means Whisper returned an empty transcript. Use `test_live_speech.py` to separate microphone problems from the full assistant. Jarvis only transcribes English. Very short speech (under about 0.45 seconds after the start event) is ignored until more speech or the 20 second cap arrives.

### Text-to-speech

`windows_tts.py` is the isolated check. Failure there is a Windows voice problem, not an Ollama problem. The synthesizer selects `Microsoft Zira Desktop` by name. If that voice is not installed, PowerShell fails and Jarvis reports `Something went wrong` or `Windows TTS did not generate an audio file.` Install or enable that desktop voice, then rerun `uv run python windows_tts.py`.

Playback uses `winsound` synchronously. `Playback error` during a reply means the WAV could not be played. `jarvis-output.wav` is overwritten on every reply and left on disk.

### Piper experiment

`test_piper.py` expects `agent/en_US-lessac-medium.onnx` next to the committed `en_US-lessac-medium.onnx.json` config (English, US, Lessac dataset, medium quality, 22050 Hz). If the weights file is missing, the script tells you to run the Piper voice download first. That download command is not in this repository. Piper output is not what `jarvis.py` speaks.

## Security

- Put `OLLAMA_BASE_URL`, `OLLAMA_USERNAME`, and `OLLAMA_PASSWORD` only in `agent/.env`.
- `.gitignore` ignores `.env` and `.env.*`, with an exception for `.env.example`. Do not commit secrets, and do not replace the placeholders in `.env.example` with real values.
- The client sends those credentials on every Ollama request. Use the HTTPS ngrok URL, not a plaintext host.
- `*.wav` is gitignored. `jarvis-output.wav`, `jarvis-test.wav`, and `microphone-test.wav` contain generated or recorded speech. Temporary transcription WAVs are deleted after Whisper runs; the spoken reply WAV is not.
- `*.onnx` is gitignored. Downloaded Piper weights stay on the machine that needs the Piper test. The JSON config in git has no credentials.
- Whisper and Silero weight caches are created on first use by those libraries. Treat them as local model files, not as project source.
- `diagnose_ollama.py` prints a slice of the HTTP body. Do not point it at a URL whose response contains secrets you do not want in the terminal scrollback.

## Known limitations and roadmap

Limitations of the current client:

- **Windows-only speech output.** `winsound` and `System.Speech` do not run on other operating systems. Microphone code uses `sounddevice`, but the supported path is Windows.
- **CPU transcription.** Whisper `base` is loaded with `device="cpu"` and `compute_type="int8"`. There is no GPU switch. Startup loads both Whisper and Silero before you can speak.
- **Remote latency.** The reply is a single blocking HTTP call with a 120 second timeout. Nothing is spoken until the full answer arrives. If the notebook sleeps or the tunnel drops, the voice loop cannot answer.
- **ngrok is an external moving URL.** This repo cannot renew it.
- **English in, spoken style out.** The system prompt asks for concise spoken answers without Markdown. Transcription is pinned to English in `jarvis.py`.
- **Memory is short and local.** Only the latest 20 chat messages are kept, only in the current process.
- **One default microphone.** No device argument, no push-to-talk beyond the Enter prompt, and no echo cancellation.
- **Barge-in is stop-then-rerecord.** Overlapping speech is not transcribed. The microphone monitor can also print `Warning: microphone monitor did not stop cleanly.` if that thread does not exit within 5 seconds.
- **Noise reduction is best-effort.** It is skipped for short clips and on any `noisereduce` error.
- **The spoken WAV stays in the working directory** and is overwritten each turn.

Unfinished work already visible in the tree, not a committed product plan:

- Transcribe the interrupting utterance itself, instead of asking for it again after playback stops.
- Piper (`test_piper.py`, Lessac medium voice config) is a separate experiment from the Windows voice Jarvis actually uses.
- LiveKit packages are installed and unused.
- The `agent` console script and `agent/README.md` are still placeholders. `pyproject.toml` still says `Add your description here`.

## License

No license has been specified. The repository has no `LICENSE` file, and `pyproject.toml` does not set a license field.
