# Chintu AI

Chintu AI is a Windows desktop voice-assistant project rebuilt around local audio processing, deterministic PC actions, offline speech output, and short answers grounded in keyless retrieved sources. The current checkout includes Phase 0 foundation, Phase 1 audio capture/VAD, Phase 2 local wake-word detection, Phase 3 local command transcription, Phase 4 deterministic local actions, Phase 5 offline speech output (TTS), and Phase 6 grounded retrieval and answer verification (RAG).

## Current status

| Capability | Current state | Implemented module |
|---|---|---|
| Desktop overlay | Centered frameless buttercup yellow (#FFE550) rotating ring | `chintu/ui/chintu_ui.py` |
| Runtime state and configuration | Phase 0 foundation with TOML configuration | `chintu/state.py`, `chintu/config.py` |
| Audio capture and VAD | Phase 1: persistent microphone input and utterance segmentation | `chintu/audio/capture.py`, `chintu/audio/vad.py` |
| Local wake-word detection | Phase 2: VAD-gated Vosk grammar candidate plus faster-whisper confirmation | `chintu/audio/wake.py` |
| Local command transcription | Phase 3: faster-whisper CPU/int8 with confidence filtering | `chintu/asr.py` |
| Deterministic Windows actions | Phase 4: local intent routing and Windows action execution; destructive actions require confirmation | `chintu/actions/` |
| Grounded retrieval and answer verification | Phase 6: keyless Wikipedia, Open-Meteo, DuckDuckGo with BM25 reranking and grounding verification | `chintu/rag/` |
| Offline speech output | Phase 5: Piper neural voice with pyttsx3 fallback, streaming PCM, and barge-in | `chintu/tts.py` |

All pipeline stages are implemented and run 100% locally on CPU without API keys, accounts, or cloud speech services.

## Setup

Supported target: Windows 10/11 and Python 3.10–3.12. CPU-only operation is the design target; no GPU is required.

1. Copy `config.toml.example` to `config.toml` if you want to customize local settings. `config.toml` is ignored by Git.
2. Run `run_chintu.bat`. It creates a virtual environment, installs `requirements.txt`, downloads the local VAD/wake models when needed, and starts the assistant in the background. The overlay stays hidden until the wake phrase is confirmed.
3. After setup, `launch_chintu.bat` starts the assistant with the project virtual environment. Only one Chintu listener can run per Windows session; additional launches are ignored.
4. For a dependency/device report, run `venv\Scripts\python scripts\doctor.py`.
5. Run the automated checks with `venv\Scripts\python -m pytest -q`.

Phase 1 opens one persistent microphone stream, calculates live RMS for the UI, and detects speech boundaries. Silero VAD runs on CPU when its checksum-verified model exists; otherwise the local WebRTC VAD fallback is used. A bounded in-memory frame queue and pre-roll buffer preserve speech onset and prevent a slow consumer from growing memory without bound. The runtime retries the input stream after device errors.

Phase 2 continuously checks VAD-delimited speech segments for the wake phrase while the UI remains hidden and the runtime stays asleep. Segments are sent to a local Vosk recognizer with phonetic candidates such as `"chin to"` and `"chin two"`; a candidate is confirmed by local faster-whisper base transcription over the segment's last two seconds, with RapidFuzz matching for configured phrases and phonetic variants. Only after confirmation does the overlay appear and the runtime enter its listening state. `wake_sensitivity` ranges from `0.0` (strict) to `1.0` (more permissive); `wake_cooldown_seconds` prevents repeated triggers.

Phase 3 passes only new VAD-delimited speech captured after wake confirmation to a background command-ASR worker. faster-whisper runs locally on CPU with int8 quantization; the default `base` model is loaded once and shared with wake confirmation to avoid a duplicate model in memory. Transcripts are retained as both raw text and normalized command text. The normalizer removes leading wake/filler phrases and converts cardinal number words through 99 to digits. Likely silence or hallucinations are rejected using Whisper's no-speech probability, average log probability, and compression ratio. ASR settings are in `config.toml.example`. The default `base` model is prepared with `python scripts/download_models.py --wake`; for another model size, use `python scripts/download_models.py --wake --whisper-model small.en` (replace the size as needed).

Phase 4 routes normalized commands to a deterministic local intent router and Windows action executor. Supported actions include opening apps and HTTPS websites, closing apps, volume/brightness/media controls, locking the workstation, screenshots, local date/time and system status, timers, alarms, and power operations. Closing apps and power operations require a separate spoken confirmation; shutdown/restart are scheduled with a delay and can be canceled.

Phase 5 provides offline speech output with `chintu/tts.py`. It integrates the offline Piper neural TTS engine (`piper-tts`) with a `pyttsx3` (Windows SAPI5) fallback. Audio is rendered as 16-bit/float32 PCM and played through `sounddevice.OutputStream` in 50 ms chunks. VAD-confirmed speech onset interrupts playback; utterance capture then continues and routes the completed segment to command ASR. A dedicated normalizer expands abbreviations, units, currencies, numbers, and strips markdown. The Piper voice model is optional; prepare it with `python scripts/download_models.py --tts`, otherwise the configured `auto` engine falls back to the installed system voice.

Phase 6 implements retrieval with `chintu/rag/`. General information queries route to keyless public sources: Open-Meteo for live weather and geocoding, Wikipedia REST API for encyclopedic summaries, and DuckDuckGo for web search. Extracted passages are reranked using Okapi BM25. The verifier returns extractive passages only when lexical evidence coverage and the configured relevance threshold pass; otherwise it abstains. This deterministic lexical check is not semantic fact-checking and cannot guarantee truth. Verified responses include the source title. Weather requests without an explicit location ask for one rather than assuming a default city.

The model downloader fetches the Indian-English small Vosk model from its public model endpoint and verifies the pinned archive SHA-256 before safe extraction. It prepares faster-whisper models in the local Hugging Face model cache. Runtime model loading uses `local_files_only=True`, so it does not fetch models while processing microphone audio. Use `python scripts/download_models.py --wake` to prepare the default model assets manually.

Wake-word evaluation uses labeled 16 kHz PCM16 WAV files and reports positive miss rate and negative false accepts per audio hour:

```powershell
python scripts\eval_wake.py --positive-dir eval\wake\positive --negative-dir eval\wake\negative --report eval\wake\report.json
```

Unit tests do not require a microphone; use the doctor script to check a real input device and report capture/VAD timing.

Command-ASR evaluation expects one post-wake command per 16 kHz, mono, PCM16 WAV file and a same-named `.txt` reference beside it. The report includes raw WER and normalized-command WER, including a per-utterance transcript and rejection reason:

```powershell
python scripts\eval_asr.py --audio-dir eval\asr\clips --report eval\asr\report.json
```

The evaluator runs the configured local model and does not upload audio. Its JSON report contains transcribed text, so keep the report local and do not commit personal recordings or transcripts.

## Configuration and local files

Defaults are documented in `config.toml.example`. Phase 1 audio settings include sample rate, frame size, pre-roll, speech-onset confirmation, end silence, maximum utterance duration, VAD threshold, WebRTC aggressiveness, and stream recovery interval. Phase 2 adds wake confirmation window, fuzzy-match sensitivity, and cooldown settings. Phase 3 adds the command-ASR model size and rejection thresholds. Local settings live in `config.toml`; logs are written to the rotating `logs/chintu.log`; downloaded Vosk model files belong in `models/`, and faster-whisper models are cached locally by Hugging Face. These local paths are ignored by Git. Invalid TOML or an unsupported setting is reported as a configuration error rather than silently replaced with defaults.

## No API keys, ever

The assistant does not use API keys, accounts, paid services, or hosted language models. The Phase 0 runtime makes no network requests. Later phases may use only the following keyless public services or model distribution sites:

| Service | Intended purpose | Data sent |
|---|---|---|
| Silero VAD model on GitHub | One-time local VAD model download | Public ONNX model request only; no audio |
| Indian-English Vosk model endpoint | One-time local wake model download | Public model archive request only; no audio |
| Hugging Face | One-time local faster-whisper model download | Public model files only; no audio |
| DuckDuckGo | Public web search | Search query text |
| Wikipedia/Wikidata REST APIs | Entity lookup and public reference information | Search/entity query |
| Open-Meteo | Weather and geocoding | Location query/coordinates |
| Public RSS/Atom feeds | Current-event retrieval | Feed URL requests; no audio |
| Public model distribution endpoints | One-time model downloads | Model file requests; no audio |
| Retrieved public web pages | Source text for answer grounding | Page URL requests; no audio |

Microphone audio remains on the device and is processed in memory by local VAD and local Vosk/Whisper models. Search text can be sent only to the keyless public services listed above when retrieval is implemented. Grounding, verification, and abstention are intended to reduce unsupported answers; answer quality still depends on the quality and availability of the sources.

## Architecture (planned)

```mermaid
flowchart LR
    UI[PyQt desktop UI] --> Runtime[Assistant runtime and state]
    Runtime --> Audio[Persistent audio capture and VAD]
    Audio --> Wake[Local Vosk + Whisper wake detector]
    Wake --> ASR[Local faster-whisper ASR]
    ASR --> Router[Deterministic intent router]
    Router --> Actions[Windows actions]
    Router --> Retrieval[Keyless retrieval sources]
    Retrieval --> Verify[Grounding and verification]
    Verify --> TTS[Offline Piper / pyttsx3]
    TTS --> UI
```

The diagram describes the assistant pipeline; the runtime implements the complete local loop: desktop shell, configuration, state foundation, persistent input capture, VAD, wake-word confirmation, local command ASR, deterministic Windows actions, grounded RAG retrieval, and offline TTS output.

## Development and evaluation

See [docs/DEVLOG.md](docs/DEVLOG.md) for phase-by-phase progress and [docs/results/](docs/results/) for test and evaluation reports. Run `scripts/eval_wake.py` with labeled positive and negative speech samples to measure wake miss rate and false accepts/hour; these accuracy metrics are not yet available because no labeled speech corpus has been evaluated.
