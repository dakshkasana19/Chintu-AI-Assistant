# Development Log

## 2026-10-12 — Phase 0: Foundation (partial)

- **Built and why:** Added the `chintu` package with TOML-backed defaults, a minimal runtime state tracker, rotating file logging, environment doctor/model-directory scripts, and a no-cloud regression test. Removed the legacy cloud integrations from the entry point and dependency manifest to establish a local-only baseline.
- **Key design decisions:** Runtime settings belong in ignored `config.toml`; `config.toml.example` documents defaults. Logs and downloaded models are local ignored artifacts. `llama-cpp-python` remains an optional commented dependency because its source package installation failed on this Windows/Python environment.
- **Problems and resolutions:** The initial full requirements install failed while packaging `llama-cpp-python` due to a missing vendored file. Installed the remaining dependencies individually. `trafilatura` initially failed to import because `lxml_html_clean` was absent; installed that dependency and confirmed import.
- **Validation:** `py_compile` passed for the new entry point and package modules. `pytest -q tests/test_no_cloud.py` passed (1 test). `scripts/doctor.py` reported all listed required imports OK; the model directory was not present at that time.
- **Completion status:** Partial. This is not yet a completed Phase 0 implementation: the old voice-assistant behavior was replaced by a UI bootstrap rather than migrated, the app was not smoke-tested end-to-end, and the download script does not yet download models. These must be resolved before calling the phase complete.
- **Evaluation report:** [Phase 0 results](results/phase0-2026-10-12.md).

### Project state after initial Phase 0 implementation (2026-10-12)

- The repository has a Phase 0 PyQt UI/runtime foundation. The former wake-word, speech recognition, web-answer, Windows-action, and TTS flows are not currently wired into the runtime.
- No persistent audio capture, VAD, local wake-word model, ASR model, Piper voice model, intent router, grounded RAG pipeline, or model download/checksum flow is implemented yet.
- The UI remains the existing visual overlay; its waveform is still simulated. `scripts/download_models.py` only prepares the models directory and does not fetch model files.
- Dependency import checks passed for the modules listed in `scripts/doctor.py`. `llama-cpp-python` was not installed because packaging failed on a missing vendored file; it remains optional until the local-LLM phase.
- The foundation is not a usable voice assistant yet; voice behavior remains deferred to its assigned phases.

## 2026-10-14 — Phase 0: Foundation follow-up (complete within approved scope)

- **Built and why:** Added Python 3.10-compatible TOML parsing, explicit config/state validation, moved the existing PyQt overlay into `chintu/ui/`, expanded dependency/device diagnostics, added startup/config/state smoke tests, hardened the Windows launcher, and documented setup/current capabilities in [README.md](../README.md).
- **Key design decisions:** Per user choice, Phase 0 remains foundational; the removed voice pipeline is deferred to its planned phases rather than reintroducing a temporary local audio implementation. The UI smoke test runs offscreen and needs no microphone or network. `tomli` is declared for Python 3.10 compatibility.
- **Problems and resolutions:** Pylance identified the stdlib `tomllib` import as unresolved on Python 3.10; switched to the already installed `tomli`. The test-runner integration did not discover the supplied absolute test paths, so focused tests were run directly with the configured Python interpreter.
- **Validation:** `pytest -q tests/test_foundation.py tests/test_no_cloud.py` passed (7 tests); `py_compile` passed for changed Python modules; `scripts/doctor.py` found all checked imports and enumerated the system microphone input device. Audio capture and end-to-end assistant behavior were not tested because they are deferred.
- **Evaluation:** No wake-word, ASR, retrieval, answer-quality, or latency evaluation has been run; all such metrics are N/A. See [Phase 0 results](results/phase0-2026-10-14.md).
- **Completion status:** Phase 0 foundation is complete within the user-approved scope. The project is still not ready for voice-assistant testing; begin Phase 1 only when requested.

## 2026-10-15 — Phase 1: Audio capture + VAD

- **Built and why:** Added a single persistent sounddevice 16 kHz mono InputStream delivering fixed-size frames into a bounded queue; added Silero ONNX VAD with WebRTC fallback, pre-roll, speech-onset hysteresis, configurable hangover and utterance cap; runtime retries microphone stream failures. Live input RMS is emitted to and reflected by the UI.
- **Key design decisions:** Silero runs on CPU with its 512-sample window padded from the configured 480-sample/30 ms frames. The pinned model is downloaded from the public Silero GitHub source with SHA-256 verification and resumable partial downloads. WebRTC is the offline fallback. No audio is persisted or sent over the network.
- **Problems and resolutions:** The first test expectation undercounted the pre-roll plus onset-confirmation frames; corrected expected segmentation to reflect the exact returned frame sequence. A real-runtime shutdown smoke test exposed a race when `stop()` cleared the stream while `read_frame()` was waiting; `read_frame()` now reports the closed stream as an explicit capture error, covered by a concurrent-shutdown regression test. A Python 3.10 runtime is used; all audio dependencies were checked before work, and `webrtcvad-wheels` was installed because the fallback binding was missing.
- **Validation:** Full pytest suite passed (19 tests); changed Python modules compiled; doctor reported all required imports OK, detected the system microphone, selected Silero ONNX, and measured 0.13 ms median VAD/frame and 84.04 ms first microphone frame in a single run. Offscreen Qt runtime smoke activated real microphone capture and shut down with both the worker and stream stopped.
- **Evaluation:** These are environment smoke measurements, not voice-quality benchmarks. No recorded-speech detection accuracy, false-alarm rate, or clipped-syllable evaluation was run. See [Phase 1 results](results/phase1-2026-10-15.md).
- **Known limits:** There is no ASR or wake-word consumer yet, so VAD segments are logged but not transcribed. TTS is not implemented, so mic muting and barge-in handling are deferred to the TTS phase. Test manual wake phrases only after those later phases are connected.

## 2026-10-16 — Phase 2: Local wake-word detection

- **Built and why:** Added VAD-gated local wake detection using the restricted Indian-English Vosk grammar, followed by faster-whisper base confirmation over the last configured two seconds. Confirmed phrases transition the runtime from sleeping through wake confirmation into listening; command transcription remains out of scope.
- **Key design decisions:** Defined a `WakeWordDetector` protocol with a Vosk/Whisper implementation; higher `wake_sensitivity` permits lower RapidFuzz match scores; cooldown suppresses repeat triggers. Wake inference runs on a bounded worker queue so CPU confirmation does not block microphone frame consumption. Runtime model loading is local-only.
- **Problems and resolutions:** The official `vosk-model-small-en-in-0.4` archive was available at the Vosk model endpoint and is SHA-256 pinned in the downloader. The faster-whisper base model was already cached locally; the setup downloader now prepares it if absent. Focused tests exposed ambiguous phrase ties and an overly permissive partial-string score; longer phrase tie-breaking and word-boundary n-gram scoring now make sensitivity thresholds effective. The demo initially showed its overlay at startup, which looked activated despite the state remaining asleep; it now keeps audio/wake detection active in the background but hides the overlay until wake confirmation.
- **Validation:** Full pytest suite passed (33 tests); Python compilation and wake-evaluation CLI checks passed. Real Vosk and cached faster-whisper initialized; a silence segment did not trigger; offscreen runtime kept the real microphone and wake worker active while the overlay remained hidden and state stayed `SLEEPING`, then showed the overlay after a simulated confirmed wake. Both workers and the stream shut down cleanly.
- **Evaluation:** `scripts/eval_wake.py` computes positive miss rate and negative false accepts per hour for labeled 16 kHz PCM16 WAV sets. No labeled speech set was available in this environment, so measured miss rate and false accepts/hour remain **N/A**. See [Phase 2 results](results/phase2-2026-10-16.md).
- **Known limits:** At phase completion, follow-up command ASR, actions, retrieval, and TTS were not yet connected. Phase 3 adds command ASR; actions, retrieval, and TTS remain for later phases.

## 2026-10-29 — Phase 2 Follow-up: Wake-word reliability

- **Problems faced:**
  - *Vosk dictionary drop*: The word `"chintu"` was missing from `vosk-model-small-en-in-0.4`, so Vosk discarded it from the grammar and only output `[unk]`, never triggering Whisper.
  - *Silero VAD suppression*: 30 ms frames were padded with 32 zeros without the required 64-sample context buffer, corrupting recurrent state and keeping speech scores < 0.05.
  - *Small queue buffers*: 12-frame mic queue dropped audio during model inference.
- **Failed attempts:**
  - *Custom openWakeWord (WSL)*: Synthetic model failed training (0/250 detections; abandoned).
  - *Audio waveform / DTW matching*: Ruled out due to natural pitch/volume variance.
- **What worked:**
  - *Silero VAD streaming*: Continuous 512 chunks + 64-sample context buffer without padding (speech scores jumped to 0.999+).
  - *Phonetic Vosk*: Mapped in-vocabulary phrases (`"chin to"`, `"chin two"`), which Vosk detects cleanly without warnings.
  - *Faster-Whisper confirmation*: Base model confirms trailing audio with RapidFuzz (`score=100.0`).
  - *Larger buffers*: Mic queue expanded to 64 frames (~1.9 s); `_wake_segments` to 4.
- **Validation:** 34/34 tests passed, doctor OK (0.12 ms VAD), verified live on microphone.

## 2026-10-29 — Phase 3: Command ASR

- **Built and why:** Added local faster-whisper command transcription for new VAD-delimited speech captured after wake confirmation. The wake utterance stays in the wake queue and is not retranscribed as a command.
- **Key design decisions:** CPU/int8 inference runs on a dedicated worker thread. The cached `base` Whisper model is shared with wake confirmation; an alternate configured command model is loaded separately and must be prepared explicitly. Raw transcript and normalized command remain separate, while transcript text is not written to runtime logs.
- **Filtering and evaluation:** Empty/silent segments and likely hallucinations are rejected using no-speech probability, average log probability, and compression ratio. Added `scripts/eval_asr.py` to calculate raw and normalized-command WER over paired one-utterance 16 kHz mono PCM16 WAV/reference files without sending audio over the network.
- **Validation:** Full suite passed (51 tests); final doctor run passed with 0.28 ms median VAD/frame and 373.82 ms microphone first frame on this machine. A synthetic low-level noise sample produced a likely “Chintu” hallucination, which was rejected at no-speech probability 0.65. This is a noise rejection smoke check, not a command accuracy evaluation.
- **Limits and manual check:** No live command speech or labeled corpus has been evaluated, so command WER and live recognition accuracy remain N/A. After waking, say a separate command and inspect the raw/normalized transcript values; repeat with silence/background noise. Phase 4 is not started.

## 2026-10-29 — Phase 4: Deterministic local actions

- **Built and why:** Added a deterministic intent router and Windows action executor, then connected them to the post-wake command-ASR flow. General information requests are identified but deferred until retrieval is implemented.
- **Supported actions:** App and HTTPS website launching, confirmed app closing and power operations, volume/brightness/media controls, workstation lock, screenshots, local time/date/day and system status, timers, alarms, and cancellation.
- **Safety and limits:** Destructive intents require an explicit spoken confirmation. Shutdown and restart are scheduled with a delay. Unit tests use injected platform boundaries; live Windows action behavior and spoken-command accuracy have not been manually validated.
- **Validation:** Full pytest suite passed (91 tests). No live Windows action execution was performed as part of this verification.
