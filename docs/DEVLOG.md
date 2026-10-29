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
- **Known limits:** This phase detects the wake phrase only. Follow-up command ASR, actions, retrieval, and TTS remain for later phases. A confirmed wake currently moves the state to listening and displays that command processing is not yet connected.

## 2026-10-29 — Phase 2 Follow-up: Wake-Word Reliability Diagnosis & Resolution

- **Problems faced:**
  - *Vosk vocabulary drop*: In controlled live testing, saying "Chintu" never triggered wake confirmation. Investigation revealed that the Kaldi dictionary in `vosk-model-small-en-in-0.4` had no lexical entry for the word `"chintu"`. When compiled into the recognizer FST with `["chintu", "hey chintu", "ok chintu", "[unk]"]`, Vosk logged `WARNING: Ignoring word missing in vocabulary: 'chintu'` and stripped the word completely. As a result, spoken "Chintu" was classified only as `[unk]` or `""`, and Stage 1 never produced a candidate match for Whisper confirmation.
  - *Silero VAD v5 protocol mismatch*: In `chintu/audio/vad.py`, incoming 480-sample (30 ms) frames were zero-padded by 32 samples into 512 without the mandatory 64-sample context buffer (`(1, 576)` shape expected by Silero VAD v5). Injecting 32 samples of pure silence every 30 ms corrupted the recurrent hidden state and severely suppressed speech probabilities below 0.05, preventing `_active_frames` onset hysteresis from qualifying utterances.
  - *Audio queue buffer starvation*: The capture queue was capped at 12 frames (360 ms) and `_wake_segments` was maxsize 1, causing frame drops whenever confirmation inference took ~400 ms.

- **Failed attempts at fixing:**
  - *Custom openWakeWord prototype (WSL)*: A custom openWakeWord "Chintu" model was trained in an isolated WSL environment outside this repo on synthetic Piper TTS data. The training run failed (accuracy 0.5, recall 0.0, false positives/hour 0.0; exported ONNX scores on held-out synthetic data were ~0.024 for both classes, detecting 0/250 positive examples at threshold 0.5). To reclaim disk space, the 17.28 GB ACAV100M feature file and ~204 MB checkpoint were deleted. The custom model was unviable and abandoned.
  - *Direct audio recording comparison / DTW template matching*: Evaluated comparing live audio against reference WAV recordings of the wake word. Ruled out because raw waveform comparison fails under natural vocal variance (pitch, speed, loudness, room acoustics), and acoustic template matching (MFCC + Dynamic Time Warping) is brittle across background noise and volume differences.

- **What actually worked:**
  - *Silero VAD v5 continuous streaming*: Updated `SileroOnnxVAD` to buffer incoming audio and feed continuous 512-sample chunks prepended with a rolling 64-sample context buffer into ONNX without zero-padding distortion. Added backend state reset on stream interruptions. Speech probability scores on spoken voice immediately jumped from 0.004 to 0.999+.
  - *Vosk phonetic decomposition*: Configured in-vocabulary phonetic phrases that Kaldi recognizes without warnings: `("chin to", "chin two", "chin too", "hey chin to", "hey chin two", "ok chin to", "ok chin two", "[unk]")`. In live testing, spoken "Chintu" is decoded by Vosk as `"[unk] chin two"`, reliably triggering Stage 1.
  - *Faster-Whisper base confirmation*: `faster-whisper` base (int8 on CPU) transcribes the trailing 2-second window in ~430 ms. Expanded `_PHONETIC_ALIASES` with `"chin two"`, `"chin to"`, and `"chin too"`. RapidFuzz fuzzy matching confirmed the wake phrase with 100% confidence (`phrase=chintu score=100.0 Vosk='[unk] chin two' Whisper='Chintu'`).
  - *Buffer sizing*: Increased `AudioCapture` default `queue_frames` from 12 to 64 (~1.9 s) and runtime `_wake_segments` to 4, eliminating frame drops.
  - *Excluded `ANTIGRAVITY_HANDOFF.md` from `test_no_cloud.py`*.

- **Validation:** 34/34 pytest tests passed; `scripts/doctor.py` reported OK with 0.12 ms/frame median VAD inference; live spoken verification confirmed reliable wake word detection.
- **Completion status:** Phase 2 wake-word reliability is complete and verified. Ready to proceed to Phase 3 (Command ASR).
