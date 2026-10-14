# Development Log

## 2026-10-12 — Phase 0: Foundation (partial)

- **Built and why:** Added the `chintu` package with TOML-backed defaults, a minimal runtime state tracker, rotating file logging, environment doctor/model-directory scripts, and a no-cloud regression test. Removed the legacy cloud integrations from the entry point and dependency manifest to establish a local-only baseline.
- **Key design decisions:** Runtime settings belong in ignored `config.toml`; `config.toml.example` documents defaults. Logs and downloaded models are local ignored artifacts. `llama-cpp-python` remains an optional commented dependency because its source package installation failed on this Windows/Python environment.
- **Problems and resolutions:** The initial full requirements install failed while packaging `llama-cpp-python` due to a missing vendored file. Installed the remaining dependencies individually. `trafilatura` initially failed to import because `lxml_html_clean` was absent; installed that dependency and confirmed import.
- **Validation:** `py_compile` passed for the new entry point and package modules. `pytest -q tests/test_no_cloud.py` passed (1 test). `scripts/doctor.py` reported all listed required imports OK; the model directory was not present at that time.
- **Completion status:** Partial. This is not yet a completed Phase 0 implementation: the old voice-assistant behavior was replaced by a UI bootstrap rather than migrated, the app was not smoke-tested end-to-end, and the download script does not yet download models. These must be resolved before calling the phase complete.
- **Evaluation report:** [Phase 0 results](results/phase0-2026-10-12.md).

### Current project state

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
