# Development Log

## 2026-10-12 — Phase 0: Foundation (partial)

- **Built and why:** Added the `chintu` package with TOML-backed defaults, a minimal runtime state tracker, rotating file logging, environment doctor/model-directory scripts, and a no-cloud regression test. Removed the legacy cloud integrations from the entry point and dependency manifest to establish a local-only baseline.
- **Key design decisions:** Runtime settings belong in ignored `config.toml`; `config.toml.example` documents defaults. Logs and downloaded models are local ignored artifacts. `llama-cpp-python` remains an optional commented dependency because its source package installation failed on this Windows/Python environment.
- **Problems and resolutions:** The initial full requirements install failed while packaging `llama-cpp-python` due to a missing vendored file. Installed the remaining dependencies individually. `trafilatura` initially failed to import because `lxml_html_clean` was absent; installed that dependency and confirmed import.
- **Validation:** `py_compile` passed for the new entry point and package modules. `pytest -q tests/test_no_cloud.py` passed (1 test). `scripts/doctor.py` reported all listed required imports OK; the model directory was not present at that time.
- **Completion status:** Partial. This is not yet a completed Phase 0 implementation: the old voice-assistant behavior was replaced by a UI bootstrap rather than migrated, the app was not smoke-tested end-to-end, and the download script does not yet download models. These must be resolved before calling the phase complete.
- **Evaluation report:** [Phase 0 results](results/phase0-2026-10-12.md).

### Current project state

- The repository now has Phase 0 scaffolding, but the app entry point only initializes the PyQt UI and state tracker. The former wake-word, speech recognition, web-answer, Windows-action, and TTS flows are not currently wired into the runtime.
- No persistent audio capture, VAD, local wake-word model, ASR model, Piper voice model, intent router, grounded RAG pipeline, or model download/checksum flow is implemented yet.
- The UI remains the existing visual overlay; its waveform is still simulated. `scripts/download_models.py` only prepares the models directory and does not fetch model files.
- Dependency import checks passed for the modules listed in `scripts/doctor.py`. `llama-cpp-python` was not installed, and that doctor script does not currently check every dependency declared in `requirements.txt`.
- The Phase 0 changes are saved locally. This log does not claim that the project is ready for normal voice-assistant use.
