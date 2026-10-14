# Development Log

## Phase 0 — Foundation (2026-10-12; initially partial)

- Added the `chintu` package, TOML-backed settings, runtime state tracking, rotating local logs, environment/model-directory diagnostics, and a no-cloud regression test. Removed legacy cloud integrations from the entry point and dependency manifest to establish a local-first baseline.
- Kept runtime settings, downloaded models, and generated logs local and ignored. `llama-cpp-python` remained optional after its package installation failed on missing vendored source files.
- Installed remaining dependencies individually after the full requirements install failed; resolved a `trafilatura` import issue by installing `lxml_html_clean`.
- Validation: Python compilation passed; `tests/test_no_cloud.py` passed (1 test); doctor import checks passed. At this point the UI was only a bootstrap, the model downloader was a placeholder, and the voice pipeline was not migrated, so Phase 0 was recorded as partial.
- Initial report: [Phase 0 results](../docs/results/phase0-2026-10-12.md).

## Phase 0 — Foundation follow-up (2026-10-14; complete within approved scope)

- Added Python 3.10 TOML compatibility with `tomli`, config and state validation, moved the PyQt overlay into `chintu/ui/`, expanded diagnostics, hardened the Windows launcher, documented setup, and added offscreen UI/config/state tests.
- Kept voice, wake-word, ASR, retrieval, and TTS implementation deferred to their planned phases; Phase 0 completion covers the agreed foundation rather than an end-to-end assistant.
- Resolved the Python 3.10 `tomllib` compatibility issue by using `tomli`. Focused validation passed: 7 tests, Python compilation, doctor import checks, and microphone device enumeration. Pylance reported no errors.
- Report: [Phase 0 results](../docs/results/phase0-2026-10-14.md).

## Phase 1 — Audio capture + VAD (2026-10-15)

- Added persistent 16 kHz mono microphone capture, bounded frame delivery, device recovery, and live RMS UI updates.
- Added local Silero ONNX voice activity detection with WebRTC fallback, pre-roll, onset confirmation, hangover, and an utterance-duration cap.
- Fixed a microphone shutdown race and added a concurrent shutdown regression test.
- Verified with 19 passing tests, Python compilation, the environment doctor, and an offscreen runtime smoke test confirming microphone capture starts and stops cleanly.
- Smoke measurements: 0.13 ms median VAD inference per frame and 84.04 ms microphone open-to-first-frame. These are machine-specific measurements, not benchmarks.
- Scope limit: segments are captured and logged but not transcribed or acted upon; wake-word detection, ASR, command routing, and TTS remain for later phases.
- Detailed results: [Phase 1 report](../docs/results/phase1-2026-10-15.md).
