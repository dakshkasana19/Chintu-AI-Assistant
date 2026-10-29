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

## Phase 2 — Local wake-word detection (2026-10-16)

- Implemented VAD-gated Vosk grammar detection plus local faster-whisper confirmation on the last two seconds, fuzzy phonetic matching, configurable sensitivity, cooldown, and a bounded worker queue.
- Added the checksum-pinned Indian-English Vosk model to the downloader; faster-whisper base is prepared locally. Runtime model loading is offline-only.
- Added `scripts/eval_wake.py` to report positive miss rate and negative false accepts/hour on labeled 16 kHz PCM16 WAVs.
- Verified full suite (33 tests), compilation, real Vosk/Whisper initialization and silence rejection, plus real microphone/runtime startup, hidden-until-confirmed behavior, and clean shutdown.
- Corrected demo startup appearance: microphone and wake detector remain active in the background, while the overlay appears only after wake confirmation.
- No labeled speech set was available; measured wake miss and false-accept rates remain N/A.
- Detailed results: [Phase 2 report](../docs/results/phase2-2026-10-16.md).

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

## Phase 3 — Command ASR (2026-10-29)

- Added local faster-whisper CPU/int8 transcription of VAD-delimited utterances captured after wake confirmation. The wake-word utterance itself is routed only to the wake detector.
- Added a separate bounded command queue and ASR worker, preserving the raw transcript and normalized command in runtime memory/signals. Transcript text is not written to runtime logs.
- Reused the cached `base` Whisper model with wake confirmation; other supported command model sizes can be prepared using `scripts/download_models.py --wake --whisper-model <size>`.
- Added configurable no-speech, average log-probability, and compression-ratio rejection thresholds and `scripts/eval_asr.py` for raw and normalized-command WER over paired one-utterance WAV/reference files.
- Validation: 51 tests passed; final doctor run passed (0.28 ms/frame median VAD, 373.82 ms microphone first frame); compilation and ASR evaluator CLI checks passed.
- Synthetic low-level noise produced a “Chintu” hallucination in Whisper's raw result, rejected by the no-speech threshold (0.65). This is only a noise-rejection smoke check; no live command speech or labeled corpus has been evaluated.

## 2026-10-29 — Phase 4: Deterministic local actions

- Added a deterministic intent router and Windows action executor, connected to the post-wake command-ASR flow. General information requests are identified but deferred until retrieval is implemented.
- Supports app and HTTPS website launching, confirmed app closing and power operations, volume/brightness/media controls, workstation lock, screenshots, local time/date/day and system status, timers, alarms, and cancellation.
- Destructive intents require spoken confirmation; shutdown and restart are scheduled with a delay. Unit tests use injected platform boundaries; live Windows action behavior and spoken-command accuracy have not been manually validated.
- Validation: Full pytest suite passed (91 tests). No live Windows action execution was performed as part of this verification.
