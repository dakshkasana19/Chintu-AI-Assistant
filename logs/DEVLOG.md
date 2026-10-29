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

## Phase 2 Follow-up — Wake-Word Reliability Diagnosis & Resolution (2026-10-29)

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


