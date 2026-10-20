# Antigravity Handoff Prompt — Chintu Wake-Word Reliability

You are taking over the Chintu AI repository to resolve the wake-word problem. Read this prompt and inspect the repository before proposing a solution.

## Mandatory first step: discuss before fixing

Do **not** change code, install packages, download models, or start a training run yet. First:

1. Inspect the current implementation, tests, configuration, model setup, and phase logs relevant to audio capture, VAD, wake detection, and UI visibility.
2. Explain to me what you believe is causing the wake-word failure. Do not assume the original report fully identifies the root cause. If your investigation points elsewhere (for example, VAD, microphone input, buffering, audio format, detector thresholds, or model suitability), tell me and show the evidence.
3. Describe your proposed approach, alternatives considered, expected trade-offs, required dependencies/data, privacy implications, and how you will measure success.
4. Ask whether that approach is what I want. Wait for my answer before implementing any fix.

## My wake-word requirements

- Chintu must remain on standby and continuously listen locally for the wake phrase **“Chintu”**, like Siri.
- It must not show an activated/listening state or begin command handling before the wake word has actually been confirmed.
- The overlay should stay hidden during standby and appear only after confirmation.
- Keep the microphone/audio processing local. Do not send or upload microphone audio, and do not use API keys, accounts, paid services, or hosted speech recognition.
- Do not report or promise robust recognition until it has been measured on relevant positive and negative audio.
- Preserve all current user changes. Do not lower thresholds or weaken confirmation simply to make a demo appear to work without measured evidence.

## Project goals and constraints

The product goal is a hands-free, wake-word activated Windows desktop assistant that hears accurately, speaks naturally, executes local Windows commands deterministically, and answers questions with accurate, brief, one-sentence spoken answers grounded in retrieved sources. Priority is answer accuracy, then speech accuracy, then latency. Avoid novelty and flashy features; make each feature reliable before adding another.

Hard constraints:

- Zero API keys, tokens, accounts, signups, or paid services.
- Audio stays on the device. Network access is limited to keyless public endpoints (DuckDuckGo and fetching result pages, Wikipedia/Wikidata REST APIs, Open-Meteo, public RSS feeds, and one-time public model downloads). Log and document network calls.
- Remove cloud/keyed code; do not reintroduce Gemini, Google speech recognition, Edge TTS, pygame, or PyAudio.
- Windows 10/11, Python 3.10–3.12, CPU-only machines must work.
- Degrade gracefully on missing models, unplugged microphone, offline operation, or slow local models. Never crash, guess a system action, or invent a fact; ask or abstain.
- Do not claim “zero hallucinations” or “no knowledge cutoff.” Explain that grounding, verification, and abstention reduce hallucination and quality depends on source quality.
- Keep `chintu_assistant.py` as a thin entry point. Use type hints, docstrings on public functions, small single-responsibility modules, no global mutable state, and no silent broad exception handling. Blocking inference/network work must not block the Qt GUI thread.
- Check that necessary libraries are present before each phase/step; install only when actually needed and explain the choice.
- Work one phase at a time. After a phase, run relevant tests (and `scripts/doctor.py` when available), summarize changes, say what I should manually test, then stop and wait for me to say “continue.”

## Original roadmap supplied by the user (Phase 0 through Phase 7)

### Phase 0: Foundation

Create the package layout, `config.py`, rotating file logging in `./logs` (stdlib logging, no `print`), and remove all keyed/cloud code. Fix `requirements.txt` (minimum versions): sounddevice, numpy, onnxruntime, vosk, faster-whisper, piper-tts, pyttsx3, llama-cpp-python, ddgs, requests, trafilatura, feedparser, rapidfuzz, psutil, pycaw, comtypes, screen-brightness-control, PyQt6, pytest, plus optional libzim. Update `_gitignore` (models/, logs/, local config), `run_chintu.bat` (create/activate venv, install, run `download_models.py` if models missing, launch). Move existing behavior over first with a smoke test, then rewrite internals. Add `tests/test_no_cloud.py` that scans the source tree and fails on: `google.generativeai`, `google.genai`, `openai`, `anthropic`, `edge_tts`, `recognize_google`, `"api_key"`, `"API_KEY"`.

### Phase 1: Audio capture + VAD

Persistent input stream, ring buffer with pre-roll so the first syllable is never clipped. Silero VAD decides speech start/end (end silence default ~700 ms, max utterance ~15 s, configurable). Only VAD-positive audio reaches ASR. Expose live mic RMS to the UI. Ignore the mic path while TTS plays (prevents self-triggering) but keep a lightweight “stop/cancel” barge-in check.

### Phase 2: Wake word (local, always-on, low CPU)

Vosk small English (Indian English model if available) with a restricted grammar (`["chintu","hey chintu","ok chintu","[unk]"]`), gated by VAD, behind a `WakeWordDetector` interface. Two-stage confirmation: on a Vosk hit, re-transcribe the last ~2 s with faster-whisper and fuzzy-match (RapidFuzz) against phonetic variants (chintu, chinto, chintoo, chinchu, “chin two”); only then wake. Cooldown to prevent double triggers. Config for sensitivity. `scripts/eval_wake.py` reports miss rate and false accepts/hour on recorded WAVs. Play a short generated earcon on wake (no mp3 assets).

### Phase 3: ASR

faster-whisper, int8 on CPU, default `small.en` (configurable: base.en/small/medium), loaded once in a worker thread and kept warm. `initial_prompt`/hotwords with “Chintu,” app names, and user vocabulary; beam_size 5; reject hallucinated text on silence/noise using `no_speech_threshold`, `avg_logprob`, and `compression_ratio` checks (“Thank you.”, “you,” etc.). Command-text normalization (lowercase, number words to digits, strip fillers like “please”/“can you”/“chintu”) while keeping the raw transcript for search queries. `scripts/eval_asr.py` computes WER over WAV + reference `.txt` pairs.

### Phase 4: Intent router + native OS actions

Rule-based router returns `Intent(name, slots, confidence)` using anchored patterns and RapidFuzz. Order: exit/sleep (only short exact utterances or phrases starting with “go to sleep”/“stop listening”; never mid-sentence) -> local actions -> information requests (RAG). Ambiguous confidence -> ask a short clarifying question instead of acting.

Build an app index at startup by scanning Start Menu `.lnk` files and UWP apps (`shell:AppsFolder`) plus user aliases in config. Launch via `os.startfile` or `explorer shell:AppsFolder` (never `shell=True` with string concatenation), verify, and report failure honestly.

Core actions only, done very well: open/close app, volume up/down/set/mute (pycaw), brightness, media play/pause/next/previous (media keys via ctypes), lock screen, screenshot, current time/date (local), battery/CPU/RAM (psutil), open common websites, timers/alarms with spoken notification. Destructive actions (shutdown, restart, sign out, close app with unsaved work) require spoken confirmation. “What day is Diwali”-style questions go to RAG, not the date handler. Every action returns a structured result and is unit-tested with mocked OS calls.

### Phase 5: TTS (natural, offline, fast)

Piper (`en_US-amy-medium` or best available medium/high voice; configurable voice/speed/volume) -> pyttsx3 fallback only. Split into sentences, synthesize sentence N+1 while N plays, play raw PCM through one persistent sounddevice `OutputStream` (no mp3, no pygame). Target time-to-first-audio < ~600 ms for short replies. Speakable-text normalizer: expand numbers, currency, dates, percentages, units, abbreviations; remove markdown/URLs/citations; keep Unicode. Fully interruptible by voice (“stop”/“cancel”/“quiet”), tray or hotkey. Expose real output RMS to the UI. Replace hardcoded “Sir” with configurable `user_name`; vary acknowledgements slightly, keep replies short.

### Phase 6: RAG answer pipeline (the most important phase; report eval numbers after EACH step)

**Step 6.1 Question classifier (rule-based, unit-tested):** `local_action`, `time_date`, `weather`, `entity_fact`, `definition`, `current_event` (latest/today/score/price/news/“who is the current...”), `howto`, `comparison_or_opinion`, `unanswerable/ambiguous`. Each class has its own retrieval plan and answer shape.

**Step 6.2 Query rewriting with the last 3–4 turns of memory:** resolve pronouns/follow-ups, strip filler, up to 2 alternative queries (keyword form and natural form). Memory is cleared on sleep and never persisted unless enabled by me.

**Step 6.3 Sources behind a `RetrievalSource` interface** (`name`, `can_handle(class)`, `fetch(query) -> list[Chunk]`, trust score, freshness), each mockable:

- WikidataSource + WikipediaSource: entity resolution via Wikipedia search API, structured properties from Wikidata, summaries from the REST API. First choice for entity_fact/definition.
- OfflineWikiSource (optional, if a `.zim` file exists in `./models`; libzim; `download_models.py --with-offline-wiki` fetches a Kiwix build from the public Kiwix site; document size options in README). Used when offline or DuckDuckGo fails.
- NewsFeedSource: public RSS/Atom feeds (BBC, The Hindu, NDTV, others editable in config) via feedparser, with local timestamped cache, for current_event.
- WeatherSource: Open-Meteo (with geocoding, configurable home city). time_date is local only.
- WebSearchSource: DuckDuckGo via ddgs for the long tail and corroboration; for entity_fact/definition/howto also run site-restricted variants (`site:wikipedia.org`, `site:britannica.com`, configurable trusted domains); fetch full page text for the top 3–4 results with trafilatura (timeouts, size caps, parallel threads, TTL cache, backoff on rate limits); fall back to snippets.

**Step 6.4 Source routing:** per question class pick an ordered list of sources, query them in parallel with timeouts, merge ALL chunks into one candidate pool (chunks ~100–150 words, 20% overlap) before reranking, each chunk keeping source name, trust and freshness. If high-trust sources answer with high retrieval confidence, skip DuckDuckGo entirely.

**Step 6.5 Lexical reranking (own BM25, no dependency):** tuned k1/b, stopword removal, light stemming, query-term proximity bonus, title/heading bonus, recency boost for current_event, trust-score feature, penalties for boilerplate/cookie text/reviews/forums/listicles, near-duplicate removal by shingle Jaccard. Keep the top 4–6 chunks within a token budget; expose the top score as a retrieval-confidence signal.

**Step 6.6 Extractive answerer (must be good enough to stand alone):** choose the best sentence from reranked chunks by query-term coverage, answer-type match (number for “how many,” date for “when,” name for “who”) and source trust; trim to one clause if long.

**Step 6.7 Generation with an embedded local LLM via llama-cpp-python** (no server, no Ollama, no keys) behind an `LLMProvider` interface. Candidates: Qwen2.5-3B-Instruct Q4_K_M, Qwen2.5-7B-Instruct Q4_K_M, Qwen2.5-1.5B-Instruct Q4_K_M (low-RAM fallback, auto-selected via psutil). `scripts/benchmark_models.py` runs the gold set on each and reports accuracy, abstention, hallucination and latency; the default is chosen from that report and the RAM check, not assumed. Temperature 0, max ~45 new tokens, GBNF/JSON-schema constrained output: `{"answer": "<one spoken sentence>", "source_ids": [...], "confidence": "high|medium|low"}` or `{"answer": null}`. Prompt: system rules (use ONLY the numbered sources; one sentence; lead with the answer; no filler, URLs, markdown or spoken citations; “According to X” only for contested/time-sensitive claims; return null if unsure), today’s date, 3–4 few-shot examples per question class (including abstention), then sources, then the question. If llama-cpp-python won’t install on Windows, find a prebuilt CPU wheel or tell me; never switch to a hosted API.

**Step 6.8 Verification gate (own code, unit-tested) before anything is spoken:** every number, date and capitalized named entity in the answer must appear in the cited chunks (normalized, number-word aware); otherwise retry once with a stricter prompt, then fall back to the extractive answerer, then abstain. For numeric and current_event answers require support from >=2 distinct domains or one structured source with trust>=0.9 (Wikidata/Open-Meteo); otherwise attribute explicitly (“According to the BBC...”) or abstain. Length check: one sentence, under ~30 words. Abstention is short and natural: “I couldn’t find a reliable answer for that.”

**Step 6.9 Failure handling:** structured/trusted source -> DuckDuckGo -> OfflineWikiSource -> spoken abstention. `offline_mode` config flag and tray toggle: when on (or the network is down), wake word, VAD, ASR, local actions and TTS still work and web questions get “I’m offline, so I can only help with PC commands right now.”

**Step 6.10 Evaluation harness:** `tests/gold/questions.jsonl` (~100 questions I can extend: static facts, entity properties, current facts, numeric, follow-up pairs, noisy/ambiguous names, entertainment, sports, prices, unanswerable). `scripts/eval_rag.py` reports accuracy, abstention rate, hallucination rate (answer present but unsupported), median and p95 latency per stage, per source configuration (DuckDuckGo-only vs multi-source), and diffs against the previous run. A pytest fails if accuracy drops below the recorded baseline. Log every live query (class, rewrite, sources, scores, verification result, final answer) to `./logs` so failures become new gold cases.

Stream the verified one-liner straight to TTS. While searching show “Searching...” and play a subtle earcon; do not speak filler.

Implementation order: classifier -> Wikipedia/Wikidata source + reranker -> extractive answerer -> multi-source routing -> trusted-domain DuckDuckGo + full-page fetch -> news RSS -> LLM generation -> verification gate -> weather/structured paths -> optional offline Wikipedia -> benchmark and choose the default model.

### Phase 7: UI, packaging, README

UI: keep the iridescent orb look, but drive the waveform/orb from real mic RMS (listening) and real TTS RMS (speaking). The overlay must NOT steal focus (Tool window + WindowDoesNotAcceptFocus + `WA_ShowWithoutActivating`; remove `activateWindow()`). Show on the screen containing the cursor, handle high-DPI, render full transcript and response with word-wrap and scrolling (no 180-character truncation), draggable. System tray: Mute microphone, Sleep/Wake, Offline mode, Open logs, Open `config.toml`, Quit; setQuitOnLastWindowClosed(False); clean shutdown stops threads and releases audio devices. Optional global hotkey to wake without voice. Clear states: Sleeping, Listening, Thinking, Searching, Speaking, Error (short reason), plus a visible “mic muted” indicator.

README: Mermaid architecture diagram, setup steps, a table mapping each project objective (wake word, VAD, Whisper ASR, intent routing, RAG, lexical reranking) to the exact module, a “No API keys, ever” section with a table of every network call (what, why, what data is sent), the latest eval numbers, and honest limitations (a local 3–7B model is weaker than hosted models, which is why answers are strictly grounded and the assistant abstains when sources are weak; live prices and sports scores are the weakest categories).

## Verified repository state and loose ends (2026-10-21)

- Repository: `dakshkasana19/Chintu-AI-Assistant`, branch `master`. The existing pushed history has Phase 0, Phase 1, and Phase 2 commits.
- Phase 0 and Phase 1 foundations and Phase 2 wake implementation are documented in the repo. Command ASR, intent routing/actions, RAG, and TTS are not yet wired into this runtime.
- Current Phase 2 implementation is **Vosk restricted grammar followed by faster-whisper confirmation**, behind the detector abstraction, with Silero/WebRTC VAD upstream. It has not recognized the user's spoken “Chintu” reliably.
- In controlled live tests, Whisper sometimes transcribed “Chintu” exactly or phonetically, but Vosk returned empty/`[unk]` and Silero sometimes produced a low speech score. One reported Silero maximum was 0.044 against the configured threshold 0.5. Results varied between captures; this is evidence to investigate, not proof of a single root cause. No audio was saved.
- The user selected an experimental custom openWakeWord “Chintu” model approach. A WSL-isolated prototype was created and trained outside this repository. It **failed**: trainer metrics were accuracy 0.5, recall 0.0, false positives/hour 0.0; exported ONNX scores on held-out synthetic data were about 0.024 for both classes and detected 0/250 positive examples at threshold 0.5. Do not integrate that model or claim it works.
- To free disk space, the 17.28 GB ACAV100M feature file, the ~185 MB false-positive validation feature file, and the ~204 MB Piper TTS checkpoint were deleted from the isolated WSL workspace. Training is paused. The WSL training virtual environment (~6.9 GB) and small workspace/output files remain outside this repository. The failed ONNX/TFLite prototype exports also remain there; they are not part of the app repository.
- No Chintu app source code has been changed for the custom openWakeWord experiment. The app still uses the existing Vosk/Whisper wake stack.
- The startup overlay correction is pending in the current changes: microphone and wake detection stay active during standby, the UI remains hidden until confirmation, then appears. The user expects that behavior.
- Seven pre-existing uncommitted changes are in `README.md`, `chintu_assistant.py`, `docs/DEVLOG.md`, `docs/results/phase2-2026-10-16.md`, `logs/DEVLOG.md`, `tests/test_foundation.py`, and `tests/test_wake_phase2.py`. They preserve/test hidden-until-confirmed behavior and update the Phase 2 log/report from 32 to 33 passing tests.
- The user explicitly requires that the `logs/` folder not be edited or omitted from Git until told otherwise. Preserve existing files/history there; keep and commit its existing pending change rather than deleting, rewriting, or excluding it. Append future phase entries without erasing prior logs.

## Further instructions from the user

1. Fix the wake-word problem only after completing the discussion described at the top and getting my approval for the approach.
2. Do not edit or omit the `logs/` folder in the Git repository until I tell you to do so. Preserve existing logs and append entries as needed.
3. Before starting any fix, discuss the diagnosis and planned approach with me. Explain what you intend to do and ask whether that is what I want. Do not implement until I answer.
4. Do not fixate on the issue as originally described if evidence points elsewhere. Tell me when you believe another cause or subsystem is responsible, and show why.
5. Keep changes local-first, testable, and honest about measured recognition quality. Do not upload user audio.
6. Work from the current repository state and retain unrelated user changes. Do not discard or rewrite existing work.
