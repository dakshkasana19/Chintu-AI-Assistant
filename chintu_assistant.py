import sys
import threading
import time
import signal
import os
import speech_recognition as sr
import pyttsx3
import datetime
import webbrowser
import subprocess
import random
import numpy as np
from PyQt6.QtWidgets import QApplication
from chintu_ui import ChintuCoreUI, UIBridge
from duckduckgo_search import DDGS

# --- CONFIGURATION ---
WAKE_VARIANTS = ["chintu", "chinto", "chintoo", "chintu ji"]

# ============================================================
# RESEARCH PAPER IMPLEMENTATION NOTES
# "Lost in the Middle: How Language Models Use Long Contexts"
# Liu et al., 2023
#
# Key findings applied in this code:
#
# 1. PRIMACY BIAS  → Most relevant snippet goes FIRST in context
# 2. RECENCY BIAS  → A summary/restatement goes LAST in context
# 3. QUERY-AWARE CONTEXTUALIZATION → Query placed BEFORE and
#    AFTER the retrieved documents in the prompt so the model
#    can attend to the question while reading every snippet
# 4. CONTEXT TRUNCATION → Only top-N snippets used; more docs
#    give diminishing returns (~1.5% gain beyond 3-5 results)
# 5. RELEVANCE RERANKING → Snippets scored and reordered so
#    the best match is at position 0, not buried in the middle
# ============================================================

MAX_SNIPPETS = 3          # Paper §5: performance saturates quickly
MAX_SNIPPET_CHARS = 180   # Keep each snippet concise


class ChintuAssistant:
    def __init__(self, bridge):
        self.bridge = bridge
        self.recognizer = sr.Recognizer()
        self.mic = None
        self.running = True

        self.setup_microphone()
        print("✅ Chintu ready! (VAD + DDG Web Search + Lost-in-the-Middle optimisation)")
        print(f"Listening for: {WAKE_VARIANTS}")

    # ------------------------------------------------------------------
    # MICROPHONE SETUP
    # ------------------------------------------------------------------
    def setup_microphone(self):
        """Professional VAD microphone setup"""
        try:
            self.mic = sr.Microphone()
            print("✅ Microphone ready")

            self.recognizer.energy_threshold = 350
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 0.5
            self.recognizer.non_speaking_duration = 0.3

            with self.mic as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=2.0)
                print("✅ VAD calibrated")
        except Exception as e:
            print(f"❌ Mic error: {e}")
            self.mic = None

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------
    def speak(self, text):
        """Enhanced TTS with smooth UI feedback"""
        if not self.running:
            return
        print(f"🗣️  Chintu: {text}")
        self.bridge.status_changed.emit("Speaking...")
        try:
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            engine.setProperty('voice', voices[1].id if len(voices) > 1 else voices[0].id)
            engine.setProperty('rate', 175)
            engine.say(text[:250])
            engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")
        self.bridge.status_changed.emit("STANDBY")

    # ------------------------------------------------------------------
    # WEB SEARCH  (unchanged — returns raw snippets)
    # ------------------------------------------------------------------
    def search_web(self, query):
        """Fetch raw search snippets from DuckDuckGo."""
        self.bridge.status_changed.emit("Searching Web...")
        print(f"🔍 Searching: {query}")
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=MAX_SNIPPETS + 2))
                # Return list of dicts so the ranker can work with them
                return results if results else []
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []

    # ------------------------------------------------------------------
    # RESEARCH PAPER LEARNING 5 — RELEVANCE RERANKING
    # Rank snippets by keyword overlap with the query so the most
    # relevant one ends up at position 0 (exploiting primacy bias).
    # ------------------------------------------------------------------
    def rerank_snippets(self, query, snippets):
        """
        Simple lexical reranker: score each snippet by how many
        query tokens appear in its text.  The highest-scoring
        snippet goes first (primacy bias), lowest-scoring goes last
        (recency bias is less harmful at the tail).

        Paper reference: §5 and §2.3 — placing relevant info at the
        start/end of context yields significantly higher accuracy.
        """
        query_tokens = set(query.lower().split())

        def score(snippet):
            body = (snippet.get('title', '') + ' ' + snippet.get('body', '')).lower()
            return sum(1 for tok in query_tokens if tok in body)

        ranked = sorted(snippets, key=score, reverse=True)
        print(f"🔢 Reranked {len(ranked)} snippets (best-first for primacy bias)")
        return ranked[:MAX_SNIPPETS]

    # ------------------------------------------------------------------
    # RESEARCH PAPER LEARNING 1-4 — QUERY-AWARE CONTEXT ASSEMBLY
    # Build the prompt so that:
    #   • Query appears at the TOP  (query-aware contextualization, §4.2)
    #   • Most-relevant snippet is FIRST (primacy bias, §2.3)
    #   • Least-relevant snippet is LAST  (recency bias still helps, §2.3)
    #   • Query is REPEATED at the BOTTOM (query-aware contextualization)
    #   • Total number of snippets is capped (context truncation, §5)
    # ------------------------------------------------------------------
    def build_query_aware_prompt(self, query, ranked_snippets):
        """
        Constructs a query-aware prompt following the paper's guidance.

        Structure:
            Question: <query>              ← query BEFORE documents
            [Document 1] most relevant     ← primacy position
            [Document 2] ...
            [Document 3] least relevant    ← recency position
            Given the above, answer: <query>  ← query AFTER documents
        """
        lines = [f"Question: {query}\n"]

        for i, snippet in enumerate(ranked_snippets):
            title = snippet.get('title', f'Result {i+1}')
            body  = snippet.get('body', '')[:MAX_SNIPPET_CHARS]
            lines.append(f"[Document {i+1}] ({title}): {body}")

        # Repeat query at the end — paper §4.2 shows this helps models
        # attend to the question when contextualising the last documents
        lines.append(f"\nGiven the documents above, answer concisely: {query}")

        prompt = "\n".join(lines)
        print(f"📝 Query-aware prompt built ({len(ranked_snippets)} docs, "
              f"query at top & bottom)")
        return prompt

    # ------------------------------------------------------------------
    # RESPONSE SYNTHESIS
    # Instead of random template selection, we use the structured prompt
    # to produce a deterministic, context-grounded answer.
    # ------------------------------------------------------------------
    def synthesize_response(self, query, ranked_snippets):
        """
        Use the top snippet's body as the primary answer source
        (primacy position) with a brief fallback to the last snippet
        (recency position) if the top one is too short.

        In a production system this prompt would be sent to an LLM.
        Here we extract the best sentence directly, which mirrors
        what a retrieval-augmented LLM would do after reading the
        query-aware prompt.
        """
        if not ranked_snippets:
            return "I couldn't find current information on that topic."

        # Primary source = position 0 (primacy)
        primary = ranked_snippets[0].get('body', '')[:MAX_SNIPPET_CHARS]
        source  = ranked_snippets[0].get('title', 'a web source')

        # Fallback = last position (recency) if primary is too short
        if len(primary) < 40 and len(ranked_snippets) > 1:
            primary = ranked_snippets[-1].get('body', '')[:MAX_SNIPPET_CHARS]
            source  = ranked_snippets[-1].get('title', 'another source')

        # Trim to a single clean sentence for TTS
        sentence = primary.split('.')[0].strip()
        if not sentence:
            sentence = primary[:120]

        return f"According to {source}: {sentence}."

    # ------------------------------------------------------------------
    # SMART RESPONSE ROUTER
    # ------------------------------------------------------------------
    def get_smart_response(self, query):
        """
        Route the query:
          • Instant local handlers (time, date, apps) — no LLM needed
          • Static knowledge base — no web call needed
          • Everything else → web search with Lost-in-the-Middle
            optimisations applied before synthesising the answer
        """
        query_lower = query.lower()

        # ── Instant local responses ────────────────────────────────────
        if any(word in query_lower for word in ['time', 'clock']):
            now = datetime.datetime.now()
            return f"The time is {now.strftime('%I:%M %p')} right now."

        if any(word in query_lower for word in ['date', 'today']):
            now = datetime.datetime.now()
            return f"Today is {now.strftime('%A, %B %d, %Y')}."

        # ── System apps ───────────────────────────────────────────────
        if 'notepad' in query_lower:
            subprocess.Popen(['notepad.exe'], shell=True)
            return "Opening Notepad!"
        if 'calculator' in query_lower or 'calc' in query_lower:
            subprocess.Popen(['calc.exe'], shell=True)
            return "Calculator launched!"
        if any(word in query_lower for word in ['edge', 'browser']):
            webbrowser.open('edge://newtab')
            return "Browser opening!"

        # ── Static knowledge base ─────────────────────────────────────
        knowledge = {
            "prime minister": "Narendra Modi is India's Prime Minister since 2014.",
            "president":      "Donald Trump is US President, serving second term since 2025.",
            "gandhi":         "Mahatma Gandhi led India's independence through non-violence.",
            "python":         "Python is your favourite programming language for AI and software engineering.",
            "india":          "India has 1.4+ billion people, the world's largest democracy."
        }
        for key, answer in knowledge.items():
            if key in query_lower:
                return answer

        # ── Web search with Lost-in-the-Middle optimisations ──────────
        raw_snippets = self.search_web(query)

        if not raw_snippets:
            return "Web search returned no results for that query."

        # PAPER LEARNING 5: Rerank — best snippet goes to position 0
        ranked_snippets = self.rerank_snippets(query, raw_snippets)

        # PAPER LEARNINGS 1-4: Build query-aware prompt
        # (In production, send this prompt to an LLM API)
        prompt = self.build_query_aware_prompt(query, ranked_snippets)
        print(f"\n--- Query-aware prompt (for LLM / debug) ---\n{prompt}\n---\n")

        # Synthesise a spoken response from the ranked context
        response = self.synthesize_response(query, ranked_snippets)
        return response

    # ------------------------------------------------------------------
    # WAKE WORD
    # ------------------------------------------------------------------
    def is_wake_word(self, text):
        return any(variant in text.lower().strip() for variant in WAKE_VARIANTS)

    # ------------------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------------------
    def run(self):
        """Professional VAD — continuous listening until silence"""
        if not self.mic:
            print("❌ No microphone!")
            return

        print("🚀 Chintu VAD Active!")
        print("💡 Say 'CHINTU' then speak naturally (stops after ~4s silence)")

        while self.running:
            try:
                with self.mic as source:
                    # Phase 1: Wake word detection
                    print("👂 (wake)... ", end="", flush=True)
                    wake_audio = self.recognizer.listen(
                        source, timeout=1.2, phrase_time_limit=2.2
                    )
                    wake_text = self.recognizer.recognize_google(
                        wake_audio, language='en-IN'
                    ).lower()
                    print(f"'{wake_text}'")

                    if self.is_wake_word(wake_text):
                        print("🔥 ACTIVATED!")
                        self.bridge.visibility_changed.emit(True)
                        self.speak("Go ahead Sir!")

                        # Phase 2: Continuous voice capture
                        query_parts  = []
                        silence_count = 0
                        SILENCE_LIMIT = 8  # ~4 seconds silence

                        print("🎤 (talking)... ", end="")
                        self.bridge.status_changed.emit("Listening...")

                        while silence_count < SILENCE_LIMIT and self.running:
                            try:
                                chunk_audio = self.recognizer.listen(
                                    source, timeout=0.45, phrase_time_limit=2.8
                                )
                                chunk_text = self.recognizer.recognize_google(
                                    chunk_audio, language='en-IN'
                                )
                                if chunk_text.strip():
                                    query_parts.append(chunk_text)
                                    silence_count = 0
                                    print(f"'{chunk_text}' ", end="")
                                else:
                                    silence_count += 1
                                    print(".", end="", flush=True)
                            except sr.WaitTimeoutError:
                                silence_count += 1
                                print(".", end="", flush=True)
                            except sr.UnknownValueError:
                                silence_count += 1
                                print("?", end="", flush=True)

                        print("\n✅")

                        # Process full query
                        full_query = ' '.join(query_parts).strip()
                        if full_query:
                            print(f"❓ Query: '{full_query}'")
                            self.bridge.status_changed.emit("Processing...")
                            response = self.get_smart_response(full_query)
                            self.speak(response)
                            print("✅ Response delivered!")
                        else:
                            self.speak("Yes Sir? What would you like?")

                        self.bridge.visibility_changed.emit(False)
                        print("\n👂 Listening for wake word...")
                        time.sleep(1.5)

                    else:
                        print("(no)...")

            except Exception as e:
                print(f"⚠️ Error: {e}")
                time.sleep(0.5)


# ----------------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------------
def handle_exit(sig, frame):
    print("\n👋 Chintu offline!")
    QApplication.quit()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_exit)
    app = QApplication(sys.argv)

    bridge = UIBridge()
    window = ChintuCoreUI(bridge)

    print("🚀 Launching Chintu VAD Assistant (Lost-in-the-Middle edition)...")

    def startup():
        try:
            assistant = ChintuAssistant(bridge)
            assistant.run()
        except Exception as e:
            print(f"💥 Startup failed: {e}")
            import traceback
            traceback.print_exc()

    thread = threading.Thread(target=startup, daemon=True)
    thread.start()

    print("✅ Chintu VAD ready!")
    print("🎙️  Say 'CHINTU' then speak naturally!")
    print("💡 Example: 'CHINTU what is the latest news about AI'")
    sys.exit(app.exec())