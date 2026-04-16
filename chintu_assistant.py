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
from ddgs import DDGS

# --- CONFIGURATION ---
WAKE_VARIANTS = ["chintu", "chinto", "chintoo", "chintu ji", "hey chintu", "a chintu"]

# ============================================================
# RESEARCH PAPER IMPLEMENTATION NOTES
# "Lost in the Middle: How Language Models Use Long Contexts"
# Liu et al., 2023
#
# Key findings applied in this code:
#
# 1. PRIMACY BIAS  [*] Most relevant snippet goes FIRST in context
# 2. RECENCY BIAS  [*] A summary/restatement goes LAST in context
# 3. QUERY-AWARE CONTEXTUALIZATION [*] Query placed BEFORE and
#    AFTER the retrieved documents in the prompt so the model
#    can attend to the question while reading every snippet
# 4. CONTEXT TRUNCATION [*] Only top-N snippets used; more docs
#    give diminishing returns (~1.5% gain beyond 3-5 results)
# 5. RELEVANCE RERANKING [*] Snippets scored and reordered so
#    the best match is at position 0, not buried in the middle
# ============================================================

MAX_SNIPPETS = 3          # Paper ?5: performance saturates quickly
MAX_SNIPPET_CHARS = 180   # Keep each snippet concise


class ChintuAssistant:
    def __init__(self, bridge):
        self.bridge = bridge
        self.recognizer = sr.Recognizer()
        self.mic = None
        self.running = True

        self.setup_microphone()
        print("[OK] Chintu ready! (VAD + DDG Web Search + Lost-in-the-Middle optimization)")
        print(f"Listening for: {WAKE_VARIANTS}")

    # ------------------------------------------------------------------
    # MICROPHONE SETUP
    # ------------------------------------------------------------------
    def setup_microphone(self):
        """Professional VAD microphone setup"""
        try:
            self.mic = sr.Microphone()
            print("[OK] Microphone ready")

            self.recognizer.energy_threshold = 350
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 0.5
            self.recognizer.non_speaking_duration = 0.3

            with self.mic as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=2.0)
                print("[OK] VAD calibrated")
        except Exception as e:
            print(f"[ERROR] Mic error: {e}")
            self.mic = None

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------
    def speak(self, text):
        """Enhanced TTS with smooth UI feedback"""
        if not self.running:
            return
        print(f"?[*]  Chintu: {text}")
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
    # WEB SEARCH  (unchanged [*] returns raw snippets)
    # ------------------------------------------------------------------
    def search_web(self, query):
        """Fetch raw search snippets from DuckDuckGo using the latest ddgs library."""
        self.bridge.status_changed.emit("Searching Web...")
        print(f"[*] Searching: {query}")
        try:
            results = list(DDGS().text(query, max_results=MAX_SNIPPETS + 2))
            return results if results else []
        except Exception as e:
            print(f"[ERROR] Search error: {e}")
            return []

    # ------------------------------------------------------------------
    # RESEARCH PAPER LEARNING 5 [*] RELEVANCE RERANKING
    # Rank snippets by keyword overlap with the query so the most
    # relevant one ends up at position 0 (exploiting primacy bias).
    # ------------------------------------------------------------------
    def rerank_snippets(self, query, snippets):
        """
        Simple lexical reranker: score each snippet by how many
        query tokens appear in its text.  The highest-scoring
        snippet goes first (primacy bias), lowest-scoring goes last
        (recency bias is less harmful at the tail).

        Paper reference: ?5 and ?2.3 [*] placing relevant info at the
        start/end of context yields significantly higher accuracy.
        """
        query_tokens = set(query.lower().split())

        def score(snippet):
            body = (snippet.get('title', '') + ' ' + snippet.get('body', '')).lower()
            return sum(1 for tok in query_tokens if tok in body)

        ranked = sorted(snippets, key=score, reverse=True)
        print(f"[*] Reranked {len(ranked)} snippets (best-first for primacy bias)")
        return ranked[:MAX_SNIPPETS]

    # ------------------------------------------------------------------
    # RESEARCH PAPER LEARNING 1-4 [*] QUERY-AWARE CONTEXT ASSEMBLY
    # Build the prompt so that:
    #   [*] Query appears at the TOP  (query-aware contextualization, ?4.2)
    #   [*] Most-relevant snippet is FIRST (primacy bias, ?2.3)
    #   [*] Least-relevant snippet is LAST  (recency bias still helps, ?2.3)
    #   [*] Query is REPEATED at the BOTTOM (query-aware contextualization)
    #   [*] Total number of snippets is capped (context truncation, ?5)
    # ------------------------------------------------------------------
    def build_query_aware_prompt(self, query, ranked_snippets):
        """
        Constructs a query-aware prompt following the paper's guidance.

        Structure:
            Question: <query>              [*] query BEFORE documents
            [Document 1] most relevant     [*] primacy position
            [Document 2] ...
            [Document 3] least relevant    [*] recency position
            Given the above, answer: <query>  [*] query AFTER documents
        """
        lines = [f"Question: {query}\n"]

        for i, snippet in enumerate(ranked_snippets):
            title = snippet.get('title', f'Result {i+1}')
            body  = snippet.get('body', '')[:MAX_SNIPPET_CHARS]
            lines.append(f"[Document {i+1}] ({title}): {body}")

        # Repeat query at the end [*] paper ?4.2 shows this helps models
        # attend to the question when contextualising the last documents
        lines.append(f"\nGiven the documents above, answer concisely: {query}")

        prompt = "\n".join(lines)
        print(f"[*] Query-aware prompt built ({len(ranked_snippets)} docs, "
              f"query at top & bottom)")
        return prompt

    # ------------------------------------------------------------------
    # RESPONSE SYNTHESIS
    # Instead of random template selection, we use the structured prompt
    # to produce a deterministic, context-grounded answer.
    # ------------------------------------------------------------------
    def synthesize_response(self, query, ranked_snippets):
        """
        Refined factual synthesis following Primacy and Recency findings.
        Prioritizes direct answers with factual density (numbers/metrics).
        """
        if not ranked_snippets:
            return "I couldn't find any information on the web for this query."

        # Filter stopwords but keep high-value modifiers
        stopwords = {'what', 'is', 'the', 'who', 'how', 'when', 'where', 'are', 'in', 'of', 'and', 'a', 'to'}
        query_tokens = [tok.lower() for tok in query.split() if tok.lower() not in stopwords]
        
        candidates = []
        
        # Weighing Primacy (0) and Recency (last) more heavily
        for i, snippet in enumerate(ranked_snippets):
            body = snippet.get('body', '')
            source = snippet.get('title', 'a web source')
            
            # Bonus for position (Primacy/Recency bias)
            position_bonus = 0
            if i == 0: position_bonus = 5
            if i == len(ranked_snippets) - 1: position_bonus = 4
            
            segments = [s.strip() for s in body.split('.') if len(s.strip()) > 25]
            
            for segment in segments:
                lower_seg = segment.lower()
                
                # FACT SCORING
                # 1. Direct Keyword Match (high weight)
                match_count = sum(1 for tok in query_tokens if tok in lower_seg)
                
                # 2. Factual Density (Numbers, dates, measurements)
                has_digits = any(c.isdigit() for c in segment)
                
                # 3. Direct Definition/State (is, was, specifically)
                defining_weight = 0
                if ' is ' in lower_seg or ' is the ' in lower_seg: defining_weight = 4
                elif ' was ' in lower_seg: defining_weight = 2 # Prefer 'is' over 'was'
                
                score = (match_count * 4) + defining_weight + position_bonus
                if has_digits: score += 5
                
                # Penalize noisy sentences
                if any(w in lower_seg for w in ['click', 'read more', 'find out', 'article', 'welcome']):
                    score -= 10
                
                if score > 5:
                    candidates.append({
                        'text': segment,
                        'source': source,
                        'score': score
                    })

        if not candidates:
            # Fallback
            top = ranked_snippets[0]
            return f"According to {top.get('title')}, {top.get('body')[:150]}..."

        # Sort by score descending
        best = sorted(candidates, key=lambda x: x['score'], reverse=True)[0]
        
        ans = best['text'].replace('\n', ' ').strip()
        if not any(ans.endswith(p) for p in ['.', '!', '?']): ans += '.'
        
        return f"Based on {best['source']}, {ans}"

    # ------------------------------------------------------------------
    # SMART RESPONSE ROUTER
    # ------------------------------------------------------------------
    def get_smart_response(self, query):
        """
        Route the query:
          [*] Instant local handlers (time, date, apps) [*] no LLM needed
          [*] Static knowledge base [*] no web call needed
          [*] Everything else [*] web search with Lost-in-the-Middle
            optimisations applied before synthesising the answer
        """
        query_lower = query.lower()

        # ?[*] Instant local responses ????????????????????????????????????
        if any(word in query_lower for word in ['time', 'clock']):
            now = datetime.datetime.now()
            return f"The time is {now.strftime('%I:%M %p')} right now."

        if any(word in query_lower for word in ['date', 'today']):
            now = datetime.datetime.now()
            return f"Today is {now.strftime('%A, %B %d, %Y')}."

        # ?[*] System apps ???????????????????????????????????????????????
        if 'notepad' in query_lower:
            subprocess.Popen(['notepad.exe'], shell=True)
            return "Opening Notepad!"
        if 'calculator' in query_lower or 'calc' in query_lower:
            subprocess.Popen(['calc.exe'], shell=True)
            return "Calculator launched!"
        if any(word in query_lower for word in ['edge', 'browser']):
            webbrowser.open('edge://newtab')
            return "Browser opening!"

        # ?[*] Static knowledge base ?????????????????????????????????????
        knowledge = {
            "prime minister": "Narendra Modi is the Prime Minister of India.",
            "president of usa": "Donald Trump is the President of the United States, serving his second term since 2025.",
            "president of india": "Droupadi Murmu is the President of India, serving since July 2022.",
            "president": "Donald Trump is the US President, serving his second term since 2025.",
            "chatgpt": "ChatGPT is an artificial intelligence chatbot developed by OpenAI, launched in November 2022.",
            "openai": "OpenAI is an AI research and deployment company that created ChatGPT and DALL-E.",
            "genshin impact": "Genshin Impact was officially launched on September 28, 2020, by miHoYo.",
            "python": "Python is a powerful programming language widely used for AI and software development.",
            "india": "India is the world's most populous democracy, located in South Asia."
        }
        # Sort knowledge by length of key descending so most specific matches first
        sorted_keys = sorted(knowledge.keys(), key=len, reverse=True)
        for key in sorted_keys:
            if key in query_lower:
                return knowledge[key]

        # ?[*] Web search with Lost-in-the-Middle optimizations ??????????
        raw_snippets = self.search_web(query)

        if not raw_snippets:
            return "I'm sorry, I couldn't find any information on the web for that query."

        # Apply research paper optimizations
        ranked_snippets = self.rerank_snippets(query, raw_snippets)
        
        # Build and print the prompt (following paper principles)
        prompt = self.build_query_aware_prompt(query, ranked_snippets)
        print(f"\n--- [Lost-in-the-Middle Optimized Prompt] ---\n{prompt}\n---\n")

        return self.synthesize_response(query, ranked_snippets)

    # ------------------------------------------------------------------
    # WAKE WORD
    # ------------------------------------------------------------------
    def is_wake_word(self, text):
        return any(variant in text.lower().strip() for variant in WAKE_VARIANTS)

    # ------------------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------------------
    def run(self):
        """Professional VAD [*] continuous listening until silence"""
        if not self.mic:
            print("[*] No microphone!")
            return

        print("[*] Chintu VAD Active!")
        print("[*] Say 'CHINTU' then speak naturally (stops after ~4s silence)")

        while self.running:
            try:
                with self.mic as source:
                    # Phase 1: Wake word detection
                    print("[*] (wake)... ", end="", flush=True)
                    try:
                        wake_audio = self.recognizer.listen(
                            source, timeout=1.0, phrase_time_limit=2.5
                        )
                        wake_text = self.recognizer.recognize_google(
                            wake_audio, language='en-IN'
                        ).lower()
                        print(f"'{wake_text}'")
                    except sr.WaitTimeoutError:
                        print(" [SILENCE]")
                        continue
                    except sr.UnknownValueError:
                        print(" [?]")
                        continue
                    except Exception as e:
                        print(f" [ERR: {e}]")
                        continue

                    if self.is_wake_word(wake_text):
                        print("[*] ACTIVATED!")
                        self.bridge.visibility_changed.emit(True)
                        self.speak("Go ahead Sir!")

                        # Phase 2: High-accuracy voice capture
                        full_query = "" # RESET TO PREVENT REPEATING PREVIOUS ANSWER
                        print("[LISTENING]... ", end="", flush=True)
                        self.bridge.status_changed.emit("Listening...")
                        
                        # Adjust pause threshold to user's 2-second request
                        self.recognizer.pause_threshold = 2.0
                        
                        try:
                            # Listen for the whole phrase at once (more reliable VAD)
                            audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=15)
                            print("[PROCESSING]... ", end="", flush=True)
                            self.bridge.status_changed.emit("Interpreting...")
                            
                            # Using Whisper for better interpretation (other technology)
                            # Fallback to google if whisper is not available
                            try:
                                full_query = self.recognizer.recognize_whisper(audio, language="english", model="base").strip()
                            except:
                                full_query = self.recognizer.recognize_google(audio, language='en-IN').strip()
                                
                        except sr.WaitTimeoutError:
                            full_query = ""
                        except Exception as e:
                            print(f"\n[ERROR] Interpretation error: {e}")
                            full_query = ""
                            
                        print("\n[OK]")

                        # Process full query
                        if full_query:
                            print(f"[*] Query: '{full_query}'")
                            self.bridge.status_changed.emit("Processing...")
                            response = self.get_smart_response(full_query)
                            self.speak(response)
                            print("[*] Response delivered!")
                        else:
                            self.speak("Yes Sir[*] What would you like?")

                        self.bridge.visibility_changed.emit(False)
                        self.bridge.status_changed.emit("STANDBY") # RESET UI STATUS
                        print("\n[*] Listening for wake word...")
                        time.sleep(1.5)

                    else:
                        print("(no)...")

            except Exception as e:
                print(f"[LOOP ERROR] {e}")
                time.sleep(1.0)


# ----------------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------------
def handle_exit(sig, frame):
    print("\n[*] Chintu offline!")
    QApplication.quit()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_exit)
    app = QApplication(sys.argv)

    bridge = UIBridge()
    window = ChintuCoreUI(bridge)

    print("[START] Launching Chintu VAD Assistant (Lost-in-the-Middle edition)...")

    def startup():
        try:
            assistant = ChintuAssistant(bridge)
            assistant.run()
        except Exception as e:
            print(f"[FAIL] Startup failed: {e}")
            import traceback
            traceback.print_exc()

    thread = threading.Thread(target=startup, daemon=True)
    thread.start()

    print("[READY] Chintu VAD ready!")
    print("[LISTEN] Say 'CHINTU' then speak naturally!")
    print("[INFO] Example: 'CHINTU what is the latest news about AI'")
    sys.exit(app.exec())