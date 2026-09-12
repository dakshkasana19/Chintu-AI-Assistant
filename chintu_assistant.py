import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')

import threading
import time
import signal
import re
import datetime
import webbrowser
import subprocess
import requests
import asyncio
import tempfile
import edge_tts
import pygame
import speech_recognition as sr
import pyttsx3
from PyQt6.QtWidgets import QApplication
from chintu_ui import ChintuCoreUI, UIBridge
from ddgs import DDGS

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Optional Gemini support if API key is provided
GEMINI_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
gemini_model = None
if GEMINI_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_KEY)
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        print("[AI] Gemini Generative AI enabled for advanced synthesis.")
    except Exception as e:
        print(f"[AI] Gemini init notice: {e}")

# --- CONFIGURATION ---
WAKE_VARIANTS = [
    "chintu", "hey chintu", "chinto", "chintoo", "chintu ji",
    "ok chintu", "a chintu", "hello chintu", "hi chintu", "wake up chintu"
]

EXIT_PHRASES = [
    "exit", "sleep", "go to sleep", "goodbye", "bye chintu", "bye bye",
    "quit", "stop listening", "dismiss", "close chintu", "shut down",
    "sleep mode", "go to sleep chintu"
]


class ChintuAssistant:
    """
    Chintu - Siri for PC Voice Assistant
    - Standby Sleep Mode with Wake Word Activation
    - Continuous Multi-turn Conversational Loop (Active Mode)
    - Dynamic Live Web Search & Summarization (Zero hardcoded answers)
    - Windows PC Utility Commands (Apps, System, Web shortcuts)
    - Exit command triggers Sleep Mode
    """
    def __init__(self, bridge):
        self.bridge = bridge
        self.recognizer = sr.Recognizer()
        self.mic = None
        self.running = True
        self.tts_lock = threading.Lock()

        self.setup_microphone()
        print("[OK] Chintu AI Assistant (Siri for PC edition) Ready!")
        print(f"[*] Wake words: {WAKE_VARIANTS[:4]}...")
        print(f"[*] Exit words: {EXIT_PHRASES[:4]}...")

    # ------------------------------------------------------------------
    # MICROPHONE SETUP & CALIBRATION
    # ------------------------------------------------------------------
    def setup_microphone(self):
        try:
            self.mic = sr.Microphone()
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 1.0
            self.recognizer.non_speaking_duration = 0.3

            with self.mic as source:
                print("[*] Calibrating microphone for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1.5)
                print("[OK] Microphone calibrated and ready.")
        except Exception as e:
            print(f"[ERROR] Mic setup error: {e}")
            self.mic = None

    # ------------------------------------------------------------------
    # TTS (ULTRA-REALISTIC NEURAL TEXT TO SPEECH)
    # ------------------------------------------------------------------
    def speak(self, text):
        """Converts text to spoken audio with natural Microsoft Neural Voice."""
        if not self.running or not text:
            return

        speech_text = self.clean_snippet(text)
        speech_text = re.sub(r'[*_#`~\[\]()]', '', speech_text)
        speech_text = re.sub(r'https?://\S+', '', speech_text).strip()
        if not speech_text:
            return

        print(f"\n[CHINTU]: {speech_text}")
        self.bridge.status_changed.emit("Speaking...")

        # 1. Try Microsoft Edge Neural Voice (ultra-lifelike Siri-quality voice)
        try:
            temp_mp3 = os.path.join(tempfile.gettempdir(), f"chintu_{int(time.time()*1000)}.mp3")

            async def _synth():
                comm = edge_tts.Communicate(speech_text, "en-US-AriaNeural", rate="+4%")
                await comm.save(temp_mp3)

            asyncio.run(_synth())

            if os.path.exists(temp_mp3):
                pygame.mixer.init()
                pygame.mixer.music.load(temp_mp3)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy() and self.running:
                    time.sleep(0.04)
                pygame.mixer.music.unload()
                pygame.mixer.quit()
                try:
                    os.remove(temp_mp3)
                except Exception:
                    pass
                return
        except Exception as e:
            print(f"[Neural TTS Notice]: {e}")

        # 2. Fallback to pyttsx3 offline engine
        with self.tts_lock:
            try:
                engine = pyttsx3.init()
                voices = engine.getProperty('voices')
                selected_voice = voices[0].id
                for v in voices:
                    if 'zira' in v.name.lower() or 'female' in v.name.lower() or 'hazel' in v.name.lower():
                        selected_voice = v.id
                        break
                engine.setProperty('voice', selected_voice)
                engine.setProperty('rate', 180)
                engine.say(speech_text[:350])
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                print(f"[TTS Error]: {e}")

    # ------------------------------------------------------------------
    # WEB SEARCH & DYNAMIC INTELLIGENCE PIPELINE
    # ------------------------------------------------------------------
    def clean_snippet(self, text):
        if not text:
            return ""
        # 1. Remove date prefixes: "Month Day, Year - ", "Day Month Year - ", "X days ago - "
        date_pattern = (
            r'^\s*(?:'
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{2,4}|'
            r'\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*,?\s+\d{2,4}|'
            r'\d+\s+(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\s+ago|'
            r'\d{4}\s*[-—]\s*\d{4}'
            r')\s*[-—–·•:|\.]\s*'
        )
        text = re.sub(date_pattern, '', text, flags=re.IGNORECASE)
        # 2. Strip any leftover date stamps at the start
        text = re.sub(r'^(?:[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})\s*[-—–·•]\s*', '', text)
        # 3. Remove conversational filler
        text = re.sub(r'^(?:Well,\s*|Basically,\s*|Actually,\s*)', '', text, flags=re.IGNORECASE)
        # 4. Remove phonetic pronunciation guides e.g. (/ˈɒtəwə/) or (listen)
        text = re.sub(r'\s*\([^)]*(?:/|ˈ|ˌ|ə|ɛ|ɪ|ʊ|ʌ|æ|ɑ|ɔ|θ|ð|ʃ|ʒ|ŋ|ʔ|listen)[^)]*\)', '', text)
        # 5. Normalize non-ascii characters to ascii for TTS & console safety
        text = text.encode('ascii', 'ignore').decode('ascii')
        text = re.sub(r'[\t]+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def query_wikipedia(self, query):
        """Fetch authoritative, review-free summary from Wikipedia full-text search API."""
        try:
            clean_q = re.sub(
                r'^(who|what|when|where|why|how|is|are|was|were|tell|me|about|the|a|an)\s+',
                '', query.strip(), flags=re.IGNORECASE
            ).strip(' ?.')
            if not clean_q:
                clean_q = query.strip(' ?.')

            search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={requests.utils.quote(clean_q)}&format=json&utf8=1"
            res = requests.get(search_url, headers={'User-Agent': 'ChintuAI/2.0'}, timeout=3.0).json()
            items = res.get('query', {}).get('search', [])
            if items:
                best_title = items[0]['title']
                summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(best_title)}"
                s_res = requests.get(summary_url, headers={'User-Agent': 'ChintuAI/2.0'}, timeout=3.0).json()
                extract = s_res.get('extract')
                if extract and len(extract) > 30:
                    clean_extract = self.clean_snippet(extract)
                    sentences = re.split(r'(?<=[.!?])\s+', clean_extract)
                    return " ".join(sentences[:2]).strip()
        except Exception:
            pass
        return None

    def search_duckduckgo(self, query):
        """Fetch live web results using DDGS."""
        try:
            return list(DDGS().text(query, max_results=5))
        except Exception as e:
            print(f"[DDG Search Notice]: {e}")
            return []

    def synthesize_summary(self, query, ddg_results):
        """Synthesize a concise, factual, review-free answer from search snippets."""
        if not ddg_results:
            return "I searched the web, but couldn't find a direct answer for that query."

        # If Gemini AI is active, use it to generate the perfect spoken summary
        if gemini_model:
            try:
                context = "\n".join([f"- {r.get('title')}: {r.get('body')}" for r in ddg_results[:4]])
                prompt = (
                    f"User asked: '{query}'\n\n"
                    f"Web search results:\n{context}\n\n"
                    "Think about the question, understand the facts, and provide a direct, concise, natural 1-2 sentence spoken answer like Siri. "
                    "Do NOT read out reviews, author names, dates, or citations."
                )
                ai_resp = gemini_model.generate_content(prompt)
                if ai_resp and ai_resp.text:
                    return self.clean_snippet(ai_resp.text.strip())
            except Exception as e:
                print(f"[Gemini fallback]: {e}")

        # Built-in zero-config intelligent NLP extraction
        stopwords = {'what', 'is', 'the', 'who', 'how', 'when', 'where', 'are', 'in', 'of', 'and', 'a', 'to', 'for', 'was', 'tell', 'me', 'about'}
        query_words = set(re.findall(r'\w+', query.lower())) - stopwords

        candidates = []
        for r in ddg_results:
            body = self.clean_snippet(r.get('body', ''))
            sentences = re.split(r'(?<=[.!?])\s+', body)
            for s in sentences:
                s = self.clean_snippet(s)
                if len(s) < 22 or len(s) > 230:
                    continue
                s_lower = s.lower()
                
                # Skip navigation, junk text, metadata, and user reviews / subjective opinions
                junk_markers = [
                    'cookie', 'subscribe', 'sign in', 'click here', 'privacy policy',
                    'read more', 'key words', 'keywords', 'doi:', 'abstract:', 'issn:',
                    'reviewer', 'critics', 'allmusic', 'imdb', 'in my opinion', 'i think',
                    'i am not going to be original', 'user review', 'star rating', 'blog post',
                    'commented', 'wrote that'
                ]
                if any(junk in s_lower for junk in junk_markers):
                    continue

                s_words = set(re.findall(r'\w+', s_lower))
                overlap = len(query_words.intersection(s_words))
                score = overlap * 4

                # Bonus for definitive, factual sentence structures
                if any(s_lower.startswith(w) for w in ['it is', 'he is', 'she is', 'they are', 'this is']):
                    score += 4
                if ' is a ' in s_lower or ' is an ' in s_lower or ' was a ' in s_lower:
                    score += 5
                if any(c.isdigit() for c in s):
                    score += 2

                candidates.append((score, s))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_sentence = candidates[0][1]
            if not any(best_sentence.endswith(p) for p in ['.', '!', '?']):
                best_sentence += '.'
            return best_sentence

        # Fallback to first snippet body with dates removed
        fallback = self.clean_snippet(ddg_results[0].get('body', ''))
        return fallback[:160] + ("." if not fallback.endswith(".") else "")

    def answer_web_query(self, query):
        """Unified web intelligence pipeline with entity comprehension."""
        self.bridge.status_changed.emit("Thinking...")
        query_clean = query.strip()

        # 1. Try Wikipedia first for high-quality encyclopedic entity definitions (zero reviews/dates)
        wiki_ans = self.query_wikipedia(query_clean)
        if wiki_ans and len(wiki_ans) > 25:
            return wiki_ans

        # 2. Live Web Search via DuckDuckGo with deep NLP synthesis
        ddg_results = self.search_duckduckgo(query_clean)
        return self.synthesize_summary(query_clean, ddg_results)

    # ------------------------------------------------------------------
    # HIGH-ACCURACY SPEECH TRANSCRIPTION (WHISPER + GOOGLE HYBRID)
    # ------------------------------------------------------------------
    def transcribe_audio(self, audio):
        """Multi-tier speech recognition with Whisper neural engine and Google fallback."""
        if not audio:
            return ""

        # 1. Try Google en-US (cleanest recognition for standard English, Western names, TV shows)
        try:
            text = self.recognizer.recognize_google(audio, language='en-US').strip()
            if text:
                return text
        except Exception:
            pass

        # 2. Try OpenAI Whisper (local neural acoustic model, highly resilient to accents)
        try:
            text = self.recognizer.recognize_whisper(audio, model="tiny").strip()
            if text:
                return text
        except Exception:
            pass

        # 3. Try Google en-IN (handles Indian regional pronunciations)
        try:
            text = self.recognizer.recognize_google(audio, language='en-IN').strip()
            if text:
                return text
        except Exception:
            pass

        return ""

    # ------------------------------------------------------------------
    # PC ACTIONS & SYSTEM CONTROLS (Siri for PC)
    # ------------------------------------------------------------------
    def handle_pc_action(self, query):
        """Executes PC automation commands or returns None if not a system command."""
        q = query.lower().strip()

        # 1. Time / Clock
        if any(w in q for w in ['what time', 'current time', 'tell me the time', 'clock']):
            now = datetime.datetime.now()
            return f"The current time is {now.strftime('%I:%M %p')}."

        # 2. Date / Calendar
        if any(w in q for w in ['what date', 'today date', "today's date", 'what day', 'current date']):
            now = datetime.datetime.now()
            return f"Today is {now.strftime('%A, %B %d, %Y')}."

        # 3. Application Launching
        app_map = {
            'notepad': (['notepad.exe'], "Notepad"),
            'calculator': (['calc.exe'], "Calculator"),
            'calc': (['calc.exe'], "Calculator"),
            'chrome': (['start', 'chrome'], "Google Chrome"),
            'edge': (['start', 'msedge'], "Microsoft Edge"),
            'explorer': (['explorer.exe'], "File Explorer"),
            'file explorer': (['explorer.exe'], "File Explorer"),
            'task manager': (['taskmgr.exe'], "Task Manager"),
            'settings': (['start', 'ms-settings:'], "Windows Settings"),
            'terminal': (['wt.exe'], "Windows Terminal"),
            'cmd': (['cmd.exe'], "Command Prompt"),
            'command prompt': (['cmd.exe'], "Command Prompt"),
            'vs code': (['code'], "VS Code"),
            'vscode': (['code'], "VS Code"),
        }

        if q.startswith("open ") or q.startswith("launch ") or q.startswith("start "):
            target = re.sub(r'^(open|launch|start)\s+', '', q).strip()

            # Web shortcuts
            web_shortcuts = {
                'youtube': 'https://www.youtube.com',
                'google': 'https://www.google.com',
                'github': 'https://www.github.com',
                'reddit': 'https://www.reddit.com',
                'chatgpt': 'https://chatgpt.com',
                'twitter': 'https://twitter.com',
                'x': 'https://x.com',
                'spotify': 'https://open.spotify.com'
            }
            if target in web_shortcuts:
                webbrowser.open(web_shortcuts[target])
                return f"Opening {target.capitalize()} in your browser."

            # System apps
            for app_key, (cmd, name) in app_map.items():
                if target == app_key or app_key in target:
                    try:
                        subprocess.Popen(cmd, shell=True)
                        return f"Opening {name}."
                    except Exception as e:
                        return f"Failed to open {name}: {e}"

        return None

    # ------------------------------------------------------------------
    # WAKE & EXIT DETECTION
    # ------------------------------------------------------------------
    def is_wake_word(self, text):
        clean = text.lower().strip()
        return any(variant in clean for variant in WAKE_VARIANTS)

    def is_exit_phrase(self, text):
        clean = text.lower().strip()
        return any(phrase in clean for phrase in EXIT_PHRASES)

    # ------------------------------------------------------------------
    # ACTIVE CONVERSATION SESSION LOOP
    # ------------------------------------------------------------------
    def run_conversation_session(self):
        """
        Active continuous conversational session.
        Stays awake and listens continuously until the user says 'exit'.
        """
        print("\n>>> [SESSION STARTED] Chintu is in active conversation mode!")
        self.bridge.visibility_changed.emit(True)
        self.bridge.status_changed.emit("Listening...")
        self.bridge.query_changed.emit("")
        self.bridge.response_changed.emit("I'm listening, Sir. Ask anything or say 'exit'.")
        
        # Give Qt event loop a moment to paint the window on screen
        time.sleep(0.15)
        self.speak("I'm listening, Sir!")

        idle_silence_count = 0
        max_idle_silence = 3  # ~20 seconds of absolute silence before auto-sleep

        while self.running:
            self.bridge.status_changed.emit("Listening...")
            print("\n[*] [ACTIVE LISTENING] Speak your question or say 'exit'...")

            # Configure VAD for active dialog
            self.recognizer.pause_threshold = 1.0
            audio = None

            try:
                with self.mic as source:
                    audio = self.recognizer.listen(source, timeout=6.5, phrase_time_limit=14.0)
            except sr.WaitTimeoutError:
                idle_silence_count += 1
                print(f"[*] (Silence timeout {idle_silence_count}/{max_idle_silence})")
                if idle_silence_count >= max_idle_silence:
                    print("[*] Prolonged inactivity. Entering Sleep Mode.")
                    self.bridge.status_changed.emit("Sleep Mode")
                    self.bridge.response_changed.emit("Going to sleep now. Just say 'Hey Chintu' whenever you need me!")
                    self.speak("Going to sleep now. Call me whenever you need me.")
                    self.bridge.visibility_changed.emit(False)
                    break
                continue
            except Exception as e:
                print(f"[-] Mic capture error: {e}")
                time.sleep(0.4)
                continue

            if not audio:
                continue

            # Context manager is completely released here!
            self.bridge.status_changed.emit("Interpreting...")
            print("[*] Processing speech (Whisper + Google)...")

            user_speech = self.transcribe_audio(audio)
            if not user_speech:
                print("[-] Could not transcribe speech clearly. Listening again...")
                continue

            # Reset idle counter on speech
            idle_silence_count = 0
            print(f"[USER]: {user_speech}")
            self.bridge.query_changed.emit(user_speech)

            # Check for Exit Keyword
            if self.is_exit_phrase(user_speech):
                print("[*] Exit command detected! Transitioning to Sleep Mode...")
                self.bridge.status_changed.emit("Going to sleep...")
                self.bridge.response_changed.emit("Going to sleep now. Just say 'Hey Chintu' whenever you need me!")
                self.speak("Going to sleep now. Have a great time!")
                self.bridge.visibility_changed.emit(False)
                break

            # If user just repeats the wake word while already in active session
            if self.is_wake_word(user_speech) and len(user_speech.split()) <= 2:
                ack = "Yes Sir, I'm right here! Go ahead and ask your question."
                self.bridge.response_changed.emit(ack)
                self.speak(ack)
                continue

            # Process User Command / Question
            self.bridge.status_changed.emit("Thinking...")

            # 1. Check PC actions
            response = self.handle_pc_action(user_speech)

            # 2. If not PC action, search the web dynamically
            if not response:
                response = self.answer_web_query(user_speech)

            # Update UI and Speak Response
            self.bridge.response_changed.emit(response)
            self.speak(response)

            # Keep conversation alive!
            print("[*] Done answering. Remaining active in conversation!")
            time.sleep(0.4)

        print("<<< [SESSION ENDED] Chintu returned to Standby Sleep Mode.\n")

    # ------------------------------------------------------------------
    # MAIN WAKE-WORD STANDBY LOOP (Sleep Mode)
    # ------------------------------------------------------------------
    def run(self):
        """Continuously listens in background for wake word."""
        if not self.mic:
            print("[ERROR] No working microphone found!")
            return

        print("\n=======================================================")
        print("  CHINTU VOICE ASSISTANT - SLEEP MODE ACTIVE")
        print("  Say 'HEY CHINTU' to activate and start talking!")
        print("=======================================================\n")

        while self.running:
            wake_detected = False
            try:
                # Capture short audio snippet
                with self.mic as source:
                    wake_audio = self.recognizer.listen(
                        source, timeout=1.5, phrase_time_limit=3.0
                    )
                # Microphone context is closed here!

                try:
                    wake_text = self.recognizer.recognize_google(
                        wake_audio, language='en-IN'
                    ).lower()
                    print(f"[*] (Standby heard): '{wake_text}'")
                    if self.is_wake_word(wake_text):
                        wake_detected = True
                except (sr.WaitTimeoutError, sr.UnknownValueError):
                    pass
                except Exception:
                    pass

            except (sr.WaitTimeoutError, sr.UnknownValueError):
                continue
            except Exception as e:
                time.sleep(0.5)
                continue

            if wake_detected:
                print("\n[WAKE WORD DETECTED] Activating Chintu!")
                self.run_conversation_session()
                print("\n[STANDBY] Back in Sleep Mode. Listening for 'Hey Chintu'...")
                time.sleep(1.0)


# ----------------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------------
def handle_exit(sig, frame):
    print("\n[*] Exiting Chintu Assistant...")
    QApplication.quit()
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_exit)
    app = QApplication(sys.argv)

    bridge = UIBridge()
    window = ChintuCoreUI(bridge)

    assistant = ChintuAssistant(bridge)

    # Run assistant background thread
    assistant_thread = threading.Thread(target=assistant.run, daemon=True)
    assistant_thread.start()

    sys.exit(app.exec())