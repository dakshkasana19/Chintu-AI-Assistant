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
WAKE_VARIANTS = ["chintu", "chinto", "chintoo", "chintu ji"]

class ChintuAssistant:
    def __init__(self, bridge):
        self.bridge = bridge
        self.recognizer = sr.Recognizer()
        self.mic = None
        self.running = True
        
        self.setup_microphone()
        print("✅ Chintu ready! (VAD + DDG Web Search)")
        print(f"Listening for: {WAKE_VARIANTS}")

    def setup_microphone(self):
        """Professional VAD microphone setup"""
        try:
            self.mic = sr.Microphone()
            print("✅ Microphone ready")
            
            # Optimized VAD settings
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

    def speak(self, text):
        """Enhanced TTS with smooth UI feedback"""
        if not self.running: return
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

    def search_web(self, query):
        """Real-time web search (FREE forever)"""
        self.bridge.status_changed.emit("Searching Web...")
        print(f"🔍 {query}")
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3))
                if results:
                    snippets = []
                    for r in results[:3]:
                        snippet = f"{r.get('title', 'Web')}: {r.get('body', '')[:120]}"
                        snippets.append(snippet)
                    return " | ".join(snippets)
                return "No recent web information available."
        except Exception as e:
            print(f"❌ Search: {e}")
            return "Web search temporarily down."

    def get_smart_response(self, query):
        """AI-powered response routing"""
        query_lower = query.lower()
        
        # Instant local responses
        if any(word in query_lower for word in ['time', 'clock']):
            now = datetime.datetime.now()
            return f"The time is {now.strftime('%I:%M %p')} right now."
        
        if any(word in query_lower for word in ['date', 'today']):
            now = datetime.datetime.now()
            return f"Today is {now.strftime('%A, %B %d, %Y')}."
        
        # System apps
        if 'notepad' in query_lower:
            subprocess.Popen(['notepad.exe'], shell=True)
            return "Opening Notepad!"
        if 'calculator' in query_lower or 'calc' in query_lower:
            subprocess.Popen(['calc.exe'], shell=True)
            return "Calculator launched!"
        if any(word in query_lower for word in ['edge', 'browser']):
            webbrowser.open('edge://newtab')
            return "Browser opening!"
        
        # 2026 Knowledge Base
        knowledge = {
            "prime minister": "Narendra Modi is India's Prime Minister since 2014.",
            "president": "Donald Trump is US President, serving second term since 2025.",
            "gandhi": "Mahatma Gandhi led India's independence through non-violence.",
            "python": "Python is your favorite programming language for AI and software engineering.",
            "india": "India has 1.4+ billion people, world's largest democracy."
        }
        
        for key, answer in knowledge.items():
            if key in query_lower:
                return answer
        
        # Complex queries → Web search + synthesis
        web_context = self.search_web(query)
        synthesis_templates = [
            f"From recent web sources: {web_context[:140]}.",
            f"Current information shows: {web_context[:120]}. Interesting!",
            f"Web search reveals: {web_context[:110]}. That's current data."
        ]
        return random.choice(synthesis_templates)

    def is_wake_word(self, text):
        """Robust wake word detection"""
        return any(variant in text.lower().strip() for variant in WAKE_VARIANTS)

    def run(self):
        """Professional VAD - Continuous listening until silence"""
        if not self.mic:
            print("❌ No microphone!")
            return

        print("🚀 Chintu VAD Active!")
        print("💡 Say 'CHINTU' then speak naturally (stops after 3s silence)")
        
        while self.running:
            try:
                with self.mic as source:
                    # Phase 1: Wake word detection
                    print("👂 (wake)... ", end="", flush=True)
                    wake_audio = self.recognizer.listen(source, timeout=1.2, phrase_time_limit=2.2)
                    wake_text = self.recognizer.recognize_google(wake_audio, language='en-IN').lower()
                    print(f"'{wake_text}'")
                    
                    if self.is_wake_word(wake_text):
                        print("🔥 ACTIVATED!")
                        self.bridge.visibility_changed.emit(True)
                        self.speak("Go ahead Sir!")
                        
                        # Phase 2: Continuous voice capture
                        query_parts = []
                        silence_count = 0
                        SILENCE_LIMIT = 8  # ~4 seconds silence
                        
                        print("🎤 (talking)... ", end="")
                        self.bridge.status_changed.emit("Listening...")
                        
                        while silence_count < SILENCE_LIMIT and self.running:
                            try:
                                chunk_audio = self.recognizer.listen(source, timeout=0.45, phrase_time_limit=2.8)
                                chunk_text = self.recognizer.recognize_google(chunk_audio, language='en-IN')
                                
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
                            print(f"❓ '{full_query}'")
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

def handle_exit(sig, frame):
    print("\n👋 Chintu offline!")
    QApplication.quit()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_exit)
    app = QApplication(sys.argv)
    
    bridge = UIBridge()
    window = ChintuCoreUI(bridge)
    
    print("🚀 Launching Chintu VAD Assistant...")
    
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
    print("🎙️ Say 'CHINTU' then speak naturally!")
    print("💡 Example: 'CHINTU tell me about cats they purr and hunt mice'")
    sys.exit(app.exec())
