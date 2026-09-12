import asyncio
import edge_tts
import pygame
import os
import time

VOICE = "en-US-AriaNeural" # Ultra-clear, natural Siri-like neural voice

async def generate_speech(text, output_file):
    communicate = edge_tts.Communicate(text, VOICE)
    await communicate.save(output_file)

def speak_edge(text):
    temp_file = "scratch/voice_test.mp3"
    asyncio.run(generate_speech(text, temp_file))
    
    pygame.mixer.init()
    pygame.mixer.music.load(temp_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.05)
    pygame.mixer.quit()
    print("Spoken with Neural Voice successfully!")

if __name__ == "__main__":
    speak_edge("Hello! I am Chintu, your voice assistant. My voice is now clear, natural, and human-like!")
