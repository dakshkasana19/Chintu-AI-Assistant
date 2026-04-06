import pvporcupine
from pvrecorder import PvRecorder
import struct
import sys

# !!! REPLACE THIS WITH YOUR REAL ACCESS KEY !!!
ACCESS_KEY = "HJC5DMc4md4+2s4Qs9RnNB8IGyQAogbiHDAQJs32/ORqsczwjHw/cQ==" 

# We use the built-in "computer" wake word for this test
WAKE_WORD = ["computer"] 

# Configuration - Porcupine requires the audio be processed in specific frame sizes
recorder = None
porcupine = None

try:
    # 1. Initialize Porcupine
    porcupine = pvporcupine.create(
        access_key=ACCESS_KEY,
        keywords=WAKE_WORD
    )

    # 2. Initialize the audio recorder using Porcupine's required frame length
    recorder = PvRecorder(
        frame_length=porcupine.frame_length,
        device_index=-1  # -1 uses the default microphone
    )

    recorder.start()
    print(f"Aura is listening in the background for: {' '.join(WAKE_WORD)}...")

    # 3. Main listening loop
    while True:
        # Read a frame of audio (in 16-bit format)
        pcm = recorder.read()

        # Process the audio frame with Porcupine
        keyword_index = porcupine.process(pcm)

        # Check if a wake word was detected
        if keyword_index >= 0:
            print(f"\n[WAKE WORD DETECTED: {WAKE_WORD[keyword_index]}]")
            print("Aura activated! This is where command listening would start.")
            break # Exit the test loop

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Clean up resources
    if recorder is not None:
        recorder.stop()
        recorder.delete()
    if porcupine is not None:
        porcupine.delete()

print("Test script finished.")