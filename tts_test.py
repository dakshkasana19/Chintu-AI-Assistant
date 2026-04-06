import pyttsx3

# Initialize the engine
engine = pyttsx3.init()

# Define the text the assistant will say
text = "Hello! I am Aura, your voice assistant. Testing complete."

# Queue the text to be spoken
engine.say(text)

# Run the speaking process
engine.runAndWait()

print("Test finished.")