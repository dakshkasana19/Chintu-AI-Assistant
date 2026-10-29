"""Local microphone capture, voice activity, and wake-word detection."""

from .capture import AudioCapture, AudioCaptureError
from .vad import SpeechSegment, VoiceActivityDetector, create_vad_backend
from .wake import WakeWordDetector, WakeWordMatch, VoskWhisperWakeWordDetector, VOSK_GRAMMAR

__all__ = [
    "AudioCapture",
    "AudioCaptureError",
    "SpeechSegment",
    "VoiceActivityDetector",
    "WakeWordDetector",
    "WakeWordMatch",
    "VoskWhisperWakeWordDetector",
    "VOSK_GRAMMAR",
    "create_vad_backend",
]
