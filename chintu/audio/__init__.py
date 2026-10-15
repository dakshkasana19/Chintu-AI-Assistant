"""Local microphone capture, voice activity, and wake-word detection."""

from .capture import AudioCapture, AudioCaptureError
from .vad import SpeechSegment, VoiceActivityDetector, create_vad_backend
from .wake import WakeWordDetector, WakeWordMatch, VoskWhisperWakeWordDetector

__all__ = [
    "AudioCapture",
    "AudioCaptureError",
    "SpeechSegment",
    "VoiceActivityDetector",
    "WakeWordDetector",
    "WakeWordMatch",
    "VoskWhisperWakeWordDetector",
    "create_vad_backend",
]
