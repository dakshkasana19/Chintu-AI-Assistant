"""Local microphone capture and voice-activity detection."""

from .capture import AudioCapture, AudioCaptureError
from .vad import SpeechSegment, VoiceActivityDetector, create_vad_backend

__all__ = [
    "AudioCapture",
    "AudioCaptureError",
    "SpeechSegment",
    "VoiceActivityDetector",
    "create_vad_backend",
]
