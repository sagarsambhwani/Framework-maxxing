"""Audio Processing, Voice Activity Detection & Speech-to-Text Package.

Exports:
    - audio_service: Preconfigured global AudioService instance.
    - AudioService: Orchestrator class combining VAD, Whisper, and telemetry.
    - WhisperTranscriber: In-memory Groq Whisper client.
    - AudioVADEngine: Acoustic VAD and container validation engine.
    - VADMetrics, TranscribeResult, AudioValidationResult: Strongly-typed schemas.
"""

from src.audio.models import VADMetrics, TranscribeResult, AudioValidationResult
from src.audio.service import AudioService, audio_service
from src.audio.transcriber import WhisperTranscriber
from src.audio.vad import AudioVADEngine

__all__ = [
    "AudioService",
    "audio_service",
    "WhisperTranscriber",
    "AudioVADEngine",
    "VADMetrics",
    "TranscribeResult",
    "AudioValidationResult",
]
