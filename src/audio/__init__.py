"""Audio Processing, Voice Activity Detection & Speech-to-Text Package.

Exports:
    - audio_service: Preconfigured global AudioService instance.
    - AudioService: Orchestrator class combining VAD, Whisper, and telemetry.
    - WhisperTranscriber: In-memory Groq Whisper client.
    - AudioVADEngine: Acoustic VAD and container validation engine.
    - VADMetrics, TranscribeResult, AudioValidationResult: Strongly-typed schemas.
"""

from src.audio.models import VADMetrics, TranscribeResult, AudioValidationResult
from src.audio.transcriber import WhisperTranscriber
from src.audio.vad import AudioVADEngine

# Lazy loading via PEP 562 so importing vad or models doesn't trigger network/tracer checks
def __getattr__(name: str):
    if name in ("AudioService", "audio_service"):
        from src.audio import service
        return getattr(service, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AudioService",
    "audio_service",
    "WhisperTranscriber",
    "AudioVADEngine",
    "VADMetrics",
    "TranscribeResult",
    "AudioValidationResult",
]
