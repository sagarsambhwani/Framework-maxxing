"""Groq Whisper In-Memory Audio Transcriber.

Decouples Speech-to-Text transcription from FastAPI route handlers.
Provides high-performance in-memory streaming directly to Groq Whisper Large v3 Turbo,
eliminating disk I/O latency and temporary file leaks.
"""

import io
import time
from typing import Optional, Any
from openai import OpenAI

from src.audio.models import TranscribeResult, VADMetrics
from src.common.config import settings
from src.common.logging import term_log, debug_log, Colors


class WhisperTranscriber:
    """In-memory Speech-to-Text transcriber using Groq Whisper Large v3 Turbo."""

    DEFAULT_MODEL = "whisper-large-v3-turbo"
    GROQ_BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        client: Optional[Any] = None
    ):
        """Initializes the Whisper transcriber.

        Args:
            api_key: Groq API key (defaults to settings.GROQ_API_KEY).
            model: Target transcription model slug.
            client: Optional pre-configured OpenAI-compatible client (for testing/mocking).
        """
        self.api_key = api_key or settings.GROQ_API_KEY
        self.model = model or self.DEFAULT_MODEL
        self._client = client

    @property
    def client(self) -> OpenAI:
        """Lazily creates and returns the Groq OpenAI-compatible client."""
        if self._client is None:
            self._client = OpenAI(
                base_url=self.GROQ_BASE_URL,
                api_key=self.api_key
            )
        return self._client

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        filename: str = "speech.wav",
        vad_metrics: Optional[VADMetrics] = None,
        language: Optional[str] = None
    ) -> TranscribeResult:
        """Transcribes audio payload completely in-memory without disk writes.

        Args:
            audio_bytes: Raw binary content of the audio file.
            filename: Virtual filename with audio extension (e.g. 'speech.wav', 'audio.webm').
            vad_metrics: Optional VAD turn telemetry attached from edge or local VAD.
            language: Optional 2-letter ISO language code (e.g. 'en', 'es', 'fr').

        Returns:
            TranscribeResult containing the text, latency, and associated metadata.

        Raises:
            ValueError: If audio_bytes is empty.
            Exception: Upstream API errors from Groq.
        """
        if not audio_bytes or len(audio_bytes) == 0:
            raise ValueError("Cannot transcribe empty audio payload (0 bytes).")

        t0 = time.time()
        debug_log(
            "🔍 [DEBUG:AUDIO_STT]",
            f"Streaming {len(audio_bytes)} audio bytes to Groq ({self.model}) as '{filename}'"
        )

        # Wrap raw bytes in an in-memory buffer with filename attribute
        audio_stream = io.BytesIO(audio_bytes)
        audio_stream.name = filename

        # Prepare kwargs for the transcription request
        kwargs: dict[str, Any] = {
            "model": self.model,
            "file": audio_stream,
            "response_format": "text"
        }
        if language:
            kwargs["language"] = language

        # Dispatch inference to Groq LPU
        response = self.client.audio.transcriptions.create(**kwargs)

        duration_s = round(time.time() - t0, 3)
        transcribed_text = str(response).strip()

        term_log(
            "🎙️ [WHISPER TURBO]",
            f"Transcribed in {Colors.GREEN}{duration_s}s{Colors.END}: "
            f"'{Colors.BOLD}{transcribed_text}{Colors.END}'",
            Colors.GREEN
        )

        return TranscribeResult(
            text=transcribed_text,
            duration_s=duration_s,
            model=self.model,
            vad_metrics=vad_metrics,
            metadata={"payload_bytes": len(audio_bytes), "filename": filename}
        )
