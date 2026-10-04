"""Unit and Integration Tests for Audio Service, VAD Engine & Transcriber.

Validates:
    1. VAD & Audio Pydantic schemas and boundary validation.
    2. Payload validation (empty, corrupt, valid format detection).
    3. Acoustic RMS analysis (silence vs. synthetic speech).
    4. In-memory WhisperTranscriber with mock client.
    5. AudioService silence filtering & transcription orchestration.
    6. FastAPI POST /api/transcribe HTTP endpoint.
"""

import io
import wave
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from src.audio.models import VADMetrics, TranscribeResult, AudioValidationResult
from src.audio.transcriber import WhisperTranscriber
from src.audio.vad import AudioVADEngine
from src.audio.service import AudioService
from src.server.app import create_app


# Helper to synthesize in-memory PCM WAV audio
def generate_synthetic_wav(freq_hz: float = 440.0, duration_s: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generates an in-memory WAV file with a sine wave tone or silence."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
        if freq_hz > 0:
            signal = (np.sin(2 * np.pi * freq_hz * t) * 16000).astype(np.int16)
        else:
            signal = np.zeros(len(t), dtype=np.int16)
        w.writeframes(signal.tobytes())
    return buf.getvalue()


# =============================================================================
# 1. Models & Validation Tests
# =============================================================================
def test_vad_metrics_validation():
    """Verifies VADMetrics data types and bounds."""
    metrics = VADMetrics(
        is_speech=True,
        confidence=0.95,
        speech_duration_s=2.5,
        silence_duration_s=0.6,
        vad_latency_ms=1.5
    )
    assert metrics.is_speech is True
    assert metrics.confidence == 0.95
    assert metrics.speech_duration_s == 2.5

    # Out of bounds confidence should raise ValidationError
    with pytest.raises(Exception):
        VADMetrics(confidence=1.5)


def test_audio_vad_engine_payload_validation():
    """Verifies payload size and magic byte format checks."""
    vad = AudioVADEngine()

    # Empty payload
    empty_res = vad.validate_payload(b"")
    assert empty_res.valid is False
    assert empty_res.format == "empty"

    # Corrupt / too small payload
    small_res = vad.validate_payload(b"RIFFshort")
    assert small_res.valid is False
    assert small_res.format == "corrupt"

    # Valid WAV payload
    wav_bytes = generate_synthetic_wav(freq_hz=440.0, duration_s=0.5)
    valid_res = vad.validate_payload(wav_bytes)
    assert valid_res.valid is True
    assert valid_res.format == "wav"
    assert valid_res.size_bytes == len(wav_bytes)


# =============================================================================
# 2. Acoustic Analysis (Silence vs Speech) Tests
# =============================================================================
def test_audio_vad_acoustic_silence():
    """Verifies that pure silence is classified as is_speech=False."""
    vad = AudioVADEngine()
    silence_wav = generate_synthetic_wav(freq_hz=0, duration_s=1.0)
    metrics = vad.analyze_audio(silence_wav)

    assert metrics.is_speech is False
    assert metrics.confidence == 0.0
    assert metrics.speech_duration_s == 0.0


def test_audio_vad_acoustic_speech():
    """Verifies that vocal-frequency audio is classified as is_speech=True."""
    vad = AudioVADEngine()
    speech_wav = generate_synthetic_wav(freq_hz=440.0, duration_s=1.0)
    metrics = vad.analyze_audio(speech_wav)

    assert metrics.is_speech is True
    assert metrics.confidence > 0.5
    assert metrics.speech_duration_s > 0.5


# =============================================================================
# 3. Whisper Transcriber Tests
# =============================================================================
def test_whisper_transcriber_mock():
    """Verifies in-memory transcription with a mocked client."""
    mock_client = MagicMock()
    mock_client.audio.transcriptions.create.return_value = "Synthetic Speech Transcript"

    transcriber = WhisperTranscriber(client=mock_client)
    audio_bytes = generate_synthetic_wav(freq_hz=440.0, duration_s=0.5)
    result = transcriber.transcribe_bytes(audio_bytes, filename="test.wav")

    assert result.text == "Synthetic Speech Transcript"
    assert result.model == "whisper-large-v3-turbo"
    assert result.metadata["filename"] == "test.wav"
    mock_client.audio.transcriptions.create.assert_called_once()


def test_whisper_transcriber_empty_raises():
    """Verifies that transcribing 0 bytes raises ValueError."""
    transcriber = WhisperTranscriber()
    with pytest.raises(ValueError, match="empty audio payload"):
        transcriber.transcribe_bytes(b"")


# =============================================================================
# 4. AudioService Orchestration Tests
# =============================================================================
def test_audio_service_silence_skips_whisper():
    """Verifies that AudioService skips Whisper when silence is detected."""
    import asyncio
    mock_transcriber = MagicMock()
    mock_transcriber.model = "whisper-large-v3-turbo"
    service = AudioService(transcriber=mock_transcriber, vad_engine=AudioVADEngine())

    silence_wav = generate_synthetic_wav(freq_hz=0, duration_s=1.0)
    result = asyncio.run(service.transcribe(silence_wav, filename="silence.wav", filter_silence=True))

    assert result.text == ""
    assert result.metadata.get("skipped_reason") == "No speech detected by VAD"
    mock_transcriber.transcribe_bytes.assert_not_called()


def test_audio_service_speech_transcription():
    """Verifies that AudioService transcribes speech and emits telemetry."""
    import asyncio
    mock_transcriber = MagicMock()
    mock_transcriber.model = "whisper-large-v3-turbo"
    mock_transcriber.transcribe_bytes.return_value = TranscribeResult(
        text="Recognized voice query",
        duration_s=0.15
    )
    service = AudioService(transcriber=mock_transcriber, vad_engine=AudioVADEngine())

    speech_wav = generate_synthetic_wav(freq_hz=440.0, duration_s=1.0)
    result = asyncio.run(service.transcribe(speech_wav, filename="speech.wav", session_id="test-session"))

    assert result.text == "Recognized voice query"
    assert result.duration_s == 0.15
    mock_transcriber.transcribe_bytes.assert_called_once()


# =============================================================================
# 5. FastAPI /api/transcribe Integration Tests
# =============================================================================
def test_server_transcribe_endpoint_empty_file():
    """Verifies /api/transcribe returns 400 when empty file is uploaded."""
    app = create_app()
    client = TestClient(app)

    response = client.post("/api/transcribe", files={"file": ("empty.wav", b"", "audio/wav")})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_server_transcribe_endpoint_valid_audio():
    """Verifies /api/transcribe returns 200 with mock AudioService."""
    app = create_app()
    client = TestClient(app)

    speech_wav = generate_synthetic_wav(freq_hz=440.0, duration_s=0.5)
    with patch("src.server.app.audio_service.transcribe") as mock_svc_transcribe:
        mock_svc_transcribe.return_value = TranscribeResult(
            text="End-to-end API test",
            duration_s=0.21
        )
        response = client.post("/api/transcribe", files={"file": ("speech.wav", speech_wav, "audio/wav")})

        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "End-to-end API test"
        assert data["duration_s"] == 0.21
