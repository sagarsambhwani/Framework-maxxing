"""VAD Benchmark: Energy VAD vs Silero ML Neural Network.

Generates 4 distinct audio signals and prints a side-by-side comparison table.
"""

import os
import sys
import wave
import scipy.signal
import win32com.client
import numpy as np
import onnxruntime as ort

# Load model
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(root_dir, "src", "audio", "weights", "silero_vad.onnx")
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    sys.exit(1)

session = ort.InferenceSession(model_path)

def evaluate_audio_sample(audio_16k):
    rms = float(np.sqrt(np.mean(audio_16k ** 2))) if len(audio_16k) > 0 else 0.0
    energy_speech = rms >= 0.015

    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros((1, 64), dtype=np.float32)
    sr = np.array(16000, dtype=np.int64)

    probs = []
    for i in range(0, len(audio_16k), 512):
        chunk = audio_16k[i : i + 512]
        if len(chunk) < 512:
            break
        x = np.concatenate([context, chunk.reshape(1, -1)], axis=1)
        context = x[:, -64:]
        out, state = session.run(None, {"input": x, "state": state, "sr": sr})
        probs.append(float(out[0][0]))

    peak_prob = max(probs) if probs else 0.0
    ml_speech = peak_prob >= 0.5

    return {
        "rms": rms,
        "energy_speech": energy_speech,
        "peak_prob": peak_prob,
        "ml_speech": ml_speech,
    }

duration_s = 1.5
n_samples = int(16000 * duration_s)
t = np.linspace(0, duration_s, n_samples, endpoint=False)

# 1. Room Silence
silence = np.zeros(n_samples, dtype=np.float32)

# 2. Loud Keyboard Clicks (spikes)
clicks = np.zeros(n_samples, dtype=np.float32)
for pos in range(1600, n_samples - 100, 3200):
    clicks[pos : pos + 30] = 0.85

# 3. Desk Bump / Thump (35Hz shock)
thump = (0.7 * np.sin(2 * np.pi * 35 * t)).astype(np.float32)

# 4. Spoken Human Voice via Windows SAPI
voice = win32com.client.Dispatch("SAPI.SpVoice")
stream = win32com.client.Dispatch("SAPI.SpFileStream")
temp_wav = os.path.join(root_dir, "temp_benchmark_speech.wav")
stream.Open(temp_wav, 3, False)
voice.AudioOutputStream = stream
voice.Speak("What is the weather like today?")
stream.Close()

with wave.open(temp_wav, "rb") as w:
    sr_file = w.getframerate()
    frames = w.readframes(w.getnframes())
    audio_pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

num_samples_speech = int(len(audio_pcm) * 16000 / sr_file)
speech = scipy.signal.resample(audio_pcm, num_samples_speech).astype(np.float32)
if os.path.exists(temp_wav):
    os.remove(temp_wav)

signals = [
    ("1. Room Silence", silence),
    ("2. Loud Keyboard Clicks", clicks),
    ("3. Heavy Desk Thump", thump),
    ("4. Spoken Human Voice", speech),
]

print("=" * 95)
print(f"{'Audio Signal':<25} | {'Volume (RMS)':<12} | {'Current Energy VAD':<20} | {'Silero ML Neural Net':<25}")
print("=" * 95)

for name, sig in signals:
    res = evaluate_audio_sample(sig)
    energy_str = "TRIGGERED (Speech)" if res["energy_speech"] else "REJECTED (Silence)"
    ml_str = f"Peak: {res['peak_prob']*100:5.1f}% -> " + ("CONFIRMED (Speech)" if res["ml_speech"] else "REJECTED (Noise)")
    print(f"{name:<25} | {res['rms']:<12.4f} | {energy_str:<20} | {ml_str:<25}")

print("=" * 95)
