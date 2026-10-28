"""AI music detection based on lofcz/ai-music-detector ("fakeprint" model).

This is a dependency-light (numpy/scipy/onnxruntime, no torch) port of the
original `inference.py`. Every step mirrors the torchaudio pipeline the model
was trained with:

1. decode at native sample rate (ffmpeg)
2. resample each channel to 16 kHz (port of torchaudio sinc_interp_hann)
3. truncate to 300 s, average channels to mono
4. power STFT (n_fft=8192, hop=4096, periodic Hann, centered, reflect pad)
5. mean dB spectrum, 1-8 kHz band, minus lower hull -> normalized "fakeprint"
6. ONNX classifier -> probability that the track is AI-generated
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.ndimage import minimum_filter1d

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "ai_music_detector.onnx"

SAMPLE_RATE = 16000
N_FFT = 8192
MAX_DURATION = 300  # seconds
FREQ_MIN = 1000
FREQ_MAX = 8000
HULL_AREA = 10
MAX_DB = 5
MIN_DB = -45
THRESHOLD = 0.5


class AudioError(Exception):
    pass


# --------------------------------------------------------------------------- audio


def probe(path: str) -> tuple[int, int, float | None]:
    """Return (sample_rate, channels, duration) of the first audio stream."""
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "a:0",
                "-show_entries", "stream=sample_rate,channels:format=duration",
                "-of", "json", path,
            ],
            capture_output=True, check=True, timeout=60,
        ).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        raise AudioError("Datei konnte nicht gelesen werden (kein gültiges Audio?)") from e
    info = json.loads(out or b"{}")
    streams = info.get("streams") or []
    if not streams:
        raise AudioError("Keine Audiospur gefunden")
    sr = int(streams[0]["sample_rate"])
    ch = int(streams[0]["channels"])
    dur = info.get("format", {}).get("duration")
    return sr, ch, float(dur) if dur not in (None, "N/A") else None


def decode(path: str) -> tuple[np.ndarray, int]:
    """Decode to float32 [channels, samples] at the native sample rate."""
    sr, ch, _ = probe(path)
    # Only decode what we need after resampling (+1 s margin for the filter).
    max_seconds = MAX_DURATION + 1
    try:
        raw = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-nostdin", "-i", path,
                "-map", "0:a:0", "-t", str(max_seconds),
                "-f", "f32le", "-acodec", "pcm_f32le", "-",
            ],
            capture_output=True, check=True, timeout=300,
        ).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        raise AudioError("Audio konnte nicht dekodiert werden") from e
    audio = np.frombuffer(raw, dtype=np.float32)
    n = len(audio) // ch
    if n == 0:
        raise AudioError("Audiodatei ist leer")
    return audio[: n * ch].reshape(n, ch).T.copy(), sr


def _sinc_resample_kernel(orig: int, new: int, lowpass_filter_width: int = 6, rolloff: float = 0.99):
    """Port of torchaudio.functional._get_sinc_resample_kernel (sinc_interp_hann)."""
    base_freq = min(orig, new) * rolloff
    width = math.ceil(lowpass_filter_width * orig / base_freq)
    idx = np.arange(-width, width + orig, dtype=np.float64)[None, :] / orig
    t = np.arange(0, -new, -1, dtype=np.float64)[:, None] / new + idx
    t *= base_freq
    t = np.clip(t, -lowpass_filter_width, lowpass_filter_width)
    window = np.cos(t * math.pi / lowpass_filter_width / 2) ** 2
    t *= math.pi
    scale = base_freq / orig
    with np.errstate(invalid="ignore", divide="ignore"):
        kernels = np.where(t == 0, 1.0, np.sin(t) / t)
    kernels *= window * scale
    return kernels.astype(np.float32), width


def resample(waveform: np.ndarray, orig_freq: int, new_freq: int) -> np.ndarray:
    """Port of torchaudio.functional.resample. waveform: [..., time] float32."""
    if orig_freq == new_freq:
        return waveform
    g = math.gcd(orig_freq, new_freq)
    orig, new = orig_freq // g, new_freq // g
    kernels, width = _sinc_resample_kernel(orig, new)

    shape = waveform.shape
    wav = waveform.reshape(-1, shape[-1])
    length = wav.shape[-1]
    padded = np.pad(wav, ((0, 0), (width, width + orig)))
    k = kernels.shape[1]
    out = []
    for row in padded:
        frames = np.lib.stride_tricks.sliding_window_view(row, k)[::orig]
        out.append((frames @ kernels.T).reshape(-1))
    res = np.stack(out)
    target_length = math.ceil(new * length / orig)
    return res[:, :target_length].reshape(*shape[:-1], -1)


def load_audio(path: str) -> np.ndarray:
    """Load audio as [channels, samples] at 16 kHz, max 300 s (like the original)."""
    audio, sr = decode(path)
    audio = resample(audio, sr, SAMPLE_RATE)
    return audio[:, : MAX_DURATION * SAMPLE_RATE]


# --------------------------------------------------------------------------- features


def power_spectrogram(mono: np.ndarray) -> np.ndarray:
    """torchaudio.transforms.Spectrogram(n_fft=8192, power=2) for a 1-D signal -> [freq, time]."""
    hop = N_FFT // 2
    pad = N_FFT // 2
    if len(mono) <= pad:
        raise AudioError("Audio ist zu kurz für die Analyse")
    x = np.pad(mono.astype(np.float32), pad, mode="reflect")
    n = 1 + (len(x) - N_FFT) // hop
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::hop][:n] * window
    spec = np.fft.rfft(frames, axis=-1)
    return (spec.real**2 + spec.imag**2).T.astype(np.float32)


FREQ_BINS = np.linspace(0, SAMPLE_RATE / 2, num=N_FFT // 2 + 1)
FREQ_MASK = (FREQ_BINS >= FREQ_MIN) & (FREQ_BINS <= FREQ_MAX)
FREQ_RANGE = FREQ_BINS[FREQ_MASK]


def compute_fakeprint(audio: np.ndarray) -> np.ndarray:
    mono = audio.mean(axis=0) if audio.shape[0] > 1 else audio[0]
    spec = power_spectrogram(mono)
    spec_db = 10 * np.log10(np.clip(spec, 1e-10, 1e6))
    mean_spectrum = spec_db.mean(axis=1)
    freq_spectrum = mean_spectrum[FREQ_MASK]
    hull = minimum_filter1d(freq_spectrum, size=HULL_AREA, mode="nearest")
    hull = np.clip(hull, MIN_DB, None)
    residue = np.clip(freq_spectrum - hull, 0, MAX_DB)
    return (residue / (residue.max() + 1e-6)).astype(np.float32)


# --------------------------------------------------------------------------- model


@dataclass
class Result:
    probability: float
    is_ai: bool
    label: str
    confidence: float
    duration_analyzed: float
    processing_ms: int
    fakeprint: list[float]
    fakeprint_freqs: list[float]

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class Detector:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.expected_features = inp.shape[1]

    def predict_fakeprint(self, fakeprint: np.ndarray) -> float:
        if len(fakeprint) != self.expected_features:
            fakeprint = np.interp(
                np.linspace(0, 1, self.expected_features), np.linspace(0, 1, len(fakeprint)), fakeprint
            ).astype(np.float32)
        out = self.session.run(None, {self.input_name: fakeprint.reshape(1, -1)})
        return float(out[0][0, 0])

    def analyze(self, path: str, plot_points: int = 700) -> Result:
        start = time.perf_counter()
        audio = load_audio(path)
        if audio.shape[1] < SAMPLE_RATE:
            raise AudioError("Audio ist kürzer als 1 Sekunde")
        fakeprint = compute_fakeprint(audio)
        p = self.predict_fakeprint(fakeprint)
        is_ai = p >= THRESHOLD

        # Downsample (max-pool) the fakeprint for the chart so peaks stay visible.
        chunks = np.array_split(np.arange(len(fakeprint)), plot_points)
        fp_plot = [round(float(fakeprint[c].max()), 4) for c in chunks]
        freqs = [round(float(FREQ_RANGE[c].mean()), 1) for c in chunks]

        return Result(
            probability=p,
            is_ai=is_ai,
            label="AI-generiert" if is_ai else "Menschlich / echt",
            confidence=abs(p - 0.5) * 2,
            duration_analyzed=audio.shape[1] / SAMPLE_RATE,
            processing_ms=int((time.perf_counter() - start) * 1000),
            fakeprint=fp_plot,
            fakeprint_freqs=freqs,
        )
