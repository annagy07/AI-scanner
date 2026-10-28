import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import detector as det
from app.main import app
from app.sources import spotify_track_id

HERE = Path(__file__).parent


def make_audio(path: Path, seconds: int = 12, rate: int = 44100):
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
         f"anoisesrc=d={seconds}:c=pink:r={rate}:a=0.3", "-f", "lavfi", "-i",
         f"sine=f=440:d={seconds}:r={rate}", "-filter_complex", "amix=inputs=2,aformat=channel_layouts=stereo",
         str(path)],
        check=True,
    )


def test_resample_matches_torchaudio():
    d = json.loads((HERE / "resample_test_data.json").read_text())
    out = det.resample(np.array(d["input"], dtype=np.float32)[None], d["orig_freq"], d["new_freq"])[0]
    np.testing.assert_allclose(out, d["expected_output"], atol=1e-5)


def test_spectrogram_shape():
    spec = det.power_spectrogram(np.random.randn(16000 * 3).astype(np.float32))
    assert spec.shape == (det.N_FFT // 2 + 1, 1 + 16000 * 3 // (det.N_FFT // 2))


def test_fakeprint_dimensions():
    fp = det.compute_fakeprint(np.random.randn(2, 16000 * 5).astype(np.float32))
    assert fp.shape == (3585,) and 0 <= fp.min() and fp.max() <= 1


@pytest.mark.parametrize("ext", ["wav", "mp3"])
def test_api_upload(tmp_path, ext):
    f = tmp_path / f"t.{ext}"
    make_audio(f)
    client = TestClient(app)
    r = client.post("/api/analyze/file", files={"file": (f.name, f.read_bytes())})
    assert r.status_code == 200, r.text
    d = r.json()
    assert 0 <= d["probability"] <= 1
    assert abs(d["duration_analyzed"] - 12) < 0.1
    assert len(d["fakeprint"]) == len(d["fakeprint_freqs"])


def test_api_rejects_non_audio(tmp_path):
    client = TestClient(app)
    r = client.post("/api/analyze/file", files={"file": ("x.mp3", b"not audio")})
    assert r.status_code == 422


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://open.spotify.com/track/4uLU6hMCjMI75M1A2tKUQC?si=abc", "4uLU6hMCjMI75M1A2tKUQC"),
        ("https://open.spotify.com/intl-de/track/4uLU6hMCjMI75M1A2tKUQC", "4uLU6hMCjMI75M1A2tKUQC"),
        ("spotify:track:4uLU6hMCjMI75M1A2tKUQC", "4uLU6hMCjMI75M1A2tKUQC"),
        ("https://open.spotify.com/album/4uLU6hMCjMI75M1A2tKUQC", None),
    ],
)
def test_spotify_track_id(url, expected):
    assert spotify_track_id(url) == expected


def _write_mp3(path: Path, x: np.ndarray, sr: int):
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-", "-b:a", "192k", str(path)],
        input=x.astype(np.float32).tobytes(), check=True,
    )


def test_vocoder_like_artifacts_are_flagged(tmp_path):
    """The model weights peaks on a 200 Hz grid; a faint comb on top of noise must flip the verdict."""
    sr, rng = 44100, np.random.default_rng(0)
    t = np.arange(sr * 30) / sr
    noise = np.cumsum(rng.standard_normal(len(t)))
    noise -= np.convolve(noise, np.ones(400) / 400, "same")
    noise /= np.abs(noise).max() * 2
    comb = sum(np.sin(2 * np.pi * f * t) for f in range(200, 16000, 200)) * 0.002

    d = det.Detector()
    _write_mp3(tmp_path / "human.mp3", noise, sr)
    _write_mp3(tmp_path / "ai.mp3", noise + comb, sr)
    assert d.analyze(str(tmp_path / "human.mp3")).probability < 0.1
    assert d.analyze(str(tmp_path / "ai.mp3")).probability > 0.9
