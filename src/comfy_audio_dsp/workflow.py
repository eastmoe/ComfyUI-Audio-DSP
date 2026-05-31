from __future__ import annotations

import json
import math

import numpy as np
import torch
from scipy import signal

from .analysis import bpm_tempo_detector, key_pitch_detector, onset_detector
from .common import AUDIO_EPS, amp_to_db, audio_waveform, copy_audio


def _mono(audio: dict) -> tuple[np.ndarray, int, torch.Tensor]:
    waveform, sample_rate = audio_waveform(audio)
    mono = waveform.mean(dim=(0, 1)).detach().cpu().numpy().astype(np.float32, copy=False)
    return mono, sample_rate, waveform


def _timbre(mono: np.ndarray, sample_rate: int) -> dict:
    if mono.size < 8:
        return {"centroid_hz": 0.0, "rolloff_hz": 0.0, "brightness": "silent"}
    n = min(4096, max(256, 2 ** int(math.floor(math.log2(max(8, mono.size))))))
    windowed = mono[:n] * np.hanning(n).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    total = float(np.sum(spectrum) + 1.0e-8)
    centroid = float(np.sum(freqs * spectrum) / total)
    cumsum = np.cumsum(spectrum)
    rolloff = float(freqs[min(len(freqs) - 1, int(np.searchsorted(cumsum, cumsum[-1] * 0.85)))])
    brightness = "bright" if centroid > 3500.0 else "dark" if centroid < 900.0 else "balanced"
    return {"centroid_hz": round(centroid, 2), "rolloff_hz": round(rolloff, 2), "brightness": brightness}


def audio_feature_to_text(audio: dict, include_bpm: bool, include_key: bool, include_loudness: bool, include_timbre: bool) -> tuple[dict, str]:
    mono, sample_rate, waveform = _mono(audio)
    features: dict[str, object] = {"duration_s": round(float(waveform.shape[-1] / sample_rate), 3), "sample_rate": sample_rate, "channels": int(waveform.shape[1])}
    phrases = []
    if include_bpm:
        _a, bpm, beats = bpm_tempo_detector(audio, 60.0, 200.0)
        features["bpm"] = round(float(bpm), 2)
        features["beats"] = json.loads(beats)[:16] if beats and beats.startswith("[") else []
        if bpm > 0.0:
            phrases.append(f"{bpm:.1f} BPM")
    if include_key:
        _a, key, pitch_hz, _text = key_pitch_detector(audio, 50.0, 1200.0)
        features["key"] = key
        features["pitch_hz"] = round(float(pitch_hz), 2)
        if key != "Unknown":
            phrases.append(f"{key} key")
    if include_loudness:
        rms = float(np.sqrt(np.mean(mono * mono) + 1.0e-8))
        peak = float(np.max(np.abs(mono)) + 1.0e-8)
        rms_db = 20.0 * math.log10(max(rms, 1.0e-8))
        peak_db = 20.0 * math.log10(max(peak, 1.0e-8))
        features["rms_dbfs"] = round(rms_db, 2)
        features["peak_dbfs"] = round(peak_db, 2)
        phrases.append(f"{rms_db:.1f} dBFS RMS")
    if include_timbre:
        timbre = _timbre(mono, sample_rate)
        features["timbre"] = timbre
        phrases.append(str(timbre["brightness"]))
    features["prompt_text"] = ", ".join(phrases) if phrases else "audio clip"
    return copy_audio(audio, waveform), json.dumps(features, ensure_ascii=False)


def beat_slicer(audio: dict, sensitivity: float, min_gap_ms: float, padding_ms: float) -> tuple[dict, str, int]:
    waveform, sample_rate = audio_waveform(audio)
    _a, onset_text, count = onset_detector(audio, sensitivity, min_gap_ms)
    onsets = json.loads(onset_text) if onset_text else []
    duration = waveform.shape[-1] / sample_rate
    points = [0.0] + [float(t) for t in onsets if 0.0 < float(t) < duration] + [duration]
    pad = max(0.0, float(padding_ms) / 1000.0)
    slices = []
    for index, (start, end) in enumerate(zip(points[:-1], points[1:], strict=False)):
        if end <= start:
            continue
        slices.append({"index": index, "start_s": round(max(0.0, start - pad), 4), "end_s": round(min(duration, end + pad), 4)})
    return copy_audio(audio, waveform), json.dumps(slices, ensure_ascii=False), int(len(slices) if slices else count)


def _reference_quality(audio: dict, reference_audio: dict) -> tuple[float, dict]:
    waveform, sample_rate = audio_waveform(audio)
    reference, ref_rate = audio_waveform(reference_audio)
    if ref_rate != sample_rate:
        return 0.0, {"error": "sample_rate_mismatch"}
    length = min(waveform.shape[-1], reference.shape[-1])
    channels = min(waveform.shape[1], reference.shape[1])
    if length <= 0 or channels <= 0:
        return 0.0, {"error": "empty_audio"}
    x = waveform[:, :channels, :length].reshape(-1).detach().cpu().numpy()
    y = reference[:, :channels, :length].reshape(-1).detach().cpu().numpy()
    noise = x - y
    snr = 10.0 * math.log10(float(np.mean(y * y) + 1.0e-8) / float(np.mean(noise * noise) + 1.0e-8))
    corr = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1.0e-8 and np.std(y) > 1.0e-8 else 0.0
    score = float(np.clip(0.5 + snr / 60.0 + 0.25 * corr, 0.0, 1.0))
    return score, {"mode": "reference", "snr_db": round(snr, 2), "correlation": round(corr, 4)}


def audio_quality_estimator(audio: dict, quality_mode: str, reference_audio: dict | None = None) -> tuple[dict, float, str]:
    waveform, sample_rate = audio_waveform(audio)
    if reference_audio is not None:
        score, details = _reference_quality(audio, reference_audio)
        return copy_audio(audio, waveform), score, json.dumps(details, ensure_ascii=False)
    mono = waveform.mean(dim=1)
    peak_db = float(amp_to_db(torch.amax(torch.abs(mono)) + AUDIO_EPS).item())
    rms_db = float(amp_to_db(torch.sqrt(torch.mean(mono * mono) + AUDIO_EPS)).item())
    silence_ratio = float(torch.mean((torch.abs(mono) < 1.0e-4).to(torch.float32)).item())
    clipped_ratio = float(torch.mean((torch.abs(mono) > 0.995).to(torch.float32)).item())
    crest = peak_db - rms_db
    mode_bias = 0.05 if quality_mode == "music" else 0.0
    score = 1.0 - min(0.45, clipped_ratio * 8.0) - min(0.25, max(0.0, silence_ratio - 0.35) * 0.5)
    score -= min(0.2, abs(crest - (12.0 if quality_mode == "speech" else 14.0)) / 80.0)
    score = float(np.clip(score + mode_bias, 0.0, 1.0))
    details = {"mode": quality_mode, "score_0_1": round(score, 4), "peak_dbfs": round(peak_db, 2), "rms_dbfs": round(rms_db, 2), "crest_db": round(crest, 2), "clipped_ratio": round(clipped_ratio, 6), "silence_ratio": round(silence_ratio, 4)}
    return copy_audio(audio, waveform), score, json.dumps(details, ensure_ascii=False)
