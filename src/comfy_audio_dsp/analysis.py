from __future__ import annotations

import json
import math

import numpy as np
import torch
from scipy import signal

from .common import AUDIO_EPS, amp_to_db, audio_waveform, copy_audio
from .dynamics import _integrated_lufs, _loudness_weighted, _power_to_lufs, _weighted_channel_power
from .pitch_time import PITCH_KEYS


def _to_numpy(waveform: torch.Tensor) -> np.ndarray:
    return waveform.detach().cpu().numpy().astype(np.float32, copy=False)


def _image(canvas: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.clip(canvas, 0.0, 1.0).astype(np.float32)).unsqueeze(0)


def _canvas(width: int, height: int) -> np.ndarray:
    image = np.full((max(32, int(height)), max(32, int(width)), 3), 0.055, dtype=np.float32)
    image[::32, :, :] = 0.12
    image[:, ::32, :] = 0.12
    return image


def _draw_polyline(canvas: np.ndarray, points: np.ndarray, color: tuple[float, float, float]) -> None:
    if len(points) < 2:
        return
    height, width, _channels = canvas.shape
    pts = np.round(points).astype(np.int32)
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:], strict=False):
        steps = max(abs(int(x1 - x0)), abs(int(y1 - y0)), 1)
        xs = np.linspace(x0, x1, steps + 1).astype(np.int32)
        ys = np.linspace(y0, y1, steps + 1).astype(np.int32)
        mask = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        canvas[ys[mask], xs[mask]] = color


def _mono_np(audio: dict) -> tuple[np.ndarray, int]:
    waveform, sample_rate = audio_waveform(audio)
    mono = waveform.mean(dim=1).detach().cpu().numpy().astype(np.float32, copy=False)
    return mono, sample_rate


def rms_meter(audio: dict, window_ms: float) -> tuple[dict, float, str]:
    waveform, sample_rate = audio_waveform(audio)
    window = max(1, int(round(sample_rate * float(window_ms) / 1000.0)))
    segment = waveform[..., -window:] if waveform.shape[-1] > window else waveform
    per_channel = amp_to_db(torch.sqrt(torch.mean(segment * segment, dim=-1) + AUDIO_EPS))
    overall = float(torch.mean(per_channel).item())
    text = json.dumps({"overall_dbfs": overall, "channels_dbfs": per_channel.mean(dim=0).detach().cpu().tolist()}, ensure_ascii=False)
    return copy_audio(audio, waveform), overall, text


def peak_meter(audio: dict, overload_db: float) -> tuple[dict, float, bool, str]:
    waveform, _sample_rate = audio_waveform(audio)
    per_channel = amp_to_db(torch.amax(torch.abs(waveform), dim=-1) + AUDIO_EPS)
    overall = float(torch.amax(per_channel).item())
    overload = bool(overall >= float(overload_db))
    text = json.dumps({"peak_dbfs": overall, "overload": overload, "channels_dbfs": per_channel.mean(dim=0).detach().cpu().tolist()}, ensure_ascii=False)
    return copy_audio(audio, waveform), overall, overload, text


def _window_lufs(weighted: torch.Tensor, sample_rate: int, seconds: float) -> torch.Tensor:
    frame = max(1, int(round(seconds * sample_rate)))
    segment = weighted[..., -frame:] if weighted.shape[-1] > frame else weighted
    return _power_to_lufs(_weighted_channel_power(segment))


def lufs_meter(audio: dict) -> tuple[dict, float, float, float, float, str]:
    waveform, sample_rate = audio_waveform(audio)
    weighted = _loudness_weighted(waveform, sample_rate)
    integrated = _integrated_lufs(weighted, sample_rate)
    short_term = _window_lufs(weighted, sample_rate, 3.0)
    momentary = _window_lufs(weighted, sample_rate, 0.4)
    block = max(1, int(round(3.0 * sample_rate)))
    hop = max(1, int(round(1.0 * sample_rate)))
    values = []
    for start in range(0, max(1, weighted.shape[-1] - block + 1), hop):
        values.append(_power_to_lufs(_weighted_channel_power(weighted[..., start : min(start + block, weighted.shape[-1])])))
    stacked = torch.stack(values, dim=-1) if values else short_term.view(-1, 1)
    lra = torch.quantile(stacked, 0.95, dim=-1) - torch.quantile(stacked, 0.10, dim=-1)
    result = {
        "integrated_lufs": integrated.detach().cpu().tolist(),
        "short_term_lufs": short_term.detach().cpu().tolist(),
        "momentary_lufs": momentary.detach().cpu().tolist(),
        "lra_lu": lra.detach().cpu().tolist(),
    }
    return copy_audio(audio, waveform), float(integrated.mean().item()), float(short_term.mean().item()), float(momentary.mean().item()), float(lra.mean().item()), json.dumps(result, ensure_ascii=False)


def spectral_analyzer(audio: dict, fft_size: int, min_db: float, max_db: float, width: int, height: int) -> tuple[dict, torch.Tensor, str]:
    waveform, sample_rate = audio_waveform(audio)
    mono = waveform.mean(dim=1).detach().cpu().numpy()
    fft_size = max(64, int(fft_size))
    segment = mono[:, -fft_size:] if mono.shape[-1] > fft_size else np.pad(mono, ((0, 0), (max(0, fft_size - mono.shape[-1]), 0)))
    window = np.hanning(fft_size).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(segment * window[None, :], axis=-1)).mean(axis=0)
    db = 20.0 * np.log10(np.maximum(spectrum, 1.0e-8))
    freqs = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    canvas = _canvas(width, height)
    norm = np.clip((db - float(min_db)) / max(float(max_db - min_db), 1.0), 0.0, 1.0)
    xs = np.linspace(0, canvas.shape[1] - 1, len(norm))
    ys = (1.0 - norm) * (canvas.shape[0] - 1)
    _draw_polyline(canvas, np.stack([xs, ys], axis=1), (0.2, 0.82, 0.95))
    bins = np.linspace(0, len(freqs) - 1, min(96, len(freqs))).astype(int)
    text = json.dumps({"sample_rate": sample_rate, "frequencies_hz": freqs[bins].round(2).tolist(), "magnitudes_db": db[bins].round(2).tolist()}, ensure_ascii=False)
    return copy_audio(audio, waveform), _image(canvas), text


def spectrogram_visualizer(audio: dict, fft_size: int, hop_size: int, min_db: float, max_db: float, width: int, height: int) -> tuple[dict, torch.Tensor]:
    waveform, sample_rate = audio_waveform(audio)
    mono = waveform.mean(dim=(0, 1)).detach().cpu().numpy().astype(np.float32)
    fft_size = max(64, int(fft_size))
    hop_size = max(1, int(hop_size))
    _, _, stft = signal.stft(mono, fs=sample_rate, nperseg=fft_size, noverlap=max(0, fft_size - hop_size), boundary=None)
    spec = 20.0 * np.log10(np.maximum(np.abs(stft), 1.0e-8))
    spec = np.clip((spec - float(min_db)) / max(float(max_db - min_db), 1.0), 0.0, 1.0)
    spec = signal.resample(signal.resample(spec, max(32, int(height)), axis=0), max(32, int(width)), axis=1)
    spec = np.flipud(spec)
    canvas = np.stack([spec * 0.15, spec * 0.75, spec], axis=-1).astype(np.float32)
    return copy_audio(audio, waveform), _image(canvas)


def waveform_visualizer(audio: dict, width: int, height: int, seconds: float) -> tuple[dict, torch.Tensor]:
    waveform, sample_rate = audio_waveform(audio)
    samples = max(1, int(round(float(seconds) * sample_rate))) if seconds > 0.0 else waveform.shape[-1]
    data = waveform[0, : min(2, waveform.shape[1]), -samples:].detach().cpu().numpy()
    canvas = _canvas(width, height)
    center = canvas.shape[0] // 2
    for channel in range(data.shape[0]):
        values = signal.resample(data[channel], canvas.shape[1])
        y_scale = canvas.shape[0] * (0.32 if data.shape[0] == 1 else 0.22)
        offset = center if data.shape[0] == 1 else int(canvas.shape[0] * (0.32 + 0.36 * channel))
        xs = np.arange(canvas.shape[1])
        ys = offset - values * y_scale
        _draw_polyline(canvas, np.stack([xs, ys], axis=1), (0.3, 0.9, 0.45) if channel == 0 else (0.95, 0.45, 0.35))
    return copy_audio(audio, waveform), _image(canvas)


def phase_correlation_meter(audio: dict) -> tuple[dict, float, str]:
    waveform, _sample_rate = audio_waveform(audio)
    if waveform.shape[1] < 2:
        return copy_audio(audio, waveform), 1.0, json.dumps({"correlation": 1.0, "status": "mono"}, ensure_ascii=False)
    left = waveform[:, 0].reshape(-1)
    right = waveform[:, 1].reshape(-1)
    corr = torch.sum(left * right) / (torch.sqrt(torch.sum(left * left) * torch.sum(right * right)) + AUDIO_EPS)
    value = float(torch.clamp(corr, -1.0, 1.0).item())
    status = "phase_risk" if value < 0.0 else "ok"
    return copy_audio(audio, waveform), value, json.dumps({"correlation": value, "status": status}, ensure_ascii=False)


def goniometer_vectorscope(audio: dict, width: int, height: int) -> tuple[dict, torch.Tensor]:
    waveform, _sample_rate = audio_waveform(audio)
    source = waveform if waveform.shape[1] >= 2 else waveform.repeat(1, 2, 1)
    left = source[0, 0].detach().cpu().numpy()
    right = source[0, 1].detach().cpu().numpy()
    points = min(6000, left.shape[-1])
    if points < left.shape[-1]:
        idx = np.linspace(0, left.shape[-1] - 1, points).astype(int)
        left = left[idx]
        right = right[idx]
    canvas = _canvas(width, height)
    scale = 0.48 * min(canvas.shape[0], canvas.shape[1])
    x = (left - right) * scale + canvas.shape[1] * 0.5
    y = -(left + right) * scale + canvas.shape[0] * 0.5
    pts = np.round(np.stack([x, y], axis=1)).astype(np.int32)
    mask = (pts[:, 0] >= 0) & (pts[:, 0] < canvas.shape[1]) & (pts[:, 1] >= 0) & (pts[:, 1] < canvas.shape[0])
    canvas[pts[mask, 1], pts[mask, 0]] = (0.2, 0.85, 0.7)
    return copy_audio(audio, waveform), _image(canvas)


def bpm_tempo_detector(audio: dict, min_bpm: float, max_bpm: float) -> tuple[dict, float, str]:
    mono, sample_rate = _mono_np(audio)
    x = mono.mean(axis=0)
    frame = max(1, int(round(0.046 * sample_rate)))
    hop = max(1, frame // 2)
    energy = []
    for start in range(0, max(1, len(x) - frame + 1), hop):
        chunk = x[start : start + frame]
        energy.append(float(np.sqrt(np.mean(chunk * chunk) + 1.0e-8)))
    env = np.maximum(0.0, np.diff(np.asarray(energy, dtype=np.float32), prepend=0.0))
    if len(env) < 4 or np.max(env) <= 0.0:
        return copy_audio(audio, audio_waveform(audio)[0]), 0.0, "[]"
    corr = signal.correlate(env - np.mean(env), env - np.mean(env), mode="full")[len(env) - 1 :]
    frame_rate = sample_rate / hop
    min_lag = max(1, int(round(frame_rate * 60.0 / max(float(max_bpm), 1.0))))
    max_lag = min(len(corr) - 1, int(round(frame_rate * 60.0 / max(float(min_bpm), 1.0))))
    if max_lag <= min_lag:
        bpm = 0.0
    else:
        lag = int(np.argmax(corr[min_lag:max_lag])) + min_lag
        bpm = 60.0 * frame_rate / max(lag, 1)
    threshold = np.mean(env) + np.std(env)
    peaks, _ = signal.find_peaks(env, height=threshold, distance=max(1, int(frame_rate * 60.0 / max(bpm, 240.0) * 0.5)))
    beats = (peaks * hop / sample_rate).round(3).tolist()
    return copy_audio(audio, audio_waveform(audio)[0]), float(bpm), json.dumps(beats, ensure_ascii=False)


def key_pitch_detector(audio: dict, min_hz: float, max_hz: float) -> tuple[dict, str, float, str]:
    mono, sample_rate = _mono_np(audio)
    x = mono.mean(axis=0)
    x = x - np.mean(x)
    corr = signal.correlate(x, x, mode="full", method="fft")[len(x) - 1 :]
    min_lag = max(1, int(sample_rate / max(float(max_hz), 1.0)))
    max_lag = min(len(corr) - 1, int(sample_rate / max(float(min_hz), 1.0)))
    if max_lag <= min_lag or float(corr[0]) <= 1.0e-8:
        pitch = 0.0
        key = "Unknown"
    else:
        lag = int(np.argmax(corr[min_lag:max_lag])) + min_lag
        pitch = float(sample_rate / max(lag, 1))
        midi = int(round(69.0 + 12.0 * math.log2(max(pitch, 1.0e-6) / 440.0)))
        key = PITCH_KEYS[midi % 12]
    text = json.dumps({"key": key, "pitch_hz": pitch}, ensure_ascii=False)
    return copy_audio(audio, audio_waveform(audio)[0]), key, pitch, text


def onset_detector(audio: dict, sensitivity: float, min_gap_ms: float) -> tuple[dict, str, int]:
    mono, sample_rate = _mono_np(audio)
    x = mono.mean(axis=0)
    frame = max(1, int(round(0.020 * sample_rate)))
    hop = max(1, frame // 2)
    values = []
    for start in range(0, max(1, len(x) - frame + 1), hop):
        chunk = x[start : start + frame]
        values.append(float(np.sqrt(np.mean(chunk * chunk) + 1.0e-8)))
    flux = np.maximum(0.0, np.diff(np.asarray(values, dtype=np.float32), prepend=0.0))
    threshold = np.mean(flux) + (2.2 - 1.8 * max(0.0, min(float(sensitivity), 1.0))) * np.std(flux)
    distance = max(1, int(round((float(min_gap_ms) / 1000.0) * sample_rate / hop)))
    peaks, _ = signal.find_peaks(flux, height=threshold, distance=distance)
    times = (peaks * hop / sample_rate).round(4).tolist()
    return copy_audio(audio, audio_waveform(audio)[0]), json.dumps(times, ensure_ascii=False), int(len(times))


def silence_detector(audio: dict, threshold_db: float, min_duration_ms: float) -> tuple[dict, str, int]:
    waveform, sample_rate = audio_waveform(audio)
    mono = waveform.mean(dim=1)
    frame = max(1, int(round(0.020 * sample_rate)))
    hop = max(1, frame // 2)
    values = []
    for start in range(0, max(1, mono.shape[-1] - frame + 1), hop):
        chunk = mono[..., start : start + frame]
        values.append(torch.sqrt(torch.mean(chunk * chunk) + AUDIO_EPS))
    db = amp_to_db(torch.stack(values)).detach().cpu().numpy()
    silent = db < float(threshold_db)
    min_frames = max(1, int(round((float(min_duration_ms) / 1000.0) * sample_rate / hop)))
    ranges = []
    start = None
    for index, flag in enumerate(silent.tolist() + [False]):
        if flag and start is None:
            start = index
        if not flag and start is not None:
            if index - start >= min_frames:
                ranges.append([round(start * hop / sample_rate, 4), round(index * hop / sample_rate, 4)])
            start = None
    return copy_audio(audio, waveform), json.dumps(ranges, ensure_ascii=False), int(len(ranges))
