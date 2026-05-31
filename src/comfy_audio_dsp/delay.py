from __future__ import annotations

import math

import numpy as np
import torch

from .common import audio_waveform, butter_sos, clamp01, copy_audio, mix_audio, sos_filter_waveform

NOTE_VALUES = {
    "1/1": 4.0,
    "1/2": 2.0,
    "1/4": 1.0,
    "1/8": 0.5,
    "1/16": 0.25,
    "1/32": 0.125,
    "1/4 dotted": 1.5,
    "1/8 dotted": 0.75,
    "1/16 dotted": 0.375,
    "1/4 triplet": 2.0 / 3.0,
    "1/8 triplet": 1.0 / 3.0,
    "1/16 triplet": 1.0 / 6.0,
}


def _to_numpy(waveform: torch.Tensor) -> np.ndarray:
    return waveform.detach().cpu().numpy().astype(np.float32, copy=False)


def _from_numpy(data: np.ndarray, like: torch.Tensor) -> torch.Tensor:
    return torch.from_numpy(data.astype(np.float32, copy=False)).to(device=like.device, dtype=like.dtype)


def _delay_feedback(x: np.ndarray, delay_samples: int, feedback: float) -> np.ndarray:
    delay_samples = max(1, int(delay_samples))
    feedback = max(-0.98, min(float(feedback), 0.98))
    y = np.zeros_like(x, dtype=np.float32)
    for n in range(delay_samples, x.shape[-1]):
        y[n] = x[n - delay_samples] + y[n - delay_samples] * feedback
    return y


def _fractional_read(x: np.ndarray, index: float) -> float:
    if index <= 0.0 or index >= x.shape[-1] - 1:
        return 0.0
    left = int(math.floor(index))
    frac = index - left
    return float(x[left] * (1.0 - frac) + x[left + 1] * frac)


def _modulated_delay(x: np.ndarray, sample_rate: int, delay_ms: float, depth_ms: float, rate_hz: float, phase: float = 0.0) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float32)
    base = sample_rate * delay_ms / 1000.0
    depth = sample_rate * depth_ms / 1000.0
    for n in range(x.shape[-1]):
        lfo = math.sin(2.0 * math.pi * rate_hz * n / sample_rate + phase)
        out[n] = _fractional_read(x, n - base - depth * lfo)
    return out


def simple_delay(audio: dict, delay_ms: float, feedback: float, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    data = _to_numpy(waveform)
    delay_samples = int(round(sample_rate * delay_ms / 1000.0))
    wet = np.empty_like(data)
    for b in range(data.shape[0]):
        for c in range(data.shape[1]):
            wet[b, c] = _delay_feedback(data[b, c], delay_samples, feedback)
    return copy_audio(audio, mix_audio(waveform, _from_numpy(wet, waveform), mix))


def tempo_synced_delay(audio: dict, bpm: float, note_value: str, feedback: float, mix: float) -> dict:
    beat_ms = 60000.0 / max(float(bpm), 1.0)
    delay_ms = beat_ms * NOTE_VALUES[str(note_value)]
    return simple_delay(audio, delay_ms, feedback, mix)


def ping_pong_delay(audio: dict, delay_ms: float, feedback: float, width: float, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    data = _to_numpy(waveform)
    channels = max(2, data.shape[1])
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    wet = np.zeros_like(data[:, :channels], dtype=np.float32)
    delay_samples = max(1, int(round(sample_rate * delay_ms / 1000.0)))
    feedback = max(-0.98, min(float(feedback), 0.98))
    width = clamp01(width)
    for b in range(data.shape[0]):
        for n in range(delay_samples, data.shape[-1]):
            wet[b, 0, n] = data[b, 1, n - delay_samples] + wet[b, 1, n - delay_samples] * feedback
            wet[b, 1, n] = data[b, 0, n - delay_samples] + wet[b, 0, n - delay_samples] * feedback
        if channels > 2:
            wet[b, 2:] = data[b, 2:] * 0.0
    wet_tensor = _from_numpy(wet[:, : waveform.shape[1]], waveform)
    if waveform.shape[1] >= 2:
        mid = wet_tensor.mean(dim=1, keepdim=True)
        wet_tensor = mid + (wet_tensor - mid) * width
    return copy_audio(audio, mix_audio(waveform, wet_tensor, mix))


def multi_tap_delay(audio: dict, mix: float, *tap_values: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    wet = torch.zeros_like(waveform)
    for index in range(0, len(tap_values), 2):
        delay_ms = float(tap_values[index])
        gain = float(tap_values[index + 1])
        delay = max(0, int(round(sample_rate * delay_ms / 1000.0)))
        if delay == 0:
            wet = wet + waveform * gain
        elif delay < waveform.shape[-1]:
            wet[..., delay:] = wet[..., delay:] + waveform[..., :-delay] * gain
    return copy_audio(audio, mix_audio(waveform, wet, mix))


def dub_delay(audio: dict, delay_ms: float, feedback: float, tone_hz: float, wow_depth_ms: float, wow_rate_hz: float, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    data = _to_numpy(waveform)
    delay_samples = int(round(sample_rate * delay_ms / 1000.0))
    wet = np.empty_like(data)
    for b in range(data.shape[0]):
        for c in range(data.shape[1]):
            delayed = _delay_feedback(data[b, c], delay_samples, feedback)
            flutter = _modulated_delay(delayed, sample_rate, max(delay_ms * 0.08, 2.0), wow_depth_ms, wow_rate_hz, phase=c * math.pi * 0.5)
            wet[b, c] = 0.75 * delayed + 0.25 * flutter
    wet_tensor = _from_numpy(wet, waveform)
    wet_tensor = sos_filter_waveform(wet_tensor, butter_sos(sample_rate, "lowpass", tone_hz, order=2), zero_phase=False)
    wet_tensor = torch.tanh(wet_tensor * 1.8) / 1.8
    return copy_audio(audio, mix_audio(waveform, wet_tensor, mix))


def filtered_delay(audio: dict, delay_ms: float, feedback: float, low_cut_hz: float, high_cut_hz: float, mix: float) -> dict:
    delayed = simple_delay(audio, delay_ms, feedback, 1.0)
    waveform, sample_rate = audio_waveform(audio)
    wet = delayed["waveform"]
    if low_cut_hz > 20.0:
        wet = sos_filter_waveform(wet, butter_sos(sample_rate, "highpass", low_cut_hz, order=2), zero_phase=False)
    if high_cut_hz < sample_rate * 0.45:
        wet = sos_filter_waveform(wet, butter_sos(sample_rate, "lowpass", high_cut_hz, order=2), zero_phase=False)
    return copy_audio(audio, mix_audio(waveform, wet, mix))


def stereo_spread_delay(audio: dict, base_delay_ms: float, spread_ms: float, feedback: float, width: float, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    data = _to_numpy(waveform)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    wet = np.empty_like(data)
    for c in range(data.shape[1]):
        side = -0.5 if c % 2 == 0 else 0.5
        delay_ms = max(0.0, float(base_delay_ms) + side * float(spread_ms))
        delay_samples = int(round(sample_rate * delay_ms / 1000.0))
        for b in range(data.shape[0]):
            wet[b, c] = _delay_feedback(data[b, c], delay_samples, feedback)
    wet_tensor = _from_numpy(wet[:, : waveform.shape[1]], waveform)
    if waveform.shape[1] >= 2:
        mid = wet_tensor.mean(dim=1, keepdim=True)
        wet_tensor = mid + (wet_tensor - mid) * float(width)
    return copy_audio(audio, mix_audio(waveform, wet_tensor, mix))


def reverse_delay(audio: dict, delay_ms: float, feedback: float, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    reversed_audio = {"waveform": torch.flip(waveform, dims=(-1,)), "sample_rate": sample_rate}
    wet = simple_delay(reversed_audio, delay_ms, feedback, 1.0)["waveform"]
    wet = torch.flip(wet, dims=(-1,))
    return copy_audio(audio, mix_audio(waveform, wet, mix))


def granular_delay(audio: dict, delay_ms: float, grain_ms: float, density: float, pitch_semitones: float, feedback: float, seed: int, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    from .pitch_time import granular_processor, pitch_shifter

    delayed = simple_delay(audio, delay_ms, feedback, 1.0)
    grains = granular_processor(delayed, grain_ms, max(1.0, density), 0.0, grain_ms * 0.5, 0.35, 0.0, seed, 1.0)
    if abs(float(pitch_semitones)) > 1.0e-4:
        grains = pitch_shifter(grains, pitch_semitones, 0.0, 1.0)
    return copy_audio(audio, mix_audio(waveform, grains["waveform"], mix))


def slap_echo(audio: dict, style: str, mix: float) -> dict:
    if style == "wide":
        return stereo_spread_delay(audio, 95.0, 18.0, 0.08, 1.35, mix)
    if style == "rockabilly":
        return filtered_delay(audio, 115.0, 0.18, 140.0, 5200.0, mix)
    return filtered_delay(audio, 85.0, 0.05, 120.0, 7000.0, mix)
