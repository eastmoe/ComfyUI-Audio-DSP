from __future__ import annotations

import math

import numpy as np
import torch

from .common import audio_waveform, ba_filter_waveform, butter_sos, clamp01, copy_audio, db_to_amp, mix_audio, sos_filter_waveform

WAVEFORMS = ["sine", "triangle", "square"]


def _to_numpy(waveform: torch.Tensor) -> np.ndarray:
    return waveform.detach().cpu().numpy().astype(np.float32, copy=False)


def _from_numpy(data: np.ndarray, like: torch.Tensor) -> torch.Tensor:
    return torch.from_numpy(data.astype(np.float32, copy=False)).to(device=like.device, dtype=like.dtype)


def _lfo(length: int, sample_rate: int, rate_hz: float, waveform: str = "sine", phase: float = 0.0) -> np.ndarray:
    t = np.arange(length, dtype=np.float32) / float(sample_rate)
    phase_value = (t * max(float(rate_hz), 0.0) + phase / (2.0 * math.pi)) % 1.0
    if waveform == "triangle":
        return (4.0 * np.abs(phase_value - 0.5) - 1.0).astype(np.float32)
    if waveform == "square":
        return np.where(phase_value < 0.5, 1.0, -1.0).astype(np.float32)
    return np.sin(2.0 * math.pi * phase_value).astype(np.float32)


def _fractional_delay(x: np.ndarray, delays: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float32)
    length = x.shape[-1]
    for n in range(length):
        index = n - float(delays[n])
        if 0.0 <= index < length - 1:
            left = int(math.floor(index))
            frac = index - left
            out[n] = x[left] * (1.0 - frac) + x[left + 1] * frac
    return out


def _mod_delay_channel(x: np.ndarray, sample_rate: int, base_ms: float, depth_ms: float, rate_hz: float, phase: float = 0.0, waveform: str = "sine") -> np.ndarray:
    lfo = _lfo(x.shape[-1], sample_rate, rate_hz, waveform, phase)
    delays = sample_rate * (float(base_ms) + float(depth_ms) * (0.5 + 0.5 * lfo)) / 1000.0
    return _fractional_delay(x, delays)


def chorus(audio: dict, voices: int, delay_ms: float, depth_ms: float, rate_hz: float, feedback: float, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    data = _to_numpy(waveform)
    wet = np.zeros_like(data, dtype=np.float32)
    voices = max(1, min(int(voices), 8))
    for b in range(data.shape[0]):
        for c in range(data.shape[1]):
            channel = np.zeros(data.shape[-1], dtype=np.float32)
            last = np.zeros_like(channel)
            for voice in range(voices):
                phase = 2.0 * math.pi * voice / voices + c * math.pi * 0.5
                delayed = _mod_delay_channel(data[b, c] + last * float(feedback), sample_rate, delay_ms, depth_ms, rate_hz, phase)
                channel += delayed / voices
                last = delayed
            wet[b, c] = channel
    return copy_audio(audio, mix_audio(waveform, _from_numpy(wet, waveform), mix))


def flanger(audio: dict, base_delay_ms: float, depth_ms: float, rate_hz: float, feedback: float, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    data = _to_numpy(waveform)
    wet = np.empty_like(data)
    feedback = max(-0.95, min(float(feedback), 0.95))
    for b in range(data.shape[0]):
        for c in range(data.shape[1]):
            delayed = _mod_delay_channel(data[b, c], sample_rate, base_delay_ms, depth_ms, rate_hz, c * math.pi)
            wet[b, c] = delayed + feedback * np.concatenate([np.zeros(1, dtype=np.float32), delayed[:-1]])
    return copy_audio(audio, mix_audio(waveform, _from_numpy(wet, waveform), mix))


def _first_order_allpass(x: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    y = np.zeros_like(x, dtype=np.float32)
    x1 = 0.0
    y1 = 0.0
    for n in range(x.shape[-1]):
        a = float(coeff[n])
        y[n] = -a * x[n] + x1 + a * y1
        x1 = float(x[n])
        y1 = float(y[n])
    return y


def phaser(audio: dict, stages: int, rate_hz: float, depth: float, feedback: float, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    data = _to_numpy(waveform)
    wet = np.empty_like(data)
    stages = max(1, min(int(stages), 12))
    lfo_base = 0.5 + 0.5 * _lfo(data.shape[-1], sample_rate, rate_hz)
    coeff = 0.15 + clamp01(depth) * 0.75 * lfo_base
    for b in range(data.shape[0]):
        for c in range(data.shape[1]):
            y = data[b, c].copy()
            for _ in range(stages):
                y = _first_order_allpass(y, coeff)
            wet[b, c] = y + float(feedback) * (y - data[b, c])
    return copy_audio(audio, mix_audio(waveform, _from_numpy(wet, waveform), mix))


def tremolo(audio: dict, rate_hz: float, depth: float, waveform_shape: str, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    lfo = torch.from_numpy(_lfo(waveform.shape[-1], sample_rate, rate_hz, waveform_shape)).to(device=waveform.device, dtype=waveform.dtype)
    gain = 1.0 - clamp01(depth) * (0.5 + 0.5 * lfo)
    wet = waveform * gain.view(1, 1, -1)
    return copy_audio(audio, mix_audio(waveform, wet, mix))


def vibrato(audio: dict, depth_ms: float, rate_hz: float, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    data = _to_numpy(waveform)
    wet = np.empty_like(data)
    for b in range(data.shape[0]):
        for c in range(data.shape[1]):
            wet[b, c] = _mod_delay_channel(data[b, c], sample_rate, depth_ms, depth_ms, rate_hz, c * math.pi * 0.5)
    return copy_audio(audio, mix_audio(waveform, _from_numpy(wet, waveform), mix))


def rotary_speaker(audio: dict, low_rate_hz: float, high_rate_hz: float, depth: float, crossover_hz: float, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    low = sos_filter_waveform(waveform, butter_sos(sample_rate, "lowpass", crossover_hz, order=2), zero_phase=False)
    high = sos_filter_waveform(waveform, butter_sos(sample_rate, "highpass", crossover_hz, order=2), zero_phase=False)
    length = waveform.shape[-1]
    low_lfo = torch.from_numpy(_lfo(length, sample_rate, low_rate_hz)).to(device=waveform.device, dtype=waveform.dtype)
    high_lfo = torch.from_numpy(_lfo(length, sample_rate, high_rate_hz, phase=math.pi * 0.5)).to(device=waveform.device, dtype=waveform.dtype)
    low_wet = low * (1.0 + 0.25 * clamp01(depth) * low_lfo.view(1, 1, -1))
    high_wet = high * (1.0 + 0.45 * clamp01(depth) * high_lfo.view(1, 1, -1))
    wet = low_wet + high_wet
    if wet.shape[1] >= 2:
        wet[:, 0] *= 1.0 - 0.25 * clamp01(depth) * high_lfo
        wet[:, 1] *= 1.0 + 0.25 * clamp01(depth) * high_lfo
    return copy_audio(audio, mix_audio(waveform, wet, mix))


def ring_modulator(audio: dict, carrier_hz: float, depth: float, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    t = torch.arange(waveform.shape[-1], device=waveform.device, dtype=waveform.dtype) / float(sample_rate)
    carrier = torch.sin(2.0 * math.pi * float(carrier_hz) * t)
    wet = waveform * ((1.0 - clamp01(depth)) + clamp01(depth) * carrier.view(1, 1, -1))
    return copy_audio(audio, mix_audio(waveform, wet, mix))


def auto_panner(audio: dict, rate_hz: float, depth: float, waveform_shape: str, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    source = waveform if waveform.shape[1] >= 2 else waveform.repeat(1, 2, 1)
    lfo = torch.from_numpy(_lfo(source.shape[-1], sample_rate, rate_hz, waveform_shape)).to(device=source.device, dtype=source.dtype)
    pan = clamp01(depth) * lfo
    left = source[:, :1] * torch.sqrt(torch.clamp((1.0 - pan) * 0.5, min=0.0)).view(1, 1, -1)
    right = source[:, 1:2] * torch.sqrt(torch.clamp((1.0 + pan) * 0.5, min=0.0)).view(1, 1, -1)
    wet = torch.cat([left, right, source[:, 2:]], dim=1)
    return copy_audio(audio, mix_audio(source, wet, mix))


def uni_vibe(audio: dict, rate_hz: float, depth: float, chorus_mix: float, mix: float) -> dict:
    phased = phaser(audio, 4, rate_hz, depth, 0.15, 1.0)
    chorused = chorus(phased, 2, 7.0, 2.5 * clamp01(depth), rate_hz, 0.05, chorus_mix)
    waveform, _sample_rate = audio_waveform(audio)
    return copy_audio(audio, mix_audio(waveform, chorused["waveform"], mix))
