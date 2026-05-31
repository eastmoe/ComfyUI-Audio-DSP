from __future__ import annotations

import math

import numpy as np
import torch
from scipy import signal

from .common import copy_audio

NOISE_TYPES = ["white", "pink", "brown"]
SWEEP_MODES = ["linear", "logarithmic"]
OSCILLATOR_WAVES = ["sine", "triangle", "saw", "square"]


def _audio(waveform: torch.Tensor, sample_rate: int) -> dict:
    return {"waveform": waveform.to(dtype=torch.float32), "sample_rate": int(sample_rate)}


def _time(duration_s: float, sample_rate: int) -> torch.Tensor:
    length = max(1, int(round(float(duration_s) * int(sample_rate))))
    return torch.arange(length, dtype=torch.float32) / float(sample_rate)


def sine_wave_generator(frequency_hz: float, amplitude: float, duration_s: float, sample_rate: int, channels: int) -> dict:
    t = _time(duration_s, sample_rate)
    wave = torch.sin(2.0 * math.pi * float(frequency_hz) * t) * float(amplitude)
    return _audio(wave.view(1, 1, -1).repeat(1, max(1, int(channels)), 1), sample_rate)


def noise_generator(noise_type: str, amplitude: float, duration_s: float, sample_rate: int, channels: int, seed: int) -> dict:
    rng = np.random.default_rng(int(seed))
    length = max(1, int(round(float(duration_s) * int(sample_rate))))
    data = rng.standard_normal((1, max(1, int(channels)), length), dtype=np.float32)
    if noise_type == "pink":
        freqs = np.fft.rfftfreq(length, d=1.0 / sample_rate)
        scale = 1.0 / np.sqrt(np.maximum(freqs, freqs[1] if len(freqs) > 1 else 1.0))
        data = np.fft.irfft(np.fft.rfft(data, axis=-1) * scale[None, None, :], n=length, axis=-1).astype(np.float32)
    elif noise_type == "brown":
        data = np.cumsum(data, axis=-1).astype(np.float32)
    data = data / max(float(np.max(np.abs(data))), 1.0e-6) * float(amplitude)
    return _audio(torch.from_numpy(data), sample_rate)


def sweep_chirp(start_hz: float, end_hz: float, amplitude: float, duration_s: float, sample_rate: int, mode: str, channels: int) -> dict:
    length = max(1, int(round(float(duration_s) * int(sample_rate))))
    t = np.arange(length, dtype=np.float32) / float(sample_rate)
    method = "logarithmic" if mode == "logarithmic" and start_hz > 0.0 and end_hz > 0.0 else "linear"
    data = signal.chirp(t, f0=max(0.01, float(start_hz)), f1=max(0.01, float(end_hz)), t1=max(float(duration_s), 1.0e-6), method=method).astype(np.float32)
    wave = torch.from_numpy(data * float(amplitude)).view(1, 1, -1).repeat(1, max(1, int(channels)), 1)
    return _audio(wave, sample_rate)


def impulse(amplitude: float, duration_s: float, sample_rate: int, position_ms: float, click_ms: float, channels: int) -> dict:
    length = max(1, int(round(float(duration_s) * int(sample_rate))))
    wave = torch.zeros(1, max(1, int(channels)), length, dtype=torch.float32)
    start = max(0, min(length - 1, int(round(float(position_ms) * sample_rate / 1000.0))))
    click = max(1, int(round(float(click_ms) * sample_rate / 1000.0)))
    end = min(length, start + click)
    envelope = torch.linspace(1.0, 0.0, end - start, dtype=torch.float32)
    wave[..., start:end] = float(amplitude) * envelope.view(1, 1, -1)
    return _audio(wave, sample_rate)


def oscillator_multiwave(waveform_shape: str, frequency_hz: float, amplitude: float, duty_cycle: float, duration_s: float, sample_rate: int, channels: int) -> dict:
    t = _time(duration_s, sample_rate)
    phase = (t * float(frequency_hz)) % 1.0
    if waveform_shape == "triangle":
        wave = 4.0 * torch.abs(phase - 0.5) - 1.0
    elif waveform_shape == "saw":
        wave = 2.0 * phase - 1.0
    elif waveform_shape == "square":
        wave = torch.where(phase < max(0.01, min(float(duty_cycle), 0.99)), 1.0, -1.0)
    else:
        wave = torch.sin(2.0 * math.pi * phase)
    return _audio((wave * float(amplitude)).view(1, 1, -1).repeat(1, max(1, int(channels)), 1), sample_rate)


def click_track_metronome(bpm: float, beats_per_bar: int, bars: int, sample_rate: int, accent_frequency_hz: float, beat_frequency_hz: float, amplitude: float) -> dict:
    beats = max(1, int(beats_per_bar) * max(1, int(bars)))
    beat_seconds = 60.0 / max(float(bpm), 1.0)
    duration = beats * beat_seconds
    length = max(1, int(round(duration * int(sample_rate))))
    wave = torch.zeros(1, 1, length, dtype=torch.float32)
    click_len = max(1, int(round(0.035 * sample_rate)))
    env = torch.exp(-torch.linspace(0.0, 7.0, click_len))
    t = torch.arange(click_len, dtype=torch.float32) / float(sample_rate)
    for beat in range(beats):
        start = int(round(beat * beat_seconds * sample_rate))
        end = min(length, start + click_len)
        freq = float(accent_frequency_hz) if beat % max(1, int(beats_per_bar)) == 0 else float(beat_frequency_hz)
        click = torch.sin(2.0 * math.pi * freq * t[: end - start]) * env[: end - start] * float(amplitude)
        wave[..., start:end] += click.view(1, 1, -1)
    return _audio(wave.repeat(1, 2, 1), sample_rate)
