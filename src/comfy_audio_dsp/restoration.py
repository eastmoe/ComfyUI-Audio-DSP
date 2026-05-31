from __future__ import annotations

import numpy as np
import torch
from scipy import signal

from .common import audio_waveform, butter_sos, clamp01, copy_audio, db_to_amp, flatten_channels, mix_audio, restore_channels, sos_filter_waveform


def _to_numpy(waveform: torch.Tensor) -> np.ndarray:
    return waveform.detach().cpu().numpy().astype(np.float32, copy=False)


def de_clip(audio: dict, threshold: float, repair_ms: float, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    flat, shape = flatten_channels(waveform)
    data = _to_numpy(flat)
    repaired = data.copy()
    threshold = max(0.05, min(float(threshold), 0.999))
    margin = max(1, int(round(float(repair_ms) * sample_rate / 1000.0)))
    for row in range(data.shape[0]):
        clipped = np.abs(data[row]) >= threshold
        if not np.any(clipped):
            continue
        mask = clipped.copy()
        if margin > 1:
            mask = signal.convolve(mask.astype(np.float32), np.ones(margin, dtype=np.float32), mode="same") > 0.0
        good = np.flatnonzero(~mask)
        bad = np.flatnonzero(mask)
        if len(good) < 2:
            repaired[row, bad] = np.median(data[row])
        else:
            interp = np.interp(bad, good, data[row, good]).astype(np.float32)
            fade = np.minimum(1.0, np.maximum(0.0, np.abs(data[row, bad]) / threshold - 1.0) * 4.0 + 0.65)
            repaired[row, bad] = data[row, bad] * (1.0 - fade) + interp * fade
    wet = restore_channels(torch.from_numpy(repaired).to(device=waveform.device, dtype=waveform.dtype), shape)
    return copy_audio(audio, mix_audio(waveform, wet, mix))


def de_reverb(audio: dict, strength: float, fft_size: int, hop_size: int, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    flat, shape = flatten_channels(waveform)
    data = _to_numpy(flat)
    fft_size = max(128, int(fft_size))
    hop_size = max(1, int(hop_size))
    strength = clamp01(strength)
    out = np.zeros_like(data, dtype=np.float32)
    for row in range(data.shape[0]):
        _, _, stft = signal.stft(data[row], fs=sample_rate, nperseg=fft_size, noverlap=max(0, fft_size - hop_size), boundary="zeros", padded=True)
        mag = np.abs(stft)
        phase = np.exp(1j * np.angle(stft))
        local_peak = np.maximum.accumulate(mag[:, ::-1], axis=1)[:, ::-1]
        tail_floor = signal.medfilt(mag, kernel_size=(1, min(9, mag.shape[1] if mag.shape[1] % 2 else mag.shape[1] - 1) or 1))
        direct = np.maximum(mag - tail_floor * (0.5 + 2.5 * strength), 0.0)
        transient_bias = np.clip(mag / np.maximum(local_peak, 1.0e-8), 0.0, 1.0)
        gain = (1.0 - strength) + strength * np.maximum(direct / np.maximum(mag, 1.0e-8), transient_bias * 0.35)
        _, y = signal.istft(mag * gain * phase, fs=sample_rate, nperseg=fft_size, noverlap=max(0, fft_size - hop_size), input_onesided=True)
        if y.shape[-1] < data.shape[-1]:
            y = np.pad(y, (0, data.shape[-1] - y.shape[-1]))
        out[row] = y[: data.shape[-1]].astype(np.float32)
    wet = restore_channels(torch.from_numpy(out).to(device=waveform.device, dtype=waveform.dtype), shape)
    return copy_audio(audio, mix_audio(waveform, wet, mix))


def bandwidth_extension(audio: dict, crossover_hz: float, amount: float, drive_db: float, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    crossover = max(1000.0, min(float(crossover_hz), sample_rate * 0.45))
    high = sos_filter_waveform(waveform, butter_sos(sample_rate, "highpass", crossover, order=4), zero_phase=True)
    low = sos_filter_waveform(waveform, butter_sos(sample_rate, "lowpass", crossover, order=4), zero_phase=True)
    driven = torch.tanh(low * float(db_to_amp(drive_db)))
    harmonics = torch.relu(driven) - torch.relu(-driven)
    harmonics = sos_filter_waveform(harmonics, butter_sos(sample_rate, "highpass", crossover, order=2), zero_phase=True)
    wet = waveform + harmonics * clamp01(amount) + high * (0.25 * clamp01(amount))
    peak = torch.amax(torch.abs(wet), dim=-1, keepdim=True)
    wet = wet / torch.clamp(peak, min=1.0)
    return copy_audio(audio, mix_audio(waveform, wet, mix))
