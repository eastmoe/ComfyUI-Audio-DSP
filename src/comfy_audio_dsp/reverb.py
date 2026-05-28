from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy import io, signal

from .common import audio_waveform, butter_sos, clamp01, copy_audio, mix_audio, resample_np, sos_filter_waveform


def _to_numpy(waveform: torch.Tensor) -> np.ndarray:
    return waveform.detach().cpu().numpy().astype(np.float32, copy=False)


def _from_numpy(data: np.ndarray, like: torch.Tensor) -> torch.Tensor:
    return torch.from_numpy(data.astype(np.float32, copy=False)).to(device=like.device, dtype=like.dtype)


def _read_wav(path: str, target_rate: int) -> np.ndarray:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Comfy-Audio-DSP: impulse response WAV not found: {path}")
    sample_rate, data = io.wavfile.read(path)
    if data.dtype.kind in {"i", "u"}:
        peak = float(np.iinfo(data.dtype).max)
        data = data.astype(np.float32) / max(peak, 1.0)
    else:
        data = data.astype(np.float32)
    if data.ndim == 1:
        data = data[None, :]
    else:
        data = data.T
    return resample_np(data, int(sample_rate), target_rate)


def _match_channels(ir: np.ndarray, channels: int) -> np.ndarray:
    if ir.shape[0] == channels:
        return ir
    if ir.shape[0] == 1:
        return np.repeat(ir, channels, axis=0)
    if channels == 1:
        return ir.mean(axis=0, keepdims=True)
    repeats = int(np.ceil(channels / ir.shape[0]))
    return np.tile(ir, (repeats, 1))[:channels]


def _fft_convolve_same(data: np.ndarray, ir: np.ndarray, normalize_ir: bool) -> np.ndarray:
    batch, channels, length = data.shape
    ir = _match_channels(ir, channels)
    if normalize_ir:
        peak = np.max(np.abs(ir), axis=1, keepdims=True)
        ir = ir / np.maximum(peak, 1.0e-6)
    wet = np.zeros_like(data, dtype=np.float32)
    for b in range(batch):
        for c in range(channels):
            wet[b, c] = signal.fftconvolve(data[b, c], ir[c], mode="full")[:length].astype(np.float32)
    return wet


def convolution_reverb(audio: dict, impulse_response_wav: str, pre_delay_ms: float, wet: float, dry: float, normalize_ir: bool) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    data = _to_numpy(waveform)
    ir = _read_wav(impulse_response_wav, sample_rate)
    wet_np = _fft_convolve_same(data, ir, bool(normalize_ir))
    wet_tensor = _from_numpy(wet_np, waveform)
    delay = max(0, int(round(sample_rate * float(pre_delay_ms) / 1000.0)))
    if delay > 0:
        wet_tensor = torch.zeros_like(wet_tensor) if delay >= wet_tensor.shape[-1] else F.pad(wet_tensor[..., :-delay], (delay, 0))
    out = waveform * float(dry) + wet_tensor * float(wet)
    return copy_audio(audio, out)


def _comb_filter(x: np.ndarray, delay: int, feedback: float, damping: float = 0.0) -> np.ndarray:
    y = np.zeros_like(x, dtype=np.float32)
    lp = 0.0
    damping = clamp01(damping)
    for n in range(x.shape[-1]):
        delayed = y[n - delay] if n >= delay else 0.0
        lp = delayed * (1.0 - damping) + lp * damping
        y[n] = x[n] + lp * feedback
    return y


def _allpass_filter(x: np.ndarray, delay: int, feedback: float) -> np.ndarray:
    y = np.zeros_like(x, dtype=np.float32)
    for n in range(x.shape[-1]):
        delayed_x = x[n - delay] if n >= delay else 0.0
        delayed_y = y[n - delay] if n >= delay else 0.0
        y[n] = -feedback * x[n] + delayed_x + feedback * delayed_y
    return y


def _schroeder_np(data: np.ndarray, sample_rate: int, decay_time_s: float, diffusion: float, damping: float = 0.15) -> np.ndarray:
    comb_ms = [29.7, 37.1, 41.1, 43.7]
    allpass_ms = [5.0, 1.7]
    out = np.zeros_like(data, dtype=np.float32)
    diffusion = clamp01(diffusion)
    for index, ms in enumerate(comb_ms):
        delay = max(1, int(round(sample_rate * ms / 1000.0)))
        feedback = 10.0 ** (-3.0 * delay / max(float(decay_time_s) * sample_rate, 1.0))
        out += _comb_filter(data, delay, feedback, damping) * (0.25 + 0.05 * index)
    for ms in allpass_ms:
        out = _allpass_filter(out, max(1, int(round(sample_rate * ms / 1000.0))), 0.35 + 0.35 * diffusion)
    peak = np.max(np.abs(out))
    return out / max(peak, 1.0) if peak > 1.0 else out


def _apply_channelwise(waveform: torch.Tensor, func) -> torch.Tensor:
    data = _to_numpy(waveform)
    wet = np.empty_like(data, dtype=np.float32)
    for b in range(data.shape[0]):
        for c in range(data.shape[1]):
            wet[b, c] = func(data[b, c]).astype(np.float32)
    return _from_numpy(wet, waveform)


def _tone_filter(waveform: torch.Tensor, sample_rate: int, low_cut_hz: float, high_cut_hz: float) -> torch.Tensor:
    wet = waveform
    if low_cut_hz > 20.0:
        wet = sos_filter_waveform(wet, butter_sos(sample_rate, "highpass", low_cut_hz, order=2), zero_phase=False)
    if high_cut_hz < sample_rate * 0.45:
        wet = sos_filter_waveform(wet, butter_sos(sample_rate, "lowpass", high_cut_hz, order=2), zero_phase=False)
    return wet


def schroeder_reverb(
    audio: dict,
    pre_delay_ms: float,
    decay_time_s: float,
    diffusion: float,
    low_cut_hz: float,
    high_cut_hz: float,
    mix: float,
) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    predelay = max(0, int(round(sample_rate * float(pre_delay_ms) / 1000.0)))
    if predelay <= 0:
        source = waveform
    elif predelay >= waveform.shape[-1]:
        source = torch.zeros_like(waveform)
    else:
        source = F.pad(waveform[..., :-predelay], (predelay, 0))
    wet = _apply_channelwise(source, lambda x: _schroeder_np(x, sample_rate, decay_time_s, diffusion))
    wet = _tone_filter(wet, sample_rate, low_cut_hz, high_cut_hz)
    return copy_audio(audio, mix_audio(waveform, wet, mix))


def freeverb_moorer_reverb(audio: dict, room_size: float, damping: float, width: float, early_reflections: float, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    room_size = clamp01(room_size)
    decay = 0.4 + room_size * 5.6
    wet = _apply_channelwise(waveform, lambda x: _schroeder_np(x, sample_rate, decay, 0.78, damping))
    data = _to_numpy(waveform)
    early = np.zeros_like(data, dtype=np.float32)
    for delay_ms, gain in [(7.0, 0.26), (13.0, 0.19), (19.0, 0.14), (29.0, 0.10)]:
        delay = max(1, int(round(sample_rate * delay_ms / 1000.0)))
        early[..., delay:] += data[..., :-delay] * gain * float(early_reflections)
    wet = wet + _from_numpy(early, waveform)
    if wet.shape[1] >= 2:
        mid = wet.mean(dim=1, keepdim=True)
        wet = mid + (wet - mid) * float(width)
    return copy_audio(audio, mix_audio(waveform, wet, mix))


def spring_reverb(audio: dict, tension: float, decay_time_s: float, tone_hz: float, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    tension = clamp01(tension)

    def process(x: np.ndarray) -> np.ndarray:
        y = x.copy()
        for delay_ms, feedback in [(8.0, 0.62), (11.3, 0.58), (17.1, 0.52), (23.7, 0.47)]:
            delay = max(1, int(round(sample_rate * (delay_ms - 3.0 * tension) / 1000.0)))
            y = _allpass_filter(y, delay, feedback)
        return _schroeder_np(y, sample_rate, max(0.1, decay_time_s), 0.55, 0.08)

    wet = _apply_channelwise(waveform, process)
    wet = sos_filter_waveform(wet, butter_sos(sample_rate, "bandpass", (max(80.0, tone_hz * 0.25), min(sample_rate * 0.45, tone_hz * 4.0)), order=2), zero_phase=False)
    return copy_audio(audio, mix_audio(waveform, wet, mix))


def plate_reverb(audio: dict, decay_time_s: float, diffusion: float, damping: float, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)

    def process(x: np.ndarray) -> np.ndarray:
        y = x.copy()
        for delay_ms in [4.7, 6.9, 11.7, 15.1]:
            y = _allpass_filter(y, max(1, int(round(sample_rate * delay_ms / 1000.0))), 0.55 + 0.25 * clamp01(diffusion))
        return _schroeder_np(y, sample_rate, decay_time_s, diffusion, damping * 0.5)

    wet = _apply_channelwise(waveform, process)
    return copy_audio(audio, mix_audio(waveform, wet, mix))


def gated_reverb(audio: dict, reverb_time_s: float, gate_time_ms: float, release_ms: float, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    wet = _apply_channelwise(waveform, lambda x: _schroeder_np(x, sample_rate, reverb_time_s, 0.7, 0.18))
    length = wet.shape[-1]
    gate_samples = max(0, int(round(sample_rate * float(gate_time_ms) / 1000.0)))
    release_samples = max(1, int(round(sample_rate * float(release_ms) / 1000.0)))
    envelope = torch.ones(length, device=waveform.device, dtype=waveform.dtype)
    if gate_samples < length:
        envelope[gate_samples:] = 0.0
        fade_end = min(length, gate_samples + release_samples)
        envelope[gate_samples:fade_end] = torch.linspace(1.0, 0.0, fade_end - gate_samples, device=waveform.device, dtype=waveform.dtype)
    wet = wet * envelope.view(1, 1, -1)
    return copy_audio(audio, mix_audio(waveform, wet, mix))


def reverse_reverb(audio: dict, reverb_time_s: float, diffusion: float, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    reversed_audio = torch.flip(waveform, dims=(-1,))
    wet = _apply_channelwise(reversed_audio, lambda x: _schroeder_np(x, sample_rate, reverb_time_s, diffusion, 0.12))
    wet = torch.flip(wet, dims=(-1,))
    return copy_audio(audio, mix_audio(waveform, wet, mix))
