from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal

AUDIO_EPS = 1.0e-8


def db_to_amp(db: float | torch.Tensor) -> float | torch.Tensor:
    if isinstance(db, torch.Tensor):
        return torch.pow(10.0, db / 20.0)
    return 10.0 ** (db / 20.0)


def amp_to_db(amp: torch.Tensor, eps: float = AUDIO_EPS) -> torch.Tensor:
    return 20.0 * torch.log10(torch.clamp(amp, min=eps))


def clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def copy_audio(audio: dict, waveform: torch.Tensor) -> dict:
    return {"waveform": waveform.to(dtype=torch.float32), "sample_rate": int(audio["sample_rate"])}


def audio_waveform(audio: dict) -> tuple[torch.Tensor, int]:
    if audio is None:
        raise ValueError("Comfy-Audio-DSP: input audio is None.")
    if "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("Comfy-Audio-DSP: expected AUDIO with waveform and sample_rate.")
    waveform = audio["waveform"]
    if not isinstance(waveform, torch.Tensor):
        raise TypeError("Comfy-Audio-DSP: audio waveform must be a torch.Tensor.")
    if waveform.ndim != 3:
        raise ValueError(f"Comfy-Audio-DSP: expected waveform shape [B, C, T], got {tuple(waveform.shape)}.")
    return waveform.to(dtype=torch.float32), int(audio["sample_rate"])


def flatten_channels(waveform: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
    shape = tuple(waveform.shape)
    return waveform.reshape(-1, shape[-1]), shape


def restore_channels(flat: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    return flat.reshape(shape)


def frame_values(
    flat: torch.Tensor,
    frame_size: int,
    mode: Literal["rms", "peak"] = "rms",
) -> tuple[torch.Tensor, int]:
    frame_size = max(1, int(frame_size))
    length = flat.shape[-1]
    pad = (-length) % frame_size
    padded = F.pad(flat, (0, pad)) if pad else flat
    frames = padded.reshape(flat.shape[0], -1, frame_size)
    if mode == "peak":
        values = frames.abs().amax(dim=-1)
    else:
        values = torch.sqrt(torch.mean(frames * frames, dim=-1) + AUDIO_EPS)
    return values, length


def time_coeff(time_ms: float, frame_seconds: float) -> float:
    if time_ms <= 0.0:
        return 0.0
    return math.exp(-frame_seconds / max(time_ms / 1000.0, 1.0e-6))


def smooth_frames(
    target: torch.Tensor,
    sample_rate: int,
    frame_size: int,
    attack_ms: float,
    release_ms: float,
) -> torch.Tensor:
    if target.shape[-1] <= 1:
        return target
    frame_seconds = frame_size / max(float(sample_rate), 1.0)
    attack_coeff = time_coeff(attack_ms, frame_seconds)
    release_coeff = time_coeff(release_ms, frame_seconds)
    smoothed = torch.empty_like(target)
    previous = target[:, 0]
    smoothed[:, 0] = previous
    for index in range(1, target.shape[-1]):
        current = target[:, index]
        coeff = torch.where(current > previous, attack_coeff, release_coeff)
        previous = coeff * previous + (1.0 - coeff) * current
        smoothed[:, index] = previous
    return smoothed


def expand_frames(frames: torch.Tensor, frame_size: int, length: int) -> torch.Tensor:
    return frames.repeat_interleave(frame_size, dim=-1)[..., :length]


def meter_envelope(
    flat: torch.Tensor,
    sample_rate: int,
    attack_ms: float,
    release_ms: float,
    mode: Literal["rms", "peak"] = "rms",
    frame_ms: float = 5.0,
) -> torch.Tensor:
    frame_size = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    values, length = frame_values(flat, frame_size, mode=mode)
    smoothed = smooth_frames(values, sample_rate, frame_size, attack_ms, release_ms)
    return expand_frames(smoothed, frame_size, length)


def normalize_frequency(sample_rate: int, frequency: float, minimum: float = 20.0) -> float:
    nyquist = sample_rate * 0.5
    return max(minimum, min(float(frequency), nyquist * 0.98))


def butter_sos(sample_rate: int, kind: str, cutoff: float | tuple[float, float], order: int = 4) -> np.ndarray:
    nyquist = sample_rate * 0.5
    if isinstance(cutoff, tuple):
        low = max(20.0, min(float(cutoff[0]), nyquist - 100.0))
        high = max(low + 20.0, min(float(cutoff[1]), nyquist - 20.0))
        wn = (low / nyquist, high / nyquist)
    else:
        wn = max(20.0 / nyquist, min(float(cutoff) / nyquist, 0.98))
    return signal.butter(max(1, int(order)), wn, btype=kind, output="sos")


def sos_filter_waveform(waveform: torch.Tensor, sos: np.ndarray, zero_phase: bool = True) -> torch.Tensor:
    device = waveform.device
    dtype = waveform.dtype
    flat, shape = flatten_channels(waveform)
    data = flat.detach().cpu().numpy()
    filtered = np.empty_like(data, dtype=np.float32)
    padlen = 3 * (2 * sos.shape[0] + 1)
    for row in range(data.shape[0]):
        if zero_phase and data.shape[1] > padlen:
            filtered[row] = signal.sosfiltfilt(sos, data[row]).astype(np.float32)
        else:
            filtered[row] = signal.sosfilt(sos, data[row]).astype(np.float32)
    return torch.from_numpy(filtered).to(device=device, dtype=dtype).reshape(shape)


def ba_filter_waveform(waveform: torch.Tensor, b: np.ndarray, a: np.ndarray, zero_phase: bool = True) -> torch.Tensor:
    device = waveform.device
    dtype = waveform.dtype
    flat, shape = flatten_channels(waveform)
    data = flat.detach().cpu().numpy()
    filtered = np.empty_like(data, dtype=np.float32)
    padlen = 3 * max(len(a), len(b))
    for row in range(data.shape[0]):
        if zero_phase and data.shape[1] > padlen:
            filtered[row] = signal.filtfilt(b, a, data[row]).astype(np.float32)
        else:
            filtered[row] = signal.lfilter(b, a, data[row]).astype(np.float32)
    return torch.from_numpy(filtered).to(device=device, dtype=dtype).reshape(shape)


def mix_audio(dry: torch.Tensor, wet: torch.Tensor, mix: float) -> torch.Tensor:
    if wet.shape[-1] < dry.shape[-1]:
        wet = F.pad(wet, (0, dry.shape[-1] - wet.shape[-1]))
    elif wet.shape[-1] > dry.shape[-1]:
        wet = wet[..., : dry.shape[-1]]
    return dry.lerp(wet, clamp01(mix))


def resample_np(data: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if int(source_rate) == int(target_rate):
        return data.astype(np.float32, copy=False)
    gcd = math.gcd(int(source_rate), int(target_rate))
    return signal.resample_poly(data, int(target_rate) // gcd, int(source_rate) // gcd, axis=-1).astype(np.float32)
