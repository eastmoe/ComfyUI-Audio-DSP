from __future__ import annotations

import numpy as np
import torch
from scipy import signal

from .common import audio_waveform, copy_audio, db_to_amp, flatten_channels, mix_audio, restore_channels


def _stft_process(waveform: torch.Tensor, sample_rate: int, fft_size: int, hop_size: int, process) -> torch.Tensor:
    flat, shape = flatten_channels(waveform)
    data = flat.detach().cpu().numpy().astype(np.float32, copy=False)
    fft_size = max(64, int(fft_size))
    hop_size = max(1, int(hop_size))
    out = np.zeros_like(data, dtype=np.float32)
    for row in range(data.shape[0]):
        _, _, stft = signal.stft(data[row], fs=sample_rate, nperseg=fft_size, noverlap=max(0, fft_size - hop_size), boundary="zeros", padded=True)
        processed = process(stft)
        _, y = signal.istft(processed, fs=sample_rate, nperseg=fft_size, noverlap=max(0, fft_size - hop_size), input_onesided=True)
        if y.shape[-1] < data.shape[-1]:
            y = np.pad(y, (0, data.shape[-1] - y.shape[-1]))
        out[row] = y[: data.shape[-1]].astype(np.float32)
    return restore_channels(torch.from_numpy(out).to(device=waveform.device, dtype=waveform.dtype), shape)


def spectral_gate(audio: dict, threshold_db: float, reduction_db: float, fft_size: int, hop_size: int, smoothing_bins: int, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    floor_gain = float(db_to_amp(-abs(float(reduction_db))))
    threshold = float(db_to_amp(threshold_db))
    smoothing = max(1, int(smoothing_bins))
    kernel = np.ones(smoothing, dtype=np.float32) / smoothing

    def process(stft: np.ndarray) -> np.ndarray:
        mag = np.abs(stft)
        gate = np.where(mag >= threshold, 1.0, floor_gain).astype(np.float32)
        if smoothing > 1:
            gate = signal.convolve2d(gate, kernel[:, None], mode="same", boundary="symm")
        return stft * gate

    wet = _stft_process(waveform, sample_rate, fft_size, hop_size, process)
    return copy_audio(audio, mix_audio(waveform, wet, mix))


def spectral_freeze(audio: dict, freeze_time_s: float, duration_s: float, fft_size: int, hop_size: int, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    target_length = waveform.shape[-1] if duration_s <= 0.0 else max(1, int(round(float(duration_s) * sample_rate)))
    flat, shape = flatten_channels(waveform)
    data = flat.detach().cpu().numpy().astype(np.float32, copy=False)
    fft_size = max(64, int(fft_size))
    hop_size = max(1, int(hop_size))
    out = np.zeros((data.shape[0], target_length), dtype=np.float32)
    frame_index = max(0, int(round(float(freeze_time_s) * sample_rate / hop_size)))
    for row in range(data.shape[0]):
        _, _, stft = signal.stft(data[row], fs=sample_rate, nperseg=fft_size, noverlap=max(0, fft_size - hop_size), boundary="zeros", padded=True)
        selected = stft[:, min(frame_index, stft.shape[1] - 1)]
        frames = max(1, int(np.ceil(target_length / hop_size)) + 2)
        frozen = np.repeat(selected[:, None], frames, axis=1)
        _, y = signal.istft(frozen, fs=sample_rate, nperseg=fft_size, noverlap=max(0, fft_size - hop_size), input_onesided=True)
        if y.shape[-1] < target_length:
            y = np.pad(y, (0, target_length - y.shape[-1]))
        out[row] = y[:target_length].astype(np.float32)
    wet_shape = (shape[0], shape[1], target_length)
    wet = torch.from_numpy(out).to(device=waveform.device, dtype=waveform.dtype).reshape(wet_shape)
    dry = waveform
    if dry.shape[-1] < target_length:
        dry = torch.nn.functional.pad(dry, (0, target_length - dry.shape[-1]))
    elif dry.shape[-1] > target_length:
        dry = dry[..., :target_length]
    return copy_audio(audio, mix_audio(dry, wet, mix))


def frequency_shifter(audio: dict, shift_hz: float, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    flat, shape = flatten_channels(waveform)
    data = flat.detach().cpu().numpy().astype(np.float32, copy=False)
    t = np.arange(data.shape[-1], dtype=np.float32) / float(sample_rate)
    oscillator = np.exp(2j * np.pi * float(shift_hz) * t)
    wet = np.empty_like(data, dtype=np.float32)
    for row in range(data.shape[0]):
        wet[row] = np.real(signal.hilbert(data[row]) * oscillator).astype(np.float32)
    wet_tensor = restore_channels(torch.from_numpy(wet).to(device=waveform.device, dtype=waveform.dtype), shape)
    return copy_audio(audio, mix_audio(waveform, wet_tensor, mix))


def spectral_blur(audio: dict, frequency_blur_bins: int, time_blur_frames: int, fft_size: int, hop_size: int, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    fb = max(1, int(frequency_blur_bins))
    tb = max(1, int(time_blur_frames))
    kernel = np.ones((fb, tb), dtype=np.float32) / float(fb * tb)

    def process(stft: np.ndarray) -> np.ndarray:
        mag = np.abs(stft)
        phase = np.exp(1j * np.angle(stft))
        blurred = signal.convolve2d(mag, kernel, mode="same", boundary="symm")
        return blurred * phase

    wet = _stft_process(waveform, sample_rate, fft_size, hop_size, process)
    return copy_audio(audio, mix_audio(waveform, wet, mix))


def spectral_noise_reduction(audio: dict, noise_profile_s: float, reduction_db: float, sensitivity: float, fft_size: int, hop_size: int, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    profile_frames = max(1, int(round(float(noise_profile_s) * sample_rate / max(int(hop_size), 1))))
    floor_gain = float(db_to_amp(-abs(float(reduction_db))))
    sensitivity = max(0.1, float(sensitivity))

    def process(stft: np.ndarray) -> np.ndarray:
        mag = np.abs(stft)
        phase = np.exp(1j * np.angle(stft))
        noise = np.mean(mag[:, : min(profile_frames, mag.shape[1])], axis=1, keepdims=True)
        over = np.maximum(mag - noise * sensitivity, 0.0)
        gain = np.maximum(over / np.maximum(mag, 1.0e-8), floor_gain)
        return mag * gain * phase

    wet = _stft_process(waveform, sample_rate, fft_size, hop_size, process)
    return copy_audio(audio, mix_audio(waveform, wet, mix))
