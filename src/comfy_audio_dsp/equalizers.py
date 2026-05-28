from __future__ import annotations

import math

import numpy as np
import torch
from scipy import signal

from .common import (
    audio_waveform,
    ba_filter_waveform,
    butter_sos,
    copy_audio,
    db_to_amp,
    flatten_channels,
    mix_audio,
    normalize_frequency,
    restore_channels,
    sos_filter_waveform,
)

FILTER_TYPES = ["low_shelf", "high_shelf", "peak", "low_pass", "high_pass", "band_pass", "band_stop", "notch"]
GRAPHIC_EQ_BANDS = {
    "10": [31.5, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0],
    "15": [25.0, 40.0, 63.0, 100.0, 160.0, 250.0, 400.0, 630.0, 1000.0, 1600.0, 2500.0, 4000.0, 6300.0, 10000.0, 16000.0],
    "31": [
        20.0,
        25.0,
        31.5,
        40.0,
        50.0,
        63.0,
        80.0,
        100.0,
        125.0,
        160.0,
        200.0,
        250.0,
        315.0,
        400.0,
        500.0,
        630.0,
        800.0,
        1000.0,
        1250.0,
        1600.0,
        2000.0,
        2500.0,
        3150.0,
        4000.0,
        5000.0,
        6300.0,
        8000.0,
        10000.0,
        12500.0,
        16000.0,
        20000.0,
    ],
}


def _biquad(sample_rate: int, kind: str, frequency: float, gain_db: float = 0.0, q: float = 0.7071) -> tuple[np.ndarray, np.ndarray]:
    frequency = normalize_frequency(sample_rate, frequency)
    q = max(0.05, float(q))
    omega = 2.0 * math.pi * frequency / sample_rate
    sin_omega = math.sin(omega)
    cos_omega = math.cos(omega)
    alpha = sin_omega / (2.0 * q)
    a_gain = 10.0 ** (gain_db / 40.0)

    if kind == "peak":
        b0 = 1.0 + alpha * a_gain
        b1 = -2.0 * cos_omega
        b2 = 1.0 - alpha * a_gain
        a0 = 1.0 + alpha / a_gain
        a1 = -2.0 * cos_omega
        a2 = 1.0 - alpha / a_gain
    elif kind in {"low_shelf", "high_shelf"}:
        sqrt_a = math.sqrt(a_gain)
        if kind == "low_shelf":
            b0 = a_gain * ((a_gain + 1.0) - (a_gain - 1.0) * cos_omega + 2.0 * sqrt_a * alpha)
            b1 = 2.0 * a_gain * ((a_gain - 1.0) - (a_gain + 1.0) * cos_omega)
            b2 = a_gain * ((a_gain + 1.0) - (a_gain - 1.0) * cos_omega - 2.0 * sqrt_a * alpha)
            a0 = (a_gain + 1.0) + (a_gain - 1.0) * cos_omega + 2.0 * sqrt_a * alpha
            a1 = -2.0 * ((a_gain - 1.0) + (a_gain + 1.0) * cos_omega)
            a2 = (a_gain + 1.0) + (a_gain - 1.0) * cos_omega - 2.0 * sqrt_a * alpha
        else:
            b0 = a_gain * ((a_gain + 1.0) + (a_gain - 1.0) * cos_omega + 2.0 * sqrt_a * alpha)
            b1 = -2.0 * a_gain * ((a_gain - 1.0) + (a_gain + 1.0) * cos_omega)
            b2 = a_gain * ((a_gain + 1.0) + (a_gain - 1.0) * cos_omega - 2.0 * sqrt_a * alpha)
            a0 = (a_gain + 1.0) - (a_gain - 1.0) * cos_omega + 2.0 * sqrt_a * alpha
            a1 = 2.0 * ((a_gain - 1.0) - (a_gain + 1.0) * cos_omega)
            a2 = (a_gain + 1.0) - (a_gain - 1.0) * cos_omega - 2.0 * sqrt_a * alpha
    elif kind == "low_pass":
        b0 = (1.0 - cos_omega) * 0.5
        b1 = 1.0 - cos_omega
        b2 = (1.0 - cos_omega) * 0.5
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_omega
        a2 = 1.0 - alpha
    elif kind == "high_pass":
        b0 = (1.0 + cos_omega) * 0.5
        b1 = -(1.0 + cos_omega)
        b2 = (1.0 + cos_omega) * 0.5
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_omega
        a2 = 1.0 - alpha
    elif kind == "band_pass":
        b0 = alpha
        b1 = 0.0
        b2 = -alpha
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_omega
        a2 = 1.0 - alpha
    elif kind in {"band_stop", "notch"}:
        b0 = 1.0
        b1 = -2.0 * cos_omega
        b2 = 1.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_omega
        a2 = 1.0 - alpha
    else:
        raise ValueError(f"Unsupported biquad filter type: {kind}")

    return np.array([b0, b1, b2], dtype=np.float64) / a0, np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)


def _apply_biquad(waveform: torch.Tensor, sample_rate: int, kind: str, frequency: float, gain_db: float = 0.0, q: float = 0.7071) -> torch.Tensor:
    b, a = _biquad(sample_rate, kind, frequency, gain_db=gain_db, q=q)
    return ba_filter_waveform(waveform, b, a)


def shelf_filter(audio: dict, shelf: str, frequency_hz: float, gain_db: float, q: float, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    kind = "low_shelf" if shelf == "low_shelf" else "high_shelf"
    wet = _apply_biquad(waveform, sample_rate, kind, frequency_hz, gain_db, q)
    return copy_audio(audio, mix_audio(waveform, wet, mix))


def peak_filter(audio: dict, frequency_hz: float, gain_db: float, q: float, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    wet = _apply_biquad(waveform, sample_rate, "peak", frequency_hz, gain_db, q)
    return copy_audio(audio, mix_audio(waveform, wet, mix))


def pass_filter(audio: dict, mode: str, cutoff_hz: float, order: int, q: float, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    if int(order) <= 2:
        wet = _apply_biquad(waveform, sample_rate, "low_pass" if mode == "low_pass" else "high_pass", cutoff_hz, q=q)
    else:
        wet = sos_filter_waveform(waveform, butter_sos(sample_rate, "lowpass" if mode == "low_pass" else "highpass", cutoff_hz, order=order))
    return copy_audio(audio, mix_audio(waveform, wet, mix))


def band_filter(audio: dict, mode: str, center_hz: float, bandwidth_hz: float, order: int, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    nyquist = sample_rate * 0.5
    bandwidth_hz = max(20.0, float(bandwidth_hz))
    low = max(20.0, min(float(center_hz) - bandwidth_hz * 0.5, nyquist - 100.0))
    high = max(low + 20.0, min(float(center_hz) + bandwidth_hz * 0.5, nyquist - 20.0))
    kind = "bandpass" if mode == "band_pass" else "bandstop"
    wet = sos_filter_waveform(waveform, butter_sos(sample_rate, kind, (low, high), order=order))
    return copy_audio(audio, mix_audio(waveform, wet, mix))


def notch_filter(audio: dict, frequency_hz: float, q: float, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    frequency_hz = normalize_frequency(sample_rate, frequency_hz)
    b, a = signal.iirnotch(frequency_hz, max(1.0, float(q)), fs=sample_rate)
    wet = ba_filter_waveform(waveform, b, a)
    return copy_audio(audio, mix_audio(waveform, wet, mix))


def three_band_eq(
    audio: dict,
    low_frequency_hz: float,
    low_gain_db: float,
    mid_frequency_hz: float,
    mid_gain_db: float,
    mid_q: float,
    high_frequency_hz: float,
    high_gain_db: float,
    mix: float,
) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    wet = _apply_biquad(waveform, sample_rate, "low_shelf", low_frequency_hz, low_gain_db, 0.7071)
    wet = _apply_biquad(wet, sample_rate, "peak", mid_frequency_hz, mid_gain_db, mid_q)
    wet = _apply_biquad(wet, sample_rate, "high_shelf", high_frequency_hz, high_gain_db, 0.7071)
    return copy_audio(audio, mix_audio(waveform, wet, mix))


def _apply_param_band(waveform: torch.Tensor, sample_rate: int, band_type: str, frequency: float, gain_db: float, q: float) -> torch.Tensor:
    if band_type == "low_pass":
        return sos_filter_waveform(waveform, butter_sos(sample_rate, "lowpass", frequency, order=2))
    if band_type == "high_pass":
        return sos_filter_waveform(waveform, butter_sos(sample_rate, "highpass", frequency, order=2))
    if band_type == "band_pass":
        bandwidth = max(20.0, float(frequency) / max(float(q), 0.1))
        return band_filter({"waveform": waveform, "sample_rate": sample_rate}, "band_pass", frequency, bandwidth, 2, 1.0)["waveform"]
    if band_type == "band_stop":
        bandwidth = max(20.0, float(frequency) / max(float(q), 0.1))
        return band_filter({"waveform": waveform, "sample_rate": sample_rate}, "band_stop", frequency, bandwidth, 2, 1.0)["waveform"]
    return _apply_biquad(waveform, sample_rate, band_type, frequency, gain_db, q)


def parametric_eq(audio: dict, bands: int, mix: float, *band_values) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    wet = waveform
    for index in range(max(0, min(int(bands), 8))):
        offset = index * 4
        band_type, frequency, gain_db, q = band_values[offset : offset + 4]
        wet = _apply_param_band(wet, sample_rate, str(band_type), float(frequency), float(gain_db), float(q))
    return copy_audio(audio, mix_audio(waveform, wet, mix))


def graphic_eq(audio: dict, bands: str, q: float, mix: float, *gains_db: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    wet = waveform
    centers = GRAPHIC_EQ_BANDS[str(bands)]
    full_centers = GRAPHIC_EQ_BANDS["31"]
    for center in centers:
        if center >= sample_rate * 0.49:
            continue
        gain_index = full_centers.index(center)
        wet = _apply_biquad(wet, sample_rate, "peak", center, float(gains_db[gain_index]), q)
    return copy_audio(audio, mix_audio(waveform, wet, mix))


def tilt_eq(audio: dict, pivot_hz: float, tilt_db: float, q: float, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    low_gain = -0.5 * float(tilt_db)
    high_gain = 0.5 * float(tilt_db)
    wet = _apply_biquad(waveform, sample_rate, "low_shelf", pivot_hz, low_gain, q)
    wet = _apply_biquad(wet, sample_rate, "high_shelf", pivot_hz, high_gain, q)
    return copy_audio(audio, mix_audio(waveform, wet, mix))


def riaa_eq(audio: dict, mode: str, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    flat, shape = flatten_channels(waveform)
    data = flat.detach().cpu().numpy()
    freqs = np.fft.rfftfreq(data.shape[-1], d=1.0 / sample_rate)
    omega = 2.0 * np.pi * freqs
    t1 = 3180.0e-6
    t2 = 318.0e-6
    t3 = 75.0e-6
    response = (1.0 + 1j * omega * t2) / ((1.0 + 1j * omega * t1) * (1.0 + 1j * omega * t3))
    omega_ref = 2.0 * np.pi * 1000.0
    ref = (1.0 + 1j * omega_ref * t2) / ((1.0 + 1j * omega_ref * t1) * (1.0 + 1j * omega_ref * t3))
    response = response / abs(ref)
    if mode == "pre_emphasis":
        response = 1.0 / np.maximum(np.abs(response), 1.0e-8) * np.exp(-1j * np.angle(response))
    spectrum = np.fft.rfft(data, axis=-1)
    wet_np = np.fft.irfft(spectrum * response[None, :], n=data.shape[-1], axis=-1).astype(np.float32)
    wet = restore_channels(torch.from_numpy(wet_np).to(device=waveform.device, dtype=waveform.dtype), shape)
    return copy_audio(audio, mix_audio(waveform, wet, mix))


def resonant_pass_filter(audio: dict, mode: str, cutoff_hz: float, resonance_q: float, drive_db: float, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    wet = _apply_biquad(waveform, sample_rate, "low_pass" if mode == "low_pass" else "high_pass", cutoff_hz, q=resonance_q)
    if abs(float(drive_db)) > 0.001:
        wet = torch.tanh(wet * float(db_to_amp(drive_db))) / max(float(db_to_amp(drive_db)), 1.0)
    return copy_audio(audio, mix_audio(waveform, wet, mix))
