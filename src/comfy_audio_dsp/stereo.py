from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .common import audio_waveform, butter_sos, clamp01, copy_audio, db_to_amp, mix_audio, sos_filter_waveform
from .equalizers import FILTER_TYPES, _apply_param_band

MID_SIDE_EQ_FILTER_TYPES = FILTER_TYPES


def _ensure_stereo(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.shape[1] >= 2:
        return waveform
    return waveform.repeat(1, 2, 1)


def _mid_side(waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source = _ensure_stereo(waveform)
    left = source[:, :1]
    right = source[:, 1:2]
    mid = (left + right) * 0.5
    side = (left - right) * 0.5
    return source, mid, side


def _decode_mid_side(mid: torch.Tensor, side: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
    stereo = torch.cat([mid + side, mid - side], dim=1)
    if extra is not None and extra.shape[1] > 2:
        stereo = torch.cat([stereo, extra[:, 2:]], dim=1)
    return stereo


def panner_balance(audio: dict, pan: float, equal_power: bool, mix: float) -> dict:
    waveform, _sample_rate = audio_waveform(audio)
    source = _ensure_stereo(waveform)
    pan = max(-1.0, min(float(pan), 1.0))
    if bool(equal_power):
        angle = (pan + 1.0) * math.pi * 0.25
        left_gain = math.cos(angle)
        right_gain = math.sin(angle)
    else:
        left_gain = 1.0 - max(pan, 0.0)
        right_gain = 1.0 + min(pan, 0.0)
    wet = source.clone()
    wet[:, :1] *= left_gain
    wet[:, 1:2] *= right_gain
    return copy_audio(audio, mix_audio(source, wet, mix))


def stereo_width(audio: dict, width: float, gain_db: float, mix: float) -> dict:
    waveform, _sample_rate = audio_waveform(audio)
    source, mid, side = _mid_side(waveform)
    wet = _decode_mid_side(mid, side * float(width), source) * float(db_to_amp(gain_db))
    return copy_audio(audio, mix_audio(source, wet, mix))


def mid_side_encoder(audio: dict, normalize: bool) -> dict:
    waveform, _sample_rate = audio_waveform(audio)
    source, mid, side = _mid_side(waveform)
    if bool(normalize):
        scale = math.sqrt(2.0)
        mid = mid / scale
        side = side / scale
    encoded = torch.cat([mid, side, source[:, 2:]], dim=1)
    return copy_audio(audio, encoded)


def mid_side_decoder(audio: dict, normalize: bool) -> dict:
    waveform, _sample_rate = audio_waveform(audio)
    source = _ensure_stereo(waveform)
    mid = source[:, :1]
    side = source[:, 1:2]
    if bool(normalize):
        scale = math.sqrt(2.0)
        mid = mid * scale
        side = side * scale
    return copy_audio(audio, _decode_mid_side(mid, side, source))


def mid_side_eq(
    audio: dict,
    mid_filter_type: str,
    mid_frequency_hz: float,
    mid_gain_db: float,
    mid_q: float,
    side_filter_type: str,
    side_frequency_hz: float,
    side_gain_db: float,
    side_q: float,
    mix: float,
) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    source, mid, side = _mid_side(waveform)
    mid = _apply_param_band(mid, sample_rate, mid_filter_type, mid_frequency_hz, mid_gain_db, mid_q)
    side = _apply_param_band(side, sample_rate, side_filter_type, side_frequency_hz, side_gain_db, side_q)
    wet = _decode_mid_side(mid, side, source)
    return copy_audio(audio, mix_audio(source, wet, mix))


def stereo_enhancer_haas(audio: dict, delay_ms: float, side: str, feedback: float, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    source = _ensure_stereo(waveform)
    delay = max(0, int(round(sample_rate * float(delay_ms) / 1000.0)))
    wet = source.clone()
    if delay > 0:
        delayed = torch.zeros_like(source[:, :1]) if delay >= source.shape[-1] else F.pad(source[:, :1, :-delay], (delay, 0))
        target = 1 if side == "right" else 0
        source_channel = 0 if target == 1 else 1
        wet[:, target : target + 1] = source[:, target : target + 1] + delayed * float(feedback)
        if source_channel == 1:
            delayed = torch.zeros_like(source[:, 1:2]) if delay >= source.shape[-1] else F.pad(source[:, 1:2, :-delay], (delay, 0))
            wet[:, target : target + 1] = source[:, target : target + 1] + delayed * float(feedback)
    return copy_audio(audio, mix_audio(source, wet, mix))


def swap_channels(audio: dict) -> dict:
    waveform, _sample_rate = audio_waveform(audio)
    source = _ensure_stereo(waveform)
    swapped = source.clone()
    swapped[:, :1] = source[:, 1:2]
    swapped[:, 1:2] = source[:, :1]
    return copy_audio(audio, swapped)


def mono_maker(audio: dict, cutoff_hz: float, slope_order: int, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    source = _ensure_stereo(waveform)
    low = sos_filter_waveform(source[:, :2], butter_sos(sample_rate, "lowpass", cutoff_hz, order=slope_order), zero_phase=False)
    high = source[:, :2] - low
    mono_low = low.mean(dim=1, keepdim=True).repeat(1, 2, 1)
    wet = torch.cat([mono_low + high, source[:, 2:]], dim=1)
    return copy_audio(audio, mix_audio(source, wet, mix))
