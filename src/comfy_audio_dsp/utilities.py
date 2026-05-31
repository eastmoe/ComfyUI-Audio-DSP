from __future__ import annotations

import json
import math

import torch
import torch.nn.functional as F

from .common import AUDIO_EPS, amp_to_db, audio_waveform, butter_sos, copy_audio, db_to_amp, resample_np, sos_filter_waveform
from .dynamics import _integrated_lufs, _loudness_weighted

FADE_CURVES = ["linear", "exponential", "s_curve"]
NORMALIZE_MODES = ["peak", "rms", "lufs"]
FORMAT_MODES = ["mono_mix", "mono_left", "mono_right", "stereo_duplicate"]


def gain_trim(audio: dict, gain_db: float) -> dict:
    waveform, _sample_rate = audio_waveform(audio)
    return copy_audio(audio, waveform * float(db_to_amp(gain_db)))


def phase_inverter(audio: dict, invert_left: bool, invert_right: bool) -> dict:
    waveform, _sample_rate = audio_waveform(audio)
    out = waveform.clone()
    if out.shape[1] == 1:
        if bool(invert_left) or bool(invert_right):
            out = -out
    else:
        if bool(invert_left):
            out[:, :1] = -out[:, :1]
        if bool(invert_right):
            out[:, 1:2] = -out[:, 1:2]
    return copy_audio(audio, out)


def dc_offset_remover(audio: dict, highpass_hz: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    if highpass_hz <= 0.0:
        return copy_audio(audio, waveform - waveform.mean(dim=-1, keepdim=True))
    return copy_audio(audio, sos_filter_waveform(waveform, butter_sos(sample_rate, "highpass", highpass_hz, order=2), zero_phase=False))


def _fade_curve(length: int, curve: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    x = torch.linspace(0.0, 1.0, max(1, int(length)), device=device, dtype=dtype)
    if curve == "exponential":
        return x * x
    if curve == "s_curve":
        return x * x * (3.0 - 2.0 * x)
    return x


def fade_in_out(audio: dict, fade_in_ms: float, fade_out_ms: float, curve: str) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    env = torch.ones(waveform.shape[-1], device=waveform.device, dtype=waveform.dtype)
    fade_in = min(waveform.shape[-1], max(0, int(round(float(fade_in_ms) * sample_rate / 1000.0))))
    fade_out = min(waveform.shape[-1], max(0, int(round(float(fade_out_ms) * sample_rate / 1000.0))))
    if fade_in > 0:
        env[:fade_in] *= _fade_curve(fade_in, curve, waveform.device, waveform.dtype)
    if fade_out > 0:
        env[-fade_out:] *= torch.flip(_fade_curve(fade_out, curve, waveform.device, waveform.dtype), dims=(0,))
    return copy_audio(audio, waveform * env.view(1, 1, -1))


def audio_trim_crop(audio: dict, start_s: float, end_s: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    start = max(0, int(round(float(start_s) * sample_rate)))
    end = waveform.shape[-1] if end_s <= 0.0 else min(waveform.shape[-1], int(round(float(end_s) * sample_rate)))
    end = max(start + 1, end)
    return copy_audio(audio, waveform[..., start:end])


def silence_trimmer(audio: dict, threshold_db: float, padding_ms: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    mono = waveform.abs().amax(dim=1).squeeze(0)
    threshold = float(db_to_amp(threshold_db))
    active = torch.nonzero(mono > threshold, as_tuple=False).flatten()
    if active.numel() == 0:
        return copy_audio(audio, waveform[..., :1] * 0.0)
    padding = max(0, int(round(float(padding_ms) * sample_rate / 1000.0)))
    start = max(0, int(active[0].item()) - padding)
    end = min(waveform.shape[-1], int(active[-1].item()) + padding + 1)
    return copy_audio(audio, waveform[..., start:end])


def normalize_audio(audio: dict, mode: str, target_db: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    if mode == "rms":
        current = amp_to_db(torch.sqrt(torch.mean(waveform * waveform, dim=(1, 2), keepdim=True) + AUDIO_EPS))
    elif mode == "lufs":
        current = _integrated_lufs(_loudness_weighted(waveform, sample_rate), sample_rate).view(-1, 1, 1)
    else:
        current = amp_to_db(torch.amax(torch.abs(waveform), dim=(1, 2), keepdim=True) + AUDIO_EPS)
    gain_db = float(target_db) - current
    return copy_audio(audio, waveform * db_to_amp(gain_db))


def resample_change_sample_rate(audio: dict, target_sample_rate: int) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    data = waveform.detach().cpu().numpy().astype("float32", copy=False)
    out = resample_np(data, sample_rate, int(target_sample_rate))
    return {"waveform": torch.from_numpy(out).to(device=waveform.device, dtype=waveform.dtype), "sample_rate": int(target_sample_rate)}


def format_converter(audio: dict, mode: str) -> dict:
    waveform, _sample_rate = audio_waveform(audio)
    if mode == "stereo_duplicate":
        out = waveform.repeat(1, 2, 1)[:, :2] if waveform.shape[1] == 1 else waveform[:, :2]
    elif mode == "mono_left":
        out = waveform[:, :1]
    elif mode == "mono_right":
        out = waveform[:, 1:2] if waveform.shape[1] > 1 else waveform[:, :1]
    else:
        out = waveform.mean(dim=1, keepdim=True)
    return copy_audio(audio, out)


def audio_info(audio: dict) -> tuple[dict, int, float, int, int, str]:
    waveform, sample_rate = audio_waveform(audio)
    duration = waveform.shape[-1] / max(float(sample_rate), 1.0)
    text = json.dumps({"sample_rate": sample_rate, "duration_s": duration, "channels": waveform.shape[1], "samples": waveform.shape[-1]}, ensure_ascii=False)
    return copy_audio(audio, waveform), int(sample_rate), float(duration), int(waveform.shape[1]), int(waveform.shape[-1]), text


def delay_compensation(audio: dict, delay_samples: int, delay_ms: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    total = max(0, int(delay_samples) + int(round(float(delay_ms) * sample_rate / 1000.0)))
    return copy_audio(audio, F.pad(waveform, (total, 0)))


def loop_duplicator(audio: dict, loops: int, target_duration_s: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    if target_duration_s > 0.0:
        target = max(1, int(round(float(target_duration_s) * sample_rate)))
        repeats = max(1, int(math.ceil(target / waveform.shape[-1])))
        out = waveform.repeat(1, 1, repeats)[..., :target]
    else:
        out = waveform.repeat(1, 1, max(1, int(loops)))
    return copy_audio(audio, out)


def reverse_audio(audio: dict) -> dict:
    waveform, _sample_rate = audio_waveform(audio)
    return copy_audio(audio, torch.flip(waveform, dims=(-1,)))
