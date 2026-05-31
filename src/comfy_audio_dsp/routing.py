from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .common import AUDIO_EPS, audio_waveform, copy_audio, db_to_amp, meter_envelope
from .dynamics import _compressor_gain_db


def _empty_like(audio: dict) -> dict:
    waveform, _sample_rate = audio_waveform(audio)
    return copy_audio(audio, torch.zeros_like(waveform))


def _match_audio(audio: dict | None, sample_rate: int, length: int, channels: int, like: torch.Tensor) -> torch.Tensor:
    if audio is None:
        return torch.zeros(like.shape[0], channels, length, device=like.device, dtype=like.dtype)
    waveform, input_rate = audio_waveform(audio)
    if input_rate != sample_rate:
        raise ValueError("Comfy-Audio-DSP: routing nodes require matching sample rates.")
    if waveform.shape[-1] < length:
        waveform = F.pad(waveform, (0, length - waveform.shape[-1]))
    elif waveform.shape[-1] > length:
        waveform = waveform[..., :length]
    if waveform.shape[1] < channels:
        repeats = int(math.ceil(channels / waveform.shape[1]))
        waveform = waveform.repeat(1, repeats, 1)[:, :channels]
    elif waveform.shape[1] > channels:
        waveform = waveform[:, :channels]
    return waveform.to(device=like.device, dtype=like.dtype)


def _pan_stereo(waveform: torch.Tensor, pan: float) -> torch.Tensor:
    if waveform.shape[1] == 1:
        waveform = waveform.repeat(1, 2, 1)
    pan = max(-1.0, min(float(pan), 1.0))
    angle = (pan + 1.0) * math.pi * 0.25
    out = waveform[:, :2].clone()
    out[:, :1] *= math.cos(angle)
    out[:, 1:2] *= math.sin(angle)
    if waveform.shape[1] > 2:
        out = torch.cat([out, waveform[:, 2:]], dim=1)
    return out


def audio_mixer(audio_1: dict, tracks: list[tuple[dict | None, float, float, bool, bool]], master_gain_db: float) -> dict:
    base, sample_rate = audio_waveform(audio_1)
    length = max([base.shape[-1]] + [audio_waveform(track[0])[0].shape[-1] for track in tracks if track[0] is not None])
    channels = max(2, max([base.shape[1]] + [audio_waveform(track[0])[0].shape[1] for track in tracks if track[0] is not None]))
    solo_active = any(solo for _audio, _gain, _pan, _mute, solo in tracks)
    out = torch.zeros(base.shape[0], channels, length, device=base.device, dtype=base.dtype)
    for track_audio, gain_db, pan, mute, solo in tracks:
        if track_audio is None or bool(mute) or (solo_active and not bool(solo)):
            continue
        track = _match_audio(track_audio, sample_rate, length, channels, base) * float(db_to_amp(gain_db))
        out += _pan_stereo(track, pan)
    out *= float(db_to_amp(master_gain_db))
    peak = torch.amax(torch.abs(out), dim=(1, 2), keepdim=True)
    out = torch.where(peak > 1.0, out / (peak + AUDIO_EPS), out)
    return copy_audio(audio_1, out)


def audio_selector(index: int, audios: list[dict | None]) -> dict:
    available = [audio for audio in audios if audio is not None]
    if not available:
        raise ValueError("Comfy-Audio-DSP: selector needs at least one audio input.")
    selected = max(1, min(int(index), len(audios))) - 1
    return audios[selected] if audios[selected] is not None else _empty_like(available[0])


def audio_splitter(audio: dict) -> tuple[dict, dict, dict, dict]:
    waveform, _sample_rate = audio_waveform(audio)
    outputs = []
    for index in range(4):
        channel = waveform[:, index : index + 1] if index < waveform.shape[1] else torch.zeros_like(waveform[:, :1])
        outputs.append(copy_audio(audio, channel))
    return tuple(outputs)


def audio_merger(audio_1: dict, audios: list[dict | None], output_mode: str) -> dict:
    base, sample_rate = audio_waveform(audio_1)
    inputs = [audio for audio in [audio_1, *audios] if audio is not None]
    length = max(audio_waveform(audio)[0].shape[-1] for audio in inputs)
    channels = 2 if output_mode == "stereo" else len(inputs)
    out_channels = []
    for audio in inputs:
        wave = _match_audio(audio, sample_rate, length, 1, base)
        out_channels.append(wave[:, :1])
    if output_mode == "stereo":
        left = sum(out_channels[0::2]) if out_channels[0::2] else torch.zeros_like(out_channels[0])
        right = sum(out_channels[1::2]) if out_channels[1::2] else left
        out = torch.cat([left, right], dim=1)
    else:
        out = torch.cat(out_channels[:channels], dim=1)
    return copy_audio(audio_1, out)


def crossfader(audio_a: dict, audio_b: dict, fade: float, equal_power: bool) -> dict:
    a, sample_rate = audio_waveform(audio_a)
    b = _match_audio(audio_b, sample_rate, max(a.shape[-1], audio_waveform(audio_b)[0].shape[-1]), max(a.shape[1], audio_waveform(audio_b)[0].shape[1]), a)
    a = _match_audio(audio_a, sample_rate, b.shape[-1], b.shape[1], a)
    fade = max(0.0, min(float(fade), 1.0))
    if bool(equal_power):
        a_gain = math.cos(fade * math.pi * 0.5)
        b_gain = math.sin(fade * math.pi * 0.5)
    else:
        a_gain = 1.0 - fade
        b_gain = fade
    return copy_audio(audio_a, a * a_gain + b * b_gain)


def sidechain_gate_compressor(
    audio: dict,
    sidechain: dict,
    mode: str,
    threshold_db: float,
    ratio: float,
    attack_ms: float,
    release_ms: float,
    range_db: float,
    mix: float,
) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    key = _match_audio(sidechain, sample_rate, waveform.shape[-1], waveform.shape[1], waveform)
    key_flat = key.reshape(-1, key.shape[-1])
    env = meter_envelope(key_flat, sample_rate, attack_ms, release_ms, mode="rms")
    if mode == "gate":
        level = 20.0 * torch.log10(torch.clamp(env, min=AUDIO_EPS))
        gain_db = torch.where(level >= float(threshold_db), torch.zeros_like(level), torch.full_like(level, -abs(float(range_db))))
    else:
        gain_db = _compressor_gain_db(env, threshold_db, ratio, knee_db=3.0)
    gain = db_to_amp(gain_db).reshape(waveform.shape)
    wet = waveform * gain
    return copy_audio(audio, waveform.lerp(wet, max(0.0, min(float(mix), 1.0))))


def send_return_loop(audio: dict, return_audio: dict, send_level_db: float, return_level_db: float, dry_level_db: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    returned = _match_audio(return_audio, sample_rate, waveform.shape[-1], waveform.shape[1], waveform)
    out = waveform * float(db_to_amp(dry_level_db)) + returned * float(db_to_amp(send_level_db)) * float(db_to_amp(return_level_db))
    return copy_audio(audio, out)


def multiband_crossover(audio: dict, bands: str, crossover_low_hz: float, crossover_mid_hz: float, crossover_high_hz: float) -> tuple[dict, dict, dict, dict]:
    waveform, sample_rate = audio_waveform(audio)
    nyquist = sample_rate * 0.5
    c1 = max(20.0, min(float(crossover_low_hz), nyquist - 100.0))
    c2 = max(c1 + 20.0, min(float(crossover_mid_hz), nyquist - 50.0))
    c3 = max(c2 + 20.0, min(float(crossover_high_hz), nyquist - 20.0))
    spectrum = torch.fft.rfft(waveform, dim=-1)
    freqs = torch.fft.rfftfreq(waveform.shape[-1], d=1.0 / sample_rate).to(device=waveform.device)
    if str(bands) == "3":
        masks = [freqs <= c1, (freqs > c1) & (freqs <= c2), freqs > c2, torch.zeros_like(freqs, dtype=torch.bool)]
    else:
        masks = [freqs <= c1, (freqs > c1) & (freqs <= c2), (freqs > c2) & (freqs <= c3), freqs > c3]
    outputs = []
    for mask in masks:
        band = torch.fft.irfft(spectrum * mask.view(1, 1, -1), n=waveform.shape[-1], dim=-1)
        outputs.append(copy_audio(audio, band))
    return tuple(outputs)
