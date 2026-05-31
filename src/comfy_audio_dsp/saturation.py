from __future__ import annotations

import torch
import torch.nn.functional as F

from .common import audio_waveform, butter_sos, copy_audio, db_to_amp, mix_audio, sos_filter_waveform


def _tone_stack(waveform: torch.Tensor, sample_rate: int, tone: float) -> torch.Tensor:
    cutoff = 800.0 + max(0.0, min(float(tone), 1.0)) * 11000.0
    return sos_filter_waveform(waveform, butter_sos(sample_rate, "lowpass", cutoff, order=2), zero_phase=False)


def soft_clipper(audio: dict, drive_db: float, curve: str, output_gain_db: float, mix: float) -> dict:
    waveform, _sample_rate = audio_waveform(audio)
    driven = waveform * float(db_to_amp(drive_db))
    if curve == "cubic":
        wet = torch.clamp(driven - (driven**3) / 3.0, min=-2.0 / 3.0, max=2.0 / 3.0) * 1.5
    else:
        wet = torch.tanh(driven)
    wet = wet * float(db_to_amp(output_gain_db))
    return copy_audio(audio, mix_audio(waveform, wet, mix))


def hard_clipper(audio: dict, threshold: float, output_gain_db: float, mix: float) -> dict:
    waveform, _sample_rate = audio_waveform(audio)
    threshold = max(0.01, float(threshold))
    wet = torch.clamp(waveform, min=-threshold, max=threshold) / threshold
    wet = wet * float(db_to_amp(output_gain_db))
    return copy_audio(audio, mix_audio(waveform, wet, mix))


def tube_saturation(audio: dict, drive_db: float, asymmetry: float, bias: float, output_gain_db: float, mix: float) -> dict:
    waveform, _sample_rate = audio_waveform(audio)
    x = waveform * float(db_to_amp(drive_db)) + float(bias)
    pos = torch.tanh(x * (1.0 + max(0.0, float(asymmetry))))
    neg = torch.tanh(x * (1.0 - min(0.9, max(0.0, float(asymmetry)))))
    wet = torch.where(x >= 0.0, pos, neg) - torch.tanh(torch.tensor(float(bias), device=waveform.device, dtype=waveform.dtype))
    wet = wet * float(db_to_amp(output_gain_db))
    return copy_audio(audio, mix_audio(waveform, wet, mix))


def tape_saturation(audio: dict, drive_db: float, compression: float, tone: float, wow: float, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    x = waveform * float(db_to_amp(drive_db))
    wet = torch.tanh(x + 0.15 * x * x) / 1.15
    comp = max(0.0, min(float(compression), 1.0))
    kernel = max(1, sample_rate // 100)
    if kernel % 2 == 0:
        kernel += 1
    envelope = torch.sqrt(F.avg_pool1d((wet.reshape(-1, 1, wet.shape[-1]) ** 2), kernel_size=kernel, stride=1, padding=kernel // 2).reshape_as(wet) + 1.0e-8)
    wet = wet / (1.0 + comp * envelope * 3.0)
    if wow > 0.0:
        t = torch.arange(wet.shape[-1], device=wet.device, dtype=wet.dtype) / float(sample_rate)
        wet = wet * (1.0 + 0.015 * float(wow) * torch.sin(2.0 * torch.pi * 0.45 * t).view(1, 1, -1))
    wet = _tone_stack(wet, sample_rate, tone)
    return copy_audio(audio, mix_audio(waveform, wet, mix))


def fuzz(audio: dict, drive_db: float, gate: float, tone: float, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    x = waveform * float(db_to_amp(drive_db))
    wet = torch.sign(x) * (1.0 - torch.exp(-torch.abs(x) * 6.0))
    if gate > 0.0:
        wet = torch.where(torch.abs(waveform) < float(gate), torch.zeros_like(wet), wet)
    wet = _tone_stack(wet, sample_rate, tone)
    return copy_audio(audio, mix_audio(waveform, wet, mix))


def bit_crusher(audio: dict, bit_depth: int, downsample_factor: int, mix: float) -> dict:
    waveform, _sample_rate = audio_waveform(audio)
    levels = max(2, 2 ** max(1, int(bit_depth)) - 1)
    wet = torch.round(torch.clamp(waveform, -1.0, 1.0) * levels) / levels
    factor = max(1, int(downsample_factor))
    if factor > 1 and wet.shape[-1] > 1:
        held = wet[..., ::factor]
        wet = held.repeat_interleave(factor, dim=-1)[..., : wet.shape[-1]]
    return copy_audio(audio, mix_audio(waveform, wet, mix))


def overdrive_distortion(audio: dict, drive_db: float, tone: float, mode: str, output_gain_db: float, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    x = waveform * float(db_to_amp(drive_db))
    if mode == "distortion":
        wet = torch.tanh(x * 2.5)
    else:
        wet = x / (1.0 + torch.abs(x))
    wet = _tone_stack(wet, sample_rate, tone) * float(db_to_amp(output_gain_db))
    return copy_audio(audio, mix_audio(waveform, wet, mix))


def wavefolder(audio: dict, drive_db: float, folds: float, mix: float) -> dict:
    waveform, _sample_rate = audio_waveform(audio)
    x = waveform * float(db_to_amp(drive_db)) * max(1.0, float(folds))
    wet = torch.asin(torch.sin(x)) * (2.0 / torch.pi)
    return copy_audio(audio, mix_audio(waveform, wet, mix))


def exciter_enhancer(audio: dict, drive_db: float, crossover_hz: float, amount: float, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    high = sos_filter_waveform(waveform, butter_sos(sample_rate, "highpass", crossover_hz, order=2), zero_phase=False)
    harmonics = torch.tanh(high * float(db_to_amp(drive_db)))
    wet = waveform + harmonics * max(0.0, min(float(amount), 2.0))
    return copy_audio(audio, mix_audio(waveform, wet, mix))


def fold_clip(audio: dict, drive_db: float, fold_amount: float, clip_threshold: float, mode: str, output_gain_db: float, mix: float) -> dict:
    waveform, _sample_rate = audio_waveform(audio)
    x = waveform * float(db_to_amp(drive_db))
    folded = torch.asin(torch.sin(x * max(1.0, float(fold_amount)))) * (2.0 / torch.pi)
    threshold = max(0.01, float(clip_threshold))
    clipped = torch.clamp(x, -threshold, threshold) / threshold
    if mode == "clip_then_fold":
        wet = torch.asin(torch.sin(clipped * max(1.0, float(fold_amount)))) * (2.0 / torch.pi)
    elif mode == "fold_then_clip":
        wet = torch.clamp(folded, -threshold, threshold) / threshold
    else:
        wet = 0.5 * folded + 0.5 * clipped
    wet = wet * float(db_to_amp(output_gain_db))
    return copy_audio(audio, mix_audio(waveform, wet, mix))


def amp_simulator(audio: dict, drive_db: float, tone: float, cabinet: str, presence: float, output_gain_db: float, mix: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    x = waveform * float(db_to_amp(drive_db))
    wet = torch.tanh(x) + 0.18 * torch.tanh(x * 3.0)
    wet = wet / 1.18
    low_cut = 70.0 if cabinet == "open_back" else 95.0
    high_cut = 4200.0 + max(0.0, min(float(tone), 1.0)) * 5200.0
    wet = sos_filter_waveform(wet, butter_sos(sample_rate, "highpass", low_cut, order=2), zero_phase=False)
    wet = sos_filter_waveform(wet, butter_sos(sample_rate, "lowpass", high_cut, order=4), zero_phase=False)
    if presence > 0.0:
        high = sos_filter_waveform(wet, butter_sos(sample_rate, "highpass", 2600.0, order=2), zero_phase=False)
        wet = wet + high * max(0.0, min(float(presence), 2.0)) * 0.35
    wet = wet * float(db_to_amp(output_gain_db))
    return copy_audio(audio, mix_audio(waveform, wet, mix))
