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


def _flatten_channels(waveform: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
    shape = tuple(waveform.shape)
    return waveform.reshape(-1, shape[-1]), shape


def _restore_channels(flat: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    return flat.reshape(shape)


def _frame_values(
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


def _time_coeff(time_ms: float, frame_seconds: float) -> float:
    if time_ms <= 0.0:
        return 0.0
    return math.exp(-frame_seconds / max(time_ms / 1000.0, 1.0e-6))


def _smooth_frames(
    target: torch.Tensor,
    sample_rate: int,
    frame_size: int,
    attack_ms: float,
    release_ms: float,
) -> torch.Tensor:
    if target.shape[-1] <= 1:
        return target
    frame_seconds = frame_size / max(float(sample_rate), 1.0)
    attack_coeff = _time_coeff(attack_ms, frame_seconds)
    release_coeff = _time_coeff(release_ms, frame_seconds)
    smoothed = torch.empty_like(target)
    previous = target[:, 0]
    smoothed[:, 0] = previous
    for index in range(1, target.shape[-1]):
        current = target[:, index]
        coeff = torch.where(current > previous, attack_coeff, release_coeff)
        previous = coeff * previous + (1.0 - coeff) * current
        smoothed[:, index] = previous
    return smoothed


def _expand_frames(frames: torch.Tensor, frame_size: int, length: int) -> torch.Tensor:
    return frames.repeat_interleave(frame_size, dim=-1)[..., :length]


def _meter_envelope(
    flat: torch.Tensor,
    sample_rate: int,
    attack_ms: float,
    release_ms: float,
    mode: Literal["rms", "peak"] = "rms",
    frame_ms: float = 5.0,
) -> torch.Tensor:
    frame_size = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    values, length = _frame_values(flat, frame_size, mode=mode)
    smoothed = _smooth_frames(values, sample_rate, frame_size, attack_ms, release_ms)
    return _expand_frames(smoothed, frame_size, length)


def _compressor_gain_db(
    env: torch.Tensor,
    threshold_db: float,
    ratio: float,
    knee_db: float,
) -> torch.Tensor:
    ratio = max(float(ratio), 1.0)
    level_db = amp_to_db(env)
    over_db = level_db - threshold_db
    full_gain = (1.0 / ratio - 1.0) * over_db
    if knee_db <= 0.0:
        return torch.where(over_db > 0.0, full_gain, torch.zeros_like(full_gain))

    half_knee = knee_db * 0.5
    knee_pos = over_db + half_knee
    soft_gain = (1.0 / ratio - 1.0) * (knee_pos * knee_pos) / (2.0 * knee_db)
    return torch.where(
        over_db < -half_knee,
        torch.zeros_like(full_gain),
        torch.where(over_db > half_knee, full_gain, soft_gain),
    )


def compressor(
    audio: dict,
    threshold_db: float,
    ratio: float,
    attack_ms: float,
    release_ms: float,
    knee_db: float,
    makeup_gain_db: float,
    detector: Literal["rms", "peak"] = "rms",
    mix: float = 1.0,
) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    flat, shape = _flatten_channels(waveform)
    env = _meter_envelope(flat, sample_rate, attack_ms, release_ms, detector)
    gain_db = _compressor_gain_db(env, threshold_db, ratio, knee_db) + makeup_gain_db
    wet = flat * db_to_amp(gain_db)
    mixed = flat.lerp(wet, max(0.0, min(float(mix), 1.0)))
    return copy_audio(audio, _restore_channels(mixed, shape))


def limiter(audio: dict, threshold_db: float, release_ms: float, lookahead_ms: float) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    flat, shape = _flatten_channels(waveform)
    ceiling = float(db_to_amp(threshold_db))
    env = _meter_envelope(flat, sample_rate, 0.05, release_ms, mode="peak", frame_ms=1.0)
    gain = torch.minimum(torch.ones_like(env), torch.tensor(ceiling, device=env.device, dtype=env.dtype) / (env + AUDIO_EPS))

    lookahead = max(0, int(round(sample_rate * lookahead_ms / 1000.0)))
    if lookahead > 0 and flat.shape[-1] > 1:
        shifted = F.pad(gain[..., lookahead:], (0, lookahead), value=1.0)
        gain = torch.minimum(gain, shifted)

    wet = torch.clamp(flat * gain, min=-ceiling, max=ceiling)
    return copy_audio(audio, _restore_channels(wet, shape))


def noise_gate(
    audio: dict,
    threshold_db: float,
    attack_ms: float,
    hold_ms: float,
    release_ms: float,
    range_db: float,
) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    flat, shape = _flatten_channels(waveform)
    frame_size = max(1, int(round(sample_rate * 5.0 / 1000.0)))
    values, length = _frame_values(flat, frame_size, mode="rms")
    level_db = amp_to_db(values)
    hold_frames = max(0, int(round((hold_ms / 1000.0) * sample_rate / frame_size)))

    target_db = torch.empty_like(level_db)
    hold = torch.zeros(level_db.shape[0], device=level_db.device, dtype=torch.long)
    for index in range(level_db.shape[-1]):
        open_now = level_db[:, index] >= threshold_db
        hold = torch.where(open_now, torch.full_like(hold, hold_frames), torch.clamp(hold - 1, min=0))
        target_db[:, index] = torch.where(hold > 0, 0.0, -abs(float(range_db)))

    smooth_db = _smooth_frames(target_db, sample_rate, frame_size, attack_ms, release_ms)
    gain = db_to_amp(_expand_frames(smooth_db, frame_size, length))
    return copy_audio(audio, _restore_channels(flat * gain, shape))


def expander(
    audio: dict,
    threshold_db: float,
    ratio: float,
    attack_ms: float,
    release_ms: float,
    range_db: float,
) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    flat, shape = _flatten_channels(waveform)
    env = _meter_envelope(flat, sample_rate, attack_ms, release_ms, mode="rms")
    level_db = amp_to_db(env)
    below = torch.clamp(threshold_db - level_db, min=0.0)
    gain_db = -torch.clamp((max(float(ratio), 1.0) - 1.0) * below, max=abs(float(range_db)))
    return copy_audio(audio, _restore_channels(flat * db_to_amp(gain_db), shape))


def transient_shaper(
    audio: dict,
    attack_gain_db: float,
    sustain_gain_db: float,
    fast_ms: float,
    slow_ms: float,
    mix: float,
) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    flat, shape = _flatten_channels(waveform)
    fast = _meter_envelope(flat, sample_rate, fast_ms, max(fast_ms * 2.0, 5.0), mode="peak", frame_ms=2.0)
    slow = _meter_envelope(flat, sample_rate, slow_ms, max(slow_ms * 4.0, 50.0), mode="rms", frame_ms=5.0)
    attack_curve = torch.clamp((fast - slow) / (slow + AUDIO_EPS), min=0.0, max=1.0)
    sustain_curve = torch.clamp((slow - fast) / (slow + AUDIO_EPS), min=0.0, max=1.0)
    gain_db = attack_curve * attack_gain_db + sustain_curve * sustain_gain_db
    wet = flat * db_to_amp(gain_db)
    return copy_audio(audio, _restore_channels(flat.lerp(wet, max(0.0, min(float(mix), 1.0))), shape))


def _sos_filter_waveform(waveform: torch.Tensor, sos: np.ndarray) -> torch.Tensor:
    device = waveform.device
    dtype = waveform.dtype
    flat, shape = _flatten_channels(waveform)
    data = flat.detach().cpu().numpy()
    filtered = np.empty_like(data, dtype=np.float32)
    for row in range(data.shape[0]):
        padlen = 3 * (2 * sos.shape[0] + 1)
        if data.shape[1] > padlen:
            filtered[row] = signal.sosfiltfilt(sos, data[row]).astype(np.float32)
        else:
            filtered[row] = signal.sosfilt(sos, data[row]).astype(np.float32)
    return torch.from_numpy(filtered).to(device=device, dtype=dtype).reshape(shape)


def _butter_sos(sample_rate: int, kind: str, cutoff: float | tuple[float, float], order: int = 4) -> np.ndarray:
    nyquist = sample_rate * 0.5
    if isinstance(cutoff, tuple):
        low = max(20.0, min(float(cutoff[0]), nyquist - 100.0))
        high = max(low + 20.0, min(float(cutoff[1]), nyquist - 20.0))
        wn = (low / nyquist, high / nyquist)
    else:
        wn = max(20.0 / nyquist, min(float(cutoff) / nyquist, 0.98))
    return signal.butter(order, wn, btype=kind, output="sos")


def _split_bands(waveform: torch.Tensor, sample_rate: int, bands: int, crossovers: tuple[float, float, float]) -> list[torch.Tensor]:
    nyquist = sample_rate * 0.5
    c1 = max(40.0, min(crossovers[0], nyquist - 300.0))
    c2 = max(c1 + 50.0, min(crossovers[1], nyquist - 150.0))
    c3 = max(c2 + 50.0, min(crossovers[2], nyquist - 50.0))
    if bands == 3:
        return [
            _sos_filter_waveform(waveform, _butter_sos(sample_rate, "lowpass", c1)),
            _sos_filter_waveform(waveform, _butter_sos(sample_rate, "bandpass", (c1, c2))),
            _sos_filter_waveform(waveform, _butter_sos(sample_rate, "highpass", c2)),
        ]
    return [
        _sos_filter_waveform(waveform, _butter_sos(sample_rate, "lowpass", c1)),
        _sos_filter_waveform(waveform, _butter_sos(sample_rate, "bandpass", (c1, c2))),
        _sos_filter_waveform(waveform, _butter_sos(sample_rate, "bandpass", (c2, c3))),
        _sos_filter_waveform(waveform, _butter_sos(sample_rate, "highpass", c3)),
    ]


def de_esser(
    audio: dict,
    threshold_db: float,
    ratio: float,
    attack_ms: float,
    release_ms: float,
    frequency_low_hz: float,
    frequency_high_hz: float,
    amount: float,
) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    low = min(frequency_low_hz, frequency_high_hz)
    high = max(frequency_low_hz, frequency_high_hz)
    band = _sos_filter_waveform(waveform, _butter_sos(sample_rate, "bandpass", (low, high)))
    flat_band, shape = _flatten_channels(band)
    env = _meter_envelope(flat_band, sample_rate, attack_ms, release_ms, mode="rms", frame_ms=2.5)
    gain_db = _compressor_gain_db(env, threshold_db, ratio, knee_db=3.0)
    wet_band = _restore_channels(flat_band * db_to_amp(gain_db), shape)
    wet = waveform + (wet_band - band) * max(0.0, min(float(amount), 1.0))
    return copy_audio(audio, wet)


def multiband_compressor(
    audio: dict,
    bands: int,
    crossover_low_hz: float,
    crossover_mid_hz: float,
    crossover_high_hz: float,
    low_threshold_db: float,
    low_ratio: float,
    low_makeup_db: float,
    low_mid_threshold_db: float,
    low_mid_ratio: float,
    low_mid_makeup_db: float,
    high_mid_threshold_db: float,
    high_mid_ratio: float,
    high_mid_makeup_db: float,
    high_threshold_db: float,
    high_ratio: float,
    high_makeup_db: float,
    attack_ms: float,
    release_ms: float,
    knee_db: float,
    mix: float,
) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    band_count = 3 if str(bands) == "3" or bands == 3 else 4
    split = _split_bands(waveform, sample_rate, band_count, (crossover_low_hz, crossover_mid_hz, crossover_high_hz))
    params = [
        (low_threshold_db, low_ratio, low_makeup_db),
        (low_mid_threshold_db, low_mid_ratio, low_mid_makeup_db),
        (high_mid_threshold_db, high_mid_ratio, high_mid_makeup_db),
        (high_threshold_db, high_ratio, high_makeup_db),
    ]
    if band_count == 3:
        params = [params[0], params[1], params[3]]

    processed = []
    for band, (threshold_db, ratio, makeup_db) in zip(split, params, strict=True):
        flat, shape = _flatten_channels(band)
        env = _meter_envelope(flat, sample_rate, attack_ms, release_ms, mode="rms")
        gain_db = _compressor_gain_db(env, threshold_db, ratio, knee_db) + makeup_db
        processed.append(_restore_channels(flat * db_to_amp(gain_db), shape))

    wet = torch.stack(processed, dim=0).sum(dim=0)
    return copy_audio(audio, waveform.lerp(wet, max(0.0, min(float(mix), 1.0))))


def auto_gain_leveler(
    audio: dict,
    mode: Literal["RMS", "Peak"],
    target_db: float,
    window_ms: float,
    attack_ms: float,
    release_ms: float,
    min_gain_db: float,
    max_gain_db: float,
) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    flat, shape = _flatten_channels(waveform)
    frame_size = max(1, int(round(sample_rate * max(window_ms, 1.0) / 1000.0)))
    values, length = _frame_values(flat, frame_size, mode="peak" if mode == "Peak" else "rms")
    gain_db = target_db - amp_to_db(values)
    low = min(float(min_gain_db), float(max_gain_db))
    high = max(float(min_gain_db), float(max_gain_db))
    gain_db = torch.clamp(gain_db, min=low, max=high)
    smooth_db = _smooth_frames(gain_db, sample_rate, frame_size, attack_ms, release_ms)
    gain = db_to_amp(_expand_frames(smooth_db, frame_size, length))
    return copy_audio(audio, _restore_channels(flat * gain, shape))


def _high_shelf_biquad(sample_rate: int, frequency: float, gain_db: float, q: float = 0.7071) -> tuple[np.ndarray, np.ndarray]:
    a = 10.0 ** (gain_db / 40.0)
    omega = 2.0 * math.pi * frequency / sample_rate
    sin_omega = math.sin(omega)
    cos_omega = math.cos(omega)
    alpha = sin_omega / (2.0 * q)
    sqrt_a = math.sqrt(a)
    b0 = a * ((a + 1.0) + (a - 1.0) * cos_omega + 2.0 * sqrt_a * alpha)
    b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_omega)
    b2 = a * ((a + 1.0) + (a - 1.0) * cos_omega - 2.0 * sqrt_a * alpha)
    a0 = (a + 1.0) - (a - 1.0) * cos_omega + 2.0 * sqrt_a * alpha
    a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_omega)
    a2 = (a + 1.0) - (a - 1.0) * cos_omega - 2.0 * sqrt_a * alpha
    return np.array([b0, b1, b2], dtype=np.float64) / a0, np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)


def _loudness_weighted(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    device = waveform.device
    dtype = waveform.dtype
    flat, shape = _flatten_channels(waveform)
    data = flat.detach().cpu().numpy()
    b_shelf, a_shelf = _high_shelf_biquad(sample_rate, min(1681.974, sample_rate * 0.45), 4.0)
    sos_hp = _butter_sos(sample_rate, "highpass", 38.0, order=2)
    out = np.empty_like(data, dtype=np.float32)
    for row in range(data.shape[0]):
        y = signal.lfilter(b_shelf, a_shelf, data[row])
        y = signal.sosfilt(sos_hp, y)
        out[row] = y.astype(np.float32)
    return torch.from_numpy(out).to(device=device, dtype=dtype).reshape(shape)


def _channel_weights(channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    base = [1.0, 1.0, 1.0, 1.41, 1.41]
    weights = base[:channels] + [1.0] * max(0, channels - len(base))
    return torch.tensor(weights, device=device, dtype=dtype).view(1, channels, 1)


def _weighted_channel_power(weighted: torch.Tensor) -> torch.Tensor:
    weights = _channel_weights(weighted.shape[1], weighted.device, weighted.dtype)
    return torch.mean((weighted * weighted) * weights, dim=2).sum(dim=1)


def _power_to_lufs(power: torch.Tensor) -> torch.Tensor:
    return -0.691 + 10.0 * torch.log10(torch.clamp(power, min=AUDIO_EPS))


def _integrated_lufs(weighted: torch.Tensor, sample_rate: int) -> torch.Tensor:
    _batch, _channels, length = weighted.shape
    block = max(1, int(round(0.400 * sample_rate)))
    hop = max(1, int(round(0.100 * sample_rate)))
    if length <= block:
        return _power_to_lufs(_weighted_channel_power(weighted))

    powers = []
    for start in range(0, length - block + 1, hop):
        segment = weighted[..., start : start + block]
        powers.append(_weighted_channel_power(segment))
    block_power = torch.stack(powers, dim=-1)
    block_lufs = _power_to_lufs(block_power)
    abs_mask = block_lufs > -70.0
    safe_power = torch.where(abs_mask, block_power, torch.zeros_like(block_power))
    counts = torch.clamp(abs_mask.sum(dim=-1), min=1)
    prelim_power = safe_power.sum(dim=-1) / counts
    relative_gate = _power_to_lufs(prelim_power) - 10.0
    rel_mask = abs_mask & (block_lufs > relative_gate[:, None])
    gated_power = torch.where(rel_mask, block_power, torch.zeros_like(block_power))
    final_counts = torch.clamp(rel_mask.sum(dim=-1), min=1)
    return _power_to_lufs(gated_power.sum(dim=-1) / final_counts)


def _batch_peak_limit(waveform: torch.Tensor, ceiling_db: float) -> torch.Tensor:
    ceiling = float(db_to_amp(ceiling_db))
    sample_peak = waveform.abs().amax(dim=(1, 2), keepdim=True)
    data = waveform.detach().cpu().numpy()
    true_peaks = []
    for batch in range(data.shape[0]):
        peak = float(np.max(np.abs(data[batch])))
        if data.shape[-1] > 1:
            for channel in range(data.shape[1]):
                oversampled = signal.resample_poly(data[batch, channel], 4, 1)
                peak = max(peak, float(np.max(np.abs(oversampled))))
        true_peaks.append(peak)
    true_peak = torch.tensor(true_peaks, device=waveform.device, dtype=waveform.dtype).view(-1, 1, 1)
    peak = torch.maximum(sample_peak, true_peak)
    gain = torch.minimum(torch.ones_like(peak), torch.tensor(ceiling, device=waveform.device, dtype=waveform.dtype) / (peak + AUDIO_EPS))
    return waveform * gain


def loudness_normalizer(
    audio: dict,
    target_lufs: float,
    measurement: Literal["integrated", "short_term", "momentary"],
    min_gain_db: float,
    max_gain_db: float,
    true_peak_ceiling_db: float,
) -> dict:
    waveform, sample_rate = audio_waveform(audio)
    weighted = _loudness_weighted(waveform, sample_rate)
    low = min(float(min_gain_db), float(max_gain_db))
    high = max(float(min_gain_db), float(max_gain_db))

    if measurement == "integrated":
        current = _integrated_lufs(weighted, sample_rate).view(-1, 1, 1)
        gain_db = torch.clamp(torch.tensor(target_lufs, device=waveform.device, dtype=waveform.dtype) - current, min=low, max=high)
        return copy_audio(audio, _batch_peak_limit(waveform * db_to_amp(gain_db), true_peak_ceiling_db))

    window_seconds = 3.0 if measurement == "short_term" else 0.4
    frame_size = max(1, int(round(window_seconds * sample_rate)))
    hop_size = max(1, int(round(0.100 * sample_rate if measurement == "short_term" else 0.050 * sample_rate)))
    batch, _channels, length = waveform.shape
    frame_powers = []
    for start in range(0, max(1, length - frame_size + 1), hop_size):
        segment = weighted[..., start : min(start + frame_size, length)]
        frame_powers.append(_weighted_channel_power(segment))
    if not frame_powers:
        frame_powers.append(_weighted_channel_power(weighted))
    lufs = _power_to_lufs(torch.stack(frame_powers, dim=-1))
    gain_db = torch.clamp(target_lufs - lufs, min=low, max=high)
    gain = _expand_frames(gain_db, hop_size, length).view(batch, 1, length)
    normalized = waveform * db_to_amp(gain)
    return copy_audio(audio, _batch_peak_limit(normalized, true_peak_ceiling_db))
